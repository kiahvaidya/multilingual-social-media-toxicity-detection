"""
Moderation domain logic and risk assessment for social media toxicity detection.

Provides:
  - 3-tier risk classification (🟢 Safe, 🟡 Review, 🔴 High Risk)
  - Assistive human review state handling (Review, Confirmed Toxic, False Positive)
  - Aggregate statistics calculation for comment batches
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.config import config

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """3-tier risk classification for assistive moderation."""
    SAFE = "Safe"
    REVIEW = "Review"
    HIGH_RISK = "High Risk"


class ReviewStatus(str, Enum):
    """Session-only human review markers."""
    PENDING = "Pending"
    REVIEW = "Review"
    CONFIRMED_TOXIC = "Confirmed Toxic"
    FALSE_POSITIVE = "False Positive"


# Categories deemed severe that automatically elevate flagged content to High Risk
SEVERE_CATEGORIES = {"severe_toxic", "threat", "identity_hate"}

# Margin below threshold to trigger a Review state for borderline scores
BORDERLINE_MARGIN = 0.08
HIGH_RISK_PROB_THRESHOLD = 0.75

# Common lexical slang / cathartic profanity markers that frequently cause false positives
SLANG_PROFANITY_MARKERS = [
    "kill", "killed", "sick", "fuck", "shit", "bawaal", "धांसू", "gadar",
    "bc", "saale", "zeher", "dead", "destroy", "murder", "c00l", "sh!rt"
]


@dataclass
class AmbiguityGateConfig:
    """Configurable parameters for JEV ambiguity gate routing."""
    borderline_margin: float = BORDERLINE_MARGIN
    elevated_sub_threshold: float = 0.40
    high_risk_prob_threshold: float = HIGH_RISK_PROB_THRESHOLD
    enable_slang_lexical_check: bool = True
    bypass_extreme_threat_threshold: float = 0.85


def should_invoke_jev(
    text: str,
    predictions: Dict[str, float],
    is_toxic_flags: Dict[str, bool],
    risk_level: RiskLevel,
    thresholds: Dict[str, float],
    gate_config: Optional[AmbiguityGateConfig] = None,
) -> Tuple[bool, str]:
    """Determine whether a comment is ambiguous and eligible for JEV contextual analysis.

    Eligibility criteria:
      1. MuRIL assigned RiskLevel.REVIEW (single moderate category, borderline, elevated sub-threshold).
      2. Any category score is within borderline_margin of its threshold.
      3. Multiple categories triggered with moderate probabilities (< 0.75) where contextual intent is uncertain.
      4. Profanity or slang markers are present in the text with moderate or elevated scores.
      5. Explicitly excludes clear safe comments (all scores << thresholds) and unambiguous extreme threats.

    Returns
    -------
    (should_invoke, reason) : tuple[bool, str]
    """
    cfg = gate_config or AmbiguityGateConfig()
    max_prob = max(predictions.values()) if predictions else 0.0
    flagged_cats = [c for c, f in is_toxic_flags.items() if f]
    t_lower = str(text).lower()

    # 1. Unambiguous Extreme Severe Toxicity Bypass:
    # If a severe category is flagged with extreme confidence (>= 0.85) without praise indicators
    has_severe = any(c in SEVERE_CATEGORIES for c in flagged_cats)
    praise_indicators = [
        "killed it", "totally sick", "bawaal", "धांसू", "gadar", "super cool",
        "super c00l", "dil jeet liya", "fuck yeah", "went so hard"
    ]
    has_praise_context = any(p in t_lower for p in praise_indicators)
    if has_severe and max_prob >= cfg.bypass_extreme_threat_threshold and not has_praise_context:
        return False, f"Clear high-risk severe toxicity (score: {max_prob:.2f} >= {cfg.bypass_extreme_threat_threshold:.2f}); bypassing JEV."

    # 2. Review Tier Trigger: comments in Review state are primary candidates
    if risk_level == RiskLevel.REVIEW:
        return True, "MuRIL assigned Review status; context requires disambiguation."

    # 3. Score Proximity to Decision Threshold
    borderline_cats = []
    for cat, prob in predictions.items():
        th = thresholds.get(cat, 0.5)
        if abs(prob - th) <= cfg.borderline_margin:
            borderline_cats.append(f"{cat} ({prob:.2f} near {th:.2f})")

    if borderline_cats:
        return True, f"Score proximity to decision threshold: {', '.join(borderline_cats)}"

    # 4. Slang / Cathartic Profanity Presence with Moderate Toxicity Signals
    if cfg.enable_slang_lexical_check:
        has_marker = any(m in t_lower for m in SLANG_PROFANITY_MARKERS)
        if has_marker and max_prob >= 0.30:
            return True, f"Colloquial slang or profanity marker detected with elevated score ({max_prob:.2f}); intent uncertain."

    # 5. Multiple categories flagged with non-extreme probability (< 0.75)
    if len(flagged_cats) >= 2 and max_prob < cfg.high_risk_prob_threshold:
        return True, f"Multiple categories flagged ({', '.join(flagged_cats)}) with moderate confidence; verifying contextual intent."

    # 6. Default: Clear case (either clear Safe or unambiguous High Risk)
    return False, f"Clear {risk_level.value} classification; bypassing JEV."


@dataclass
class ContextualModerationResult:
    """Consolidated result combining MuRIL multi-label outputs and JEV contextual moderation."""
    comment_text: str
    muril_predictions: Dict[str, float]
    is_toxic_flags: Dict[str, bool]
    thresholds: Dict[str, float]
    base_risk_level: RiskLevel
    base_badge_icon: str
    base_explanation: str
    is_ambiguous: bool
    gate_reason: str
    jev_invoked: bool
    jev_status: str  # e.g. "INVOKED_REAL", "INVOKED_MOCK", "BYPASSED_CLEAR_CASE", "UNCONFIGURED", "DISABLED", "FAILED"
    final_risk_level: RiskLevel
    final_badge_icon: str
    final_explanation: str
    jev_decision: Optional[Any] = None
    jev_error: Optional[str] = None


def moderate_comment_with_context(
    text: str,
    predictions: Dict[str, float],
    is_toxic_flags: Dict[str, bool],
    thresholds: Dict[str, float],
    jev_provider: Optional[Any] = None,
    gate_config: Optional[AmbiguityGateConfig] = None,
) -> ContextualModerationResult:
    """Execute complete assistive moderation workflow: Base Moderation -> Ambiguity Gate -> JEV -> Final Tier.

    Parameters
    ----------
    text : str
        Raw comment text.
    predictions : dict[str, float]
        MuRIL per-label probabilities.
    is_toxic_flags : dict[str, bool]
        MuRIL binary flags evaluated against frozen thresholds.
    thresholds : dict[str, float]
        Active decision thresholds.
    jev_provider : JEVDecisionProvider, optional
        Explicit provider instance. If None, resolved via get_jev_provider().
    gate_config : AmbiguityGateConfig, optional
        Customized gate parameters.

    Returns
    -------
    ContextualModerationResult
        Consolidated decision object.
    """
    # 1. Base 3-tier risk assessment
    base_risk, base_icon, base_expl = classify_risk(predictions, is_toxic_flags, thresholds)

    # 2. Ambiguity evaluation
    is_ambiguous, gate_reason = should_invoke_jev(
        text, predictions, is_toxic_flags, base_risk, thresholds, gate_config
    )

    if not is_ambiguous:
        return ContextualModerationResult(
            comment_text=text,
            muril_predictions=predictions,
            is_toxic_flags=is_toxic_flags,
            thresholds=thresholds,
            base_risk_level=base_risk,
            base_badge_icon=base_icon,
            base_explanation=base_expl,
            is_ambiguous=False,
            gate_reason=gate_reason,
            jev_invoked=False,
            jev_status="BYPASSED_CLEAR_CASE",
            final_risk_level=base_risk,
            final_badge_icon=base_icon,
            final_explanation=base_expl,
        )

    # 3. Ambiguous Case -> JEV Provider Resolution
    from src.jev import get_jev_provider, JEVProviderType

    provider = jev_provider if jev_provider is not None else get_jev_provider()

    # If provider is unconfigured or disabled
    if provider is None:
        jev_status = "DISABLED" if not config.JEV_ENABLED else "UNCONFIGURED"
        return ContextualModerationResult(
            comment_text=text,
            muril_predictions=predictions,
            is_toxic_flags=is_toxic_flags,
            thresholds=thresholds,
            base_risk_level=base_risk,
            base_badge_icon=base_icon,
            base_explanation=base_expl,
            is_ambiguous=True,
            gate_reason=gate_reason,
            jev_invoked=False,
            jev_status=jev_status,
            final_risk_level=base_risk,
            final_badge_icon=base_icon,
            final_explanation=f"{base_expl} (JEV {jev_status.lower()}; relying on base moderation)",
            jev_error=f"JEV {jev_status.lower()}",
        )

    # 4. Invoke JEV with Failure Safety
    try:
        decision = provider.assess_context(text, predictions, is_toxic_flags, base_risk)
        jev_status = "INVOKED_REAL" if provider.provider_type == JEVProviderType.REAL_JEV else "INVOKED_MOCK"

        # Badge icon mapping
        if decision.moderation == RiskLevel.HIGH_RISK:
            final_icon = "🔴"
        elif decision.moderation == RiskLevel.REVIEW:
            final_icon = "🟡"
        else:
            final_icon = "🟢"

        final_expl = f"Contextual Assessment: {decision.reason} [Intent: {decision.intent.value}, Confidence: {decision.confidence:.2f}]"

        return ContextualModerationResult(
            comment_text=text,
            muril_predictions=predictions,
            is_toxic_flags=is_toxic_flags,
            thresholds=thresholds,
            base_risk_level=base_risk,
            base_badge_icon=base_icon,
            base_explanation=base_expl,
            is_ambiguous=True,
            gate_reason=gate_reason,
            jev_invoked=True,
            jev_status=jev_status,
            final_risk_level=decision.moderation,
            final_badge_icon=final_icon,
            final_explanation=final_expl,
            jev_decision=decision,
        )

    except Exception as exc:
        logger.warning("JEV execution failed: %s. Gracefully falling back to base moderation.", exc)
        # Safe fallback: NEVER silently classify failed JEV as SAFE
        return ContextualModerationResult(
            comment_text=text,
            muril_predictions=predictions,
            is_toxic_flags=is_toxic_flags,
            thresholds=thresholds,
            base_risk_level=base_risk,
            base_badge_icon=base_icon,
            base_explanation=base_expl,
            is_ambiguous=True,
            gate_reason=gate_reason,
            jev_invoked=True,
            jev_status="FAILED",
            final_risk_level=base_risk,
            final_badge_icon=base_icon,
            final_explanation=f"{base_expl} (JEV call failed: {exc}; fallback to base moderation)",
            jev_error=str(exc),
        )


def classify_risk(
    predictions: Dict[str, float],
    is_toxic_flags: Dict[str, bool],
    thresholds: Dict[str, float],
) -> Tuple[RiskLevel, str, str]:
    """Classify the risk level of an analyzed comment into one of three tiers.

    Tiers:
      - 🔴 High Risk: Severe categories triggered (severe_toxic, threat, identity_hate),
        or multiple categories triggered, or high confidence (>= 0.75).
      - 🟡 Review: Single moderate category triggered, or unflagged but borderline
        score within BORDERLINE_MARGIN of threshold or score >= 0.40.
      - 🟢 Safe: No categories triggered and all probabilities well below thresholds.

    Parameters
    ----------
    predictions : dict[str, float]
        Per-label probabilities (0.0 - 1.0).
    is_toxic_flags : dict[str, bool]
        Per-label boolean flags evaluated against thresholds.
    thresholds : dict[str, float]
        Active thresholds used for evaluation.

    Returns
    -------
    risk_level : RiskLevel
        Risk tier (SAFE, REVIEW, or HIGH_RISK).
    badge_icon : str
        Emoji icon ("🟢", "🟡", or "🔴").
    explanation : str
        Human-readable rationale for the assigned risk level.
    """
    flagged_cats = [cat for cat, flag in is_toxic_flags.items() if flag]
    max_prob = max(predictions.values()) if predictions else 0.0

    # 1. Check for High Risk triggers
    severe_flagged = [c for c in flagged_cats if c in SEVERE_CATEGORIES]
    if severe_flagged:
        return (
            RiskLevel.HIGH_RISK,
            "🔴",
            f"Severe toxicity flagged: {', '.join(severe_flagged)}",
        )

    if len(flagged_cats) >= 2:
        return (
            RiskLevel.HIGH_RISK,
            "🔴",
            f"Multiple toxicity categories flagged ({len(flagged_cats)}): {', '.join(flagged_cats)}",
        )

    if max_prob >= HIGH_RISK_PROB_THRESHOLD and len(flagged_cats) >= 1:
        return (
            RiskLevel.HIGH_RISK,
            "🔴",
            f"High confidence toxicity ({max_prob:.2f}): {', '.join(flagged_cats)}",
        )

    # 2. Check for Review triggers
    if len(flagged_cats) == 1:
        cat = flagged_cats[0]
        prob = predictions.get(cat, 0.0)
        th = thresholds.get(cat, 0.5)
        return (
            RiskLevel.REVIEW,
            "🟡",
            f"Single category flagged ({cat}: {prob:.2f} vs {th:.2f} threshold)",
        )

    # Check for borderline unflagged scores
    borderline_cats = []
    for cat, prob in predictions.items():
        th = thresholds.get(cat, 0.5)
        if prob < th and (th - prob) <= BORDERLINE_MARGIN:
            borderline_cats.append(f"{cat} ({prob:.2f} near {th:.2f})")

    if borderline_cats:
        return (
            RiskLevel.REVIEW,
            "🟡",
            f"Borderline score near threshold: {', '.join(borderline_cats)}",
        )

    if max_prob >= 0.40:
        return (
            RiskLevel.REVIEW,
            "🟡",
            f"Elevated sub-threshold signal (peak probability: {max_prob:.2f})",
        )

    # 3. Safe
    return (
        RiskLevel.SAFE,
        "🟢",
        "All categories evaluated safely below decision thresholds.",
    )


def calculate_aggregate_statistics(
    results: List[Tuple[Dict[str, float], float, str, Dict[str, bool]]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Calculate aggregate statistics across a collection of comment predictions.

    Parameters
    ----------
    results : list of tuples
        List of (predictions, score, intensity, is_toxic_flags).
    thresholds : dict[str, float]
        Per-label thresholds.

    Returns
    -------
    dict
        Aggregated summary metrics.
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "toxic_count": 0,
            "safe_count": 0,
            "toxic_percentage": 0.0,
            "high_risk_count": 0,
            "review_count": 0,
            "safe_risk_count": 0,
            "category_counts": {k: 0 for k in thresholds.keys()},
            "category_percentages": {k: 0.0 for k in thresholds.keys()},
        }

    toxic_count = 0
    high_risk_count = 0
    review_count = 0
    safe_risk_count = 0
    category_counts = {k: 0 for k in thresholds.keys()}

    for preds, score, intensity, flags in results:
        is_toxic = any(flags.values())
        if is_toxic:
            toxic_count += 1

        for cat, flagged in flags.items():
            if flagged:
                category_counts[cat] = category_counts.get(cat, 0) + 1

        risk_level, _, _ = classify_risk(preds, flags, thresholds)
        if risk_level == RiskLevel.HIGH_RISK:
            high_risk_count += 1
        elif risk_level == RiskLevel.REVIEW:
            review_count += 1
        else:
            safe_risk_count += 1

    category_percentages = {
        k: round((v / total) * 100, 1) for k, v in category_counts.items()
    }

    return {
        "total": total,
        "toxic_count": toxic_count,
        "safe_count": total - toxic_count,
        "toxic_percentage": round((toxic_count / total) * 100, 1),
        "high_risk_count": high_risk_count,
        "review_count": review_count,
        "safe_risk_count": safe_risk_count,
        "category_counts": category_counts,
        "category_percentages": category_percentages,
    }


def get_review_status(session_reviews: dict, comment_id: str) -> str:
    """Retrieve the human review status for a comment from session state."""
    return session_reviews.get(comment_id, ReviewStatus.PENDING.value)


def set_review_status(session_reviews: dict, comment_id: str, status: str) -> None:
    """Update human review status in session state."""
    session_reviews[comment_id] = status
