"""
Moderation domain logic and risk assessment for social media toxicity detection.

Provides:
  - 3-tier risk classification (🟢 Safe, 🟡 Review, 🔴 High Risk)
  - Assistive human review state handling (Review, Confirmed Toxic, False Positive)
  - Aggregate statistics calculation for comment batches
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


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
