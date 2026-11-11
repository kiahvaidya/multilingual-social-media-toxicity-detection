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
import re
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
    "bc", "saale", "zeher", "dead", "destroy", "murder", "c00l", "sh!rt",
    "lit", "fire", "slayed", "wicked", "phaad", "faad", "jhakaas", "zabardast",
    "aag laga", "banger"
]

# ── Word Classes & Grammatical Roles Taxonomy (World-Class NLP Standard) ──

@dataclass
class SyntacticAnalysis:
    """Detailed syntactic intent and constituent decomposition."""
    is_intensified_praise: bool
    is_targeted_insult: bool
    is_untargeted_profanity: bool
    has_expletive: bool
    target_tokens: List[str]
    intensifier_tokens: List[str]
    positive_tokens: List[str]
    pejorative_tokens: List[str]
    praise_headword: Optional[str]
    insult_headword: Optional[str]
    target_entity: Optional[str]
    syntactic_role_summary: str


# Word Class 1: Degree Intensifiers & Amplifiers (Adverbial modifiers scaling adjacent predicates)
EXPLETIVE_INTENSIFIERS = {
    "fucking", "fuckin", "fkn", "f***ing", "fking", "effing", "freaking", "fricking",
    "damn", "damned", "bloody", "hella", "mad", "dead", "helluva", "badass",
    "saala", "katai", "behad", "faad", "bawaal", "ekdum", "bilkul"
}

STANDARD_INTENSIFIERS = {
    "extremely", "insanely", "ridiculously", "crazy", "so", "literally", "absolutely",
    "totally", "purely", "pure", "wildly", "immensely", "monumentally", "supremely",
    "massively", "deeply", "truly", "unbelievably", "deadass", "straight", "proper",
    "hugely", "stupidly", "seriously", "wicked"
}

# Word Class 2: Evaluative Positive / Superlative Headwords (Nouns & Adjectives of praise)
SUPERLATIVE_HONORIFICS = {
    "goat", "g.o.a.t", "legend", "beast", "king", "queen", "god", "wizard", "genius",
    "champ", "champion", "master", "maestro", "hero", "idol", "rockstar", "gem", "icon",
    "pro", "artist", "boss", "powerhouse", "monster", "titan", "banger", "masterpiece",
    "classic", "winner", "miracle", "magic", "stud", "star", "phenom", "mvp", "chief",
    "captain", "legendary", "perfection"
}

EVALUATIVE_POSITIVE_ADJECTIVES = {
    "great", "awesome", "amazing", "brilliant", "fire", "lit", "sick", "unreal", "dope",
    "clean", "flawless", "majestic", "immaculate", "good", "wonderful", "incredible",
    "exceptional", "outstanding", "stellar", "sublime", "phenomenal", "lethal", "beautiful",
    "gorgeous", "perfect", "insane", "solid", "based", "tight", "epic", "top-tier",
    "flawless", "golden", "superb", "terrific", "splendid",
    "bawaal", "gadar", "dhansu", "todu", "jhakaas", "zabardast", "shaandaar", "dumdar",
    "mast", "lajawab", "kamaal"
}

POSITIVE_PREDICATE_VERBS = [
    "killed it", "crushed it", "nailed it", "slayed", "ate that", "cooked", "went hard",
    "rocked", "owns", "wins", "rules", "delivered", "destroyed that", "bodied",
    "aag laga di", "phaad diya", "faad diya", "chha gaye", "dil jeet liya", "shabash",
    "fuck yeah", "went so hard", "let him cook", "you rock", "u rock"
]

# Word Class 3: Pejoratives / Derogatory Epithets / Insults
DEROGATORY_EPITHETS = {
    "idiot", "moron", "loser", "clown", "fool", "bastard", "bitch", "asshole", "jerk",
    "scumbag", "trash", "garbage", "scum", "creep", "coward", "hypocrite", "liar",
    "freak", "whore", "slut", "pig", "dog", "rat", "snake", "imbecile", "retard",
    "parasite", "vermin", "cunt", "jackass", "dipshit", "dirtbag", "douche", "douchebag",
    "prick", "chutiya", "kutta", "kamina", "saala", "gandu", "bhosdike", "harami",
    "nalayak", "suar", "madarchod", "behenchod", "tatti", "kachra", "bevakoof", "ullu",
    "stupid", "dumb", "pathetic", "disgusting", "vile", "useless", "worthless", "ignorant",
    "toxic", "nasty", "repulsive", "hateful", "shitty", "crappy", "lame"
}

# Word Class 4: Deictic Targets (Direct address and personal references)
SECOND_PERSON_TARGETS = {
    "you", "u", "ya", "ur", "you're", "yours", "yourself", "y'all", "thou", "thee",
    "tu", "tum", "aap", "tera", "teri", "tere", "tumhara", "apka"
}

VOCATIVE_ADDRESS = {
    "man", "bro", "brother", "dude", "mate", "friend", "fam", "boss", "bhai", "yaar",
    "dost", "mitr", "guys", "folks", "pal", "chief"
}

THIRD_PERSON_TARGETS = {
    "he", "she", "they", "him", "her", "them", "this guy", "that guy", "these people",
    "woh", "usne", "unhone"
}

# Word Class 5: Copular & Linking Verbs
COPULAS = {
    "is", "are", "'re", "'s", "am", "was", "were", "be", "being", "been",
    "look", "looks", "sound", "sounds", "seem", "seems", "acts",
    "hai", "ho", "tha", "the", "hain"
}

# Word Class 6: Negators
NEGATORS = {
    "not", "n't", "never", "no", "nahi", "nahin", "mat", "hardly", "barely"
}

# Explicit Physical Threat Action Markers (disambiguates violent praise metaphors like 'killed it' or 'katai bawaal')
EXPLICIT_THREAT_ACTIONS = [
    "kill you", "murder you", "goli maar", "jaan se maar", "haddi tod",
    "beat you until", "khatam kar dunga", "peetenge", "destroy you",
    "slay you", "cut your", "shoot you", "stab you"
]

ALL_INTENSIFIERS = EXPLETIVE_INTENSIFIERS | STANDARD_INTENSIFIERS
ALL_POSITIVE_WORDS = SUPERLATIVE_HONORIFICS | EVALUATIVE_POSITIVE_ADJECTIVES
ALL_TARGETS = SECOND_PERSON_TARGETS | VOCATIVE_ADDRESS | THIRD_PERSON_TARGETS

# Legacy praise indicators list maintained for backward compatibility
PRAISE_INDICATORS = list(ALL_POSITIVE_WORDS) + POSITIVE_PREDICATE_VERBS


def analyze_syntactic_intent(text: Optional[str]) -> SyntacticAnalysis:
    """Perform grammatical word-class tagging and syntactic dependency constituent parsing.

    Disentangles lexical expletives from communicative intent:
      - Adverbial intensifiers modifying positive superlatives (Intensified Praise)
      - Predicative insults modifying a target person (Targeted Harassment)
      - Untargeted expletive interjections (Non-targeted Vulgarity)
    """
    if not text or not str(text).strip():
        return SyntacticAnalysis(
            is_intensified_praise=False,
            is_targeted_insult=False,
            is_untargeted_profanity=False,
            has_expletive=False,
            target_tokens=[],
            intensifier_tokens=[],
            positive_tokens=[],
            pejorative_tokens=[],
            praise_headword=None,
            insult_headword=None,
            target_entity=None,
            syntactic_role_summary="Empty text",
        )

    raw_text = str(text)
    t_lower = raw_text.lower()
    norm_text = re.sub(r"[^\w\s\'-]", " ", t_lower)
    tokens = norm_text.split()

    found_targets = []
    found_intensifiers = []
    found_positives = []
    found_pejoratives = []
    has_expletive = False

    # Check multi-word verbal praise idioms
    matched_praise_idiom = None
    for idiom in POSITIVE_PREDICATE_VERBS:
        if idiom in t_lower:
            matched_praise_idiom = idiom
            break

    # Tag token word classes and track positions
    token_classes = []
    for i, tok in enumerate(tokens):
        classes = set()
        if tok in SECOND_PERSON_TARGETS:
            classes.add("TARGET_2ND")
            found_targets.append(tok)
        elif tok in VOCATIVE_ADDRESS:
            classes.add("VOCATIVE")
            found_targets.append(tok)
        elif tok in THIRD_PERSON_TARGETS:
            classes.add("TARGET_3RD")
            found_targets.append(tok)

        if tok in EXPLETIVE_INTENSIFIERS:
            classes.add("EXPLETIVE_INTENSIFIER")
            found_intensifiers.append(tok)
            has_expletive = True
        elif tok in STANDARD_INTENSIFIERS:
            classes.add("STANDARD_INTENSIFIER")
            found_intensifiers.append(tok)

        if tok in ALL_POSITIVE_WORDS:
            classes.add("POSITIVE_EVALUATIVE")
            found_positives.append(tok)

        if tok in DEROGATORY_EPITHETS:
            classes.add("PEJORATIVE")
            found_pejoratives.append(tok)

        if tok in COPULAS:
            classes.add("COPULA")

        if tok in NEGATORS:
            classes.add("NEGATOR")

        token_classes.append((tok, classes))

    # Also detect "as fuck" or "af" post-positive intensifiers
    has_post_intensifier = ("as fuck" in t_lower or " af" in t_lower)
    if has_post_intensifier:
        has_expletive = True
        found_intensifiers.append("as fuck")

    # Check for general expletive presence in text
    if any(sw in t_lower for sw in ["fuck", "shit", "bitch", "asshole", "damn", "saala", "bc", "chutiya"]):
        has_expletive = True

    # 1. Syntactic Dependency: Intensifier -> Positive Headword
    is_intensified_praise = False
    praise_head = matched_praise_idiom
    praise_target = found_targets[0] if found_targets else None

    # Check token adjacency windows: [Intensifier] + [Det/Adv]? + [Positive]
    for i, (tok, cls) in enumerate(token_classes):
        if "EXPLETIVE_INTENSIFIER" in cls or "STANDARD_INTENSIFIER" in cls:
            for lookahead in range(1, 4):
                if i + lookahead < len(token_classes):
                    next_tok, next_cls = token_classes[i + lookahead]
                    if "POSITIVE_EVALUATIVE" in next_cls:
                        negated = False
                        if i > 0 and "NEGATOR" in token_classes[i - 1][1]:
                            negated = True
                        if not negated:
                            is_intensified_praise = True
                            praise_head = next_tok
                            break
            if is_intensified_praise:
                break

    # Also check post-positive: [Positive] + "as fuck"
    if not is_intensified_praise and has_post_intensifier and found_positives:
        is_intensified_praise = True
        praise_head = found_positives[0]

    # Also check praise idiom with intensifier or target
    if not is_intensified_praise and matched_praise_idiom:
        is_intensified_praise = True
        praise_head = matched_praise_idiom

    # If positive headword is present with direct target/vocative and NO pejoratives:
    if not is_intensified_praise and found_positives and not found_pejoratives:
        is_intensified_praise = True
        praise_head = found_positives[0]

    # 2. Syntactic Dependency: Target + Pejorative (Targeted Insult)
    is_targeted_insult = False
    insult_head = None
    insult_target = None

    if found_pejoratives:
        for i, (tok, cls) in enumerate(token_classes):
            if "PEJORATIVE" in cls:
                for lookbehind in range(1, 5):
                    if i - lookbehind >= 0:
                        prev_tok, prev_cls = token_classes[i - lookbehind]
                        if any(t in prev_cls for t in ["TARGET_2ND", "TARGET_3RD", "VOCATIVE"]):
                            is_targeted_insult = True
                            insult_head = tok
                            insult_target = prev_tok
                            break
                if not is_targeted_insult:
                    for lookahead in range(1, 3):
                        if i + lookahead < len(token_classes):
                            next_tok, next_cls = token_classes[i + lookahead]
                            if any(t in next_cls for t in ["TARGET_2ND", "TARGET_3RD", "VOCATIVE"]):
                                is_targeted_insult = True
                                insult_head = tok
                                insult_target = next_tok
                                break
                if is_targeted_insult:
                    break

        if not is_targeted_insult and found_targets and not is_intensified_praise:
            is_targeted_insult = True
            insult_head = found_pejoratives[0]
            insult_target = found_targets[0]

    # Conflict Resolution: If both intensified praise and pejorative appear
    if is_intensified_praise and is_targeted_insult:
        is_intensified_praise = False

    # 3. Untargeted Profanity
    is_untargeted_profanity = False
    if has_expletive and not is_intensified_praise and not is_targeted_insult and not found_pejoratives:
        is_untargeted_profanity = True

    # Build summary
    if is_intensified_praise:
        role_summary = f"Expletive intensifier modifying positive headword '{praise_head}'"
        if praise_target:
            role_summary += f" predicated on target '{praise_target}'"
    elif is_targeted_insult:
        role_summary = f"Pejorative epithet '{insult_head}' predicated on target '{insult_target}'"
    elif is_untargeted_profanity:
        role_summary = "Untargeted colloquial expletive / profanity interjection without personal animus"
    else:
        role_summary = "Standard declarative syntax"

    return SyntacticAnalysis(
        is_intensified_praise=is_intensified_praise,
        is_targeted_insult=is_targeted_insult,
        is_untargeted_profanity=is_untargeted_profanity,
        has_expletive=has_expletive,
        target_tokens=found_targets,
        intensifier_tokens=found_intensifiers,
        positive_tokens=found_positives,
        pejorative_tokens=found_pejoratives,
        praise_headword=praise_head,
        insult_headword=insult_head,
        target_entity=insult_target or praise_target,
        syntactic_role_summary=role_summary,
    )

# ── Quoted Speech, Reporting Verbs & Counter-Speech Attribution ───
QUOTE_PATTERN = re.compile(r'["\u201C\u201D\u00AB\u00BB]([^"\u201C\u201D\u00AB\u00BB]+)["\u201C\u201D\u00AB\u00BB]|\'([^\']+)\'')

REPORTING_VERBS = [
    "called me", "said to me", "sent me", "told me", "texted me", "tweeted that",
    "commented that", "accused me of", "calling me", "saying that", "calling people",
    "someone said", "they called", "she called", "he called", "people call",
    "mujhe bola", "usne kaha", "bol raha tha", "bol rahi thi", "kehta hai", "kehti hai",
]

CONDEMNATION_MARKERS = [
    "horrible", "disgusting", "wrong", "terrible", "unacceptable", "shameful",
    "heartbreaking", "broke my heart", "made me cry", "offensive", "toxic",
    "hateful", "not okay", "not ok", "galat", "bura", "sharamnak", "galat baat",
    "stop calling", "stop saying", "don't call", "dont call", "do not call",
    "never say", "show some respect", "mat bolo", "rokna chahiye", "izzat karo",
    "hate speech", "harassment", "abusive",
]

# ── Pragmatic Irony & Sarcasm Lexicons ─────────────────────────
SUPERLATIVE_PRAISE = [
    "great job", "brilliant work", "wonderful job", "absolute genius", "genius",
    "congratulations on", "amazing job", "outstanding work", "shabash",
    "kya baat hai", "wah bhai", "what a masterclass",
]

DERISION_FAILURE = [
    "breaking", "crashing", "losing all", "ruined", "destroyed", "ruining",
    "destroying", "screwing up", "screwed up", "messing up", "messed up",
    "sab barbad", "pura kharab", "satyanas", "kachra", "disaster",
    "mess", "fail", "failing", "kharab", "barbad",
]


def detect_counter_speech_context(text: Optional[str]) -> Tuple[bool, str]:
    """Detect counter-speech or victim reporting context (ACL HateCheck F18/F19 paradigm).

    Distinguishes denouncing, quoting, or reporting abuse from actively perpetrating it.
    """
    if not text:
        return False, ""
    t_lower = str(text).lower()
    has_reporting = any(rv in t_lower for rv in REPORTING_VERBS)
    has_condemnation = any(cm in t_lower for cm in CONDEMNATION_MARKERS)
    quotes = QUOTE_PATTERN.findall(str(text))
    has_quotes = len(quotes) > 0

    if (has_reporting and has_condemnation) or (has_quotes and (has_reporting or has_condemnation)):
        return True, "Counter-speech or victim reporting abusive quote detected; intent is denouncing, not perpetrating."
    if any(cm in t_lower for cm in ["stop calling", "stop saying", "don't call", "dont call", "mat bolo", "show some respect"]):
        return True, "Corrective counter-speech urging cessation of harassment detected."
    return False, ""


def detect_sarcasm_indicators(text: Optional[str]) -> Tuple[bool, str]:
    """Detect lexical contradictions characteristic of sarcastic mockery or pragmatic irony.

    Flags comments where positive superlatives are juxtaposed with failure/derision.
    """
    if not text:
        return False, ""
    t_lower = str(text).lower()
    has_praise = any(sp in t_lower for sp in SUPERLATIVE_PRAISE)
    has_failure = any(df in t_lower for df in DERISION_FAILURE)

    if has_praise and has_failure:
        return True, "Lexical contradiction detected: praise juxtaposed with failure/derision indicates potential sarcastic mockery."
    if "slow clap" in t_lower or "oh great," in t_lower or "yeah, right," in t_lower:
        return True, "Discourse marker for sarcastic irony detected."
    return False, ""


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
    # and without counter-speech reporting context
    has_severe = any(c in SEVERE_CATEGORIES for c in flagged_cats)
    syntax = analyze_syntactic_intent(text)
    has_praise_context = (
        syntax.is_intensified_praise
        or bool(syntax.positive_tokens)
        or any(re.search(r'\b' + re.escape(p) + r'\b', t_lower) for p in POSITIVE_PREDICATE_VERBS)
    )
    is_counter_speech, counter_reason = detect_counter_speech_context(text)
    is_sarcasm, sarcasm_reason = detect_sarcasm_indicators(text)

    if has_severe and max_prob >= cfg.bypass_extreme_threat_threshold and not has_praise_context and not is_counter_speech:
        return False, f"Clear high-risk severe toxicity (score: {max_prob:.2f} >= {cfg.bypass_extreme_threat_threshold:.2f}); bypassing JEV."

    # 2. Counter-Speech / Victim Reporting Attribution Gate
    if is_counter_speech:
        return True, f"Counter-speech / victim reporting context detected ({counter_reason}); verifying intent via JEV."

    # 3. Pragmatic Irony & Sarcastic Mockery Contradiction Gate
    if is_sarcasm:
        return True, f"Potential sarcastic mockery detected ({sarcasm_reason}); verifying intent via JEV."

    # 4. Review Tier Trigger: comments in Review state are primary candidates
    if risk_level == RiskLevel.REVIEW:
        return True, "MuRIL assigned Review status; context requires disambiguation."

    # 5. Score Proximity to Decision Threshold
    borderline_cats = []
    for cat, prob in predictions.items():
        th = thresholds.get(cat, 0.5)
        if abs(prob - th) <= cfg.borderline_margin:
            borderline_cats.append(f"{cat} ({prob:.2f} near {th:.2f})")

    if borderline_cats:
        return True, f"Score proximity to decision threshold: {', '.join(borderline_cats)}"

    # 6. Slang / Cathartic Profanity Presence with Moderate Toxicity Signals
    if cfg.enable_slang_lexical_check:
        has_marker = any(m in t_lower for m in SLANG_PROFANITY_MARKERS)
        if has_marker and max_prob >= 0.30:
            return True, f"Colloquial slang or profanity marker detected with elevated score ({max_prob:.2f}); intent uncertain."

    # 7. Multiple categories flagged with non-extreme probability (< 0.75)
    if len(flagged_cats) >= 2 and max_prob < cfg.high_risk_prob_threshold:
        return True, f"Multiple categories flagged ({', '.join(flagged_cats)}) with moderate confidence; verifying contextual intent."

    # 8. Default: Clear case (either clear Safe or unambiguous High Risk)
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
    human_review_required: bool = False

    @property
    def human_review(self) -> bool:
        """Alias for human_review_required."""
        return self.human_review_required


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
    base_risk, base_icon, base_expl = classify_risk(predictions, is_toxic_flags, thresholds, text=text)

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
            human_review_required=(base_risk != RiskLevel.SAFE),
        )

    # 3. Ambiguous Case -> JEV Provider Resolution
    from src.jev import (
        get_jev_provider,
        get_jev_cache,
        sanitize_jev_error,
        JEVProviderType,
    )

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
            human_review_required=(base_risk != RiskLevel.SAFE),
        )

    # 4. Invoke JEV with Failure Safety & LRU Caching
    cache = get_jev_cache()
    provider_id = getattr(provider, "model_name", provider.__class__.__name__)
    cache_key = cache.make_key(text, predictions, provider_id)

    # Check cache for previous successful adjudication
    cached_decision = cache.get(cache_key)

    try:
        if cached_decision is not None:
            logger.debug("JEV Cache HIT in moderate_comment_with_context (key %s)", cache_key[:8])
            decision = cached_decision
        else:
            decision = provider.assess_context(text, predictions, is_toxic_flags, base_risk)
            # ONLY cache successful decisions (failed calls raise exception and are not cached)
            cache.put(cache_key, decision)

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
            human_review_required=decision.human_review_required,
        )

    except Exception as exc:
        clean_reason = sanitize_jev_error(exc)
        logger.warning("JEV execution failed: %s. Routing ambiguous comment to REVIEW.", clean_reason)
        # CRITICAL SAFE FAILURE HANDLING:
        # When JEV was invoked specifically because the comment was ambiguous/context-dependent,
        # failure of remote adjudication routes to REVIEW with human oversight.
        # This prevents transient API outages from converting ambiguous praise/slang into HIGH_RISK.
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
            final_risk_level=RiskLevel.REVIEW,
            final_badge_icon="🟡",
            final_explanation=f"Contextual adjudication unavailable; routed to human review. (Reason: {clean_reason})",
            jev_decision=None,
            jev_error=clean_reason,
            human_review_required=True,
        )


def classify_risk(
    predictions: Dict[str, float],
    is_toxic_flags: Dict[str, bool],
    thresholds: Dict[str, float],
    text: Optional[str] = None,
) -> Tuple[RiskLevel, str, str]:
    """Classify the risk level of an analyzed comment into one of three tiers.

    Tiers:
      - 🔴 High Risk: Severe categories triggered (severe_toxic, threat, identity_hate),
        or multiple categories triggered without praise context, or high confidence (>= 0.75).
      - 🟡 Review: Single moderate category triggered, or unflagged but borderline
        score within BORDERLINE_MARGIN of threshold or score >= 0.40, or multiple mild categories with praise context.
      - 🟢 Safe: No categories triggered and all probabilities well below thresholds.

    Parameters
    ----------
    predictions : dict[str, float]
        Per-label probabilities (0.0 - 1.0).
    is_toxic_flags : dict[str, bool]
        Per-label boolean flags evaluated against thresholds.
    thresholds : dict[str, float]
        Active thresholds used for evaluation.
    text : str, optional
        Raw comment text used for context disambiguation.

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
    t_lower = str(text or "").lower()

    # 1. Linguistic & Syntactic Role Analysis (Word Class Constituency Engine)
    syntax = analyze_syntactic_intent(text)
    is_counter_speech, counter_expl = detect_counter_speech_context(text)
    is_sarcasm, sarcasm_expl = detect_sarcasm_indicators(text)

    # 2. Hard Safety Guardrail: Severe Toxicity Isolation
    # Categories deemed severe that represent non-negotiable safety harms
    severe_flagged = [c for c in flagged_cats if c in SEVERE_CATEGORIES]
    if severe_flagged:
        if is_counter_speech:
            return (
                RiskLevel.REVIEW,
                "🟡",
                f"Counter-speech / victim reporting context detected ({counter_expl}); requires intent review",
            )
        # Check if severe flag is a violent metaphor within intensified praise without genuine threat actions
        if syntax.is_intensified_praise and not any(ta in t_lower for ta in EXPLICIT_THREAT_ACTIONS):
            return (
                RiskLevel.SAFE,
                "🟢",
                f"Enthusiastic praise detected: {syntax.syntactic_role_summary}; non-toxic metaphorical praise.",
            )
        return (
            RiskLevel.HIGH_RISK,
            "🔴",
            f"Severe toxicity flagged: {', '.join(severe_flagged)}",
        )

    # 3. Counter-Speech / Quoted Abuse Override
    if is_counter_speech and (len(flagged_cats) >= 1 or max_prob >= 0.35):
        return (
            RiskLevel.REVIEW,
            "🟡",
            f"Counter-speech / victim reporting context detected ({counter_expl}); requires intent review",
        )

    # 4. Pragmatic Irony & Sarcastic Mockery Contradiction
    if is_sarcasm and max_prob >= 0.30:
        return (
            RiskLevel.REVIEW,
            "🟡",
            f"Potential sarcastic mockery detected ({sarcasm_expl}); requires intent review",
        )

    # 5. Syntactic Role: Intensified Praise / Positive Evaluation
    # If an expletive acts as an adverbial intensifier modifying a positive headword,
    # the communicative polarity is praise/excitement, NOT toxicity.
    if syntax.is_intensified_praise and not is_sarcasm:
        return (
            RiskLevel.SAFE,
            "🟢",
            f"Enthusiastic praise detected: {syntax.syntactic_role_summary}; communicative intent is non-toxic.",
        )

    # 6. Syntactic Role: Targeted Insult / Harassment
    # If a pejorative epithet is predicated on a target addressee and toxicity is elevated
    if syntax.is_targeted_insult and (len(flagged_cats) >= 1 or max_prob >= 0.40):
        return (
            RiskLevel.HIGH_RISK,
            "🔴",
            f"Targeted personal insult detected: {syntax.syntactic_role_summary}",
        )

    # 7. Syntactic Role: Untargeted Profanity / Interjection
    # Untargeted swearing (without insults, threats, or hate) must NEVER escalate to High Risk
    if syntax.is_untargeted_profanity:
        return (
            RiskLevel.REVIEW,
            "🟡",
            f"Untargeted colloquial profanity or expletive interjection; no targeted insult or severe harm detected.",
        )

    # 8. Standard Multi-Category High Risk Escalation
    if len(flagged_cats) >= 2:
        return (
            RiskLevel.HIGH_RISK,
            "🔴",
            f"Multiple toxicity categories flagged ({len(flagged_cats)}): {', '.join(flagged_cats)}",
        )

    # 9. High Confidence Single Category Escalation
    if max_prob >= HIGH_RISK_PROB_THRESHOLD and len(flagged_cats) >= 1:
        if flagged_cats == ["obscene"]:
            return (
                RiskLevel.REVIEW,
                "🟡",
                f"Single category flagged (obscene: {max_prob:.2f} vs {thresholds.get('obscene', 0.35):.2f} threshold)",
            )
        return (
            RiskLevel.HIGH_RISK,
            "🔴",
            f"High confidence toxicity ({max_prob:.2f}): {', '.join(flagged_cats)}",
        )

    # 10. Single Category Flagged -> Review
    if len(flagged_cats) == 1:
        cat = flagged_cats[0]
        prob = predictions.get(cat, 0.0)
        th = thresholds.get(cat, 0.5)
        return (
            RiskLevel.REVIEW,
            "🟡",
            f"Single category flagged ({cat}: {prob:.2f} vs {th:.2f} threshold)",
        )

    # 11. Check for Borderline Unflagged Scores
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

    # 12. Safe
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
