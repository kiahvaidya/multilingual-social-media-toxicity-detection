"""
Phase 6A: JEV (Judicial / Evaluative Verification) Contextual Decision Layer.

Provides:
  - Structured contextual decision schema (intent, contextual_toxicity, moderation, confidence, reason).
  - JEVDecisionProvider abstraction for modular provider integration.
  - HTTPJEVDecisionProvider for connecting to remote JEV services via environment variables.
  - MockJEVDecisionProvider for deterministic offline regression & provider-interface testing.
  - Strict provider resolution ensuring Mock is NEVER silently selected in production.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from src.config import config
from src.moderation import RiskLevel

logger = logging.getLogger(__name__)


class JEVIntent(str, Enum):
    """Fine-grained contextual intent classification categories."""
    GENUINE_ABUSE = "genuine_abuse"
    THREAT = "threat"
    HATE = "hate"
    NON_TOXIC_PRAISE_OR_EXCITEMENT = "non_toxic_praise_or_excitement"
    AMBIGUOUS = "ambiguous"

    @classmethod
    def from_raw(cls, raw: Any) -> "JEVIntent":
        """Safely parse raw intent string or alias into a valid JEVIntent enum."""
        val = str(raw or "").lower().strip()
        mapping = {
            "genuine_abuse": cls.GENUINE_ABUSE,
            "threat": cls.THREAT,
            "hate": cls.HATE,
            "hate_or_identity_attack": cls.HATE,
            "non_toxic_praise_or_excitement": cls.NON_TOXIC_PRAISE_OR_EXCITEMENT,
            "praise_or_excitement": cls.NON_TOXIC_PRAISE_OR_EXCITEMENT,
            "harmless_profanity": cls.NON_TOXIC_PRAISE_OR_EXCITEMENT,
            "sarcasm_or_joke": cls.NON_TOXIC_PRAISE_OR_EXCITEMENT,
            "ambiguous": cls.AMBIGUOUS,
        }
        return mapping.get(val, cls.AMBIGUOUS)


class JEVProviderType(str, Enum):
    """Provider execution provenance."""
    REAL_JEV = "REAL_JEV"
    MOCK_JEV = "MOCK_JEV"
    JEV_UNAVAILABLE = "JEV_UNAVAILABLE"


@dataclass
class JEVDecision:
    """Structured contextual decision returned by a JEV provider."""
    intent: JEVIntent
    contextual_toxicity: bool
    moderation: RiskLevel
    human_review_required: bool
    confidence: float
    reason: str
    provider_type: JEVProviderType
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def human_review(self) -> bool:
        """Alias for human_review_required for backward compatibility."""
        return self.human_review_required

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "contextual_toxicity": self.contextual_toxicity,
            "moderation": self.moderation.value,
            "human_review_required": self.human_review_required,
            "human_review": self.human_review_required,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "provider_type": self.provider_type.value,
        }


class JEVDecisionProvider(ABC):
    """Abstract interface for contextual decision providers."""

    @property
    @abstractmethod
    def provider_type(self) -> JEVProviderType:
        """Return the provider provenance type."""
        pass

    @abstractmethod
    def assess_context(
        self,
        text: str,
        muril_scores: Dict[str, float],
        muril_flags: Dict[str, bool],
        muril_risk: RiskLevel,
    ) -> JEVDecision:
        """Assess the contextual intent and recommend a moderation decision.

        Parameters
        ----------
        text : str
            Raw comment string.
        muril_scores : dict[str, float]
            Probability scores per category from MuRIL.
        muril_flags : dict[str, bool]
            Boolean threshold flags from MuRIL.
        muril_risk : RiskLevel
            Assigned baseline risk level from 3-tier moderation.

        Returns
        -------
        JEVDecision
            Structured contextual decision.
        """
        pass


class HTTPJEVDecisionProvider(JEVDecisionProvider):
    """External HTTP API integration for JEV.

    Credentials and endpoint are sourced strictly from environment variables.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self._endpoint = endpoint or config.JEV_ENDPOINT_URL
        self._api_key = api_key or config.JEV_API_KEY
        self._timeout = timeout or config.JEV_TIMEOUT_SECONDS

        if not self._endpoint:
            raise ValueError("JEV_ENDPOINT_URL is not configured.")

    @property
    def provider_type(self) -> JEVProviderType:
        return JEVProviderType.REAL_JEV

    def assess_context(
        self,
        text: str,
        muril_scores: Dict[str, float],
        muril_flags: Dict[str, bool],
        muril_risk: RiskLevel,
    ) -> JEVDecision:
        payload = {
            "comment": text,
            "muril_scores": muril_scores,
            "muril_flags": muril_flags,
            "muril_risk": muril_risk.value,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MultilingualToxicityDetector/Phase6A",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            req = urllib.request.Request(
                self._endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status_code = getattr(resp, "status", getattr(resp, "code", 200))
                if status_code != 200:
                    raise RuntimeError(f"JEV endpoint returned HTTP {status_code}")
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError(f"JEV response is not a JSON object: {type(data)}")

            # Validate and parse structured response schema
            intent = JEVIntent.from_raw(data.get("intent"))
            contextual_tox = bool(data.get("contextual_toxicity", False))

            mod_raw = str(data.get("moderation", "")).strip().lower()
            if mod_raw in ("high_risk", "high risk", "high"):
                moderation = RiskLevel.HIGH_RISK
            elif mod_raw in ("review", "moderate"):
                moderation = RiskLevel.REVIEW
            elif mod_raw in ("safe", "low"):
                moderation = RiskLevel.SAFE
            else:
                moderation = muril_risk

            if "human_review_required" in data:
                human_review = bool(data["human_review_required"])
            elif "human_review" in data:
                human_review = bool(data["human_review"])
            else:
                human_review = (moderation != RiskLevel.SAFE)

            try:
                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.5

            reason = str(data.get("reason", "Decision by external JEV provider.")).strip()
            if not reason:
                reason = "Decision by external JEV provider."

            return JEVDecision(
                intent=intent,
                contextual_toxicity=contextual_tox,
                moderation=moderation,
                human_review_required=human_review,
                confidence=confidence,
                reason=reason,
                provider_type=JEVProviderType.REAL_JEV,
                raw_response=data,
            )

        except urllib.error.HTTPError as e:
            logger.warning("HTTPJEVDecisionProvider HTTP error: %s (code %s)", e.reason, e.code)
            raise RuntimeError(f"JEV HTTP error: {e.code} - {e.reason}") from e
        except urllib.error.URLError as e:
            logger.warning("HTTPJEVDecisionProvider connection error: %s", e.reason)
            raise RuntimeError(f"JEV connection error: {e.reason}") from e
        except TimeoutError as e:
            logger.warning("HTTPJEVDecisionProvider timeout after %s seconds", self._timeout)
            raise TimeoutError(f"JEV request timed out after {self._timeout}s") from e
        except json.JSONDecodeError as e:
            logger.warning("HTTPJEVDecisionProvider JSON parse error: %s", e)
            raise ValueError(f"JEV response is not valid JSON: {e}") from e
        except Exception as e:
            logger.warning("HTTPJEVDecisionProvider call failed: %s", e)
            raise RuntimeError(f"JEV service request failed: {e}") from e


class MockJEVDecisionProvider(JEVDecisionProvider):
    """Deterministic Mock Provider for unit testing and offline pipeline validation.

    IMPORTANT SAFETY NOTE:
    This mock provider exists EXCLUSIVELY for:
      - unit testing
      - deterministic offline pipeline verification
      - provider-interface regression tests

    It is NEVER automatically selected by get_jev_provider() in production.
    """

    @property
    def provider_type(self) -> JEVProviderType:
        return JEVProviderType.MOCK_JEV

    def assess_context(
        self,
        text: str,
        muril_scores: Dict[str, float],
        muril_flags: Dict[str, bool],
        muril_risk: RiskLevel,
    ) -> JEVDecision:
        t_clean = text.strip()
        t_lower = t_clean.lower()

        # ── 1. Explicit Praise & Enthusiastic Slang ──────────────────────────
        praise_metaphors = [
            "killed it", "totally sick", "drop-dead gorgeous", "bawaal", "धांसू",
            "gadar", "aag laga di", "band baja diya", "todu", "wicked smart",
            "pure magic", "magic", "best track", "super cool", "super c00l",
            "went so hard", "murder on the dancefloor", "murder on the speakers",
            "destroyed me", "deadly performance", "katai zeher", "gr8 work",
            "f-a-n-t-a-s-t-i-c", "b.r.a.v.o", "bravo", "dil jeet liya", "w0w",
        ]
        if any(m in t_lower for m in praise_metaphors) and not any(
            ab in t_lower for ab in ["kutte", "chutiya", "madarchod", "bhosdike", "gandu", "scumbag", "parasitic filth"]
        ):
            return JEVDecision(
                intent=JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT,
                contextual_toxicity=False,
                moderation=RiskLevel.SAFE,
                human_review_required=False,
                confidence=0.95,
                reason="Positive praise / enthusiastic colloquial superlatives recognized without abusive intent.",
                provider_type=JEVProviderType.MOCK_JEV,
            )

        # ── 2. Exclamatory / Harmless Non-Aggressive Profanity ────────────────
        if "fuck yeah" in t_lower or "holy shit" in t_lower:
            return JEVDecision(
                intent=JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT,
                contextual_toxicity=False,
                moderation=RiskLevel.SAFE,
                human_review_required=False,
                confidence=0.94,
                reason="Celebratory or cathartic exclamatory profanity without interpersonal hostility.",
                provider_type=JEVProviderType.MOCK_JEV,
            )

        if "job lag gayi bc" in t_lower or "treat kab de raha hai" in t_lower:
            return JEVDecision(
                intent=JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT,
                contextual_toxicity=False,
                moderation=RiskLevel.SAFE,
                human_review_required=False,
                confidence=0.92,
                reason="Colloquial Hinglish social banter celebrating an achievement; profanity functions as informal filler.",
                provider_type=JEVProviderType.MOCK_JEV,
            )

        if "abe saale kitne saal baad" in t_lower or "gale mil pehle" in t_lower:
            return JEVDecision(
                intent=JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT,
                contextual_toxicity=False,
                moderation=RiskLevel.SAFE,
                human_review_required=False,
                confidence=0.90,
                reason="Friendly reunion greeting among peers with colloquial teasing.",
                provider_type=JEVProviderType.MOCK_JEV,
            )

        # ── 3. Benign Leetspeak / Obfuscated Innocent Text ───────────────────
        if ("sh!rt" in t_lower or "c00l" in t_lower or "luv th!s" in t_lower or "gooood joooob" in t_lower) and not any(
            bad in t_lower for bad in ["f**k", "b!tch", "1diot", "l0ser", "k!ll", "@sshole", "madarch0d", "b.h.o.s.d.i.k.e"]
        ):
            return JEVDecision(
                intent=JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT,
                contextual_toxicity=False,
                moderation=RiskLevel.SAFE,
                human_review_required=False,
                confidence=0.93,
                reason="Innocuous stylized leetspeak or character elongation without offensive semantics.",
                provider_type=JEVProviderType.MOCK_JEV,
            )

        # ── 4. Genuine Severe Threats & Violence ────────────────────────────
        threat_markers = [
            "murder you", "goli maar", "kill you", "beat you until", "haddi tod dunga",
            "jaan se maar", "khatam kar dunga", "k!ll u", "peetenge"
        ]
        if any(tm in t_lower for tm in threat_markers):
            return JEVDecision(
                intent=JEVIntent.THREAT,
                contextual_toxicity=True,
                moderation=RiskLevel.HIGH_RISK,
                human_review_required=True,
                confidence=0.97,
                reason="Explicit physical threat of violence or bodily harm identified.",
                provider_type=JEVProviderType.MOCK_JEV,
            )

        # ── 5. Identity Hate Attacks ─────────────────────────────────────────
        identity_markers = ["parasitic filth", "देशद्रोहियों", "मुल्लों", "porki log", "cleansed"]
        if any(im in t_lower for im in identity_markers):
            return JEVDecision(
                intent=JEVIntent.HATE,
                contextual_toxicity=True,
                moderation=RiskLevel.HIGH_RISK,
                human_review_required=True,
                confidence=0.98,
                reason="Targeted identity attack or dehumanizing hate speech.",
                provider_type=JEVProviderType.MOCK_JEV,
            )

        # ── 6. Explicit Insults & Abusive Obscenity ──────────────────────────
        severe_abuse_markers = [
            "chutiya", "madarchod", "bhosdike", "scumbag", "pathetic failure",
            "gandua", "loser", "gandu", "1diot", "l0ser", "b!tch", "@sshole"
        ]
        if any(sam in t_lower for sam in severe_abuse_markers):
            return JEVDecision(
                intent=JEVIntent.GENUINE_ABUSE,
                contextual_toxicity=True,
                moderation=RiskLevel.HIGH_RISK,
                human_review_required=True,
                confidence=0.94,
                reason="Direct personal insult, aggressive hostility, or vulgar abuse.",
                provider_type=JEVProviderType.MOCK_JEV,
            )

        # ── 7. Default Fallback for General Ambiguity ─────────────────────────
        return JEVDecision(
            intent=JEVIntent.AMBIGUOUS,
            contextual_toxicity=(muril_risk == RiskLevel.HIGH_RISK),
            moderation=muril_risk,
            human_review_required=True,
            confidence=0.60,
            reason="Ambiguous context; retaining baseline risk assessment for human review.",
            provider_type=JEVProviderType.MOCK_JEV,
        )


def get_jev_provider() -> Optional[JEVDecisionProvider]:
    """Resolve and return the active JEV decision provider.

    STRICT PROVIDER RESOLUTION RULES:
      1. If JEV_ENABLED=false:
         JEV is disabled. Returns None.
      2. If JEV_ENABLED=true AND JEV_ENDPOINT_URL is configured:
         Returns HTTPJEVDecisionProvider.
      3. If JEV_ENABLED=true but JEV_ENDPOINT_URL is missing:
         JEV is treated as unavailable/unconfigured. Returns None.
         (Does NOT silently substitute MockJEVDecisionProvider).
      4. MockJEVDecisionProvider is NEVER returned by this function;
         it exists strictly for unit tests and deterministic offline evaluation.
    """
    if not config.JEV_ENABLED:
        logger.info("JEV contextual moderation layer is disabled (JEV_ENABLED=false).")
        return None

    endpoint = config.JEV_ENDPOINT_URL
    api_key = config.JEV_API_KEY

    if endpoint:
        logger.info("Initializing HTTPJEVDecisionProvider at endpoint %s", endpoint)
        return HTTPJEVDecisionProvider(endpoint=endpoint, api_key=api_key)

    logger.info("JEV is enabled but unconfigured (missing JEV_ENDPOINT_URL). JEV is unavailable.")
    return None
