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
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from src.config import config
from src.moderation import RiskLevel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def sanitize_jev_error(exc: Exception) -> str:
    """Sanitize error messages to eliminate API keys, credentials, or internal URLs."""
    raw = str(exc)
    # Redact sensitive credentials and patterns
    redacted = re.sub(r'(?i)(key|token|auth)=([a-zA-Z0-9_\-]+)', r'\1=[REDACTED]', raw)
    redacted = re.sub(r'AIza[0-9A-Za-z\-_]{35}', '[REDACTED_API_KEY]', redacted)
    redacted = re.sub(r'https?://[^\s\'"]+', '[URL]', redacted)

    raw_lower = raw.lower()
    if "429" in raw or "resource_exhausted" in raw_lower or "quota" in raw_lower:
        return "provider quota exceeded or rate limit (429)"
    elif "timed out" in raw_lower or "timeout" in raw_lower or "deadline" in raw_lower:
        return "provider request timed out"
    elif "getaddrinfo failed" in raw_lower or "connection" in raw_lower or "dns" in raw_lower:
        return "network or connection failure"
    elif any(code in raw for code in ("500", "502", "503", "504")) or "unavailable" in raw_lower:
        return "provider service temporarily unavailable (5xx)"
    elif "json" in raw_lower or "parse" in raw_lower:
        return "malformed provider response"
    else:
        first_line = redacted.splitlines()[0].strip()
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        return first_line or "provider request failed"


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


class JEVCache:
    """Thread-safe bounded in-memory LRU cache for successful JEV decisions."""

    def __init__(self, capacity: int = 1000):
        self._capacity = max(1, capacity)
        self._cache: OrderedDict[str, JEVDecision] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def make_key(
        self,
        text: str,
        muril_scores: Dict[str, float],
        model_name: Optional[str] = None,
    ) -> str:
        """Create a deterministic cache key from normalized text and rounded scores."""
        norm_text = " ".join(str(text or "").strip().lower().split())
        rounded_scores = {k: round(float(v), 4) for k, v in sorted(muril_scores.items())}
        raw_key = json.dumps({
            "text": norm_text,
            "scores": rounded_scores,
            "model": str(model_name or "default"),
        }, sort_keys=True)
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[JEVDecision]:
        """Retrieve a cached decision. Updates LRU order on hit."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, decision: JEVDecision) -> None:
        """Cache a successful JEV decision. Only non-None decisions are cached."""
        if decision is None:
            return
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._capacity:
                    self._cache.popitem(last=False)
            self._cache[key] = decision

    def clear(self) -> None:
        """Clear all cached decisions and reset hit/miss telemetry."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def stats(self) -> Dict[str, int]:
        """Return cache telemetry."""
        with self._lock:
            return {
                "size": len(self._cache),
                "capacity": self._capacity,
                "hits": self._hits,
                "misses": self._misses,
            }


_GLOBAL_JEV_CACHE = JEVCache(capacity=1000)


def get_jev_cache() -> JEVCache:
    """Retrieve the global in-memory LRU decision cache."""
    return _GLOBAL_JEV_CACHE


def reset_jev_cache() -> None:
    """Reset the global in-memory decision cache (primarily for tests)."""
    _GLOBAL_JEV_CACHE.clear()


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
        cache = get_jev_cache()
        cache_key = cache.make_key(text, muril_scores, "http_provider")
        cached_decision = cache.get(cache_key)
        if cached_decision is not None:
            return cached_decision

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

            decision = JEVDecision(
                intent=intent,
                contextual_toxicity=contextual_tox,
                moderation=moderation,
                human_review_required=human_review,
                confidence=confidence,
                reason=reason,
                provider_type=JEVProviderType.REAL_JEV,
                raw_response=data,
            )
            cache.put(cache_key, decision)
            return decision

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


class GeminiJEVResponseSchema(BaseModel):
    """Strict structured JSON response schema for Gemini JEV contextual adjudication."""
    intent: str = Field(description="genuine_abuse, threat, hate, non_toxic_praise_or_excitement, or ambiguous")
    contextual_toxicity: bool = Field(description="Whether the comment is contextually toxic")
    moderation: str = Field(description="SAFE, REVIEW, or HIGH_RISK")
    human_review_required: bool = Field(description="Whether human review is required")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    reason: str = Field(description="Clear concise justification")


class GeminiJEVDecisionProvider(JEVDecisionProvider):
    """Official Google Gemini AI integration for contextual JEV decision adjudication.

    Connects via the official google-genai SDK using GEMINI_API_KEY and JEV_GEMINI_MODEL.
    Configured for maximum determinism (temperature=0.0, seed=42, strict response schema).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.0,
        seed: int = 42,
    ):
        self._api_key = (api_key or config.GEMINI_API_KEY).strip()
        self._model_name = (model_name or config.JEV_GEMINI_MODEL).strip() or "gemini-2.5-flash"
        self._timeout = float(timeout if timeout is not None else config.JEV_TIMEOUT_SECONDS)
        self._temperature = float(temperature)
        self._seed = int(seed)

        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self._client = None

    @property
    def provider_type(self) -> JEVProviderType:
        return JEVProviderType.REAL_JEV

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def seed(self) -> int:
        return self._seed

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            timeout_ms = max(15000, int(self._timeout * 1000))
            http_options = types.HttpOptions(timeout=timeout_ms)
            self._client = genai.Client(api_key=self._api_key, http_options=http_options)
        return self._client

    def _build_prompt(
        self,
        text: str,
        muril_scores: Dict[str, float],
        muril_flags: Dict[str, bool],
        muril_risk: RiskLevel,
    ) -> str:
        flagged_cats = [c for c, f in muril_flags.items() if f]
        return f"""You are an expert multilingual content moderation and contextual intent adjudicator for social media text (English, Hindi in Devanagari script, and Hinglish / Romanized Hindi).

An initial baseline neural toxicity classifier (MuRIL) analyzed the following comment and flagged potential toxicity:

[INPUT DATA]
- Comment: "{text}"
- Classifier Probability Scores: {json.dumps(muril_scores)}
- Active Flagged Categories: {json.dumps(flagged_cats)}
- Baseline Risk Tier: {muril_risk.value}

[YOUR TASK]
Perform nuanced contextual adjudication to determine whether the comment represents genuine toxicity/abuse or an innocuous false positive.
Specifically distinguish genuine abuse from:
1. Non-toxic profanity & excitement (e.g., "Fuck yeah, we passed!", "holy shit that's awesome")
2. Colloquial praise metaphors & slang (e.g., "you killed it", "this is sick", "bhai bawaal banaya hai", "aag laga di", "katai zeher")
3. Friendly peer banter & informal teasing (e.g., "abe saale kitne din baad mila", "job lag gayi bc treat de")
4. Harmless obfuscation or leetspeak (e.g., "c00l sh!rt", "gr8 work")
5. Genuine abuse, hostility, targeted insults, threats, or hate speech (e.g., "you're a fucking idiot and I hope you get hurt", "tu ek number ka kutta hai")

[DECISION RULES]
- If the comment is benign praise, excitement, friendly banter, or non-aggressive slang:
  intent: "non_toxic_praise_or_excitement"
  contextual_toxicity: false
  moderation: "SAFE"
  human_review_required: false

- If the comment contains explicit threats, targeted abuse, harassment, or hate speech:
  intent: "genuine_abuse" (or "threat" or "hate" as applicable)
  contextual_toxicity: true
  moderation: "HIGH_RISK"
  human_review_required: true

- If context is genuinely ambiguous, veiled, or uncertain:
  intent: "ambiguous"
  contextual_toxicity: true if any severe category probability >= 0.5 else false
  moderation: "REVIEW"
  human_review_required: true

[OUTPUT REQUIREMENTS]
Respond strictly with a valid JSON object matching this schema:
{{
  "intent": "genuine_abuse" | "threat" | "hate" | "non_toxic_praise_or_excitement" | "ambiguous",
  "contextual_toxicity": true | false,
  "moderation": "SAFE" | "REVIEW" | "HIGH_RISK",
  "human_review_required": true | false,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<clear concise explanation of the contextual justification>"
}}"""

    def _parse_and_validate_response(
        self,
        raw_text: str,
        muril_risk: RiskLevel,
    ) -> JEVDecision:
        if not raw_text or not raw_text.strip():
            raise ValueError("Gemini returned empty response.")

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gemini response is not valid JSON: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Gemini response is not a JSON object: {type(data)}")

        # 1. intent
        raw_intent = data.get("intent")
        if not raw_intent:
            raise ValueError("Missing 'intent' field in Gemini response.")
        intent = JEVIntent.from_raw(raw_intent)

        # 2. contextual_toxicity
        if "contextual_toxicity" not in data:
            raise ValueError("Missing 'contextual_toxicity' field in Gemini response.")
        contextual_tox = bool(data["contextual_toxicity"])

        # 3. moderation
        mod_raw = str(data.get("moderation", "")).strip().lower()
        if mod_raw in ("safe", "low"):
            moderation = RiskLevel.SAFE
        elif mod_raw in ("review", "moderate"):
            moderation = RiskLevel.REVIEW
        elif mod_raw in ("high_risk", "high risk", "high"):
            moderation = RiskLevel.HIGH_RISK
        else:
            raise ValueError(f"Invalid 'moderation' value in Gemini response: {data.get('moderation')}")

        # 4. human_review_required
        if "human_review_required" in data:
            human_review = bool(data["human_review_required"])
        elif "human_review" in data:
            human_review = bool(data["human_review"])
        else:
            human_review = (moderation != RiskLevel.SAFE)

        # 5. confidence
        try:
            confidence = float(data.get("confidence", 0.5))
            if confidence < 0.0 or confidence > 1.0:
                confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        # 6. reason
        reason = str(data.get("reason", "")).strip()
        if not reason:
            raise ValueError("Missing or empty 'reason' field in Gemini response.")

        return JEVDecision(
            intent=intent,
            contextual_toxicity=contextual_tox,
            moderation=moderation,
            human_review_required=human_review,
            confidence=confidence,
            reason=reason,
            provider_type=JEVProviderType.REAL_JEV,
            raw_response={"model": self._model_name, "data": data},
        )

    def assess_context(
        self,
        text: str,
        muril_scores: Dict[str, float],
        muril_flags: Dict[str, bool],
        muril_risk: RiskLevel,
    ) -> JEVDecision:
        cache = get_jev_cache()
        cache_key = cache.make_key(text, muril_scores, self._model_name)
        cached_decision = cache.get(cache_key)
        if cached_decision is not None:
            logger.debug("Gemini JEV cache hit for model %s (hash: %s)", self._model_name, cache_key[:8])
            return cached_decision

        import time as _time
        client = self._get_client()
        prompt = self._build_prompt(text, muril_scores, muril_flags, muril_risk)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                from google.genai import types

                config_params = types.GenerateContentConfig(
                    temperature=self._temperature,
                    seed=self._seed,
                    response_mime_type="application/json",
                    response_schema=GeminiJEVResponseSchema,
                )
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config_params,
                )
                raw_text = getattr(response, "text", "") or ""
                decision = self._parse_and_validate_response(raw_text, muril_risk)
                # Successful decision stored in LRU cache
                cache.put(cache_key, decision)
                return decision
            except Exception as e:
                err_str = str(e)
                if any(k in err_str for k in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED")) and attempt < max_attempts - 1:
                    logger.warning("GeminiJEV transient failure (attempt %d/%d): %s; retrying...", attempt + 1, max_attempts, sanitize_jev_error(e))
                    _time.sleep(1.0 * (attempt + 1))
                    continue
                logger.warning("GeminiJEVDecisionProvider call failed: %s", sanitize_jev_error(e))
                raise


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
      2. If JEV_ENABLED=true AND GEMINI_API_KEY is configured:
         Returns GeminiJEVDecisionProvider with JEV_GEMINI_MODEL.
      3. If JEV_ENABLED=true AND JEV_ENDPOINT_URL is configured:
         Returns HTTPJEVDecisionProvider.
      4. If JEV_ENABLED=true but neither GEMINI_API_KEY nor JEV_ENDPOINT_URL is configured:
         JEV is treated as unavailable/unconfigured. Returns None.
         (Does NOT silently substitute MockJEVDecisionProvider).
      5. MockJEVDecisionProvider is NEVER returned by this function;
         it exists strictly for unit tests and deterministic offline evaluation.
    """
    if not config.JEV_ENABLED:
        logger.info("JEV contextual moderation layer is disabled (JEV_ENABLED=false).")
        return None

    # Priority 1: Gemini Provider (official LLM integration)
    gemini_key = config.GEMINI_API_KEY
    if gemini_key:
        logger.info("Initializing GeminiJEVDecisionProvider with model %s", config.JEV_GEMINI_MODEL)
        return GeminiJEVDecisionProvider(api_key=gemini_key, model_name=config.JEV_GEMINI_MODEL)

    # Priority 2: Generic HTTP REST Microservice
    endpoint = config.JEV_ENDPOINT_URL
    api_key = config.JEV_API_KEY
    if endpoint:
        logger.info("Initializing HTTPJEVDecisionProvider at endpoint %s", endpoint)
        return HTTPJEVDecisionProvider(endpoint=endpoint, api_key=api_key)

    logger.info("JEV is enabled but unconfigured (missing GEMINI_API_KEY and JEV_ENDPOINT_URL). JEV is unavailable.")
    return None
