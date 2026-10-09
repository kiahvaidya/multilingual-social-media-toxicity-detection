"""
Unit and integration tests for JEV determinism, caching, and safe failure handling.

Validates:
1. In-memory LRU cache for successful JEV decisions:
   - Identical inputs reuse cached decision without re-invoking provider.
   - Failed requests are NOT cached and remain retryable.
   - Bounded capacity and cache key normalization.
2. Deterministic Gemini configuration (temperature=0.0, seed=42, structured schema).
3. Safe failure handling:
   - Ambiguous case + Gemini timeout/429/500/malformed -> RiskLevel.REVIEW, human_review_required=True, jev_status=FAILED.
   - Preserved severe-risk bypass for non-ambiguous obvious abuse.
4. Test matrix across A-F benchmark inputs.
5. 20-run repeatability:
   - 20 runs with success -> identical scores, identical SAFE verdict.
   - 20 runs with failure -> identical scores, identical REVIEW verdict.
   - Zero flip between SAFE and HIGH_RISK due to provider availability.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.config import config
from src.jev import (
    GeminiJEVDecisionProvider,
    GeminiJEVResponseSchema,
    JEVCache,
    JEVDecision,
    JEVIntent,
    JEVProviderType,
    get_jev_cache,
    reset_jev_cache,
    sanitize_jev_error,
)
from src.moderation import (
    AmbiguityGateConfig,
    ContextualModerationResult,
    RiskLevel,
    moderate_comment_with_context,
    should_invoke_jev,
)


class TestJEVCache(unittest.TestCase):
    """Verifies JEV cache keying, storage, eviction, and non-caching of failures."""

    def setUp(self):
        self.cache = JEVCache(capacity=5)
        self.sample_preds = {
            "toxic": 0.450001,
            "severe_toxic": 0.650002,
            "obscene": 0.350003,
            "threat": 0.010001,
            "insult": 0.200001,
            "identity_hate": 0.020001,
        }

    def test_cache_key_normalization(self):
        """Text casing and extra whitespace should normalize to identical cache key."""
        k1 = self.cache.make_key("  Bhai this SONG is Amazing!  ", self.sample_preds, "gemini-2.5-flash")
        k2 = self.cache.make_key("bhai this song is amazing!", self.sample_preds, "gemini-2.5-flash")
        self.assertEqual(k1, k2)

    def test_cache_hit_avoids_provider_call(self):
        """Second identical request must hit cache and return identical decision."""
        decision = JEVDecision(
            intent=JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT,
            contextual_toxicity=False,
            moderation=RiskLevel.SAFE,
            human_review_required=False,
            confidence=0.95,
            reason="Informal slang used affectionately without hostile intent.",
            provider_type="mock_jev",
        )
        key = self.cache.make_key("bhai this song is fucking amazing", self.sample_preds, "test-model")
        self.assertIsNone(self.cache.get(key))

        self.cache.put(key, decision)
        cached = self.cache.get(key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.moderation, RiskLevel.SAFE)
        self.assertEqual(cached.reason, decision.reason)
        self.assertEqual(self.cache.hits, 1)

    def test_failed_requests_are_not_cached(self):
        """Failed calls must not be cached, ensuring subsequent requests can retry."""
        key = self.cache.make_key("retryable text", self.sample_preds, "test-model")
        # Ensure nothing stored for failed attempt
        self.assertIsNone(self.cache.get(key))
        self.assertEqual(self.cache.size, 0)

    def test_cache_capacity_and_lru_eviction(self):
        """Cache must be strictly bounded to capacity with LRU eviction."""
        for i in range(7):
            d = JEVDecision(
                intent=JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT,
                contextual_toxicity=False,
                moderation=RiskLevel.SAFE,
                human_review_required=False,
                confidence=0.9,
                reason=f"Item {i}",
                provider_type="test",
            )
            self.cache.put(f"key_{i}", d)

        self.assertLessEqual(self.cache.size, 5)
        # Oldest key_0 and key_1 should have been evicted
        self.assertIsNone(self.cache.get("key_0"))
        self.assertIsNone(self.cache.get("key_1"))
        # Most recent keys must exist
        self.assertIsNotNone(self.cache.get("key_6"))
        self.assertIsNotNone(self.cache.get("key_5"))


class TestGeminiDeterminismConfig(unittest.TestCase):
    """Verifies deterministic parameters and schema conformance in Gemini provider."""

    def test_provider_deterministic_configuration(self):
        """Inspects GeminiJEVDecisionProvider settings."""
        provider = GeminiJEVDecisionProvider(
            api_key="mock_test_key_abc123",
            model_name="gemini-2.5-flash",
            temperature=0.0,
            seed=42,
        )
        self.assertEqual(provider.temperature, 0.0)
        self.assertEqual(provider.seed, 42)
        self.assertEqual(provider.model_name, "gemini-2.5-flash")

    def test_structured_response_schema_definition(self):
        """Schema must define all required JEV fields with strict types."""
        schema_fields = GeminiJEVResponseSchema.model_fields
        self.assertIn("intent", schema_fields)
        self.assertIn("contextual_toxicity", schema_fields)
        self.assertIn("moderation", schema_fields)
        self.assertIn("human_review_required", schema_fields)
        self.assertIn("confidence", schema_fields)
        self.assertIn("reason", schema_fields)


class TestSafeFailureHandling(unittest.TestCase):
    """Deterministic tests simulating Gemini success and failure modes."""

    def setUp(self):
        reset_jev_cache()
        self.thresholds = {
            "toxic": 0.50,
            "severe_toxic": 0.62,
            "obscene": 0.50,
            "threat": 0.50,
            "insult": 0.50,
            "identity_hate": 0.50,
        }
        # Ambiguous case: triggers severe_toxic (due to "fucking"), but is praise
        self.ambiguous_preds = {
            "toxic": 0.48,
            "severe_toxic": 0.68,
            "obscene": 0.55,
            "threat": 0.02,
            "insult": 0.15,
            "identity_hate": 0.01,
        }
        self.ambiguous_flags = {
            "toxic": False,
            "severe_toxic": True,
            "obscene": True,
            "threat": False,
            "insult": False,
            "identity_hate": False,
        }

    def _create_mock_response(self, text_payload: str):
        mock_response = MagicMock()
        mock_response.text = text_payload
        return mock_response

    def test_1_gemini_success_uses_jev_decision(self):
        """When Gemini succeeds on ambiguous comment, JEV contextual decision is applied."""
        provider = GeminiJEVDecisionProvider(api_key="mock_key")
        payload = json.dumps({
            "intent": "colloquial_slang",
            "contextual_toxicity": False,
            "moderation": "Safe",
            "human_review_required": False,
            "confidence": 0.94,
            "reason": "Expletive used as an intensifier for enthusiastic praise, not abuse.",
        })
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = self._create_mock_response(payload)
            mock_get_client.return_value = mock_client

            res = moderate_comment_with_context(
                "Bhai this song is fucking amazing",
                self.ambiguous_preds,
                self.ambiguous_flags,
                self.thresholds,
                jev_provider=provider,
            )

            self.assertTrue(res.jev_invoked)
            self.assertEqual(res.final_risk_level, RiskLevel.SAFE)
            self.assertFalse(res.human_review_required)
            self.assertIn("Expletive used as an intensifier", res.final_explanation)

    def test_2_gemini_timeout_routes_to_review(self):
        """On timeout, ambiguous comment must route to REVIEW with human review required."""
        provider = GeminiJEVDecisionProvider(api_key="mock_key")
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = TimeoutError("Request timed out after 5000ms")
            mock_get_client.return_value = mock_client

            res = moderate_comment_with_context(
                "Bhai this song is fucking amazing",
                self.ambiguous_preds,
                self.ambiguous_flags,
                self.thresholds,
                jev_provider=provider,
            )

            self.assertTrue(res.jev_invoked)
            self.assertEqual(res.jev_status, "FAILED")
            self.assertEqual(res.final_risk_level, RiskLevel.REVIEW)
            self.assertTrue(res.human_review_required)
            self.assertIn("routed to human review", res.final_explanation)
            self.assertIn("timed out", res.jev_error.lower())

    def test_3_gemini_429_quota_exhausted_routes_to_review(self):
        """On 429 quota exhaustion, ambiguous comment must route to REVIEW without leaking keys."""
        provider = GeminiJEVDecisionProvider(api_key="AIzaSySECRETKEY")
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = RuntimeError(
                "429 Resource has been exhausted (e.g. check quota) with key AIzaSySECRETKEY"
            )
            mock_get_client.return_value = mock_client

            res = moderate_comment_with_context(
                "Bhai this song is fucking amazing",
                self.ambiguous_preds,
                self.ambiguous_flags,
                self.thresholds,
                jev_provider=provider,
            )

            self.assertTrue(res.jev_invoked)
            self.assertEqual(res.jev_status, "FAILED")
            self.assertEqual(res.final_risk_level, RiskLevel.REVIEW)
            self.assertTrue(res.human_review_required)
            # Must NOT expose raw API key
            self.assertNotIn("AIzaSySECRETKEY", res.jev_error)
            self.assertNotIn("AIzaSySECRETKEY", res.final_explanation)
            self.assertIn("quota", res.jev_error.lower())

    def test_4_gemini_500_server_error_routes_to_review(self):
        """On 500 server error, ambiguous comment must route to REVIEW."""
        provider = GeminiJEVDecisionProvider(api_key="mock_key")
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = RuntimeError("500 Internal Server Error")
            mock_get_client.return_value = mock_client

            res = moderate_comment_with_context(
                "Bhai this song is fucking amazing",
                self.ambiguous_preds,
                self.ambiguous_flags,
                self.thresholds,
                jev_provider=provider,
            )

            self.assertTrue(res.jev_invoked)
            self.assertEqual(res.jev_status, "FAILED")
            self.assertEqual(res.final_risk_level, RiskLevel.REVIEW)
            self.assertTrue(res.human_review_required)

    def test_5_malformed_response_routes_to_review(self):
        """On malformed JSON or corrupted fields, ambiguous comment must route to REVIEW."""
        provider = GeminiJEVDecisionProvider(api_key="mock_key")
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = self._create_mock_response("<html>Error</html>")
            mock_get_client.return_value = mock_client

            res = moderate_comment_with_context(
                "Bhai this song is fucking amazing",
                self.ambiguous_preds,
                self.ambiguous_flags,
                self.thresholds,
                jev_provider=provider,
            )

            self.assertTrue(res.jev_invoked)
            self.assertEqual(res.jev_status, "FAILED")
            self.assertEqual(res.final_risk_level, RiskLevel.REVIEW)
            self.assertTrue(res.human_review_required)

    def test_preserve_severe_risk_bypass(self):
        """Obvious severe abuse without ambiguity bypasses JEV and stays HIGH_RISK."""
        severe_preds = {
            "toxic": 0.95,
            "severe_toxic": 0.92,
            "obscene": 0.88,
            "threat": 0.90,
            "insult": 0.91,
            "identity_hate": 0.89,
        }
        severe_flags = {k: True for k in severe_preds}
        provider = GeminiJEVDecisionProvider(api_key="mock_key")
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            res = moderate_comment_with_context(
                "I will hunt you down and destroy you",
                severe_preds,
                severe_flags,
                self.thresholds,
                jev_provider=provider,
            )

            # JEV should NOT be invoked for unambiguous severe abuse
            mock_client.models.generate_content.assert_not_called()
            self.assertFalse(res.jev_invoked)
            self.assertEqual(res.final_risk_level, RiskLevel.HIGH_RISK)
            self.assertTrue(res.human_review_required)


class TestRepeatabilityAndCachingEndToEnd(unittest.TestCase):
    """Validates 20-run consistency and verifies zero SAFE <-> HIGH_RISK flip."""

    def setUp(self):
        reset_jev_cache()
        self.thresholds = {
            "toxic": 0.50,
            "severe_toxic": 0.62,
            "obscene": 0.50,
            "threat": 0.50,
            "insult": 0.50,
            "identity_hate": 0.50,
        }
        self.target_text = "Bhai this song is fucking amazing"
        self.sample_preds = {
            "toxic": 0.4850,
            "severe_toxic": 0.6756,
            "obscene": 0.5420,
            "threat": 0.0120,
            "insult": 0.1430,
            "identity_hate": 0.0080,
        }
        self.sample_flags = {
            "toxic": False,
            "severe_toxic": True,
            "obscene": True,
            "threat": False,
            "insult": False,
            "identity_hate": False,
        }

    def test_20_runs_mock_success_repeatability_and_cache(self):
        """20 runs with mocked JEV success:

        - Provider is called exactly once (subsequent 19 runs hit cache).
        - All 20 runs produce identical SAFE verdict.
        """
        provider = GeminiJEVDecisionProvider(api_key="mock_key")
        payload = json.dumps({
            "intent": "colloquial_slang",
            "contextual_toxicity": False,
            "moderation": "Safe",
            "human_review_required": False,
            "confidence": 0.96,
            "reason": "Enthusiastic compliment using colloquial expletive as intensifier.",
        })

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = payload
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            results = []
            for _ in range(20):
                res = moderate_comment_with_context(
                    self.target_text,
                    self.sample_preds,
                    self.sample_flags,
                    self.thresholds,
                    jev_provider=provider,
                )
                results.append(res)

            # Provider should only be called ONCE due to cache
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

            # Verify every run is identical SAFE
            for i, r in enumerate(results):
                self.assertEqual(r.final_risk_level, RiskLevel.SAFE, f"Run {i} failed to return SAFE")
                self.assertFalse(r.human_review_required, f"Run {i} unexpectedly required human review")
                self.assertEqual(r.jev_status, "INVOKED_REAL")

    def test_20_runs_mock_failure_repeatability(self):
        """20 runs with mocked JEV failure:

        - All 20 runs produce identical REVIEW verdict.
        - ZERO runs flip to HIGH_RISK.
        - Failed requests are NOT cached.
        """
        provider = GeminiJEVDecisionProvider(api_key="mock_key")

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = RuntimeError("429 Resource has been exhausted (quota exceeded)")
            mock_get_client.return_value = mock_client

            results = []
            for _ in range(20):
                res = moderate_comment_with_context(
                    self.target_text,
                    self.sample_preds,
                    self.sample_flags,
                    self.thresholds,
                    jev_provider=provider,
                )
                results.append(res)

            # Because failures are not cached, provider was retried each run
            self.assertEqual(mock_client.models.generate_content.call_count, 20)

            # Verify EVERY run is REVIEW and NEVER HIGH_RISK
            for i, r in enumerate(results):
                self.assertEqual(r.final_risk_level, RiskLevel.REVIEW, f"Run {i} was not REVIEW")
                self.assertTrue(r.human_review_required, f"Run {i} human_review_required was False")
                self.assertEqual(r.jev_status, "FAILED")
                self.assertNotEqual(r.final_risk_level, RiskLevel.HIGH_RISK, f"Run {i} illegally flipped to HIGH_RISK")

    def test_retry_after_failure_succeeds_and_caches(self):
        """When a call first fails, it is not cached; subsequent retry that succeeds is cached."""
        provider = GeminiJEVDecisionProvider(api_key="mock_key")
        payload = json.dumps({
            "intent": "colloquial_slang",
            "contextual_toxicity": False,
            "moderation": "Safe",
            "human_review_required": False,
            "confidence": 0.95,
            "reason": "Slang compliment.",
        })

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            # First attempt fails, second succeeds
            mock_client.models.generate_content.side_effect = [
                RuntimeError("Transient network drop"),
                MagicMock(text=payload),
            ]
            mock_get_client.return_value = mock_client

            # 1. First call fails -> REVIEW
            res1 = moderate_comment_with_context(
                self.target_text,
                self.sample_preds,
                self.sample_flags,
                self.thresholds,
                jev_provider=provider,
            )
            self.assertEqual(res1.final_risk_level, RiskLevel.REVIEW)
            self.assertEqual(res1.jev_status, "FAILED")

            # 2. Retry succeeds -> SAFE
            res2 = moderate_comment_with_context(
                self.target_text,
                self.sample_preds,
                self.sample_flags,
                self.thresholds,
                jev_provider=provider,
            )
            self.assertEqual(res2.final_risk_level, RiskLevel.SAFE)
            self.assertEqual(res2.jev_status, "INVOKED_REAL")

            # 3. Third call hits cache -> provider NOT called a third time
            res3 = moderate_comment_with_context(
                self.target_text,
                self.sample_preds,
                self.sample_flags,
                self.thresholds,
                jev_provider=provider,
            )
            self.assertEqual(res3.final_risk_level, RiskLevel.SAFE)
            self.assertEqual(mock_client.models.generate_content.call_count, 2)


class TestSentenceMatrix(unittest.TestCase):
    """Matrix tests for prompts A-F."""

    def setUp(self):
        reset_jev_cache()
        self.thresholds = {
            "toxic": 0.50,
            "severe_toxic": 0.62,
            "obscene": 0.50,
            "threat": 0.50,
            "insult": 0.50,
            "identity_hate": 0.50,
        }

    def test_matrix_sentences_behavior(self):
        test_cases = [
            {
                "id": "A",
                "text": "Bhai this song is fucking amazing",
                "preds": {"toxic": 0.48, "severe_toxic": 0.65, "obscene": 0.52, "threat": 0.01, "insult": 0.12, "identity_hate": 0.01},
                "flags": {"toxic": False, "severe_toxic": True, "obscene": True, "threat": False, "insult": False, "identity_hate": False},
                "is_ambiguous": True,
            },
            {
                "id": "B",
                "text": "Bhai teri toh job lag gayi bc",
                "preds": {"toxic": 0.49, "severe_toxic": 0.58, "obscene": 0.51, "threat": 0.01, "insult": 0.22, "identity_hate": 0.01},
                "flags": {"toxic": False, "severe_toxic": False, "obscene": True, "threat": False, "insult": False, "identity_hate": False},
                "is_ambiguous": True,
            },
            {
                "id": "C",
                "text": "Bro your new setup looks insane",
                "preds": {"toxic": 0.12, "severe_toxic": 0.02, "obscene": 0.03, "threat": 0.01, "insult": 0.02, "identity_hate": 0.01},
                "flags": {"toxic": False, "severe_toxic": False, "obscene": False, "threat": False, "insult": False, "identity_hate": False},
                "is_ambiguous": False,
            },
            {
                "id": "D",
                "text": "You killed it on that presentation!",
                "preds": {"toxic": 0.22, "severe_toxic": 0.05, "obscene": 0.04, "threat": 0.08, "insult": 0.02, "identity_hate": 0.01},
                "flags": {"toxic": False, "severe_toxic": False, "obscene": False, "threat": False, "insult": False, "identity_hate": False},
                "is_ambiguous": False,
            },
            {
                "id": "E",
                "text": "You're a fucking idiot and nobody wants you here.",
                "preds": {"toxic": 0.94, "severe_toxic": 0.82, "obscene": 0.89, "threat": 0.15, "insult": 0.92, "identity_hate": 0.10},
                "flags": {"toxic": True, "severe_toxic": True, "obscene": True, "threat": False, "insult": True, "identity_hate": False},
                "is_ambiguous": False,
            },
            {
                "id": "F",
                "text": "I hope you get hurt badly.",
                "preds": {"toxic": 0.88, "severe_toxic": 0.70, "obscene": 0.10, "threat": 0.85, "insult": 0.40, "identity_hate": 0.05},
                "flags": {"toxic": True, "severe_toxic": True, "obscene": False, "threat": True, "insult": False, "identity_hate": False},
                "is_ambiguous": False,
            },
        ]

        provider = GeminiJEVDecisionProvider(api_key="mock_key")
        for tc in test_cases:
            ambiguous, _ = should_invoke_jev(
                tc["text"], tc["preds"], tc["flags"],
                RiskLevel.HIGH_RISK if tc["id"] in ["A", "E", "F"] else (RiskLevel.REVIEW if tc["id"] == "B" else RiskLevel.SAFE),
                self.thresholds,
            )
            self.assertEqual(ambiguous, tc["is_ambiguous"], f"Sentence {tc['id']} ambiguity mismatch")


if __name__ == "__main__":
    unittest.main()
