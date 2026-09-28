"""
Phase 6A: Unit tests for JEV (Judicial / Evaluative Verification) Contextual Decision Layer.

Verifies:
  1. Clearly safe comment bypasses JEV.
  2. Clearly high-confidence severe toxic comment bypasses JEV.
  3. Borderline comment invokes JEV.
  4. Ambiguous/slang-style comment invokes JEV.
  5. JEV returns SAFE correctly (disambiguating benign slang/profanity).
  6. JEV returns REVIEW correctly.
  7. JEV returns HIGH_RISK correctly.
  8. Malformed JEV response is handled safely without crashing.
  9. JEV timeout is handled safely without crashing.
  10. Missing JEV configuration falls back to deterministic base moderation.
  11. Mock provider is NEVER auto-selected by get_jev_provider in production.
  12. Frozen Phase 4C thresholds remain strictly unchanged.
  13. ContextualModerationResult fields and serialization integrity.
"""

import io
import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from src.config import config
from src.jev import (
    HTTPJEVDecisionProvider,
    JEVDecision,
    JEVIntent,
    JEVProviderType,
    MockJEVDecisionProvider,
    get_jev_provider,
)
from src.moderation import (
    AmbiguityGateConfig,
    RiskLevel,
    classify_risk,
    moderate_comment_with_context,
    should_invoke_jev,
)


class TestPhase6AJEV(unittest.TestCase):
    """Test suite for Phase 6A JEV contextual moderation layer."""

    def setUp(self):
        self.thresholds = dict(config.FROZEN_MURIL_THRESHOLDS)
        self.mock_provider = MockJEVDecisionProvider()

    # ── 1. Clearly Safe Bypass ───────────────────────────────────────────────
    def test_clearly_safe_comment_bypasses_jev(self):
        text = "Have a wonderful day and enjoy the beautiful sunshine!"
        preds = {cat: 0.05 for cat in self.thresholds}
        flags = {cat: False for cat in self.thresholds}
        base_risk, _, _ = classify_risk(preds, flags, self.thresholds)

        should_invoke, reason = should_invoke_jev(text, preds, flags, base_risk, self.thresholds)
        self.assertFalse(should_invoke)
        self.assertIn("Safe", reason)

        ctx_res = moderate_comment_with_context(
            text, preds, flags, self.thresholds, jev_provider=self.mock_provider
        )
        self.assertFalse(ctx_res.jev_invoked)
        self.assertEqual(ctx_res.jev_status, "BYPASSED_CLEAR_CASE")
        self.assertEqual(ctx_res.final_risk_level, RiskLevel.SAFE)
        self.assertEqual(ctx_res.final_badge_icon, "🟢")

    # ── 2. Clearly High-Confidence Severe Toxic Bypass ────────────────────────
    def test_clearly_high_confidence_severe_threat_bypasses_jev(self):
        text = "I will find you and murder you in cold blood."
        preds = {cat: 0.10 for cat in self.thresholds}
        preds["threat"] = 0.95
        preds["toxic"] = 0.92
        flags = {cat: (preds[cat] >= self.thresholds[cat]) for cat in self.thresholds}
        base_risk, _, _ = classify_risk(preds, flags, self.thresholds)

        should_invoke, reason = should_invoke_jev(text, preds, flags, base_risk, self.thresholds)
        self.assertFalse(should_invoke)
        self.assertIn("severe toxicity", reason.lower())

        ctx_res = moderate_comment_with_context(
            text, preds, flags, self.thresholds, jev_provider=self.mock_provider
        )
        self.assertFalse(ctx_res.jev_invoked)
        self.assertEqual(ctx_res.jev_status, "BYPASSED_CLEAR_CASE")
        self.assertEqual(ctx_res.final_risk_level, RiskLevel.HIGH_RISK)
        self.assertEqual(ctx_res.final_badge_icon, "🔴")

    # ── 3. Borderline Comment Invokes JEV ─────────────────────────────────────
    def test_borderline_comment_invokes_jev(self):
        text = "Your argument is completely flawed and unsupported."
        # toxic threshold is 0.45; score 0.43 is within 0.08 margin
        preds = {cat: 0.10 for cat in self.thresholds}
        preds["toxic"] = 0.43
        flags = {cat: (preds[cat] >= self.thresholds[cat]) for cat in self.thresholds}
        base_risk, _, _ = classify_risk(preds, flags, self.thresholds)

        should_invoke, reason = should_invoke_jev(text, preds, flags, base_risk, self.thresholds)
        self.assertTrue(should_invoke)
        self.assertTrue("threshold" in reason.lower() or "review" in reason.lower())

    # ── 4. Ambiguous Slang / Profanity Invokes JEV ────────────────────────────
    def test_ambiguous_slang_invokes_jev(self):
        text = "Bro that guitar solo was totally sick you killed it!"
        preds = {cat: 0.15 for cat in self.thresholds}
        preds["toxic"] = 0.50  # flagged toxic by lexical match
        preds["threat"] = 0.48  # near threat threshold
        flags = {cat: (preds[cat] >= self.thresholds[cat]) for cat in self.thresholds}
        base_risk, _, _ = classify_risk(preds, flags, self.thresholds)

        should_invoke, reason = should_invoke_jev(text, preds, flags, base_risk, self.thresholds)
        self.assertTrue(should_invoke)
        self.assertTrue("slang" in reason.lower() or "review" in reason.lower() or "threshold" in reason.lower())

    # ── 5. JEV Returns SAFE Correctly (Resolves False Positive) ───────────────
    def test_jev_returns_safe_correctly(self):
        text = "Fuck yeah! We finally passed all unit tests after three days of debugging!"
        preds = {cat: 0.10 for cat in self.thresholds}
        preds["toxic"] = 0.65
        preds["obscene"] = 0.68
        flags = {cat: (preds[cat] >= self.thresholds[cat]) for cat in self.thresholds}

        ctx_res = moderate_comment_with_context(
            text, preds, flags, self.thresholds, jev_provider=self.mock_provider
        )
        self.assertTrue(ctx_res.jev_invoked)
        self.assertEqual(ctx_res.jev_status, "INVOKED_MOCK")
        self.assertEqual(ctx_res.final_risk_level, RiskLevel.SAFE)
        self.assertEqual(ctx_res.final_badge_icon, "🟢")
        self.assertEqual(ctx_res.jev_decision.intent, JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT)
        self.assertFalse(ctx_res.jev_decision.human_review_required)

    # ── 6. JEV Returns REVIEW Correctly ───────────────────────────────────────
    def test_jev_returns_review_correctly(self):
        text = "I don't know if this is genuine criticism or low-key hostility."
        preds = {cat: 0.20 for cat in self.thresholds}
        preds["toxic"] = 0.46
        flags = {cat: (preds[cat] >= self.thresholds[cat]) for cat in self.thresholds}

        # Custom mock provider returning REVIEW
        custom_provider = MagicMock()
        custom_provider.provider_type = JEVProviderType.MOCK_JEV
        custom_provider.assess_context.return_value = JEVDecision(
            intent=JEVIntent.AMBIGUOUS,
            contextual_toxicity=False,
            moderation=RiskLevel.REVIEW,
            human_review_required=True,
            confidence=0.65,
            reason="Ambiguous borderline criticism requiring human inspection.",
            provider_type=JEVProviderType.MOCK_JEV,
        )

        ctx_res = moderate_comment_with_context(
            text, preds, flags, self.thresholds, jev_provider=custom_provider
        )
        self.assertTrue(ctx_res.jev_invoked)
        self.assertEqual(ctx_res.final_risk_level, RiskLevel.REVIEW)
        self.assertEqual(ctx_res.final_badge_icon, "🟡")
        self.assertTrue(ctx_res.jev_decision.human_review_required)

    # ── 7. JEV Returns HIGH_RISK Correctly ────────────────────────────────────
    def test_jev_returns_high_risk_correctly(self):
        text = "You are a pathetic failure and a worthless scumbag."
        preds = {cat: 0.20 for cat in self.thresholds}
        preds["toxic"] = 0.60
        preds["insult"] = 0.62
        flags = {cat: (preds[cat] >= self.thresholds[cat]) for cat in self.thresholds}

        ctx_res = moderate_comment_with_context(
            text, preds, flags, self.thresholds, jev_provider=self.mock_provider
        )
        self.assertTrue(ctx_res.jev_invoked)
        self.assertEqual(ctx_res.final_risk_level, RiskLevel.HIGH_RISK)
        self.assertEqual(ctx_res.final_badge_icon, "🔴")
        self.assertEqual(ctx_res.jev_decision.intent, JEVIntent.GENUINE_ABUSE)
        self.assertTrue(ctx_res.jev_decision.human_review_required)

    # ── 8. Malformed JEV Response Handled Safely ──────────────────────────────
    @patch("urllib.request.urlopen")
    def test_malformed_jev_response_handled_safely(self, mock_urlopen):
        # Simulate non-JSON response from server
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"<html>502 Bad Gateway</html>"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        provider = HTTPJEVDecisionProvider(
            endpoint="https://example.com/jev", api_key="test_key"
        )
        text = "Bro this guitar solo was totally sick!"
        preds = {cat: 0.40 for cat in self.thresholds}
        flags = {cat: False for cat in self.thresholds}

        # Must not crash
        ctx_res = moderate_comment_with_context(
            text, preds, flags, self.thresholds, jev_provider=provider
        )
        self.assertTrue(ctx_res.jev_invoked)
        self.assertEqual(ctx_res.jev_status, "FAILED")
        self.assertIsNotNone(ctx_res.jev_error)
        self.assertEqual(ctx_res.final_risk_level, ctx_res.base_risk_level)

    # ── 9. JEV Timeout Handled Safely ─────────────────────────────────────────
    @patch("urllib.request.urlopen")
    def test_jev_timeout_handled_safely(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("Connection timed out after 3.0s")

        provider = HTTPJEVDecisionProvider(
            endpoint="https://example.com/jev", api_key="test_key", timeout=1.0
        )
        text = "Bhai bawaal cheez banayi tune!"
        preds = {cat: 0.42 for cat in self.thresholds}
        flags = {cat: False for cat in self.thresholds}

        # Must not crash and fall back to base moderation
        ctx_res = moderate_comment_with_context(
            text, preds, flags, self.thresholds, jev_provider=provider
        )
        self.assertTrue(ctx_res.jev_invoked)
        self.assertEqual(ctx_res.jev_status, "FAILED")
        self.assertIn("timed out", ctx_res.jev_error.lower())
        self.assertEqual(ctx_res.final_risk_level, ctx_res.base_risk_level)

    # ── 10. Missing Configuration Fallback ────────────────────────────────────
    def test_missing_jev_config_falls_back_to_deterministic_moderation(self):
        with patch.dict("os.environ", {"JEV_ENABLED": "true", "JEV_ENDPOINT_URL": ""}):
            provider = get_jev_provider()
            self.assertIsNone(provider)

            text = "Bhai bawaal cheez banayi tune!"
            preds = {cat: 0.42 for cat in self.thresholds}
            flags = {cat: False for cat in self.thresholds}

            ctx_res = moderate_comment_with_context(
                text, preds, flags, self.thresholds, jev_provider=None
            )
            self.assertFalse(ctx_res.jev_invoked)
            self.assertEqual(ctx_res.jev_status, "UNCONFIGURED")
            self.assertEqual(ctx_res.final_risk_level, ctx_res.base_risk_level)

    # ── 11. Mock Provider NEVER Auto-Selected in Production ───────────────────
    def test_mock_provider_never_auto_selected(self):
        # Case A: JEV disabled -> returns None
        with patch.dict("os.environ", {"JEV_ENABLED": "false"}):
            self.assertIsNone(get_jev_provider())

        # Case B: JEV enabled but unconfigured -> returns None (NOT Mock)
        with patch.dict("os.environ", {"JEV_ENABLED": "true", "JEV_ENDPOINT_URL": ""}):
            self.assertIsNone(get_jev_provider())

        # Case C: JEV enabled and configured -> returns HTTPJEVDecisionProvider
        with patch.dict(
            "os.environ",
            {
                "JEV_ENABLED": "true",
                "JEV_ENDPOINT_URL": "https://api.moderation.service/v1/assess",
                "JEV_API_KEY": "secret_test_key",
            },
        ):
            prov = get_jev_provider()
            self.assertIsNotNone(prov)
            self.assertIsInstance(prov, HTTPJEVDecisionProvider)
            self.assertEqual(prov.provider_type, JEVProviderType.REAL_JEV)

    # ── 12. Frozen Thresholds Integrity ───────────────────────────────────────
    def test_frozen_thresholds_remain_strictly_unchanged(self):
        expected_thresholds = {
            "toxic": 0.45,
            "severe_toxic": 0.62,
            "obscene": 0.65,
            "threat": 0.59,
            "insult": 0.61,
            "identity_hate": 0.60,
        }
        self.assertEqual(config.FROZEN_MURIL_THRESHOLDS, expected_thresholds)
        self.assertEqual(self.thresholds, expected_thresholds)

    # ── 13. Decision Contract & Serialization ─────────────────────────────────
    def test_jev_decision_contract_and_serialization(self):
        decision = JEVDecision(
            intent=JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT,
            contextual_toxicity=False,
            moderation=RiskLevel.SAFE,
            human_review_required=False,
            confidence=0.95234,
            reason="Celebratory exclamation.",
            provider_type=JEVProviderType.REAL_JEV,
        )
        d = decision.to_dict()
        self.assertEqual(d["intent"], "non_toxic_praise_or_excitement")
        self.assertFalse(d["contextual_toxicity"])
        self.assertEqual(d["moderation"], "Safe")
        self.assertFalse(d["human_review_required"])
        self.assertFalse(d["human_review"])
        self.assertEqual(d["confidence"], 0.9523)
        self.assertEqual(d["provider_type"], "REAL_JEV")


if __name__ == "__main__":
    unittest.main()
