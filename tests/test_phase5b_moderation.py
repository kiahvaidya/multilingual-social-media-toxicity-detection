"""
Unit tests for Phase 5B Moderation UX, risk assessment, and aggregate statistics.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from src.config import config
from src.moderation import (
    RiskLevel,
    ReviewStatus,
    classify_risk,
    calculate_aggregate_statistics,
    get_review_status,
    set_review_status,
)


@pytest.fixture
def standard_thresholds():
    return dict(config.FROZEN_MURIL_THRESHOLDS)


class TestRiskClassification:
    def test_safe_classification(self, standard_thresholds):
        preds = {k: 0.05 for k in standard_thresholds}
        flags = {k: False for k in standard_thresholds}
        level, icon, explanation = classify_risk(preds, flags, standard_thresholds)
        assert level == RiskLevel.SAFE
        assert icon == "🟢"
        assert "safely below" in explanation.lower()

    def test_high_risk_severe_category(self, standard_thresholds):
        # Trigger 'threat' which is in SEVERE_CATEGORIES
        preds = {k: 0.1 for k in standard_thresholds}
        flags = {k: False for k in standard_thresholds}
        preds["threat"] = 0.65
        flags["threat"] = True

        level, icon, explanation = classify_risk(preds, flags, standard_thresholds)
        assert level == RiskLevel.HIGH_RISK
        assert icon == "🔴"
        assert "threat" in explanation.lower()

    def test_high_risk_multiple_categories(self, standard_thresholds):
        # Trigger 2 non-severe categories (toxic and insult)
        preds = {k: 0.1 for k in standard_thresholds}
        flags = {k: False for k in standard_thresholds}
        preds["toxic"] = 0.50
        flags["toxic"] = True
        preds["insult"] = 0.63
        flags["insult"] = True

        level, icon, explanation = classify_risk(preds, flags, standard_thresholds)
        assert level == RiskLevel.HIGH_RISK
        assert icon == "🔴"
        assert "multiple" in explanation.lower()

    def test_high_risk_high_confidence(self, standard_thresholds):
        # Single category flagged with very high confidence (>= 0.75)
        preds = {k: 0.1 for k in standard_thresholds}
        flags = {k: False for k in standard_thresholds}
        preds["toxic"] = 0.82
        flags["toxic"] = True

        level, icon, explanation = classify_risk(preds, flags, standard_thresholds)
        assert level == RiskLevel.HIGH_RISK
        assert icon == "🔴"
        assert "high confidence" in explanation.lower()

    def test_review_single_moderate_category(self, standard_thresholds):
        # Single standard category flagged with moderate probability (< 0.75)
        preds = {k: 0.1 for k in standard_thresholds}
        flags = {k: False for k in standard_thresholds}
        preds["toxic"] = 0.48
        flags["toxic"] = True

        level, icon, explanation = classify_risk(preds, flags, standard_thresholds)
        assert level == RiskLevel.REVIEW
        assert icon == "🟡"
        assert "single category flagged" in explanation.lower()

    def test_review_borderline_unflagged(self, standard_thresholds):
        # No category flagged, but 'toxic' is at 0.42 (threshold is 0.45, difference is 0.03 <= 0.08)
        preds = {k: 0.1 for k in standard_thresholds}
        flags = {k: False for k in standard_thresholds}
        preds["toxic"] = 0.42

        level, icon, explanation = classify_risk(preds, flags, standard_thresholds)
        assert level == RiskLevel.REVIEW
        assert icon == "🟡"
        assert "borderline" in explanation.lower()

    def test_review_elevated_sub_threshold(self, standard_thresholds):
        # No category flagged, but peak score is 0.41 (>= 0.40)
        preds = {k: 0.05 for k in standard_thresholds}
        flags = {k: False for k in standard_thresholds}
        preds["obscene"] = 0.41  # obscene threshold is 0.65, so not within 0.08, but >= 0.40

        level, icon, explanation = classify_risk(preds, flags, standard_thresholds)
        assert level == RiskLevel.REVIEW
        assert icon == "🟡"
        assert "elevated sub-threshold" in explanation.lower()


class TestAggregateStatistics:
    def test_empty_results(self, standard_thresholds):
        stats = calculate_aggregate_statistics([], standard_thresholds)
        assert stats["total"] == 0
        assert stats["toxic_count"] == 0
        assert stats["toxic_percentage"] == 0.0
        assert stats["high_risk_count"] == 0

    def test_batch_statistics_calculation(self, standard_thresholds):
        sample_results = [
            # 1. Safe comment
            (
                {k: 0.05 for k in standard_thresholds},
                0.05,
                "Low",
                {k: False for k in standard_thresholds},
            ),
            # 2. Review comment (borderline toxic at 0.42)
            (
                {k: (0.42 if k == "toxic" else 0.05) for k in standard_thresholds},
                0.42,
                "Moderate",
                {k: False for k in standard_thresholds},
            ),
            # 3. High Risk comment (threat flagged)
            (
                {k: (0.80 if k == "threat" else 0.1) for k in standard_thresholds},
                0.80,
                "High",
                {k: (k == "threat") for k in standard_thresholds},
            ),
            # 4. Toxic comment (toxic & insult flagged)
            (
                {k: (0.70 if k in ("toxic", "insult") else 0.1) for k in standard_thresholds},
                0.70,
                "Moderate",
                {k: (k in ("toxic", "insult")) for k in standard_thresholds},
            ),
        ]

        stats = calculate_aggregate_statistics(sample_results, standard_thresholds)
        assert stats["total"] == 4
        assert stats["toxic_count"] == 2  # items 3 and 4 have flags
        assert stats["safe_count"] == 2
        assert stats["toxic_percentage"] == 50.0
        assert stats["high_risk_count"] == 2  # item 3 (threat) and item 4 (multiple)
        assert stats["review_count"] == 1  # item 2
        assert stats["safe_risk_count"] == 1  # item 1
        assert stats["category_counts"]["threat"] == 1
        assert stats["category_counts"]["toxic"] == 1
        assert stats["category_counts"]["insult"] == 1


class TestHumanReviewSessionWorkflow:
    def test_session_review_lifecycle(self):
        session_store = {}
        cid = "comment_abc123"

        # Default is PENDING
        assert get_review_status(session_store, cid) == ReviewStatus.PENDING.value

        # Mark for Review
        set_review_status(session_store, cid, ReviewStatus.REVIEW.value)
        assert get_review_status(session_store, cid) == ReviewStatus.REVIEW.value

        # Mark Confirmed Toxic
        set_review_status(session_store, cid, ReviewStatus.CONFIRMED_TOXIC.value)
        assert get_review_status(session_store, cid) == ReviewStatus.CONFIRMED_TOXIC.value

        # Mark False Positive
        set_review_status(session_store, cid, ReviewStatus.FALSE_POSITIVE.value)
        assert get_review_status(session_store, cid) == ReviewStatus.FALSE_POSITIVE.value
