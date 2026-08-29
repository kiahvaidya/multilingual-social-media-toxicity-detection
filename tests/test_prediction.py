"""
Tests for src.model — model loading and prediction.

These tests use the real serialized model files.
Skip gracefully if .pkl files are missing (CI environment).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.config import config

# Skip all tests in this module if model files are not present
_model_files_exist = (
    config.MODEL_PATH.exists()
    and config.VECTORIZER_PATH.exists()
    and config.COLUMNS_PATH.exists()
)
pytestmark = pytest.mark.skipif(
    not _model_files_exist,
    reason="Model .pkl files not found — skipping prediction tests",
)


@pytest.fixture(scope="module")
def loaded_model():
    """Load model once for all tests in this module."""
    from src.model import load_model
    return load_model()


# ── Model Loading ────────────────────────────────────────

class TestModelLoading:
    def test_model_loads_successfully(self, loaded_model):
        model, vectorizer, labels = loaded_model
        assert model is not None
        assert vectorizer is not None
        assert labels is not None

    def test_labels_are_correct(self, loaded_model):
        _, _, labels = loaded_model
        expected = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        assert list(labels) == expected

    def test_labels_count(self, loaded_model):
        _, _, labels = loaded_model
        assert len(labels) == 6


# ── Single Prediction ───────────────────────────────────

class TestPrediction:
    def test_safe_text_returns_low_score(self, loaded_model):
        from src.model import predict_comment
        model, vectorizer, labels = loaded_model
        preds, score, intensity = predict_comment(
            "Have a wonderful day!", model, vectorizer, labels
        )
        assert score < 0.5
        assert intensity in ("Low", "Moderate")

    def test_toxic_text_returns_high_score(self, loaded_model):
        from src.model import predict_comment
        model, vectorizer, labels = loaded_model
        preds, score, intensity = predict_comment(
            "You are a stupid idiot and I hate you",
            model, vectorizer, labels,
        )
        assert score > 0.3  # Should detect some toxicity

    def test_prediction_returns_all_labels(self, loaded_model):
        from src.model import predict_comment
        model, vectorizer, labels = loaded_model
        preds, _, _ = predict_comment("test text", model, vectorizer, labels)
        assert set(preds.keys()) == set(labels)

    def test_prediction_values_are_floats(self, loaded_model):
        from src.model import predict_comment
        model, vectorizer, labels = loaded_model
        preds, score, intensity = predict_comment("hello", model, vectorizer, labels)
        for v in preds.values():
            assert isinstance(v, float)
        assert isinstance(score, float)
        assert isinstance(intensity, str)

    def test_scores_between_zero_and_one(self, loaded_model):
        from src.model import predict_comment
        model, vectorizer, labels = loaded_model
        preds, score, _ = predict_comment("some random text", model, vectorizer, labels)
        for v in preds.values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= score <= 1.0

    def test_intensity_categories(self, loaded_model):
        from src.model import predict_comment
        model, vectorizer, labels = loaded_model
        _, _, intensity = predict_comment("hello", model, vectorizer, labels)
        assert intensity in ("Low", "Moderate", "High")

    def test_empty_string(self, loaded_model):
        from src.model import predict_comment
        model, vectorizer, labels = loaded_model
        preds, score, intensity = predict_comment("", model, vectorizer, labels)
        # Should not crash, should return valid structure
        assert len(preds) == 6
        assert isinstance(score, float)


# ── Batch Prediction ─────────────────────────────────────

class TestBatchPrediction:
    def test_batch_returns_correct_count(self, loaded_model):
        from src.model import predict_batch
        model, vectorizer, labels = loaded_model
        texts = ["hello", "you suck", "nice day"]
        results = predict_batch(texts, model, vectorizer, labels)
        assert len(results) == 3

    def test_batch_matches_individual(self, loaded_model):
        from src.model import predict_comment, predict_batch
        model, vectorizer, labels = loaded_model
        texts = ["hello world", "this is terrible"]

        individual = [
            predict_comment(t, model, vectorizer, labels) for t in texts
        ]
        batched = predict_batch(texts, model, vectorizer, labels)

        for (i_preds, i_score, i_int), (b_preds, b_score, b_int) in zip(individual, batched):
            assert i_preds == b_preds
            assert abs(i_score - b_score) < 1e-6
            assert i_int == b_int

    def test_empty_batch(self, loaded_model):
        from src.model import predict_batch
        model, vectorizer, labels = loaded_model
        results = predict_batch([], model, vectorizer, labels)
        assert results == []
