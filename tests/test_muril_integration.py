"""
Unit and integration tests for Phase 5A MuRIL production integration.

Verifies:
  1. Model & tokenizer loading
  2. Device detection & CPU fallback
  3. English inference (safe, toxic, threat)
  4. Hinglish & Hindi inference (toxic, insult, code-mixed)
  5. Batch prediction consistency with single prediction
  6. Frozen Phase 4C threshold loading & application
  7. Graceful error handling for missing checkpoints
  8. Logistic Regression baseline regression tests
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
import torch
import numpy as np

from src.config import config
from src.model import (
    MuRILForMultiLabelToxicity,
    get_device,
    load_muril_model,
    load_muril_thresholds,
    predict_comment_muril,
    predict_batch_muril,
    load_model as load_logistic_model,
    predict_comment as predict_comment_logistic,
    predict_batch as predict_batch_logistic,
)

# Check if MuRIL checkpoint exists
_muril_ckpt_exists = config.MURIL_CHECKPOINT_PATH.exists()
pytest_muril = pytest.mark.skipif(
    not _muril_ckpt_exists,
    reason="muril_toxicity_model.pt not found — skipping MuRIL tests",
)


@pytest.fixture(scope="module")
def loaded_muril():
    """Load MuRIL engine once for all tests in this module."""
    if not _muril_ckpt_exists:
        pytest.skip("MuRIL checkpoint not available")
    return load_muril_model()


# ── 1. Model Loading & Configuration ──────────────────────────

class TestMuRILLoading:
    @pytest_muril
    def test_muril_loads_successfully(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        assert isinstance(model, MuRILForMultiLabelToxicity)
        assert tokenizer is not None
        assert isinstance(labels, list)
        assert len(labels) == 6
        assert isinstance(thresholds, dict)
        assert isinstance(device, torch.device)

    @pytest_muril
    def test_muril_labels_match_order(self, loaded_muril):
        _, _, labels, _, _ = loaded_muril
        expected = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        assert labels == expected

    @pytest_muril
    def test_muril_frozen_thresholds_exact(self, loaded_muril):
        _, _, _, thresholds, _ = loaded_muril
        expected = {
            "toxic": 0.45,
            "severe_toxic": 0.62,
            "obscene": 0.65,
            "threat": 0.59,
            "insult": 0.61,
            "identity_hate": 0.60,
        }
        for k, v in expected.items():
            assert abs(thresholds[k] - v) < 1e-4

    def test_device_detection(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "cpu")

    def test_missing_checkpoint_raises_filenotfound(self, tmp_path):
        fake_path = tmp_path / "non_existent.pt"
        with pytest.raises(FileNotFoundError):
            load_muril_model(checkpoint_path=fake_path)


# ── 2. Multilingual Inference Tests ───────────────────────────

class TestMultilingualInference:
    @pytest_muril
    def test_english_safe_comment(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        text = "This is a truly insightful and well-researched article. Thank you!"
        preds, score, intensity, flags = predict_comment_muril(
            text, model, tokenizer, labels, device, thresholds
        )
        assert score < 0.45
        assert intensity in ("Low", "Moderate")
        assert flags["toxic"] is False
        assert flags["threat"] is False

    @pytest_muril
    def test_english_toxic_insult_comment(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        text = "You are a complete idiot, a disgrace, and a total moron."
        preds, score, intensity, flags = predict_comment_muril(
            text, model, tokenizer, labels, device, thresholds
        )
        assert preds["toxic"] >= 0.45
        assert flags["toxic"] is True
        assert intensity in ("High", "Moderate")

    @pytest_muril
    def test_english_threat_comment(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        text = "I will find where you live and kill you."
        preds, score, intensity, flags = predict_comment_muril(
            text, model, tokenizer, labels, device, thresholds
        )
        assert preds["threat"] >= 0.40
        assert flags["threat"] is True or preds["toxic"] >= 0.45

    @pytest_muril
    def test_hinglish_toxic_comment(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        text = "Tu chutiya hai behenchod chup baith bilkul."
        preds, score, intensity, flags = predict_comment_muril(
            text, model, tokenizer, labels, device, thresholds
        )
        assert preds["toxic"] >= 0.45
        assert flags["toxic"] is True

    @pytest_muril
    def test_hindi_toxic_comment(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        text = "तू सबसे बड़ा कुत्ता और कमीना है, तुझे मार डालूंगा।"
        preds, score, intensity, flags = predict_comment_muril(
            text, model, tokenizer, labels, device, thresholds
        )
        assert preds["toxic"] >= 0.45
        assert flags["toxic"] is True

    @pytest_muril
    def test_prediction_output_structure(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        preds, score, intensity, flags = predict_comment_muril(
            "Sample evaluation comment", model, tokenizer, labels, device, thresholds
        )
        assert set(preds.keys()) == set(labels)
        assert set(flags.keys()) == set(labels)
        for val in preds.values():
            assert 0.0 <= val <= 1.0
            assert isinstance(val, float)
        assert isinstance(score, float)
        assert intensity in ("Low", "Moderate", "High")


# ── 3. Batch Inference & Consistency ──────────────────────────

class TestBatchInference:
    @pytest_muril
    def test_batch_prediction_matches_single(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        texts = [
            "Great work on this project!",
            "You are an idiot.",
            "Bhai mast video!",
            "तू बहुत बुरा इंसान है।",
        ]
        batch_results = predict_batch_muril(
            texts, model, tokenizer, labels, device, batch_size=2, thresholds=thresholds
        )
        assert len(batch_results) == 4

        for text, (b_preds, b_score, b_intensity, b_flags) in zip(texts, batch_results):
            s_preds, s_score, s_intensity, s_flags = predict_comment_muril(
                text, model, tokenizer, labels, device, thresholds
            )
            for k in labels:
                assert abs(b_preds[k] - s_preds[k]) < 1e-3
            assert abs(b_score - s_score) < 1e-3
            assert b_intensity == s_intensity
            assert b_flags == s_flags

    @pytest_muril
    def test_empty_batch(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        assert predict_batch_muril([], model, tokenizer, labels, device) == []


# ── 4. Logistic Regression Regression Tests ────────────────────

class TestLogisticRegressionFallback:
    def test_logistic_model_loads_and_predicts(self):
        model, vectorizer, labels = load_logistic_model()
        preds, score, intensity = predict_comment_logistic(
            "Hello world", model, vectorizer, labels
        )
        assert isinstance(preds, dict)
        assert len(preds) == 6
        assert isinstance(score, float)
        assert intensity in ("Low", "Moderate", "High")

    def test_logistic_batch_predicts(self):
        model, vectorizer, labels = load_logistic_model()
        texts = ["Safe text", "You are stupid"]
        results = predict_batch_logistic(texts, model, vectorizer, labels)
        assert len(results) == 2

    def test_api_health_and_text_analysis_fallback(self):
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        h_resp = client.get("/api/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert h_data["status"] == "ONLINE"
        assert "active_engine" in h_data

        t_resp = client.post("/api/analyze/text", json={"text": "Shut up you idiot"})
        assert t_resp.status_code == 200
        t_data = t_resp.json()
        assert t_data["is_toxic"] is True
        assert t_data["risk_level"] in ("Review", "High Risk")
