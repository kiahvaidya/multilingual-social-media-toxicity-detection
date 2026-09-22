"""
Phase 5C: Output Parity and Optimization Validation Tests.

Compares pre-optimization baseline behavior (fixed padding="max_length", torch.no_grad)
against post-optimization implementation (dynamic padding=True, torch.inference_mode).

Verifies:
  1. Score parity: maximum & mean absolute score difference.
  2. Flag parity: exact boolean match across all multi-label thresholds.
  3. Category parity: exact match of active flagged categories.
  4. Single-comment prediction consistency.
  5. Batch prediction consistency across varying batch sizes.
  6. Empty & whitespace input handling.
  7. Multilingual inputs (English, Hindi Devanagari, Hinglish).
  8. Frozen threshold application integrity.
  9. Logistic Regression baseline fallback functionality.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
import torch
import numpy as np

from src.config import config
from src.model import (
    MuRILForMultiLabelToxicity,
    load_muril_model,
    predict_comment_muril,
    predict_batch_muril,
    load_model as load_logistic_model,
    predict_comment as predict_comment_logistic,
    predict_batch as predict_batch_logistic,
    _compute_intensity,
)

_muril_ckpt_exists = config.MURIL_CHECKPOINT_PATH.exists()
pytest_muril = pytest.mark.skipif(
    not _muril_ckpt_exists,
    reason="muril_toxicity_model.pt not found — skipping MuRIL tests",
)

REPRESENTATIVE_PARITY_CORPUS = [
    # English safe & toxic
    "This is a wonderful, insightful, and helpful contribution.",
    "You are a total idiot, a disgrace, and a moron.",
    "I will find where you live and kill you tonight.",
    "Holy shit, that match was absolutely legendary and thrilling!",
    # Hindi Devanagari
    "यह वीडियो बहुत ही ज्ञानवर्धक और उपयोगी है, धन्यवाद।",
    "तू एक नंबर का झूठा, चोर और बेईमान इंसान है।",
    "तुझे जान से मार डालूंगा कुत्ते, बच के रहना।",
    "भाई क्या धांसू काम किया है, दिल जीत लिया!",
    # Hinglish & code-mixed
    "Bhai content bahut badhiya banaya hai, maza aa gaya.",
    "Tu sabse bada chutiya hai behenchod, chup kar ke baith.",
    "Raste me mil kabhi, goli maar denge tereko.",
    "Bro tune to stage pe sach me aag laga di, deadly performance!",
    # Slang & colloquial
    "That workout literally destroyed me, I am completely dead.",
    "Lmao get wrecked trash, go touch grass you brainless bot.",
    # Obfuscated
    "Nice sh!rt, looks super c00l on you bro!",
    "You are a f**king 1diot and a total l0ser.",
    # Edge cases
    "",
    "   ",
    "1234567890",
    "ok",
    "A " * 50,  # Long sequence testing truncation
]


def _pre_optimization_predict_comment(
    text: str,
    model: MuRILForMultiLabelToxicity,
    tokenizer,
    labels: List[str],
    device: torch.device,
    thresholds: Dict[str, float],
) -> Tuple[Dict[str, float], float, str, Dict[str, bool]]:
    """Pre-optimization baseline: padding='max_length' and torch.no_grad()."""
    model.eval()
    raw_text = str(text).strip() if text is not None else ""

    encoding = tokenizer(
        raw_text,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    predictions: Dict[str, float] = {}
    is_toxic_flags: Dict[str, bool] = {}

    for i, label in enumerate(labels):
        val = float(probs[i]) if i < len(probs) else 0.0
        val_rounded = float(round(val, 4))
        predictions[label] = val_rounded
        th = float(thresholds.get(label, 0.5))
        is_toxic_flags[label] = bool(val >= th)

    toxic_score = max(predictions.values()) if predictions else 0.0
    intensity = _compute_intensity(toxic_score)

    return predictions, toxic_score, intensity, is_toxic_flags


def _pre_optimization_predict_batch(
    texts: List[str],
    model: MuRILForMultiLabelToxicity,
    tokenizer,
    labels: List[str],
    device: torch.device,
    batch_size: int,
    thresholds: Dict[str, float],
) -> List[Tuple[Dict[str, float], float, str, Dict[str, bool]]]:
    """Pre-optimization baseline batch: padding='max_length' and torch.no_grad()."""
    if not texts:
        return []

    model.eval()
    total_samples = len(texts)
    results = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_texts = [str(t).strip() for t in texts[start_idx:end_idx]]

        encoding = tokenizer(
            batch_texts,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            batch_probs = torch.sigmoid(logits).cpu().numpy()

        for probs in batch_probs:
            predictions: Dict[str, float] = {}
            is_toxic_flags: Dict[str, bool] = {}

            for i, label in enumerate(labels):
                val = float(probs[i]) if i < len(probs) else 0.0
                val_rounded = float(round(val, 4))
                predictions[label] = val_rounded
                th = float(thresholds.get(label, 0.5))
                is_toxic_flags[label] = bool(val >= th)

            toxic_score = max(predictions.values()) if predictions else 0.0
            intensity = _compute_intensity(toxic_score)
            results.append((predictions, toxic_score, intensity, is_toxic_flags))

    return results


@pytest.fixture(scope="module")
def loaded_muril():
    if not _muril_ckpt_exists:
        pytest.skip("MuRIL checkpoint not available")
    return load_muril_model()


class TestPhase5COutputParity:
    """Rigorous parity verification between pre- and post-optimization."""

    @pytest_muril
    def test_single_prediction_parity_and_divergence(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril

        max_score_diff = 0.0
        sum_score_diff = 0.0
        total_evaluations = 0
        flag_mismatches = 0
        category_mismatches = 0

        for text in REPRESENTATIVE_PARITY_CORPUS:
            pre_preds, pre_score, pre_int, pre_flags = _pre_optimization_predict_comment(
                text, model, tokenizer, labels, device, thresholds
            )
            post_preds, post_score, post_int, post_flags = predict_comment_muril(
                text, model, tokenizer, labels, device, thresholds
            )

            # Check per-label score differences
            for lbl in labels:
                diff = abs(pre_preds[lbl] - post_preds[lbl])
                max_score_diff = max(max_score_diff, diff)
                sum_score_diff += diff
                total_evaluations += 1

                if pre_flags[lbl] != post_flags[lbl]:
                    flag_mismatches += 1

            # Check category parity
            pre_active = sorted([k for k, v in pre_flags.items() if v])
            post_active = sorted([k for k, v in post_flags.items() if v])
            if pre_active != post_active:
                category_mismatches += 1

            assert pre_int == post_int

        mean_score_diff = sum_score_diff / max(total_evaluations, 1)

        print("\n--- Phase 5C Single Prediction Parity Metrics ---")
        print(f"Total Comments Tested: {len(REPRESENTATIVE_PARITY_CORPUS)}")
        print(f"Max Absolute Score Difference:  {max_score_diff:.6f}")
        print(f"Mean Absolute Score Difference: {mean_score_diff:.6f}")
        print(f"Flag Mismatches:                {flag_mismatches}")
        print(f"Category Mismatches:            {category_mismatches}")

        if max_score_diff == 0.0 and flag_mismatches == 0 and category_mismatches == 0:
            print("Report: 'No observed output difference in the tested parity corpus.'")

        # Tolerances: Rounded to 4 decimals, parity should be exact or at floating precision limit (< 1e-4)
        assert max_score_diff < 1e-4
        assert mean_score_diff < 1e-4
        assert flag_mismatches == 0
        assert category_mismatches == 0

    @pytest_muril
    def test_batch_prediction_parity(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril

        corpus = [t for t in REPRESENTATIVE_PARITY_CORPUS if t.strip()]

        pre_batch = _pre_optimization_predict_batch(
            corpus, model, tokenizer, labels, device, batch_size=4, thresholds=thresholds
        )
        post_batch = predict_batch_muril(
            corpus, model, tokenizer, labels, device, batch_size=4, thresholds=thresholds
        )

        assert len(pre_batch) == len(post_batch)

        max_diff = 0.0
        flag_mismatches = 0

        for (pre_p, pre_s, pre_i, pre_f), (post_p, post_s, post_i, post_f) in zip(pre_batch, post_batch):
            for lbl in labels:
                diff = abs(pre_p[lbl] - post_p[lbl])
                max_diff = max(max_diff, diff)
                if pre_f[lbl] != post_f[lbl]:
                    flag_mismatches += 1
            assert pre_i == post_i

        print(f"\nBatch Parity Max Diff: {max_diff:.6f}, Flag Mismatches: {flag_mismatches}")
        assert max_diff < 1e-3
        assert flag_mismatches == 0

    @pytest_muril
    def test_batch_matches_single_after_optimization(self, loaded_muril):
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
    def test_empty_input_handling(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        preds, score, intensity, flags = predict_comment_muril(
            "", model, tokenizer, labels, device, thresholds
        )
        assert isinstance(preds, dict)
        assert len(preds) == len(labels)
        for val in preds.values():
            assert 0.0 <= val <= 1.0
        assert isinstance(score, float)
        assert intensity in ("Low", "Moderate", "High")
        assert isinstance(flags, dict)

        assert predict_batch_muril([], model, tokenizer, labels, device) == []

    @pytest_muril
    def test_multilingual_text_handling(self, loaded_muril):
        model, tokenizer, labels, thresholds, device = loaded_muril
        # Devanagari
        preds_hi, _, _, _ = predict_comment_muril(
            "नमस्ते दोस्त, आप कैसे हैं?", model, tokenizer, labels, device, thresholds
        )
        assert "toxic" in preds_hi
        # Hinglish
        preds_hing, _, _, _ = predict_comment_muril(
            "Bhai sab theek hai, kya chal raha hai?", model, tokenizer, labels, device, thresholds
        )
        assert "toxic" in preds_hing

    @pytest_muril
    def test_frozen_thresholds_strictly_applied(self, loaded_muril):
        _, _, _, thresholds, _ = loaded_muril
        assert thresholds["toxic"] == 0.45
        assert thresholds["severe_toxic"] == 0.62
        assert thresholds["obscene"] == 0.65
        assert thresholds["threat"] == 0.59
        assert thresholds["insult"] == 0.61
        assert thresholds["identity_hate"] == 0.60

    def test_logistic_regression_fallback(self):
        model, vectorizer, labels = load_logistic_model()
        preds, score, intensity = predict_comment_logistic(
            "Hello world test fallback", model, vectorizer, labels
        )
        assert isinstance(preds, dict)
        assert len(preds) == 6
        assert isinstance(score, float)
        assert intensity in ("Low", "Moderate", "High")
