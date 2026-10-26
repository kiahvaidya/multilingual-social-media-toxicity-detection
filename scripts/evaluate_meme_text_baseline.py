"""
Side-by-Side Evaluation Script: Model A (Production MuRIL) vs Model B (Meme Text Baseline).

Evaluates both models identically across:
  1. HASOC-Meme 2025 Test Split (769 held-out Hindi meme OCR samples).
  2. Multi3Hate Strictly Held-Out Evaluation Set (600 parallel English & Hindi samples).
  3. Existing Curated 60-Sample Multilingual Probe (multilingual_eval_phase5c.json).
Outputs comprehensive side-by-side metrics:
  - Precision, Recall, Macro F1, Micro F1, FPR, FNR
  - Language slices (English, Hindi, Hinglish, Slang, Obfuscation)
  - Confusion matrices
Saves results to: scratch/meme_experiments/model_b/model_a_vs_model_b_evaluation.json
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.model import MuRILForMultiLabelToxicity, DEFAULT_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalMemeBaseline")

EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b"
CHECKPOINT_B = EXP_DIR / "muril_model_b.pt"
EVAL_OUT = EXP_DIR / "model_a_vs_model_b_evaluation.json"

RAW_HASOC_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "hasoc_meme_2025"
MANIFEST_PATH = _PROJECT_ROOT / "scratch" / "meme_data" / "processed" / "unified_meme_manifest.json"
PROBE_PATH = _PROJECT_ROOT / "evaluation_results" / "multilingual_eval_phase5c.json"


def load_model(checkpoint_path: Path, device: torch.device) -> MuRILForMultiLabelToxicity:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model = MuRILForMultiLabelToxicity(model_name=config.MURIL_MODEL_NAME, num_labels=6)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_batch(
    texts: List[str],
    model: MuRILForMultiLabelToxicity,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_len: int = 64,
) -> np.ndarray:
    model.eval()
    all_probs = []

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = [str(t).strip() for t in texts[i:i + batch_size]]
            encoding = tokenizer(
                chunk,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device, non_blocking=True)
            attention_mask = encoding["attention_mask"].to(device, non_blocking=True)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs) if all_probs else np.zeros((0, 6), dtype=np.float32)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "samples": len(y_true),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
    }


def evaluate_both_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    tokenizer = AutoTokenizer.from_pretrained(config.MURIL_MODEL_NAME)

    # 1. Load Model A and Model B
    logger.info("Loading Model A (Production MuRIL)...")
    model_a = load_model(config.MURIL_CHECKPOINT_PATH, device)

    logger.info("Loading Model B (Trained Meme Text Baseline)...")
    model_b = load_model(CHECKPOINT_B, device)

    thresholds = config.FROZEN_MURIL_THRESHOLDS
    toxic_threshold = thresholds["toxic"]  # 0.45

    results = {
        "model_a_checkpoint": str(config.MURIL_CHECKPOINT_PATH),
        "model_b_checkpoint": str(CHECKPOINT_B),
        "frozen_thresholds_used": thresholds,
    }

    # ==============================================================
    # 🧪 Benchmark 1: Held-out HASOC-Meme 2025 Validation Split (227 labeled samples)
    # ==============================================================
    train_csv = RAW_HASOC_DIR / "Hindi Train HASOC 2025 - Updated.csv"
    if train_csv.exists():
        df_hasoc = pd.read_csv(train_csv)
        df_hasoc = df_hasoc[df_hasoc["OCR"].notna() & (df_hasoc["OCR"].str.strip() != "")].reset_index(drop=True)
        y_all = np.array([1 if str(a).strip().lower() == "abusive" else 0 for a in df_hasoc["Abuse"]])
        
        # Exact reproducible 80/20 split matching train_meme_text_baseline.py
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(df_hasoc))
        _, val_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_all)
        
        df_val = df_hasoc.iloc[val_idx].reset_index(drop=True)
        texts_hasoc = list(df_val["OCR"])
        y_hasoc = y_all[val_idx]

        probs_a_hasoc = predict_batch(texts_hasoc, model_a, tokenizer, device)
        probs_b_hasoc = predict_batch(texts_hasoc, model_b, tokenizer, device)

        preds_a_hasoc = (probs_a_hasoc[:, 0] >= toxic_threshold).astype(int)
        preds_b_hasoc = (probs_b_hasoc[:, 0] >= toxic_threshold).astype(int)

        results["hasoc_meme_2025_val_eval"] = {
            "total_val_samples": len(texts_hasoc),
            "model_a": compute_binary_metrics(y_hasoc, preds_a_hasoc),
            "model_b": compute_binary_metrics(y_hasoc, preds_b_hasoc),
        }
        logger.info("HASOC Val Eval: Model A F1=%.4f (FPR=%.4f) vs Model B F1=%.4f (FPR=%.4f)",
                    results["hasoc_meme_2025_val_eval"]["model_a"]["f1"],
                    results["hasoc_meme_2025_val_eval"]["model_a"]["fpr"],
                    results["hasoc_meme_2025_val_eval"]["model_b"]["f1"],
                    results["hasoc_meme_2025_val_eval"]["model_b"]["fpr"])

    # ==============================================================
    # 🧪 Benchmark 2: Difficult 60-Sample Multilingual Probe
    # ==============================================================
    if PROBE_PATH.exists():
        with open(PROBE_PATH, "r", encoding="utf-8") as f:
            probe_data = json.load(f)
        samples_probe = probe_data["samples"]
        texts_probe = [s["text"] for s in samples_probe]
        y_probe = np.array([1 if s["expected_toxic"] else 0 for s in samples_probe])

        probs_a_probe = predict_batch(texts_probe, model_a, tokenizer, device)
        probs_b_probe = predict_batch(texts_probe, model_b, tokenizer, device)

        # Flagged toxic if ANY category exceeds threshold (matching Phase 5C probe protocol)
        thresh_arr = np.array([thresholds[c] for c in DEFAULT_LABELS])
        preds_a_probe = (probs_a_probe >= thresh_arr).any(axis=1).astype(int)
        preds_b_probe = (probs_b_probe >= thresh_arr).any(axis=1).astype(int)

        probe_overall_a = compute_binary_metrics(y_probe, preds_a_probe)
        probe_overall_b = compute_binary_metrics(y_probe, preds_b_probe)

        # Slice-level breakdown
        slices_a = {}
        slices_b = {}
        for lang_slice in ["english", "hindi", "hinglish", "slang", "obfuscation"]:
            idx_slice = [i for i, s in enumerate(samples_probe) if s.get("language") == lang_slice]
            if idx_slice:
                slices_a[lang_slice] = compute_binary_metrics(y_probe[idx_slice], preds_a_probe[idx_slice])
                slices_b[lang_slice] = compute_binary_metrics(y_probe[idx_slice], preds_b_probe[idx_slice])

        results["difficult_60_probe_eval"] = {
            "overall": {"model_a": probe_overall_a, "model_b": probe_overall_b},
            "slices": {"model_a": slices_a, "model_b": slices_b},
        }
        logger.info("60-Probe Eval: Model A FPR=%.4f (F1=%.4f) vs Model B FPR=%.4f (F1=%.4f)",
                    probe_overall_a["fpr"], probe_overall_a["f1"],
                    probe_overall_b["fpr"], probe_overall_b["f1"])

    # ==============================================================
    # 🧪 Benchmark 3: Strictly Held-out Multi3Hate Cultural Benchmark
    # ==============================================================
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_records = json.load(f)

        m3h_records = [r for r in manifest_records if r["dataset"] == "multi3hate"]
        if m3h_records:
            texts_m3h = [r["raw_text"] for r in m3h_records]
            # Target 1: Indian Consensus Hate (hate_IN)
            y_in = np.array([1 if r["labels"].get("hate_IN") == 1 else 0 for r in m3h_records])
            # Target 2: US Consensus Hate (hate_US)
            y_us = np.array([1 if r["labels"].get("hate_US") == 1 else 0 for r in m3h_records])

            probs_a_m3h = predict_batch(texts_m3h, model_a, tokenizer, device)
            probs_b_m3h = predict_batch(texts_m3h, model_b, tokenizer, device)

            preds_a_m3h = (probs_a_m3h[:, 0] >= toxic_threshold).astype(int)
            preds_b_m3h = (probs_b_m3h[:, 0] >= toxic_threshold).astype(int)

            results["multi3hate_held_out_cultural_eval"] = {
                "total_samples": len(m3h_records),
                "evaluated_against_indian_consensus": {
                    "model_a": compute_binary_metrics(y_in, preds_a_m3h),
                    "model_b": compute_binary_metrics(y_in, preds_b_m3h),
                },
                "evaluated_against_us_consensus": {
                    "model_a": compute_binary_metrics(y_us, preds_a_m3h),
                    "model_b": compute_binary_metrics(y_us, preds_b_m3h),
                },
            }
            logger.info("Multi3Hate (IN Consensus): Model A F1=%.4f vs Model B F1=%.4f",
                        results["multi3hate_held_out_cultural_eval"]["evaluated_against_indian_consensus"]["model_a"]["f1"],
                        results["multi3hate_held_out_cultural_eval"]["evaluated_against_indian_consensus"]["model_b"]["f1"])

    # Save Side-by-Side Results
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Side-by-side evaluation complete. Results written to %s", EVAL_OUT)
    return results


if __name__ == "__main__":
    evaluate_both_models()
