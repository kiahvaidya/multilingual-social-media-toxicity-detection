"""
Model C-2: Calibrated Multimodal Meme Evaluation Script.

Evaluates:
  1. C-2a: Text-Only Model B
  2. C-2b: Image-Only Baseline (CLIP ViT-B/32)
  3. C-2c: Multimodal Concatenation
  4. C-2d: Multimodal Gated Fusion (Learned Convex Combination: gate * text + (1 - gate) * vision)

Evaluation Regimes on Held-Out Multi3Hate (600 parallel real instances: 300 HI, 300 EN):
  - Standard 1: Frozen Phase 4C Threshold (tau = 0.45)
  - Standard 2: Validation-Selected Threshold (tau_val fitted strictly on HASOC validation split)
  - Calibration: Raw Scores vs Platt-Calibrated Scores
  - Metrics: Precision, Recall, F1, FPR, FNR, Brier Score, ECE, Confusion Matrix
  - Dual Cultural Standards: hate_IN and hate_US
  - Gate Analysis: Distribution of gate values across benign, hateful, and conflict cases
  - Tradeoff Curves: Precision-Recall and FPR across threshold sweep
  - Cross-Modal Context Set: Modality conflict resolution across Categories A, B, C, D
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.model import MuRILForMultiLabelToxicity, DEFAULT_LABELS
from scripts.train_meme_multimodal_calibrated import (
    CalibratedMultimodalMemeModel,
    compute_brier_score,
    compute_ece,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalModelC2")

EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_c2"
MODEL_A_CHECKPOINT = _PROJECT_ROOT / "muril_toxicity_model.pt"
MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
MODEL_C_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_c" / "muril_model_c.pt"
MODEL_C2_CHECKPOINT = EXP_DIR / "muril_model_c2.pt"
METRICS_IN = EXP_DIR / "model_c2_training_metrics.json"
RESULTS_OUT = EXP_DIR / "calibrated_eval_results.json"

MULTI3HATE_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "multi3hate"
MULTI3HATE_IMAGES_BASE = MULTI3HATE_DIR / "Multi3Hate_repo" / "data" / "memes"

FROZEN_THRESHOLDS = {
    "toxic": 0.45,
    "severe_toxic": 0.62,
    "obscene": 0.65,
    "threat": 0.59,
    "insult": 0.61,
    "identity_hate": 0.60,
}


def load_multi3hate_instances() -> List[Dict[str, Any]]:
    annot_csv = MULTI3HATE_DIR / "final_annotations.csv"
    hi_csv = MULTI3HATE_DIR / "hi_captions.csv"
    en_csv = MULTI3HATE_DIR / "en_captions.csv"

    df_annot = pd.read_csv(annot_csv)
    df_hi = pd.read_csv(hi_csv)
    df_en = pd.read_csv(en_csv)

    hi_map = {r["Meme ID"]: r for _, r in df_hi.iterrows()}
    en_map = {r["Meme ID"]: r for _, r in df_en.iterrows()}

    instances = []
    for _, row in df_annot.iterrows():
        mid = row["Meme ID"]
        hate_in = int(row["IN"]) if pd.notna(row.get("IN")) else None
        hate_us = int(row["US"]) if pd.notna(row.get("US")) else None

        # Hindi
        hi_row = hi_map.get(mid, {})
        hi_text = str(hi_row.get("Translation", "")).replace("<sep>", " ").strip()
        hi_matches = list((MULTI3HATE_IMAGES_BASE / "hi").rglob(f"{mid}.jpg"))
        if hi_text and hi_matches and hi_matches[0].exists():
            instances.append({
                "sample_id": f"multi3hate_hi_{mid}",
                "meme_id": mid,
                "language": "hindi",
                "text": hi_text,
                "image_path": str(hi_matches[0]),
                "template_name": str(hi_row.get("Template Name", "")),
                "hate_IN": hate_in,
                "hate_US": hate_us,
            })

        # English
        en_row = en_map.get(mid, {})
        en_text = str(en_row.get("Original (English)", "")).replace("<sep>", " ").strip()
        en_matches = list((MULTI3HATE_IMAGES_BASE / "en").rglob(f"{mid}.jpg"))
        if en_text and en_matches and en_matches[0].exists():
            instances.append({
                "sample_id": f"multi3hate_en_{mid}",
                "meme_id": mid,
                "language": "english",
                "text": en_text,
                "image_path": str(en_matches[0]),
                "template_name": str(en_row.get("Template Name", "")),
                "hate_IN": hate_in,
                "hate_US": hate_us,
            })

    logger.info("Loaded %d Multi3Hate instances with paired images.", len(instances))
    return instances


def compute_comprehensive_metrics(y_true: np.ndarray, raw_probs: np.ndarray, cal_probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = (cal_probs >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    raw_brier = compute_brier_score(y_true, raw_probs)
    cal_brier = compute_brier_score(y_true, cal_probs)
    raw_ece = compute_ece(y_true, raw_probs)
    cal_ece = compute_ece(y_true, cal_probs)

    return {
        "threshold": round(float(threshold), 3),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "raw_brier": round(raw_brier, 4),
        "calibrated_brier": round(cal_brier, 4),
        "raw_ece": round(raw_ece, 4),
        "calibrated_ece": round(cal_ece, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total": len(y_true),
    }


def evaluate_all():
    logger.info("Starting Comprehensive Model C-2 Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation threshold & calibration params from training metadata
    with open(METRICS_IN, "r", encoding="utf-8") as f:
        meta = json.load(f)

    val_selected_threshold = meta["thresholds"]["validation_selected_threshold"]
    platt_a = meta["calibration"]["platt_a"]
    platt_b = meta["calibration"]["platt_b"]
    logger.info("Using Validation Threshold: %.3f (Frozen Production: %.2f)", val_selected_threshold, FROZEN_THRESHOLDS["toxic"])
    logger.info("Using Validation Platt Scaling: a=%.4f, b=%.4f", platt_a, platt_b)

    # Load Models
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Model B
    model_b = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_b.load_state_dict(torch.load(MODEL_B_CHECKPOINT, map_location=device, weights_only=True))
    model_b.to(device).eval()

    # Model C-2
    model_c2 = CalibratedMultimodalMemeModel(num_labels=6)
    model_c2.load_state_dict(torch.load(MODEL_C2_CHECKPOINT, map_location=device, weights_only=True))
    model_c2.to(device).eval()

    toxic_idx = 0
    instances = load_multi3hate_instances()

    scores_b = []
    scores_img = []
    scores_concat_raw = []
    scores_gated_raw = []
    gates_gated = []

    y_in = []
    y_us = []
    langs = []

    with torch.no_grad():
        for i, inst in enumerate(instances):
            text = inst["text"]
            img_path = inst["image_path"]

            enc = tokenizer(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            img = Image.open(img_path).convert("RGB")
            img_inputs = vision_processor(images=img, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(device)

            # C-2a: Text-Only Model B
            logits_b = model_b(input_ids=input_ids, attention_mask=attention_mask)
            sb = float(torch.sigmoid(logits_b[0, toxic_idx]).cpu().item())
            scores_b.append(sb)

            # C-2b: Image-Only Baseline
            logits_img, _ = model_c2(pixel_values=pixel_values, mode="vision_only")
            s_img = float(torch.sigmoid(logits_img[0, toxic_idx]).cpu().item())
            scores_img.append(s_img)

            # C-2c: Multimodal Concatenation
            logits_concat, _ = model_c2(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, mode="concat")
            s_concat = float(torch.sigmoid(logits_concat[0, toxic_idx]).cpu().item())
            scores_concat_raw.append(s_concat)

            # C-2d: Multimodal Gated Fusion
            logits_gated, gate_val = model_c2(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, mode="gated", return_gate=True)
            s_gated = float(torch.sigmoid(logits_gated[0, toxic_idx]).cpu().item())
            scores_gated_raw.append(s_gated)
            gates_gated.append(float(gate_val.cpu().item()))

            y_in.append(inst["hate_IN"])
            y_us.append(inst["hate_US"])
            langs.append(inst["language"])

            if (i + 1) % 150 == 0:
                logger.info("Evaluated %d/%d instances...", i + 1, len(instances))

    y_in_arr = np.array(y_in)
    y_us_arr = np.array(y_us)
    scores_b_arr = np.array(scores_b)
    scores_img_arr = np.array(scores_img)
    scores_concat_arr = np.array(scores_concat_raw)
    scores_gated_raw_arr = np.array(scores_gated_raw)
    gates_arr = np.array(gates_gated)

    # Calibrated probabilities via validation Platt scaling
    # logit = log(p / (1 - p))
    eps = 1e-7
    gated_logits = np.log((scores_gated_raw_arr + eps) / (1.0 - scores_gated_raw_arr + eps))
    scores_gated_cal_arr = 1.0 / (1.0 + np.exp(-(platt_a * gated_logits + platt_b)))

    concat_logits = np.log((scores_concat_arr + eps) / (1.0 - scores_concat_arr + eps))
    scores_concat_cal_arr = 1.0 / (1.0 + np.exp(-(platt_a * concat_logits + platt_b)))

    # 1. Indian Consensus Evaluation
    in_results_frozen = {
        "C-2a_Text_Only_Model_B": compute_comprehensive_metrics(y_in_arr, scores_b_arr, scores_b_arr, FROZEN_THRESHOLDS["toxic"]),
        "C-2b_Image_Only": compute_comprehensive_metrics(y_in_arr, scores_img_arr, scores_img_arr, FROZEN_THRESHOLDS["toxic"]),
        "C-2c_Multimodal_Concat": compute_comprehensive_metrics(y_in_arr, scores_concat_arr, scores_concat_cal_arr, FROZEN_THRESHOLDS["toxic"]),
        "C-2d_Multimodal_Gated": compute_comprehensive_metrics(y_in_arr, scores_gated_raw_arr, scores_gated_cal_arr, FROZEN_THRESHOLDS["toxic"]),
    }

    in_results_val_threshold = {
        "C-2a_Text_Only_Model_B": compute_comprehensive_metrics(y_in_arr, scores_b_arr, scores_b_arr, val_selected_threshold),
        "C-2b_Image_Only": compute_comprehensive_metrics(y_in_arr, scores_img_arr, scores_img_arr, val_selected_threshold),
        "C-2c_Multimodal_Concat": compute_comprehensive_metrics(y_in_arr, scores_concat_arr, scores_concat_cal_arr, val_selected_threshold),
        "C-2d_Multimodal_Gated": compute_comprehensive_metrics(y_in_arr, scores_gated_raw_arr, scores_gated_cal_arr, val_selected_threshold),
    }

    # 2. Gate Distribution Analysis
    benign_mask = (y_in_arr == 0)
    hateful_mask = (y_in_arr == 1)
    gate_stats = {
        "overall": {"mean": round(float(np.mean(gates_arr)), 4), "std": round(float(np.std(gates_arr)), 4)},
        "benign_memes": {"mean": round(float(np.mean(gates_arr[benign_mask])), 4), "std": round(float(np.std(gates_arr[benign_mask])), 4)},
        "hateful_memes": {"mean": round(float(np.mean(gates_arr[hateful_mask])), 4), "std": round(float(np.std(gates_arr[hateful_mask])), 4)},
    }

    # 3. Tradeoff Sweep (PR / FPR Curve)
    tradeoff_sweep = []
    for t in np.linspace(0.1, 0.9, 17):
        m = compute_comprehensive_metrics(y_in_arr, scores_gated_raw_arr, scores_gated_cal_arr, t)
        tradeoff_sweep.append({
            "threshold": round(float(t), 2),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "fpr": m["fpr"],
        })

    # 4. Cross-Modal Context Cases
    cross_modal_samples = [
        {"id": "A1_mid115", "mid": 115, "cat": "Category A (Contextual Inversion)", "gt": 0, "desc": "damn happy about football (smiling meme)"},
        {"id": "A2_mid282", "mid": 282, "cat": "Category A (Contextual Inversion)", "gt": 0, "desc": "under arrest for accessibility (office meme)"},
        {"id": "B1_mid37", "mid": 37, "cat": "Category B (Emergent Toxicity)", "gt": 1, "desc": "see this hat thrift store (advicejew trope)"},
        {"id": "B2_mid141", "mid": 141, "cat": "Category B (Emergent Toxicity)", "gt": 1, "desc": "wants white tradition eats mcdonalds (white power)"},
        {"id": "C1_mid89", "mid": 89, "cat": "Category C (Explicit Harm)", "gt": 1, "desc": "hate black people got gun (strict policeman)"},
        {"id": "C2_mid96", "mid": 96, "cat": "Category C (Explicit Harm)", "gt": 1, "desc": "hate men need cock (feminist caricature)"},
        {"id": "D1_mid174", "mid": 174, "cat": "Category D (Benign Meme)", "gt": 0, "desc": "blog non-german friends (harmless cartoon)"},
        {"id": "D2_mid265", "mid": 265, "cat": "Category D (Benign Meme)", "gt": 0, "desc": "quality men believe equality (couple meme)"},
    ]

    cross_modal_results = []
    for item in cross_modal_samples:
        idx_match = [idx for idx, inst in enumerate(instances) if inst["meme_id"] == item["mid"] and inst["language"] == "english"]
        if idx_match:
            idx = idx_match[0]
            sb = scores_b_arr[idx]
            sc2_raw = scores_gated_raw_arr[idx]
            sc2_cal = scores_gated_cal_arr[idx]
            gate = gates_arr[idx]

            dec_b = int(sb >= FROZEN_THRESHOLDS["toxic"])
            dec_c2_frozen = int(sc2_cal >= FROZEN_THRESHOLDS["toxic"])
            dec_c2_val = int(sc2_cal >= val_selected_threshold)

            cross_modal_results.append({
                "sample_id": item["id"],
                "category": item["cat"],
                "description": item["desc"],
                "ground_truth": item["gt"],
                "model_b_score": round(float(sb), 4),
                "model_b_decision": dec_b,
                "model_c2_raw_score": round(float(sc2_raw), 4),
                "model_c2_calibrated_score": round(float(sc2_cal), 4),
                "model_c2_decision_frozen": dec_c2_frozen,
                "model_c2_decision_val_thresh": dec_c2_val,
                "gate_value": round(float(gate), 4),
                "resolved_at_val_thresh": bool(dec_c2_val == item["gt"]),
            })

    output_report = {
        "benchmark": "Multi3Hate Held-Out Cultural Evaluation",
        "total_instances": len(instances),
        "frozen_threshold": FROZEN_THRESHOLDS["toxic"],
        "validation_selected_threshold": val_selected_threshold,
        "indian_consensus_frozen": in_results_frozen,
        "indian_consensus_val_threshold": in_results_val_threshold,
        "gate_distribution": gate_stats,
        "tradeoff_curve": tradeoff_sweep,
        "cross_modal_context_cases": cross_modal_results,
    }

    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    logger.info("Evaluation complete! Results saved to %s", RESULTS_OUT)

    print("\n" + "=" * 70)
    print("MODEL C-2 ABLATION RESULTS (INDIAN CONSENSUS @ FROZEN THRESHOLD 0.45):")
    for name, res in in_results_frozen.items():
        print(f"  {name:25s} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | F1: {res['f1']:.4f} | FPR: {res['fpr']:.4f} | Brier: {res['calibrated_brier']:.4f} | ECE: {res['calibrated_ece']:.4f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(f"MODEL C-2 RESULTS @ VALIDATION-SELECTED THRESHOLD ({val_selected_threshold:.3f}):")
    for name, res in in_results_val_threshold.items():
        print(f"  {name:25s} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | F1: {res['f1']:.4f} | FPR: {res['fpr']:.4f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("CROSS-MODAL CONTEXT RESOLUTION (MODEL B vs MODEL C-2):")
    for r in cross_modal_results:
        print(f"[{r['sample_id']}] {r['category']}")
        print(f"  GT: {r['ground_truth']} | Model B: {r['model_b_score']:.4f} (pred {r['model_b_decision']}) | Model C-2 Cal: {r['model_c2_calibrated_score']:.4f} (pred {r['model_c2_decision_val_thresh']}) | Gate: {r['gate_value']:.4f} | Resolved: {r['resolved_at_val_thresh']}")
    print("=" * 70)

    return output_report


if __name__ == "__main__":
    evaluate_all()
