"""
Model C: Multimodal Meme Baseline Evaluation Script.

Evaluates:
  1. Model A: Production MuRIL (muril_toxicity_model.pt) with frozen Phase 4C thresholds.
  2. Model B: Meme-text adapted MuRIL (muril_model_b.pt).
  3. Image-Only Baseline: CLIP Vision ViT encoder + classification projection.
  4. Model C: Multimodal MuRIL + CLIP Vision fusion model.

Evaluated on:
  - Multi3Hate benchmark (600 parallel real instances: 300 English, 300 Hindi).
  - Evaluated against both Indian consensus (hate_IN) and US consensus (hate_US).
  - Strictly zero threshold tuning on Multi3Hate (frozen Phase 4C thresholds).
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
from scripts.train_meme_multimodal_baseline import MultimodalMemeToxicityModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalModelC")

EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_c"
MODEL_A_CHECKPOINT = _PROJECT_ROOT / "muril_toxicity_model.pt"
MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
MODEL_C_CHECKPOINT = EXP_DIR / "muril_model_c.pt"
RESULTS_OUT = EXP_DIR / "multimodal_eval_results.json"

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


def load_multi3hate_paired_benchmark() -> List[Dict[str, Any]]:
    annot_csv = MULTI3HATE_DIR / "final_annotations.csv"
    hi_csv = MULTI3HATE_DIR / "hi_captions.csv"
    en_csv = MULTI3HATE_DIR / "en_captions.csv"

    if not annot_csv.exists() or not hi_csv.exists() or not en_csv.exists():
        raise FileNotFoundError("Multi3Hate annotation files missing.")

    df_annot = pd.read_csv(annot_csv)
    df_hi = pd.read_csv(hi_csv)
    df_en = pd.read_csv(en_csv)

    hi_map = {r["Meme ID"]: r for _, r in df_hi.iterrows()}
    en_map = {r["Meme ID"]: r for _, r in df_en.iterrows()}

    instances = []
    missing_images = 0

    for _, row in df_annot.iterrows():
        mid = row["Meme ID"]
        hate_in = int(row["IN"]) if pd.notna(row.get("IN")) else None
        hate_us = int(row["US"]) if pd.notna(row.get("US")) else None

        # 1. Hindi Instance
        hi_row = hi_map.get(mid, {})
        hi_text = str(hi_row.get("Translation", "")).replace("<sep>", " ").strip()
        hi_matches = list((MULTI3HATE_IMAGES_BASE / "hi").rglob(f"{mid}.jpg"))
        hi_img_path = hi_matches[0] if hi_matches else None

        if hi_text and hi_img_path and hi_img_path.exists():
            instances.append({
                "sample_id": f"multi3hate_hi_{mid}",
                "meme_id": mid,
                "language": "hindi",
                "text": hi_text,
                "image_path": str(hi_img_path),
                "template_name": str(hi_row.get("Template Name", "")),
                "hate_IN": hate_in,
                "hate_US": hate_us,
            })
        else:
            missing_images += 1

        # 2. English Instance
        en_row = en_map.get(mid, {})
        en_text = str(en_row.get("Original (English)", "")).replace("<sep>", " ").strip()
        en_matches = list((MULTI3HATE_IMAGES_BASE / "en").rglob(f"{mid}.jpg"))
        en_img_path = en_matches[0] if en_matches else None

        if en_text and en_img_path and en_img_path.exists():
            instances.append({
                "sample_id": f"multi3hate_en_{mid}",
                "meme_id": mid,
                "language": "english",
                "text": en_text,
                "image_path": str(en_img_path),
                "template_name": str(en_row.get("Template Name", "")),
                "hate_IN": hate_in,
                "hate_US": hate_us,
            })
        else:
            missing_images += 1

    logger.info("Loaded %d paired Multi3Hate instances (%d missing images)", len(instances), missing_images)
    return instances


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total": len(y_true),
    }


def evaluate_models():
    logger.info("Starting Multi3Hate Multimodal Benchmark Evaluation...")
    instances = load_multi3hate_paired_benchmark()

    # Load Tokenizer & Vision Processor
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model A: Production MuRIL
    logger.info("Loading Model A (Production MuRIL)...")
    model_a = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_a.load_state_dict(torch.load(MODEL_A_CHECKPOINT, map_location=device, weights_only=True))
    model_a.to(device).eval()

    # Model B: Meme-text MuRIL
    logger.info("Loading Model B (Meme Text Baseline)...")
    model_b = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_b.load_state_dict(torch.load(MODEL_B_CHECKPOINT, map_location=device, weights_only=True))
    model_b.to(device).eval()

    # Model C: Multimodal Baseline
    logger.info("Loading Model C (Multimodal MuRIL + Vision)...")
    model_c = MultimodalMemeToxicityModel(num_labels=6)
    model_c.load_state_dict(torch.load(MODEL_C_CHECKPOINT, map_location=device, weights_only=True))
    model_c.to(device).eval()

    toxic_idx = DEFAULT_LABELS.index("toxic")
    threshold = FROZEN_THRESHOLDS["toxic"]

    results = {
        "model_a": {"scores": [], "preds": []},
        "model_b": {"scores": [], "preds": []},
        "model_c_multimodal": {"scores": [], "preds": []},
        "model_c_image_only": {"scores": [], "preds": []},
        "ground_truth_in": [],
        "ground_truth_us": [],
        "languages": [],
        "sample_ids": [],
    }

    with torch.no_grad():
        for i, inst in enumerate(instances):
            text = inst["text"]
            img_path = inst["image_path"]

            # 1. Text tokenization
            enc = tokenizer(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # 2. Image processing
            img = Image.open(img_path).convert("RGB")
            img_inputs = vision_processor(images=img, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(device)

            # --- Model A Inference ---
            logits_a = model_a(input_ids=input_ids, attention_mask=attention_mask)
            score_a = float(torch.sigmoid(logits_a[0, toxic_idx]).cpu().item())
            pred_a = int(score_a >= threshold)

            # --- Model B Inference ---
            logits_b = model_b(input_ids=input_ids, attention_mask=attention_mask)
            score_b = float(torch.sigmoid(logits_b[0, toxic_idx]).cpu().item())
            pred_b = int(score_b >= threshold)

            # --- Model C Multimodal Inference ---
            logits_c_multi = model_c(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, mode="multimodal")
            score_c_multi = float(torch.sigmoid(logits_c_multi[0, toxic_idx]).cpu().item())
            pred_c_multi = int(score_c_multi >= threshold)

            # --- Image-Only Baseline Inference ---
            logits_c_img = model_c(pixel_values=pixel_values, mode="vision_only")
            score_c_img = float(torch.sigmoid(logits_c_img[0, toxic_idx]).cpu().item())
            pred_c_img = int(score_c_img >= threshold)

            results["model_a"]["scores"].append(score_a)
            results["model_a"]["preds"].append(pred_a)
            results["model_b"]["scores"].append(score_b)
            results["model_b"]["preds"].append(pred_b)
            results["model_c_multimodal"]["scores"].append(score_c_multi)
            results["model_c_multimodal"]["preds"].append(pred_c_multi)
            results["model_c_image_only"]["scores"].append(score_c_img)
            results["model_c_image_only"]["preds"].append(pred_c_img)

            results["ground_truth_in"].append(inst["hate_IN"])
            results["ground_truth_us"].append(inst["hate_US"])
            results["languages"].append(inst["language"])
            results["sample_ids"].append(inst["sample_id"])

            if (i + 1) % 100 == 0:
                logger.info("Processed %d/%d instances...", i + 1, len(instances))

    # Compile Evaluation Reports
    y_in = results["ground_truth_in"]
    y_us = results["ground_truth_us"]
    langs = results["languages"]

    report = {
        "benchmark": "Multi3Hate Held-Out Cultural Evaluation",
        "total_evaluated_instances": len(instances),
        "frozen_threshold_used": threshold,
        "indian_consensus": {
            "model_a_production": compute_binary_metrics(y_in, results["model_a"]["preds"]),
            "model_b_text_only": compute_binary_metrics(y_in, results["model_b"]["preds"]),
            "model_c_image_only": compute_binary_metrics(y_in, results["model_c_image_only"]["preds"]),
            "model_c_multimodal": compute_binary_metrics(y_in, results["model_c_multimodal"]["preds"]),
        },
        "us_consensus": {
            "model_a_production": compute_binary_metrics(y_us, results["model_a"]["preds"]),
            "model_b_text_only": compute_binary_metrics(y_us, results["model_b"]["preds"]),
            "model_c_image_only": compute_binary_metrics(y_us, results["model_c_image_only"]["preds"]),
            "model_c_multimodal": compute_binary_metrics(y_us, results["model_c_multimodal"]["preds"]),
        },
        "language_breakdown_in": {
            "hindi": {
                "model_a": compute_binary_metrics(
                    [y for y, l in zip(y_in, langs) if l == "hindi"],
                    [p for p, l in zip(results["model_a"]["preds"], langs) if l == "hindi"]
                ),
                "model_b": compute_binary_metrics(
                    [y for y, l in zip(y_in, langs) if l == "hindi"],
                    [p for p, l in zip(results["model_b"]["preds"], langs) if l == "hindi"]
                ),
                "model_c": compute_binary_metrics(
                    [y for y, l in zip(y_in, langs) if l == "hindi"],
                    [p for p, l in zip(results["model_c_multimodal"]["preds"], langs) if l == "hindi"]
                ),
            },
            "english": {
                "model_a": compute_binary_metrics(
                    [y for y, l in zip(y_in, langs) if l == "english"],
                    [p for p, l in zip(results["model_a"]["preds"], langs) if l == "english"]
                ),
                "model_b": compute_binary_metrics(
                    [y for y, l in zip(y_in, langs) if l == "english"],
                    [p for p, l in zip(results["model_b"]["preds"], langs) if l == "english"]
                ),
                "model_c": compute_binary_metrics(
                    [y for y, l in zip(y_in, langs) if l == "english"],
                    [p for p, l in zip(results["model_c_multimodal"]["preds"], langs) if l == "english"]
                ),
            },
        },
    }

    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation complete! Results saved to %s", RESULTS_OUT)
    print("\n" + "=" * 65)
    print("MULTI3HATE BENCHMARK RESULTS (INDIAN CONSENSUS):")
    for m_name, m_data in report["indian_consensus"].items():
        print(f"  {m_name:25s} | Prec: {m_data['precision']:.4f} | Rec: {m_data['recall']:.4f} | F1: {m_data['f1']:.4f} | FPR: {m_data['fpr']:.4f}")
    print("=" * 65)

    return report


if __name__ == "__main__":
    evaluate_models()
