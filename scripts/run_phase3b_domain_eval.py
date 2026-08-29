"""
Phase 3B Domain Evaluation & Synthetic Stress-Testing Execution Script.

Evaluates the FROZEN Phase 3A model, vectorizer, and optimal thresholds on:
  1. Real-World Social Media Dataset (dataset/real_world_domain_eval.csv)
  2. Synthetic Adversarial Stress-Test Dataset (dataset/synthetic_adversarial_stresstest.csv)

Outputs:
  - Category-by-category breakdown (Sample count, Micro/Macro F1, per-label P/R/F1)
  - Jigsaw vs Real-World distribution shift comparison
  - Error analysis of False Positives and False Negatives for threat, identity_hate, severe_toxic
  - Export to evaluation_results/domain_evaluation_results_phase3b.json
"""

import sys
import os
import json
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib

from src.config import config
from src.preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3B")

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
OUTPUT_DIR = _PROJECT_ROOT / "evaluation_results"


def load_frozen_artifacts():
    logger.info("Loading frozen Phase 3A model, vectorizer, columns, and thresholds...")
    model = joblib.load(config.MODEL_PATH)
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    labels = joblib.load(config.COLUMNS_PATH)
    
    thresholds_file = _PROJECT_ROOT / "toxic_thresholds.json"
    with open(thresholds_file, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
        
    logger.info("Loaded thresholds: %s", thresholds)
    return model, vectorizer, labels, thresholds


def evaluate_subset(df_subset, model, vectorizer, thresholds, category_name):
    """Evaluate a specific subset of comments using frozen model and thresholds."""
    texts = df_subset["comment_text"].tolist()
    y_true = df_subset[LABEL_COLS].values
    
    cleaned = [preprocess_text(t, social_media=True) for t in texts]
    X_vec = vectorizer.transform(cleaned)
    probs = model.predict_proba(X_vec)
    
    thresh_arr = np.array([thresholds.get(col, 0.5) for col in LABEL_COLS])
    y_pred = (probs >= thresh_arr).astype(int)
    
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    per_label = {}
    error_cases = []
    
    for i, col in enumerate(LABEL_COLS):
        p = float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        r = float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1 = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).tolist()
        
        per_label[col] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm,
            "threshold": thresholds.get(col, 0.5)
        }
        
    # Extract sample FP and FN for error analysis
    for idx, (t, yt, yp) in enumerate(zip(texts, y_true, y_pred)):
        for i, col in enumerate(LABEL_COLS):
            if yt[i] == 0 and yp[i] == 1:
                error_cases.append({
                    "type": "False Positive",
                    "label": col,
                    "comment": t,
                    "prob": round(float(probs[idx, i]), 4),
                    "threshold": thresholds.get(col, 0.5)
                })
            elif yt[i] == 1 and yp[i] == 0:
                error_cases.append({
                    "type": "False Negative",
                    "label": col,
                    "comment": t,
                    "prob": round(float(probs[idx, i]), 4),
                    "threshold": thresholds.get(col, 0.5)
                })
                
    return {
        "category": category_name,
        "sample_count": len(df_subset),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_label": per_label,
        "error_cases": error_cases
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model, vectorizer, labels, thresholds = load_frozen_artifacts()
    
    # 1. Load Real-World Dataset
    real_path = config.DATASET_DIR / "real_world_domain_eval.csv"
    if not real_path.exists():
        raise FileNotFoundError(f"Real-world dataset not found at {real_path}")
    real_df = pd.read_csv(real_path)
    
    # 2. Load Synthetic Stress-Test Dataset
    syn_path = config.DATASET_DIR / "synthetic_adversarial_stresstest.csv"
    if not syn_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at {syn_path}")
    syn_df = pd.read_csv(syn_path)
    
    results = {}
    
    # Evaluate Real-World Subsets
    for lang in ["english", "hinglish", "hindi_devanagari"]:
        subset = real_df[real_df["language_cat"] == lang]
        name = f"Real-World {lang.replace('_', ' ').title()}"
        res = evaluate_subset(subset, model, vectorizer, thresholds, name)
        results[lang] = res
        
    # Evaluate Synthetic Stress-Test Subset
    syn_res = evaluate_subset(syn_df, model, vectorizer, thresholds, "Synthetic Slang / Adversarial Stress-Test")
    results["synthetic_adversarial"] = syn_res
    
    # Save Report
    report_file = OUTPUT_DIR / "domain_evaluation_results_phase3b.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Saved Phase 3B Evaluation Report to %s", report_file)
    
    # Print Executive Summary Tables
    print("\n" + "=" * 90)
    print("PHASE 3B DOMAIN EVALUATION EXECUTIVE SUMMARY")
    print("=" * 90)
    headers = f"{'Evaluation Category':<40} | {'Count':<7} | {'Micro F1':<9} | {'Macro F1':<9}"
    print(headers)
    print("-" * len(headers))
    
    for key, data in results.items():
        print(f"{data['category']:<40} | {data['sample_count']:<7} | {data['micro_f1']:<9.4f} | {data['macro_f1']:<9.4f}")
        
    print("\n" + "=" * 90)
    print("PER-CATEGORY & PER-LABEL F1 SCORES (FOCUS CLASSES HIGHLIGHTED)")
    print("=" * 90)
    label_hdr = f"{'Category':<32} | {'toxic':<7} | {'sev_tox*':<8} | {'obscene':<7} | {'threat*':<7} | {'insult':<7} | {'id_hate*':<8}"
    print(label_hdr)
    print("-" * len(label_hdr))
    
    for key, data in results.items():
        p = data["per_label"]
        print(f"{data['category']:<32} | {p['toxic']['f1']:<7.4f} | {p['severe_toxic']['f1']:<8.4f} | {p['obscene']['f1']:<7.4f} | {p['threat']['f1']:<7.4f} | {p['insult']['f1']:<7.4f} | {p['identity_hate']['f1']:<8.4f}")
        
    print("=" * 90)
    print("(* Focus Classes: severe_toxic, threat, identity_hate)\n")


if __name__ == "__main__":
    main()
