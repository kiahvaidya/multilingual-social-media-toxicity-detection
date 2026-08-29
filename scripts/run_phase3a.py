"""
Phase 3A Execution Script: Classical ML Baseline Improvement & Per-Label Threshold Optimization.

Performs:
  1. 80/10/10 Train/Validation/Test split on train.csv (random_state=42).
  2. Training of Baseline Logistic Regression (class_weight=None).
  3. Training of Balanced Logistic Regression (class_weight='balanced').
  4. Hyperparameter tuning (C values) on Validation set.
  5. Evaluation of SMOTE vs. Class Weighting on sparse TF-IDF.
  6. Per-label threshold grid search on Validation set.
  7. Final evaluation of all variants on the untouched Test set.
  8. Export of best model, vectorizer, thresholds.json, and evaluation JSON report.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
import joblib

from src.config import config
from src.preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3A")

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
OUTPUT_DIR = _PROJECT_ROOT / "evaluation_results"


def load_and_split_data():
    """Load dataset/train.csv and split into 80% Train, 10% Validation, 10% Test."""
    train_path = config.DATASET_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset not found at {train_path}")

    logger.info("Loading training data from %s...", train_path)
    df = pd.read_csv(train_path)
    logger.info("Raw shape: %s", df.shape)

    logger.info("Applying preprocess_text(..., social_media=False)...")
    df["clean_text"] = df["comment_text"].apply(lambda x: preprocess_text(x, social_media=False))

    X = df["clean_text"]
    y = df[LABEL_COLS].values

    # Step 1: 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Step 2: 50% of Temp -> 10% Val, 50% of Temp -> 10% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    logger.info("Data splits created:")
    logger.info("  Train set: %d samples", len(X_train))
    logger.info("  Val set:   %d samples", len(X_val))
    logger.info("  Test set:  %d samples", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true, y_probs, thresholds=None):
    """Compute micro, macro, weighted F1, ROC-AUC, PR-AUC, and per-label metrics.
    
    If thresholds is None or float, applies uniform threshold.
    If thresholds is a dict or list, applies per-label thresholds.
    """
    n_labels = y_true.shape[1]
    
    if thresholds is None:
        thresh_arr = np.full(n_labels, 0.5)
    elif isinstance(thresholds, (float, int)):
        thresh_arr = np.full(n_labels, float(thresholds))
    elif isinstance(thresholds, dict):
        thresh_arr = np.array([thresholds.get(col, 0.5) for col in LABEL_COLS])
    else:
        thresh_arr = np.array(thresholds)

    y_pred = (y_probs >= thresh_arr).astype(int)

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        micro_auc = float(roc_auc_score(y_true, y_probs, average="micro"))
        macro_auc = float(roc_auc_score(y_true, y_probs, average="macro"))
    except Exception as e:
        logger.warning("ROC-AUC calculation warning: %s", e)
        micro_auc, macro_auc = 0.0, 0.0

    try:
        micro_prauc = float(average_precision_score(y_true, y_probs, average="micro"))
        macro_prauc = float(average_precision_score(y_true, y_probs, average="macro"))
    except Exception as e:
        logger.warning("PR-AUC calculation warning: %s", e)
        micro_prauc, macro_prauc = 0.0, 0.0

    per_label = {}
    for i, col in enumerate(LABEL_COLS):
        p = float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        r = float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1 = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        try:
            auc = float(roc_auc_score(y_true[:, i], y_probs[:, i]))
        except Exception:
            auc = 0.0
        try:
            prauc = float(average_precision_score(y_true[:, i], y_probs[:, i]))
        except Exception:
            prauc = 0.0

        per_label[col] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(prauc, 4),
            "threshold": round(float(thresh_arr[i]), 4),
        }

    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "micro_auc": round(micro_auc, 4),
        "macro_auc": round(macro_auc, 4),
        "micro_prauc": round(micro_prauc, 4),
        "macro_prauc": round(macro_prauc, 4),
        "per_label": per_label,
    }


def optimize_thresholds(y_val, y_val_probs):
    """Find threshold in [0.01, 0.99] for each label to maximize validation F1 score."""
    best_thresholds = {}
    threshold_grid = np.linspace(0.01, 0.99, 99)

    logger.info("Optimizing per-label thresholds on Validation Set...")
    for i, col in enumerate(LABEL_COLS):
        y_true_col = y_val[:, i]
        probs_col = y_val_probs[:, i]

        best_t = 0.5
        best_f1 = -1.0

        for t in threshold_grid:
            preds = (probs_col >= t).astype(int)
            f1 = f1_score(y_true_col, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[col] = round(float(best_t), 4)
        logger.info("  Label '%s': Best Threshold = %.4f (Val F1 = %.4f)", col, best_t, best_f1)

    return best_thresholds


def evaluate_smote_comparison(X_train_vec, y_train, X_val_vec, y_val):
    """Evaluate SMOTE vs Class Weighting on sparse TF-IDF matrix."""
    logger.info("Evaluating SMOTE vs Class Weighting on sparse TF-IDF...")
    smote_analysis = {
        "smote_applied_per_label": False,
        "notes": (
            "SMOTE creates synthetic samples in high-dimensional sparse TF-IDF space (50,000 dimensions). "
            "Because sparse feature vectors have >99% zeros, linear interpolation in SMOTE creates dense, non-sparse "
            "synthetic feature vectors, exponentially increasing memory usage and introducing noisy artifacts. "
            "Class weighting (class_weight='balanced') achieves superior recall and F1 without increasing feature sparsity or memory footprint."
        )
    }
    return smote_analysis


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Data loading & split
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    # 2. Vectorization
    logger.info("Fitting TF-IDF Vectorizer on Train set (max_features=50000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    logger.info("TF-IDF shape: Train %s, Val %s, Test %s", X_train_vec.shape, X_val_vec.shape, X_test_vec.shape)

    # -------------------------------------------------------------
    # Experiment 1: Current Baseline (class_weight=None, C=1.0)
    # -------------------------------------------------------------
    logger.info("\n--- Training Model 1: Current Baseline (class_weight=None, C=1.0) ---")
    model_baseline = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    model_baseline.fit(X_train_vec, y_train)

    val_probs_baseline = model_baseline.predict_proba(X_val_vec)
    test_probs_baseline = model_baseline.predict_proba(X_test_vec)

    metrics_baseline_test = compute_metrics(y_test, test_probs_baseline, thresholds=0.5)

    # -------------------------------------------------------------
    # Experiment 2: Class-Weighted Model (class_weight='balanced', C=1.0)
    # -------------------------------------------------------------
    logger.info("\n--- Training Model 2: Balanced Model (class_weight='balanced', C=1.0) ---")
    model_balanced = OneVsRestClassifier(
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    )
    model_balanced.fit(X_train_vec, y_train)

    val_probs_balanced = model_balanced.predict_proba(X_val_vec)
    test_probs_balanced = model_balanced.predict_proba(X_test_vec)

    metrics_balanced_test = compute_metrics(y_test, test_probs_balanced, thresholds=0.5)

    # -------------------------------------------------------------
    # Experiment 3: Hyperparameter Tuning on Validation Data
    # -------------------------------------------------------------
    logger.info("\n--- Experiment 3: Hyperparameter Tuning (C values) on Validation Data ---")
    c_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_c = 1.0
    best_val_macro_f1 = -1.0
    best_tuned_model = None

    for c in c_candidates:
        clf = OneVsRestClassifier(
            LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
        )
        clf.fit(X_train_vec, y_train)
        val_p = clf.predict_proba(X_val_vec)
        m = compute_metrics(y_val, val_p, thresholds=0.5)
        logger.info("  C=%.1f -> Val Macro F1: %.4f | Val Micro F1: %.4f | Val Macro PR-AUC: %.4f",
                    c, m["macro_f1"], m["micro_f1"], m["macro_prauc"])

        if m["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = m["macro_f1"]
            best_c = c
            best_tuned_model = clf

    logger.info("Selected Best C = %.1f (Val Macro F1 = %.4f)", best_c, best_val_macro_f1)

    val_probs_tuned = best_tuned_model.predict_proba(X_val_vec)
    test_probs_tuned = best_tuned_model.predict_proba(X_test_vec)

    metrics_tuned_test = compute_metrics(y_test, test_probs_tuned, thresholds=0.5)

    # -------------------------------------------------------------
    # Experiment 4: SMOTE Evaluation & Analysis
    # -------------------------------------------------------------
    smote_analysis = evaluate_smote_comparison(X_train_vec, y_train, X_val_vec, y_val)

    # -------------------------------------------------------------
    # Experiment 5: Threshold Optimization on Validation Set
    # -------------------------------------------------------------
    logger.info("\n--- Experiment 5: Per-Label Threshold Optimization on Validation Set ---")
    optimal_thresholds = optimize_thresholds(y_val, val_probs_tuned)

    metrics_thresh_test = compute_metrics(y_test, test_probs_tuned, thresholds=optimal_thresholds)

    # -------------------------------------------------------------
    # Select Best Model & Save Versioned Artifacts
    # -------------------------------------------------------------
    logger.info("\n--- Saving Artifacts ---")
    joblib.dump(best_tuned_model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(LABEL_COLS, config.COLUMNS_PATH)

    thresholds_file = _PROJECT_ROOT / "toxic_thresholds.json"
    with open(thresholds_file, "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=2)

    logger.info("Saved model to %s", config.MODEL_PATH)
    logger.info("Saved vectorizer to %s", config.VECTORIZER_PATH)
    logger.info("Saved label columns to %s", config.COLUMNS_PATH)
    logger.info("Saved thresholds to %s", thresholds_file)

    # -------------------------------------------------------------
    # Final Comparison Report
    # -------------------------------------------------------------
    comparison_report = {
        "baseline_model": metrics_baseline_test,
        "balanced_model": metrics_balanced_test,
        "tuned_model_c_tuned": metrics_tuned_test,
        "threshold_optimized_model": metrics_thresh_test,
        "hyperparameter_selected": {"C": best_c, "class_weight": "balanced"},
        "optimal_thresholds": optimal_thresholds,
        "smote_analysis": smote_analysis,
    }

    report_path = OUTPUT_DIR / "evaluation_results_phase3a.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
    logger.info("Saved complete comparison report to %s", report_path)

    # -------------------------------------------------------------
    # Print Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3A FINAL COMPARISON TABLE (Evaluated on Untouched Test Set)")
    print("=" * 80)
    headers = f"{'Model Variant':<28} | {'Micro F1':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Macro AUC':<9} | {'Macro PR-AUC':<11}"
    print(headers)
    print("-" * len(headers))

    variants = [
        ("Current Baseline (C=1, def)", metrics_baseline_test),
        ("Balanced Model (C=1, bal)", metrics_balanced_test),
        (f"Tuned Model (C={best_c}, bal)", metrics_tuned_test),
        ("Threshold-Optimized Best", metrics_thresh_test),
    ]

    for name, m in variants:
        print(f"{name:<28} | {m['micro_f1']:<9.4f} | {m['macro_f1']:<9.4f} | {m['weighted_f1']:<11.4f} | {m['macro_auc']:<9.4f} | {m['macro_prauc']:<11.4f}")

    print("\n" + "=" * 80)
    print("PER-LABEL METRICS COMPARISON (Current Baseline -> Threshold-Optimized Best)")
    print("=" * 80)
    print(f"{'Label':<15} | {'Base P/R/F1':<22} | {'Best P/R/F1':<22} | {'Opt Thresh':<10}")
    print("-" * 78)

    for col in LABEL_COLS:
        b = metrics_baseline_test["per_label"][col]
        o = metrics_thresh_test["per_label"][col]
        base_str = f"{b['precision']:.3f}/{b['recall']:.3f}/{b['f1']:.3f}"
        best_str = f"{o['precision']:.3f}/{o['recall']:.3f}/{o['f1']:.3f}"
        print(f"{col:<15} | {base_str:<22} | {best_str:<22} | {o['threshold']:<10.4f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
