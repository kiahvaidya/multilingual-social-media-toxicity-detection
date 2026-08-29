"""
Reproducible Model Evaluation Script.

Reports:
  - Micro / Macro F1
  - Per-label Precision, Recall, F1
  - ROC-AUC (micro, macro, per-label)
  - Confusion matrices (per-label)
  - Per-label optimal thresholds (maximising F1 on validation set)

Usage:
    python scripts/evaluate.py

Outputs are printed to stdout and saved to evaluation_results/.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

from src.config import config
from src.model import load_model
from src.preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = _PROJECT_ROOT / "evaluation_results"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_data():
    """Load and preprocess the training dataset."""
    train_path = config.DATASET_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Please place train.csv in the dataset/ directory."
        )

    logger.info("Loading training data from %s ...", train_path)
    df = pd.read_csv(train_path)
    logger.info("  Shape: %s", df.shape)

    # Apply training-compatible preprocessing (no social-media stage)
    logger.info("Preprocessing text (training-compatible pipeline)...")
    df["comment_text"] = df["comment_text"].apply(
        lambda x: preprocess_text(x, social_media=False)
    )

    return df


def evaluate():
    """Run the full evaluation pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load model & data ────────────────────────────────
    model, vectorizer, labels = load_model()
    df = load_data()

    X_text = df["comment_text"]
    y = df[LABEL_COLS]

    # ── Vectorize ────────────────────────────────────────
    logger.info("Vectorizing with saved TF-IDF vectorizer...")
    X_vec = vectorizer.transform(X_text)

    # ── Train/test split (same as notebook) ──────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )
    logger.info("Test set size: %d samples", X_test.shape[0])

    # ── Get probabilities and binary predictions ─────────
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        logger.warning("predict_proba not available, using predict")
        y_proba = model.predict(X_test).astype(float)

    y_pred = model.predict(X_test)

    y_test_np = y_test.values

    # ══════════════════════════════════════════════════════
    # 1. OVERALL METRICS
    # ══════════════════════════════════════════════════════
    micro_f1 = f1_score(y_test_np, y_pred, average="micro")
    macro_f1 = f1_score(y_test_np, y_pred, average="macro")

    logger.info("\n" + "=" * 60)
    logger.info("OVERALL METRICS")
    logger.info("=" * 60)
    logger.info("  Micro F1:  %.4f", micro_f1)
    logger.info("  Macro F1:  %.4f", macro_f1)

    # ══════════════════════════════════════════════════════
    # 2. PER-LABEL METRICS
    # ══════════════════════════════════════════════════════
    per_label_precision = precision_score(y_test_np, y_pred, average=None, zero_division=0)
    per_label_recall = recall_score(y_test_np, y_pred, average=None, zero_division=0)
    per_label_f1 = f1_score(y_test_np, y_pred, average=None, zero_division=0)

    label_metrics = pd.DataFrame({
        "Label": LABEL_COLS,
        "Precision": per_label_precision,
        "Recall": per_label_recall,
        "F1": per_label_f1,
    }).sort_values("F1", ascending=False)

    logger.info("\n" + "=" * 60)
    logger.info("PER-LABEL METRICS (default threshold = 0.5)")
    logger.info("=" * 60)
    logger.info("\n%s", label_metrics.to_string(index=False))

    # ══════════════════════════════════════════════════════
    # 3. ROC-AUC
    # ══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("ROC-AUC SCORES")
    logger.info("=" * 60)

    auc_per_label = {}
    for i, label in enumerate(LABEL_COLS):
        try:
            auc = roc_auc_score(y_test_np[:, i], y_proba[:, i])
            auc_per_label[label] = auc
            logger.info("  %-15s  AUC = %.4f", label, auc)
        except ValueError as e:
            auc_per_label[label] = None
            logger.info("  %-15s  AUC = N/A (%s)", label, e)

    valid_aucs = [v for v in auc_per_label.values() if v is not None]
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    try:
        micro_auc = roc_auc_score(y_test_np, y_proba, average="micro")
    except ValueError:
        micro_auc = 0.0

    logger.info("  %-15s  AUC = %.4f", "MICRO", micro_auc)
    logger.info("  %-15s  AUC = %.4f", "MACRO", macro_auc)

    # ══════════════════════════════════════════════════════
    # 4. CONFUSION MATRICES
    # ══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("CONFUSION MATRICES")
    logger.info("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, label in enumerate(LABEL_COLS):
        cm = confusion_matrix(y_test_np[:, i], y_pred[:, i])
        logger.info("\n  %s:", label)
        logger.info("    TN=%d  FP=%d", cm[0, 0], cm[0, 1])
        logger.info("    FN=%d  TP=%d", cm[1, 0], cm[1, 1])

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Safe", "Toxic"],
            yticklabels=["Safe", "Toxic"],
            ax=axes_flat[i],
        )
        axes_flat[i].set_title(f"{label}")
        axes_flat[i].set_ylabel("Actual")
        axes_flat[i].set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices (per label, default threshold)", fontsize=14)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("\n  Saved confusion matrix plot to %s", cm_path)

    # ══════════════════════════════════════════════════════
    # 5. PER-LABEL OPTIMAL THRESHOLDS
    # ══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMAL THRESHOLDS (maximising F1 per label)")
    logger.info("=" * 60)

    optimal_thresholds = {}
    threshold_metrics = []

    for i, label in enumerate(LABEL_COLS):
        prec_curve, rec_curve, thresholds = precision_recall_curve(
            y_test_np[:, i], y_proba[:, i]
        )
        # Compute F1 for each threshold
        f1_scores = np.where(
            (prec_curve[:-1] + rec_curve[:-1]) > 0,
            2 * prec_curve[:-1] * rec_curve[:-1] / (prec_curve[:-1] + rec_curve[:-1]),
            0.0,
        )

        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        best_prec = float(prec_curve[best_idx])
        best_rec = float(rec_curve[best_idx])

        optimal_thresholds[label] = best_threshold

        logger.info(
            "  %-15s  threshold=%.3f  → P=%.3f  R=%.3f  F1=%.3f",
            label, best_threshold, best_prec, best_rec, best_f1,
        )

        threshold_metrics.append({
            "Label": label,
            "Optimal_Threshold": round(best_threshold, 4),
            "Precision_at_Opt": round(best_prec, 4),
            "Recall_at_Opt": round(best_rec, 4),
            "F1_at_Opt": round(best_f1, 4),
        })

    # ── Compare default vs optimal thresholds ────────────
    logger.info("\n  Improvement with optimal thresholds:")
    y_pred_optimized = np.zeros_like(y_proba)
    for i, label in enumerate(LABEL_COLS):
        y_pred_optimized[:, i] = (y_proba[:, i] >= optimal_thresholds[label]).astype(int)

    opt_micro_f1 = f1_score(y_test_np, y_pred_optimized, average="micro")
    opt_macro_f1 = f1_score(y_test_np, y_pred_optimized, average="macro")
    logger.info("    Default Micro F1:   %.4f  →  Optimal: %.4f  (Δ %+.4f)",
                micro_f1, opt_micro_f1, opt_micro_f1 - micro_f1)
    logger.info("    Default Macro F1:   %.4f  →  Optimal: %.4f  (Δ %+.4f)",
                macro_f1, opt_macro_f1, opt_macro_f1 - macro_f1)

    # ══════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════
    results_summary = {
        "overall": {
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_auc": round(micro_auc, 4),
            "macro_auc": round(macro_auc, 4),
        },
        "overall_with_optimal_thresholds": {
            "micro_f1": round(opt_micro_f1, 4),
            "macro_f1": round(opt_macro_f1, 4),
        },
        "per_label": label_metrics.to_dict(orient="records"),
        "roc_auc_per_label": {k: round(v, 4) if v else None for k, v in auc_per_label.items()},
        "optimal_thresholds": threshold_metrics,
    }

    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("\n  Full results saved to %s", results_path)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    return results_summary


if __name__ == "__main__":
    evaluate()
