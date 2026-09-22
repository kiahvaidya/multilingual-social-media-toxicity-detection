"""
Phase 5C: Multilingual Quality Evaluation & False Positive Analysis.

Evaluates MuRIL on the curated multilingual evaluation set (evaluation_results/multilingual_eval_phase5c.json)
using the frozen Phase 4C thresholds.

Computes:
  - Precision, Recall, F1, False Positive Rate (FPR), False Negative Rate (FNR)
  - Broken down by language/category: English, Hindi, Hinglish, Slang, Obfuscation, and Overall.
  - Granular False Positive and False Negative inspection.
  - Classified error patterns (hypothesized causes):
      * enthusiastic slang interpreted as aggression
      * ambiguous wording
      * profanity used non-aggressively
      * code-mixed interpretation
      * contextual ambiguity
      * obfuscation
      * other

Outputs to: evaluation_results/multilingual_evaluation_phase5c.json
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import config
from src.model import load_muril_model, predict_comment_muril

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase5CMultilingualEval")

EVAL_SET_PATH = _PROJECT_ROOT / "evaluation_results" / "multilingual_eval_phase5c.json"
OUTPUT_REPORT_PATH = _PROJECT_ROOT / "evaluation_results" / "multilingual_evaluation_phase5c.json"

FROZEN_THRESHOLDS = config.FROZEN_MURIL_THRESHOLDS


def classify_fp_error(sample: Dict[str, Any], pred_score: float) -> str:
    """Hypothesize likely error pattern for false positives without unsubstantiated claims."""
    stype = sample.get("sample_type", "")
    ambiguity = sample.get("ambiguity", "")
    lang = sample.get("language", "")
    text = sample.get("text", "").lower()
    
    if "profanity_non_aggressive" in stype:
        return "profanity used non-aggressively"
    elif "enthusiastic_slang" in stype:
        return "enthusiastic slang interpreted as aggression"
    elif "metaphorical_slang" in stype:
        return "enthusiastic slang interpreted as aggression"
    elif ambiguity == "contextual":
        return "contextual ambiguity"
    elif lang == "obfuscation":
        return "obfuscation"
    elif lang in ("hinglish", "hindi") and ambiguity in ("lexical", "cultural"):
        return "code-mixed interpretation"
    elif ambiguity == "lexical":
        return "ambiguous wording"
    return "other"


def classify_fn_error(sample: Dict[str, Any], pred_score: float) -> str:
    """Hypothesize likely error pattern for false negatives."""
    lang = sample.get("language", "")
    stype = sample.get("sample_type", "")
    if lang == "obfuscation":
        return "adversarial obfuscation evasion"
    elif lang == "hinglish":
        return "code-mixed vocabulary gap"
    elif lang == "slang":
        return "informal colloquialism / gaming slang mismatch"
    elif "threat" in stype:
        return "indirect or subtle threat phrasing"
    return "other"


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute Precision, Recall, F1, FPR, FNR safely."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
    }


def run_evaluation():
    logger.info("Loading curated multilingual evaluation set from %s...", EVAL_SET_PATH)
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"Evaluation set not found at {EVAL_SET_PATH}")
        
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = data.get("samples", [])
    logger.info("Loaded %d curated samples.", len(samples))
    
    logger.info("Loading MuRIL model engine with frozen thresholds: %s", FROZEN_THRESHOLDS)
    model, tokenizer, labels, thresholds, device = load_muril_model()
    
    # Store predictions
    evaluation_records = []
    
    for s in samples:
        text = s["text"]
        preds, score, intensity, flags = predict_comment_muril(
            text=text,
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            device=device,
            thresholds=FROZEN_THRESHOLDS
        )
        
        # Primary binary toxicity flag (toxic threshold = 0.45)
        pred_toxic = bool(flags.get("toxic", False))
        expected_toxic = bool(s["expected_toxic"])
        
        # Predicted active categories
        pred_categories = [lbl for lbl, flg in flags.items() if flg]
        
        rec = {
            "id": s["id"],
            "text": text,
            "language": s["language"],
            "sample_type": s["sample_type"],
            "ambiguity": s["ambiguity"],
            "expected_toxic": expected_toxic,
            "expected_categories": s["expected_categories"],
            "pred_toxic": pred_toxic,
            "pred_score": score,
            "pred_categories": pred_categories,
            "predictions_raw": preds,
            "is_fp": (not expected_toxic and pred_toxic),
            "is_fn": (expected_toxic and not pred_toxic)
        }
        evaluation_records.append(rec)
        
    # Slices for evaluation
    slices = ["english", "hindi", "hinglish", "slang", "obfuscation"]
    slice_results = {}
    
    for sl in slices:
        subset = [r for r in evaluation_records if r["language"] == sl]
        y_true = [1 if r["expected_toxic"] else 0 for r in subset]
        y_pred = [1 if r["pred_toxic"] else 0 for r in subset]
        metrics = compute_metrics(y_true, y_pred)
        metrics["sample_count"] = len(subset)
        slice_results[sl] = metrics
        
    # Overall metrics
    overall_y_true = [1 if r["expected_toxic"] else 0 for r in evaluation_records]
    overall_y_pred = [1 if r["pred_toxic"] else 0 for r in evaluation_records]
    overall_metrics = compute_metrics(overall_y_true, overall_y_pred)
    overall_metrics["sample_count"] = len(evaluation_records)
    
    # Analyze False Positives
    fp_records = [r for r in evaluation_records if r["is_fp"]]
    detailed_fps = []
    for fp in fp_records:
        error_pattern = classify_fp_error(fp, fp["pred_score"])
        detailed_fps.append({
            "id": fp["id"],
            "text": fp["text"],
            "language": fp["language"],
            "sample_type": fp["sample_type"],
            "predicted_labels": fp["pred_categories"],
            "relevant_score": fp["pred_score"],
            "expected_label": False,
            "likely_error_pattern": error_pattern,
            "hypothesized_cause": (
                f"Model flagged text with score {fp['pred_score']:.4f}. "
                f"Hypothesized cause: {error_pattern}."
            )
        })
        
    # Analyze False Negatives
    fn_records = [r for r in evaluation_records if r["is_fn"]]
    detailed_fns = []
    for fn in fn_records:
        error_pattern = classify_fn_error(fn, fn["pred_score"])
        detailed_fns.append({
            "id": fn["id"],
            "text": fn["text"],
            "language": fn["language"],
            "sample_type": fn["sample_type"],
            "predicted_labels": fn["pred_categories"],
            "relevant_score": fn["pred_score"],
            "expected_label": True,
            "likely_error_pattern": error_pattern,
            "hypothesized_cause": (
                f"Model failed to detect toxicity (score {fn['pred_score']:.4f} < 0.45). "
                f"Hypothesized cause: {error_pattern}."
            )
        })
        
    final_report = {
        "evaluation_name": "Phase 5C Multilingual Quality & Error Pattern Analysis",
        "timestamp": pd.Timestamp.now().isoformat(),
        "disclaimer": (
            "This analysis is based on a hand-curated evaluation set designed to probe specific "
            "linguistic, slang, and obfuscation behaviors. It is NOT statistically representative of "
            "general social media population distributions."
        ),
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "overall_metrics": overall_metrics,
        "slice_metrics": slice_results,
        "false_positive_count": len(detailed_fps),
        "false_positive_analysis": detailed_fps,
        "false_negative_count": len(detailed_fns),
        "false_negative_analysis": detailed_fns,
    }
    
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
        
    logger.info("Evaluation report saved to %s", OUTPUT_REPORT_PATH)
    
    # Print formatted markdown table
    print("\n" + "="*85)
    print("PHASE 5C MULTILINGUAL EVALUATION RESULTS (FROZEN THRESHOLDS)")
    print("="*85)
    print(f"{'Language / Slice':<18} | {'Samples':<8} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'FPR':<8} | {'FNR':<8}")
    print("-" * 85)
    for sl, m in slice_results.items():
        print(f"{sl.capitalize():<18} | {m['sample_count']:<8} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['fpr']:<8.4f} | {m['fnr']:<8.4f}")
    print("-" * 85)
    om = overall_metrics
    print(f"{'OVERALL':<18} | {om['sample_count']:<8} | {om['precision']:<10.4f} | {om['recall']:<10.4f} | {om['f1']:<10.4f} | {om['fpr']:<8.4f} | {om['fnr']:<8.4f}")
    print("="*85)
    print(f"Total False Positives Recorded: {len(detailed_fps)}")
    for fp in detailed_fps:
        safe_text = fp['text'].encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(f"  [FP] ({fp['language']}) \"{safe_text}\" -> Pred: {fp['predicted_labels']} (Score: {fp['relevant_score']:.4f}) | Pattern: {fp['likely_error_pattern']}")
    print(f"Total False Negatives Recorded: {len(detailed_fns)}")
    for fn in detailed_fns:
        safe_text = fn['text'].encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(f"  [FN] ({fn['language']}) \"{safe_text}\" -> Pred: {fn['predicted_labels']} (Score: {fn['relevant_score']:.4f}) | Pattern: {fn['likely_error_pattern']}")
    print("="*85 + "\n")
    
    return final_report


if __name__ == "__main__":
    run_evaluation()
