"""
Phase 6A: Experimental Evaluation of Contextual Decision Layer (JEV).

Compares:
  Pipeline A: MuRIL + Deterministic Moderation (Phase 5C Baseline)
  Pipeline B: MuRIL + Ambiguity Gate + REAL JEV (Production Path)
  Pipeline C: MuRIL + Ambiguity Gate + MOCK JEV (Deterministic Regression Simulation Only)

IMPORTANT INTEGRITY PRINCIPLE:
If no real JEV endpoint/credentials are configured in the environment:
  - REAL_JEV_AVAILABLE is reported as False.
  - Zero results are fabricated.
  - Pipeline B demonstrates 100% graceful fallback to deterministic moderation.
  - MockJEV simulation results are strictly labeled as regression tests and NOT as real JEV performance.

Outputs to: evaluation_results/phase6a_experiment_results.json
"""

import io
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import config
from src.jev import (
    JEVDecision,
    JEVIntent,
    JEVProviderType,
    MockJEVDecisionProvider,
    get_jev_provider,
)
from src.model import load_muril_model, predict_comment_muril
from src.moderation import (
    RiskLevel,
    classify_risk,
    moderate_comment_with_context,
    should_invoke_jev,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase6AExperiment")

EVAL_SET_PATH = _PROJECT_ROOT / "evaluation_results" / "multilingual_eval_phase5c.json"
OUTPUT_REPORT_PATH = _PROJECT_ROOT / "evaluation_results" / "phase6a_experiment_results.json"


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculate Precision, Recall, F1, FPR, FNR."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "total": len(y_true),
    }


def run_experiment() -> Dict[str, Any]:
    logger.info("Starting Phase 6A Contextual Moderation Experiment")

    # 1. Check Real JEV Provider Availability
    real_provider = get_jev_provider()
    real_jev_available = (
        real_provider is not None
        and real_provider.provider_type == JEVProviderType.REAL_JEV
    )
    logger.info("Real JEV provider available: %s", real_jev_available)

    # 2. Load Evaluation Probe Set
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"Evaluation set not found at {EVAL_SET_PATH}")

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    samples = eval_data.get("samples", [])
    logger.info("Loaded %d probe samples from %s", len(samples), EVAL_SET_PATH)

    # 3. Load MuRIL Model & Frozen Thresholds
    muril_model, muril_tokenizer, muril_labels, muril_thresholds, muril_device = load_muril_model()
    frozen_thresholds = dict(config.FROZEN_MURIL_THRESHOLDS)

    # Ground truth labels (1 = toxic, 0 = non-toxic)
    y_true = [1 if s.get("expected_toxic", False) else 0 for s in samples]

    # Pre-compute MuRIL inferences to guarantee identical baseline inputs
    logger.info("Computing MuRIL inference across all %d samples...", len(samples))
    muril_inference_cache = []
    start_inf = time.perf_counter()
    for s in samples:
        text = s["text"]
        preds, score, intensity, flags = predict_comment_muril(
            text, muril_model, muril_tokenizer, muril_labels, muril_device, frozen_thresholds
        )
        muril_inference_cache.append((text, preds, score, intensity, flags))
    inf_duration = time.perf_counter() - start_inf
    logger.info("MuRIL inference completed in %.2fs (%.2f ms/comment)", inf_duration, (inf_duration / len(samples)) * 1000)

    # ── Pipeline A: MuRIL + Deterministic Moderation (Baseline) ───────────────
    logger.info("Evaluating Pipeline A: Deterministic Baseline...")
    pipe_a_preds = []
    pipe_a_risk_levels = []
    for text, preds, score, intensity, flags in muril_inference_cache:
        risk_level, badge_icon, explanation = classify_risk(preds, flags, frozen_thresholds)
        pipe_a_risk_levels.append(risk_level.value)
        # In deterministic moderation, toxic if any flag is raised or risk is HIGH_RISK
        is_toxic_action = (risk_level == RiskLevel.HIGH_RISK) or any(flags.values())
        pipe_a_preds.append(1 if is_toxic_action else 0)

    metrics_a = compute_metrics(y_true, pipe_a_preds)
    review_count_a = sum(1 for r in pipe_a_risk_levels if r == RiskLevel.REVIEW.value)
    metrics_a["review_rate"] = round(review_count_a / len(samples), 4)

    # ── Pipeline B: MuRIL + Ambiguity Gate + REAL JEV (Production Path) ───────
    logger.info("Evaluating Pipeline B: Real JEV Production Path...")
    pipe_b_preds = []
    pipe_b_statuses = []
    pipe_b_invocations = 0
    pipe_b_latencies = []

    for text, preds, score, intensity, flags in muril_inference_cache:
        t0 = time.perf_counter()
        ctx_res = moderate_comment_with_context(
            text, preds, flags, frozen_thresholds, jev_provider=real_provider
        )
        pipe_b_latencies.append((time.perf_counter() - t0) * 1000)
        pipe_b_statuses.append(ctx_res.jev_status)
        if ctx_res.jev_invoked:
            pipe_b_invocations += 1

        # Action: toxic if final risk level is HIGH_RISK or contextual toxicity is True
        if ctx_res.jev_decision:
            is_toxic = ctx_res.final_risk_level == RiskLevel.HIGH_RISK or ctx_res.jev_decision.contextual_toxicity
        else:
            is_toxic = (ctx_res.final_risk_level == RiskLevel.HIGH_RISK) or any(flags.values())
        pipe_b_preds.append(1 if is_toxic else 0)

    metrics_b = compute_metrics(y_true, pipe_b_preds)
    metrics_b["real_jev_available"] = real_jev_available
    metrics_b["jev_invocation_rate"] = round(pipe_b_invocations / len(samples), 4)
    metrics_b["fallback_to_base_rate"] = round(
        sum(1 for st in pipe_b_statuses if st in ("UNCONFIGURED", "DISABLED", "FAILED")) / len(samples), 4
    )
    metrics_b["mean_moderation_overhead_ms"] = round(float(sum(pipe_b_latencies) / len(pipe_b_latencies)), 2)

    # ── Pipeline C: MuRIL + Ambiguity Gate + MOCK JEV (Regression Simulation) ─
    logger.info("Evaluating Pipeline C: Mock JEV Simulation (Regression Test Only)...")
    mock_provider = MockJEVDecisionProvider()
    pipe_c_preds = []
    pipe_c_statuses = []
    pipe_c_invocations = 0
    gate_reasons = {}
    pipe_c_latencies = []

    for text, preds, score, intensity, flags in muril_inference_cache:
        t0 = time.perf_counter()
        ctx_res = moderate_comment_with_context(
            text, preds, flags, frozen_thresholds, jev_provider=mock_provider
        )
        pipe_c_latencies.append((time.perf_counter() - t0) * 1000)
        pipe_c_statuses.append(ctx_res.jev_status)
        if ctx_res.jev_invoked:
            pipe_c_invocations += 1

        reason_key = ctx_res.gate_reason[:40]
        gate_reasons[reason_key] = gate_reasons.get(reason_key, 0) + 1

        if ctx_res.jev_decision:
            is_toxic = (ctx_res.final_risk_level == RiskLevel.HIGH_RISK) or ctx_res.jev_decision.contextual_toxicity
        else:
            is_toxic = (ctx_res.final_risk_level == RiskLevel.HIGH_RISK) or any(flags.values())
        pipe_c_preds.append(1 if is_toxic else 0)

    metrics_c = compute_metrics(y_true, pipe_c_preds)
    metrics_c["is_mock_simulation"] = True
    metrics_c["jev_invocation_rate"] = round(pipe_c_invocations / len(samples), 4)
    metrics_c["bypassed_rate"] = round(sum(1 for st in pipe_c_statuses if st == "BYPASSED_CLEAR_CASE") / len(samples), 4)
    metrics_c["mean_moderation_overhead_ms"] = round(float(sum(pipe_c_latencies) / len(pipe_c_latencies)), 2)

    # Compile Final Report
    report = {
        "metadata": {
            "phase": "Phase 6A",
            "task": "JEV Contextual Moderation Layer Experimental Evaluation",
            "eval_set": eval_data.get("dataset_metadata", {}).get("name", "Curated Multilingual Probe"),
            "total_samples": len(samples),
            "real_jev_configured": real_jev_available,
            "muril_model": config.MURIL_MODEL_NAME,
            "frozen_thresholds": frozen_thresholds,
        },
        "pipeline_a_deterministic_baseline": metrics_a,
        "pipeline_b_real_jev_production": {
            "available": real_jev_available,
            "metrics": metrics_b,
            "note": "Production execution path. When credentials are not configured, gracefully falls back to deterministic base moderation with zero service interruption and zero fabrication."
        },
        "pipeline_c_mock_simulation_regression_only": {
            "is_simulation": True,
            "warning": "MOCK SIMULATION FOR REGRESSION TESTING ONLY. DO NOT cite as real JEV accuracy or vendor performance.",
            "metrics": metrics_c,
            "gate_routing_reasons": gate_reasons,
        },
    }

    # Save to disk
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", OUTPUT_REPORT_PATH)

    # Print Clean Formatted Summary to Console
    print("\n" + "=" * 80)
    print("PHASE 6A: CONTEXTUAL DECISION LAYER (JEV) EXPERIMENTAL EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Probe Samples Evaluated : {len(samples)}")
    print(f"Real JEV Configured in Env     : {real_jev_available}")
    print(f"Frozen MuRIL Thresholds        : {frozen_thresholds}")
    print("-" * 80)
    print(f"{'Metric':<25} | {'Pipeline A (Base)':<18} | {'Pipeline B (Real JEV)':<21} | {'Pipeline C (Mock Sim)*':<20}")
    print("-" * 80)
    print(f"{'Precision':<25} | {metrics_a['precision']:<18.4f} | {metrics_b['precision']:<21.4f} | {metrics_c['precision']:<20.4f}")
    print(f"{'Recall':<25} | {metrics_a['recall']:<18.4f} | {metrics_b['recall']:<21.4f} | {metrics_c['recall']:<20.4f}")
    print(f"{'F1 Score':<25} | {metrics_a['f1']:<18.4f} | {metrics_b['f1']:<21.4f} | {metrics_c['f1']:<20.4f}")
    print(f"{'False Positive Rate':<25} | {metrics_a['fpr']:<18.4f} | {metrics_b['fpr']:<21.4f} | {metrics_c['fpr']:<20.4f}")
    print(f"{'False Negative Rate':<25} | {metrics_a['fnr']:<18.4f} | {metrics_b['fnr']:<21.4f} | {metrics_c['fnr']:<20.4f}")
    print(f"{'JEV Invocation Rate':<25} | {'0.0000 (N/A)':<18} | {metrics_b['jev_invocation_rate']:<21.4f} | {metrics_c['jev_invocation_rate']:<20.4f}")
    print(f"{'Fallback / Unavail Rate':<25} | {'0.0000 (N/A)':<18} | {metrics_b['fallback_to_base_rate']:<21.4f} | {'0.0000 (N/A)':<20}")
    print("-" * 80)
    print("* Pipeline C is a deterministic mock simulation exclusively used for unit regression tests.")
    print("  It is NEVER used as evidence of real JEV production performance.")
    print("=" * 80 + "\n")

    return report


if __name__ == "__main__":
    run_experiment()
