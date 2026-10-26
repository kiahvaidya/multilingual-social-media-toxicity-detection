import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from collections import Counter
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.model import load_muril_model, predict_comment_muril, get_device
from src.moderation import (
    classify_risk,
    should_invoke_jev,
    moderate_comment_with_context,
    RiskLevel,
)
from src.jev import get_jev_provider, JEVProviderType, GeminiJEVDecisionProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUDIT_DIR = Path(__file__).parent

INPUT_20_RUN = "Bhai this song is fucking amazing"

TEST_INPUTS = {
    "A": "Bhai this song is fucking amazing",
    "B": "Bhai teri toh job lag gayi bc",
    "C": "Bro your new setup looks insane",
    "D": "You killed it on that presentation!",
    "E": "You're a fucking idiot and nobody wants you here.",
    "F": "I hope you get hurt badly.",
}


def serialize_decision(decision):
    if decision is None:
        return None
    return {
        "intent": decision.intent.value if hasattr(decision.intent, "value") else str(decision.intent),
        "contextual_toxicity": bool(decision.contextual_toxicity),
        "moderation": decision.moderation.value if hasattr(decision.moderation, "value") else str(decision.moderation),
        "human_review_required": bool(decision.human_review_required),
        "confidence": float(decision.confidence),
        "reason": str(decision.reason),
        "provider_type": decision.provider_type.value if hasattr(decision.provider_type, "value") else str(decision.provider_type),
        "raw_response": decision.raw_response,
    }


def run_single_pipeline_pass(text, model, tokenizer, labels, device, thresholds, provider):
    t0 = time.time()
    # 1. MuRIL inference
    predictions, max_toxic_score, intensity, is_toxic_flags = predict_comment_muril(
        text=text,
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        device=device,
        thresholds=thresholds,
    )
    muril_duration = time.time() - t0

    # 2. Base risk
    base_risk, base_icon, base_expl = classify_risk(predictions, is_toxic_flags, thresholds)

    # 3. Ambiguity gate
    is_ambiguous, gate_reason = should_invoke_jev(
        text, predictions, is_toxic_flags, base_risk, thresholds
    )

    # 4. Contextual moderation with JEV
    t_mod0 = time.time()
    mod_res = moderate_comment_with_context(
        text=text,
        predictions=predictions,
        is_toxic_flags=is_toxic_flags,
        thresholds=thresholds,
        jev_provider=provider,
    )
    mod_duration = time.time() - t_mod0

    return {
        "text": text,
        "muril_scores": predictions,
        "is_toxic_flags": is_toxic_flags,
        "muril_risk_level": base_risk.value,
        "is_ambiguous": is_ambiguous,
        "gate_reason": gate_reason,
        "jev_invoked": mod_res.jev_invoked,
        "jev_status": mod_res.jev_status,
        "jev_error": mod_res.jev_error,
        "jev_decision": serialize_decision(mod_res.jev_decision),
        "final_moderation": mod_res.final_risk_level.value,
        "final_explanation": mod_res.final_explanation,
        "human_review_required": mod_res.jev_decision.human_review_required if mod_res.jev_decision else (base_risk != RiskLevel.SAFE),
        "muril_duration_s": round(muril_duration, 4),
        "mod_duration_s": round(mod_duration, 4),
        "timestamp": time.time(),
    }


def main():
    logger.info("Initializing Text Model + JEV Determinism Audit...")
    device = get_device()
    logger.info("Using device: %s", device)

    # Load production model
    model, tokenizer, labels, thresholds, _ = load_muril_model(device=device)
    provider = get_jev_provider()

    logger.info("Loaded MuRIL with labels: %s", labels)
    logger.info("Loaded thresholds: %s", thresholds)
    logger.info("Active JEV Provider: %s", provider.__class__.__name__ if provider else "None")

    if provider:
        logger.info("Provider type: %s", provider.provider_type)
        if isinstance(provider, GeminiJEVDecisionProvider):
            logger.info("Gemini Model configured: %s", provider.model_name)

    # ── PART 1: 20-Run Repeatability Test on "Bhai this song is fucking amazing" ──
    logger.info("=" * 60)
    logger.info("PART 1: 20-RUN REPEATABILITY TEST ON TARGET SENTENCE")
    logger.info("Text: '%s'", INPUT_20_RUN)
    logger.info("=" * 60)

    part1_runs = []
    raw_jev_outputs = []

    for i in range(20):
        logger.info("Executing Run %d/20...", i + 1)
        res = run_single_pipeline_pass(
            text=INPUT_20_RUN,
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            device=device,
            thresholds=thresholds,
            provider=provider,
        )
        res["run_index"] = i + 1
        part1_runs.append(res)
        if res.get("jev_decision") and res["jev_decision"].get("raw_response"):
            raw_jev_outputs.append({
                "run_index": i + 1,
                "input_key": "TARGET_20_RUN",
                "text": INPUT_20_RUN,
                "raw_response": res["jev_decision"]["raw_response"],
                "error": res.get("jev_error"),
            })
        elif res.get("jev_error"):
            raw_jev_outputs.append({
                "run_index": i + 1,
                "input_key": "TARGET_20_RUN",
                "text": INPUT_20_RUN,
                "raw_response": None,
                "error": res.get("jev_error"),
            })
        # Small breath between remote calls to observe rate-limits/transients
        time.sleep(0.5)

    # ── PART 2: 10-Run Tests for Inputs A through F ──────────────────
    logger.info("=" * 60)
    logger.info("PART 2: 10-RUN TESTS ACROSS 6 BENCHMARK INPUTS (A - F)")
    logger.info("=" * 60)

    part2_results = {}
    for key, text in TEST_INPUTS.items():
        logger.info("Testing Input %s (10 runs): '%s'", key, text)
        runs_for_key = []
        for r in range(10):
            logger.info("  Input %s - Run %d/10...", key, r + 1)
            res = run_single_pipeline_pass(
                text=text,
                model=model,
                tokenizer=tokenizer,
                labels=labels,
                device=device,
                thresholds=thresholds,
                provider=provider,
            )
            res["run_index"] = r + 1
            runs_for_key.append(res)
            if res.get("jev_decision") and res["jev_decision"].get("raw_response"):
                raw_jev_outputs.append({
                    "run_index": r + 1,
                    "input_key": key,
                    "text": text,
                    "raw_response": res["jev_decision"]["raw_response"],
                    "error": res.get("jev_error"),
                })
            elif res.get("jev_error"):
                raw_jev_outputs.append({
                    "run_index": r + 1,
                    "input_key": key,
                    "text": text,
                    "raw_response": None,
                    "error": res.get("jev_error"),
                })
            time.sleep(0.5)

        # Compute summary statistics for Input key
        # 1. MuRIL mean / std for all labels
        muril_scores_by_label = {lbl: [] for lbl in labels}
        for run in runs_for_key:
            for lbl, score in run["muril_scores"].items():
                muril_scores_by_label[lbl].append(score)

        muril_stats = {
            lbl: {
                "mean": round(float(np.mean(scores)), 5),
                "std": round(float(np.std(scores)), 6),
                "min": round(float(np.min(scores)), 5),
                "max": round(float(np.max(scores)), 5),
                "is_constant": bool(np.std(scores) < 1e-6),
            }
            for lbl, scores in muril_scores_by_label.items()
        }

        # 2. JEV decision distribution
        jev_intents = Counter()
        jev_moderations = Counter()
        jev_statuses = Counter()
        final_decisions = Counter()

        for run in runs_for_key:
            jev_statuses[run["jev_status"]] += 1
            final_decisions[run["final_moderation"]] += 1
            if run["jev_decision"]:
                jev_intents[run["jev_decision"]["intent"]] += 1
                jev_moderations[run["jev_decision"]["moderation"]] += 1
            else:
                jev_intents["NONE_OR_FAILED"] += 1
                jev_moderations["NONE_OR_FAILED"] += 1

        part2_results[key] = {
            "text": text,
            "muril_stats": muril_stats,
            "jev_status_distribution": dict(jev_statuses),
            "jev_intent_distribution": dict(jev_intents),
            "jev_moderation_distribution": dict(jev_moderations),
            "final_decision_distribution": dict(final_decisions),
            "runs": runs_for_key,
        }

    # ── PART 3: Analysis of Part 1 (Target sentence) ──────────────────
    p1_muril_scores = {lbl: [r["muril_scores"][lbl] for r in part1_runs] for lbl in labels}
    p1_muril_stats = {
        lbl: {
            "mean": round(float(np.mean(scores)), 5),
            "std": round(float(np.std(scores)), 6),
            "min": round(float(np.min(scores)), 5),
            "max": round(float(np.max(scores)), 5),
            "is_constant": bool(np.std(scores) < 1e-6),
        }
        for lbl, scores in p1_muril_scores.items()
    }

    p1_final_decisions = Counter(r["final_moderation"] for r in part1_runs)
    p1_jev_statuses = Counter(r["jev_status"] for r in part1_runs)
    p1_jev_moderations = Counter(
        r["jev_decision"]["moderation"] if r["jev_decision"] else "FAILED_OR_NONE"
        for r in part1_runs
    )
    p1_jev_intents = Counter(
        r["jev_decision"]["intent"] if r["jev_decision"] else "FAILED_OR_NONE"
        for r in part1_runs
    )

    part1_summary = {
        "text": INPUT_20_RUN,
        "total_runs": 20,
        "muril_stats": p1_muril_stats,
        "jev_status_distribution": dict(p1_jev_statuses),
        "jev_intent_distribution": dict(p1_jev_intents),
        "jev_moderation_distribution": dict(p1_jev_moderations),
        "final_decision_distribution": dict(p1_final_decisions),
        "is_final_decision_deterministic": len(p1_final_decisions) == 1,
        "runs": part1_runs,
    }

    # ── PART 4: Diagnostics on Provider Configuration ─────────────────
    diagnostics = {
        "provider_class": provider.__class__.__name__ if provider else "None",
        "provider_type": provider.provider_type.value if provider else "NONE",
        "gemini_model": getattr(provider, "model_name", None),
        "config_timeout_seconds": config.JEV_TIMEOUT_SECONDS,
        "config_gemini_api_key_set": bool(config.GEMINI_API_KEY),
        "temperature_setting": 0.0,
        "seed_setting": None,  # Not passed to GenerateContentConfig
        "response_mime_type": "application/json",
        "structured_schema_enforcement": "Pydantic/JSON Schema NOT passed to API; relies on prompt string schema + JSON decode",
        "retry_logic": "Max 3 attempts, retries on 503/504/UNAVAILABLE/DEADLINE_EXCEEDED with backoff",
        "fallback_logic": "On ANY unhandled exception or failed retry, catches Exception and returns base_risk (HIGH_RISK or REVIEW)",
        "cached_responses": "None (No caching layer in Python or FastAPI)",
        "session_state": "None (Stateless REST)",
    }

    # ── SAVE ALL JSON ARTIFACTS ───────────────────────────────────────
    repeatability_file = AUDIT_DIR / "repeatability_results.json"
    with open(repeatability_file, "w", encoding="utf-8") as f:
        json.dump({
            "target_repeatability_20_runs": part1_summary,
            "inputs_a_to_f_10_runs": part2_results,
        }, f, indent=2)
    logger.info("Saved %s", repeatability_file)

    raw_jev_file = AUDIT_DIR / "raw_jev_outputs.json"
    with open(raw_jev_file, "w", encoding="utf-8") as f:
        json.dump(raw_jev_outputs, f, indent=2)
    logger.info("Saved %s", raw_jev_file)

    diagnostics_file = AUDIT_DIR / "pipeline_diagnostics.json"
    with open(diagnostics_file, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Saved %s", diagnostics_file)

    logger.info("Audit execution complete! All raw data saved.")


if __name__ == "__main__":
    main()
