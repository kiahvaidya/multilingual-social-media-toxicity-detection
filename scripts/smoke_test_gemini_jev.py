"""
Real Integration Smoke Test for Gemini-backed JEV Contextual Decision Layer.

This script executes actual Gemini API calls to validate contextual adjudication
against ambiguous or borderline comments flagged by the MuRIL toxicity classifier.

SAFETY RULE:
This script ONLY runs real API calls when GEMINI_API_KEY is present in the environment.
It is never executed as part of the automated pytest test suite.
"""

import json
import os
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.jev import GeminiJEVDecisionProvider
from src.model import load_muril_model, predict_comment_muril
from src.moderation import moderate_comment_with_context, should_invoke_jev, RiskLevel

MANUAL_CASES = [
    {
        "id": "CASE 1",
        "text": "Fuck yeah! We finally passed all the unit tests!",
        "expected_intent": "non_toxic_praise_or_excitement",
        "expected_tier": "Safe",
        "description": "Benign celebratory profanity"
    },
    {
        "id": "CASE 2",
        "text": "You're a fucking idiot and I hope you get hurt.",
        "expected_intent": "genuine_abuse or threat",
        "expected_tier": "High Risk",
        "description": "Direct insult with malicious hostility"
    },
    {
        "id": "CASE 3",
        "text": "This movie was sick!",
        "expected_intent": "non_toxic_praise_or_excitement",
        "expected_tier": "Safe",
        "description": "Colloquial praise metaphor using slang"
    },
    {
        "id": "CASE 4",
        "text": "You're sick and disgusting.",
        "expected_intent": "genuine_abuse",
        "expected_tier": "High Risk",
        "description": "Direct personal hostility using identical lexicon in abusive context"
    }
]


def run_smoke_test():
    api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("=" * 70)
        print("GEMINI JEV INTEGRATION SMOKE TEST: SKIPPED")
        print("=" * 70)
        print("Reason: GEMINI_API_KEY environment variable is not configured.")
        print("To run this smoke test, provide your API key:")
        print("  $env:GEMINI_API_KEY = 'your_key_here'; python scripts/smoke_test_gemini_jev.py")
        print("=" * 70)
        return 0

    model_name = config.JEV_GEMINI_MODEL or "gemini-2.5-flash"
    print("=" * 70)
    print("STARTING REAL GEMINI JEV INTEGRATION SMOKE TEST")
    print(f"Model: {model_name} | API Key Configured: True")
    print("=" * 70)

    try:
        provider = GeminiJEVDecisionProvider(api_key=api_key, model_name=model_name)
    except Exception as e:
        print(f"[ERROR] Failed to initialize GeminiJEVDecisionProvider: {e}")
        return 1

    print("\nLoading MuRIL primary classifier...")
    muril_model, tokenizer, labels, thresholds, device = load_muril_model()
    print(f"MuRIL loaded successfully on {device}.\n")

    summary_results = []

    for case in MANUAL_CASES:
        case_id = case["id"]
        text = case["text"]
        print(f"\n--- {case_id}: \"{text}\" ---")
        print(f"Category Description: {case['description']}")

        # 1. MuRIL Prediction
        scores, peak_score, intensity, flags = predict_comment_muril(
            text, muril_model, tokenizer, labels, device, thresholds
        )
        flagged = [cat for cat, fl in flags.items() if fl]
        print(f"MuRIL Scores: {json.dumps(scores)}")
        print(f"Flagged Categories: {flagged} | Peak Score: {peak_score:.3f}")

        # 2. Moderation with Gemini Context
        t0 = time.perf_counter()
        ctx_res = moderate_comment_with_context(
            text=text,
            predictions=scores,
            is_toxic_flags=flags,
            thresholds=thresholds,
            jev_provider=provider
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"Ambiguity Gate: is_ambiguous={ctx_res.is_ambiguous} (Reason: {ctx_res.gate_reason})")
        print(f"JEV Invoked: {ctx_res.jev_invoked} | Status: {ctx_res.jev_status} | Latency: {latency_ms:.1f}ms")
        print(f"Base Tier (Before JEV): {ctx_res.base_risk_level.value}")
        print(f"Final Tier (After JEV):  {ctx_res.final_risk_level.value} {ctx_res.final_badge_icon}")

        if ctx_res.jev_decision:
            dec = ctx_res.jev_decision
            print(f"Gemini Intent: {dec.intent.value}")
            print(f"Gemini Confidence: {dec.confidence:.2f}")
            print(f"Gemini Contextual Toxicity: {dec.contextual_toxicity}")
            print(f"Gemini Rationale: {dec.reason}")
        elif ctx_res.jev_status == "FAILED":
            print(f"[WARNING] JEV execution failed: {ctx_res.jev_error}")

        match_tier = (ctx_res.final_risk_level.value.lower() == case["expected_tier"].lower())
        status_label = "MATCH" if match_tier else "DIVERGENCE"
        print(f"Validation Target: {case['expected_tier']} -> Outcome: {status_label}")

        summary_results.append({
            "case_id": case_id,
            "text": text,
            "base_tier": ctx_res.base_risk_level.value,
            "final_tier": ctx_res.final_risk_level.value,
            "expected_tier": case["expected_tier"],
            "intent": ctx_res.jev_decision.intent.value if ctx_res.jev_decision else "N/A",
            "latency_ms": round(latency_ms, 1),
            "match": match_tier,
            "fallback_used": (ctx_res.jev_status == "FAILED")
        })

        # Respect 5 RPM free tier rate limit with a small delay between cases
        if case != MANUAL_CASES[-1]:
            print("\n[Pacing] Waiting 13s to respect Gemini API Free Tier 5 RPM rate limit...")
            time.sleep(13)

    print("\n" + "=" * 70)
    print("GEMINI JEV SMOKE TEST SUMMARY")
    print("=" * 70)
    for res in summary_results:
        print(f"[{res['case_id']}] Base: {res['base_tier']} -> Final: {res['final_tier']} | Latency: {res['latency_ms']}ms | Target: {res['expected_tier']} | Fallback: {res['fallback_used']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
