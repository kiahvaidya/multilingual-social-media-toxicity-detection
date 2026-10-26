import json
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import load_muril_model, predict_comment_muril

def run_experiment_b():
    print("Loading MuRIL model...")
    model, tokenizer, labels, thresholds, device = load_muril_model()
    
    # 1. Evaluate the 60-sample curated Phase 5C probe
    eval_set_path = _PROJECT_ROOT / "evaluation_results" / "multilingual_eval_phase5c.json"
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    samples = eval_data["samples"]
    print(f"Loaded {len(samples)} samples from Phase 5C curated probe.")
    
    probe_results = []
    category_breakdown = {}
    
    for s in samples:
        text = s["text"]
        is_toxic_expected = s["expected_toxic"]
        lang = s["language"]
        stype = s.get("sample_type", "")
        
        preds, peak, intensity, flags = predict_comment_muril(
            text, model, tokenizer, labels, device, thresholds
        )
        is_toxic_pred = any(flags.values())
        
        probe_results.append({
            "id": s["id"],
            "text": text,
            "language": lang,
            "sample_type": stype,
            "expected_toxic": is_toxic_expected,
            "predicted_toxic": is_toxic_pred,
            "peak_score": round(peak, 4),
            "flags": flags,
            "preds": {k: round(v, 4) for k, v in preds.items()}
        })
        
        cat_key = lang
        if cat_key not in category_breakdown:
            category_breakdown[cat_key] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}
        category_breakdown[cat_key]["total"] += 1
        
        if is_toxic_expected and is_toxic_pred:
            category_breakdown[cat_key]["tp"] += 1
        elif not is_toxic_expected and is_toxic_pred:
            category_breakdown[cat_key]["fp"] += 1
        elif is_toxic_expected and not is_toxic_pred:
            category_breakdown[cat_key]["fn"] += 1
        else:
            category_breakdown[cat_key]["tn"] += 1

    # 2. Evaluate Semantically Paired Contrast Sets
    pairs = [
        {
            "category": "Benign Profanity vs Targeted Profanity",
            "praise_benign": "Fuck yeah! We finally passed all the unit tests!",
            "hostile_abuse": "You're a fucking idiot and I hope you get hurt."
        },
        {
            "category": "Praise Metaphor vs Literal Threat",
            "praise_benign": "Bro, you totally killed it on stage tonight!",
            "hostile_abuse": "I am going to kill you tonight."
        },
        {
            "category": "Slang Praise vs Targeted Insult",
            "praise_benign": "This movie was sick!",
            "hostile_abuse": "You're sick and disgusting."
        },
        {
            "category": "Untargeted Profanity vs Person-Targeted Profanity",
            "praise_benign": "Damn, this traffic is so fucking annoying.",
            "hostile_abuse": "You are so fucking annoying, get out of here."
        },
        {
            "category": "Hinglish Friendly Banter vs Hinglish Abuse",
            "praise_benign": "Bhai kya bawaal cheez banayi hai tune, maza aa gaya!",
            "hostile_abuse": "Bhai tu ek number ka chutiya aur ghatiya aadmi hai."
        },
        {
            "category": "Hindi Praise vs Hindi Hostile Abuse",
            "praise_benign": "भाई क्या धांसू काम किया है आपने, सच में दिल जीत लिया!",
            "hostile_abuse": "तू बिल्कुल घटिया और नीच इंसान है, तेरी औकात नहीं है।"
        }
    ]

    pairwise_results = []
    for p in pairs:
        preds_benign, peak_b, _, flags_b = predict_comment_muril(p["praise_benign"], model, tokenizer, labels, device, thresholds)
        preds_abuse, peak_a, _, flags_a = predict_comment_muril(p["hostile_abuse"], model, tokenizer, labels, device, thresholds)
        
        # Primary head diffs
        deltas = {lbl: round(preds_abuse[lbl] - preds_benign[lbl], 4) for lbl in labels}
        
        pairwise_results.append({
            "category": p["category"],
            "praise_benign": {
                "text": p["praise_benign"],
                "peak_score": round(peak_b, 4),
                "flags": [k for k, v in flags_b.items() if v],
                "preds": {k: round(v, 4) for k, v in preds_benign.items()}
            },
            "hostile_abuse": {
                "text": p["hostile_abuse"],
                "peak_score": round(peak_a, 4),
                "flags": [k for k, v in flags_a.items() if v],
                "preds": {k: round(v, 4) for k, v in preds_abuse.items()}
            },
            "separation_delta_abuse_minus_praise": deltas,
            "peak_delta": round(peak_a - peak_b, 4)
        })

    # Overall metrics for the 60 samples
    total_tp = sum(c["tp"] for c in category_breakdown.values())
    total_fp = sum(c["fp"] for c in category_breakdown.values())
    total_fn = sum(c["fn"] for c in category_breakdown.values())
    total_tn = sum(c["tn"] for c in category_breakdown.values())
    
    summary = {
        "overall_probe_metrics": {
            "total_samples": len(samples),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0,
            "recall": round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0.0,
            "fpr": round(total_fp / (total_fp + total_tn), 4) if (total_fp + total_tn) > 0 else 0.0,
            "fnr": round(total_fn / (total_fn + total_tp), 4) if (total_fn + total_tp) > 0 else 0.0,
        },
        "by_language_metrics": {},
        "pairwise_contrast_results": pairwise_results,
        "sample_error_details": [r for r in probe_results if (r["expected_toxic"] != r["predicted_toxic"])]
    }
    
    for k, v in category_breakdown.items():
        tp, fp, fn, tn = v["tp"], v["fp"], v["fn"], v["tn"]
        summary["by_language_metrics"][k] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": v["total"],
            "fpr": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
            "fnr": round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
        }
        
    out_file = _PROJECT_ROOT / "scratch" / "exp_b_contextual_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Experiment B completed. Output saved to:", out_file)
    print("Overall Probe FPR:", summary["overall_probe_metrics"]["fpr"], "Recall:", summary["overall_probe_metrics"]["recall"])

if __name__ == "__main__":
    run_experiment_b()
