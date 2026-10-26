"""
Paired Semantic Contrast Evaluation: Model A vs Model B.

Evaluates whether Model B genuinely improves contextual semantic separation on identical
lexical pairs rather than merely shifting the global toxicity threshold:
  1. Benign profanity vs hostile profanity
  2. "You killed it!" vs "I will kill you."
  3. "This movie was sick!" vs "You're sick and disgusting."
  4. Hinglish praise/banter vs actual abuse
  5. Hindi praise vs Hindi abuse
  6. Obfuscated profanity / praise vs insult
  7. Culturally contextual meme text (friendly peer slang vs violent threat)

Outputs to: scratch/meme_experiments/model_b/paired_semantic_comparison.json
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.model import MuRILForMultiLabelToxicity, DEFAULT_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalMemePairs")

EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b"
CHECKPOINT_B = EXP_DIR / "muril_model_b.pt"
PAIRS_OUT = EXP_DIR / "paired_semantic_comparison.json"

PAIRED_CASES = [
    {
        "pair_id": 1,
        "category": "Benign Profanity vs Hostile Profanity",
        "benign_text": "Fuck yeah! We finally passed all the unit tests!",
        "hostile_text": "You're a fucking idiot and I hope you get hurt.",
        "shared_lexical_token": "fuck"
    },
    {
        "pair_id": 2,
        "category": "Praise Metaphor vs Literal Threat",
        "benign_text": "Bro, you totally killed it on stage tonight!",
        "hostile_text": "I am going to kill you tonight.",
        "shared_lexical_token": "kill / killed"
    },
    {
        "pair_id": 3,
        "category": "Slang Praise vs Targeted Insult",
        "benign_text": "This movie was sick!",
        "hostile_text": "You're sick and disgusting.",
        "shared_lexical_token": "sick"
    },
    {
        "pair_id": 4,
        "category": "Hinglish Friendly Banter vs Hinglish Abuse",
        "benign_text": "Bhai kya bawaal cheez banayi hai tune, maza aa gaya!",
        "hostile_text": "Bhai tu ek number ka chutiya aur ghatiya aadmi hai.",
        "shared_lexical_token": "bhai / bawaal"
    },
    {
        "pair_id": 5,
        "category": "Hindi Praise vs Hindi Abuse",
        "benign_text": "भाई क्या धांसू काम किया है आपने, सच में दिल जीत लिया!",
        "hostile_text": "तू बिल्कुल घटिया और नीच इंसान है, तेरी औकात नहीं है।",
        "shared_lexical_token": "colloquial tone"
    },
    {
        "pair_id": 6,
        "category": "Obfuscated Innocent vs Obfuscated Abuse",
        "benign_text": "Nice sh!rt, looks super c00l on you bro!",
        "hostile_text": "You are a f**king 1diot and a total l0ser.",
        "shared_lexical_token": "leetspeak symbols"
    },
    {
        "pair_id": 7,
        "category": "Culturally Contextual Peer Banter vs Hostile Abuse",
        "benign_text": "Bhai teri to job lag gayi bc, treat kab de raha hai party chalegi!",
        "hostile_text": "Abe saale kutte nikal yahan se varna maar maar ke bhoot bana dunga.",
        "shared_lexical_token": "bc / saale"
    },
]


def load_model(checkpoint_path: Path, device: torch.device) -> MuRILForMultiLabelToxicity:
    model = MuRILForMultiLabelToxicity(model_name=config.MURIL_MODEL_NAME, num_labels=6)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_pair(text: str, model: MuRILForMultiLabelToxicity, tokenizer: AutoTokenizer, device: torch.device) -> Dict[str, float]:
    encoding = tokenizer(
        str(text).strip(),
        truncation=True,
        max_length=64,
        padding=True,
        return_tensors="pt"
    )
    with torch.inference_mode():
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().squeeze(0).numpy()

    return {label: round(float(probs[i]), 4) for i, label in enumerate(DEFAULT_LABELS)}


def run_paired_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    tokenizer = AutoTokenizer.from_pretrained(config.MURIL_MODEL_NAME)

    logger.info("Loading Model A (Production MuRIL)...")
    model_a = load_model(config.MURIL_CHECKPOINT_PATH, device)

    logger.info("Loading Model B (Meme Text Baseline)...")
    model_b = load_model(CHECKPOINT_B, device)

    pair_results = []
    improved_count = 0
    correct_ordering_a = 0
    correct_ordering_b = 0

    print("\n" + "=" * 80)
    print("PAIRED SEMANTIC CONTRAST EVALUATION: MODEL A vs MODEL B")
    print("=" * 80)

    for case in PAIRED_CASES:
        p_id = case["pair_id"]
        cat = case["category"]
        b_text = case["benign_text"]
        h_text = case["hostile_text"]

        # Predictions for Model A
        preds_a_benign = predict_pair(b_text, model_a, tokenizer, device)
        preds_a_hostile = predict_pair(h_text, model_a, tokenizer, device)
        peak_a_benign = max(preds_a_benign.values())
        peak_a_hostile = max(preds_a_hostile.values())
        delta_a = round(peak_a_hostile - peak_a_benign, 4)
        order_ok_a = bool(delta_a > 0.05)
        if order_ok_a:
            correct_ordering_a += 1

        # Predictions for Model B
        preds_b_benign = predict_pair(b_text, model_b, tokenizer, device)
        preds_b_hostile = predict_pair(h_text, model_b, tokenizer, device)
        peak_b_benign = max(preds_b_benign.values())
        peak_b_hostile = max(preds_b_hostile.values())
        delta_b = round(peak_b_hostile - peak_b_benign, 4)
        order_ok_b = bool(delta_b > 0.05)
        if order_ok_b:
            correct_ordering_b += 1

        separation_gain = round(delta_b - delta_a, 4)
        is_improved = bool(separation_gain > 0)
        if is_improved:
            improved_count += 1

        record = {
            "pair_id": p_id,
            "category": cat,
            "shared_lexical_token": case["shared_lexical_token"],
            "benign_text": b_text,
            "hostile_text": h_text,
            "model_a": {
                "benign_peak_score": peak_a_benign,
                "hostile_peak_score": peak_a_hostile,
                "separation_delta": delta_a,
                "ordering_correct": order_ok_a,
                "benign_preds": preds_a_benign,
                "hostile_preds": preds_a_hostile,
            },
            "model_b": {
                "benign_peak_score": peak_b_benign,
                "hostile_peak_score": peak_b_hostile,
                "separation_delta": delta_b,
                "ordering_correct": order_ok_b,
                "benign_preds": preds_b_benign,
                "hostile_preds": preds_b_hostile,
            },
            "separation_improvement_delta_b_minus_a": separation_gain,
            "is_separation_improved": is_improved,
        }
        pair_results.append(record)

        print(f"\n--- [Pair {p_id}] {cat} ---")
        print(f"  Benign : \"{b_text}\"")
        print(f"  Hostile: \"{h_text}\"")
        print(f"  Model A: Benign={peak_a_benign:.4f} | Hostile={peak_a_hostile:.4f} | Delta={delta_a:+.4f} (Order OK: {order_ok_a})")
        print(f"  Model B: Benign={peak_b_benign:.4f} | Hostile={peak_b_hostile:.4f} | Delta={delta_b:+.4f} (Order OK: {order_ok_b})")
        print(f"  --> Separation Gain (Model B - Model A): {separation_gain:+.4f} ({'IMPROVED' if is_improved else 'NO GAIN'})")

    summary = {
        "total_pairs_evaluated": len(PAIRED_CASES),
        "model_a_correct_order_count": correct_ordering_a,
        "model_b_correct_order_count": correct_ordering_b,
        "pairs_with_improved_separation": improved_count,
        "pairs_with_improved_separation_pct": round(improved_count / len(PAIRED_CASES) * 100, 2),
        "mean_separation_gain": round(float(np.mean([r["separation_improvement_delta_b_minus_a"] for r in pair_results])), 4),
        "detailed_pair_evaluations": pair_results,
    }

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {improved_count}/{len(PAIRED_CASES)} pairs showed improved separation in Model B.")
    print(f"Mean Separation Gain: {summary['mean_separation_gain']:+.4f}")
    print(f"Detailed pair comparison saved to: {PAIRS_OUT}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    run_paired_evaluation()
