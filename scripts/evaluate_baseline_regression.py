"""
Phase 1 & Phase 2: Baseline Evaluation & Diagnostic Regression Suite

Verifies:
- MuRIL encoder configuration and freezing status
- Classifier dimensions and label alignment
- Raw logits, sigmoid probabilities, and threshold behavior
- Diagnostic evaluation on known failure modes and diverse linguistic variations
- Output saved to artifacts/baseline_error_analysis.csv
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.model import load_muril_model, DEFAULT_LABELS, MuRILForMultiLabelToxicity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineRegression")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def verify_model_architecture(model: MuRILForMultiLabelToxicity) -> Dict[str, Any]:
    """Inspect and mathematically verify model layers, parameters, and classification head."""
    print("\n" + "=" * 65)
    print("PHASE 2: DETAILED ARCHITECTURE & LABEL ORDERING VERIFICATION")
    print("=" * 65)

    encoder = model.encoder
    num_layers = len(encoder.encoder.layer)
    frozen_layers = []
    trainable_layers = []

    for idx, layer in enumerate(encoder.encoder.layer):
        requires_grad = any(p.requires_grad for p in layer.parameters())
        if requires_grad:
            trainable_layers.append(idx)
        else:
            frozen_layers.append(idx)

    classifier = model.classifier
    weight_shape = list(classifier.weight.shape)
    bias_shape = list(classifier.bias.shape) if classifier.bias is not None else None

    verification_info = {
        "model_name": config.MURIL_MODEL_NAME,
        "total_transformer_layers": num_layers,
        "frozen_layer_indices": frozen_layers,
        "trainable_layer_indices": trainable_layers,
        "classifier_in_features": classifier.in_features,
        "classifier_out_features": classifier.out_features,
        "classifier_weight_shape": weight_shape,
        "classifier_bias_shape": bias_shape,
        "label_order": list(DEFAULT_LABELS),
        "frozen_thresholds": dict(config.FROZEN_MURIL_THRESHOLDS),
    }

    print(f"Base Model: {verification_info['model_name']}")
    print(f"Total Transformer Layers: {num_layers}")
    print(f"Frozen Layers (requires_grad=False): {frozen_layers} ({len(frozen_layers)}/12 layers frozen!)")
    print(f"Trainable Layers: {trainable_layers}")
    print(f"Classifier Head: in_features={classifier.in_features}, out_features={classifier.out_features}")
    print(f"Label Alignment:")
    for idx, lbl in enumerate(DEFAULT_LABELS):
        print(f"  Index {idx} -> Label: '{lbl}' (Threshold: {config.FROZEN_MURIL_THRESHOLDS[lbl]})")

    assert DEFAULT_LABELS == ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], "Label ordering mismatch!"
    assert classifier.out_features == 6, f"Classifier head has {classifier.out_features} outputs instead of 6!"
    assert DEFAULT_LABELS[3] == "threat", "Index 3 is not threat!"

    return verification_info


def build_regression_test_suite() -> List[Dict[str, Any]]:
    """Build dedicated regression suite containing the 6 mandatory cases + diverse linguistic variations."""
    suite = [
        # ── 1. The 6 Mandatory Known Failure Modes ─────────────────────────
        {
            "id": "canonical_01",
            "text": "Bhai kya bawaal cheez banayi hai tune, maza hi aa gaya dekh kar!",
            "language": "hinglish",
            "category": "POSITIVE_HINGLISH_PRAISE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Enthusiastic Hinglish praise using colloquial word 'bawaal'"
        },
        {
            "id": "canonical_02",
            "text": "भाई क्या धांसू काम किया है आपने, सच में दिल जीत लिया!",
            "language": "hindi",
            "category": "POSITIVE_HINDI_PRAISE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Devanagari Hindi compliment using colloquial word 'धांसू'"
        },
        {
            "id": "canonical_03",
            "text": "Bro, this guitar solo is totally sick! You killed it!",
            "language": "english",
            "category": "POSITIVE_LEXICAL_AMBIGUITY",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "English slang praise using polysemous words 'sick' and 'killed it'"
        },
        {
            "id": "canonical_04",
            "text": "Fuck yeah! We finally passed all unit tests after three days of debugging!",
            "language": "english",
            "category": "BENIGN_PROFANITY_EXCITEMENT",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Celebratory expletive used without hostile target"
        },
        {
            "id": "canonical_05",
            "text": "I know where your office is and I am going to murder you tonight.",
            "language": "english",
            "category": "DIRECT_THREAT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
            "description": "Explicit death threat with physical location mention"
        },
        {
            "id": "canonical_06",
            "text": "You are a pathetic failure who should be fired immediately.",
            "language": "english",
            "category": "GENUINE_INSULT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
            "description": "Direct disparagement attacking competence and dignity"
        },

        # ── 2. Variations: Positive Hinglish & Hindi Slang ─────────────────
        {
            "id": "var_hing_praise_01",
            "text": "Bro tune to stage pe sach me aag laga di, deadly performance tha!",
            "language": "hinglish",
            "category": "POSITIVE_HINGLISH_PRAISE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Metaphorical praise 'aag laga di' and 'deadly'"
        },
        {
            "id": "var_hing_praise_02",
            "text": "Bhai kya faad code likha hai, ekdum mast chal raha hai!",
            "language": "hinglish",
            "category": "POSITIVE_HINGLISH_PRAISE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Slang praise 'faad code'"
        },
        {
            "id": "var_hing_praise_03",
            "text": "Bhai teri to job lag gayi bc, treat kab de raha hai party chalegi!",
            "language": "hinglish",
            "category": "BENIGN_PROFANITY_BANTER",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Colloquial non-hostile filler profanity 'bc' celebrating a friend's new job"
        },
        {
            "id": "var_hing_praise_04",
            "text": "Abe saale kitne saal baad mila hai tu, gale mil pehle!",
            "language": "hinglish",
            "category": "BENIGN_PROFANITY_BANTER",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Affectionate reunion greeting using 'abe saale'"
        },
        {
            "id": "var_hi_praise_01",
            "text": "यह नया गाना तो गदर मचा रहा है, ज़बरदस्त आवाज़ और धुन है।",
            "language": "hindi",
            "category": "POSITIVE_HINDI_PRAISE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Devanagari Hindi praise using 'गदर मचा रहा है'"
        },
        {
            "id": "var_hi_praise_02",
            "text": "अरे दोस्त तेरी तो किस्मत ही चमक गई, बधाई हो इस उपलब्धि पर।",
            "language": "hindi",
            "category": "POSITIVE_HINDI_PRAISE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Warm congratulatory Hindi message"
        },

        # ── 3. Variations: English Slang & Non-aggressive Profanity ────────
        {
            "id": "var_en_slang_01",
            "text": "Holy shit, that was the most unbelievable comeback in playoff history!",
            "language": "english",
            "category": "BENIGN_PROFANITY_EXCITEMENT",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Celebratory sports reaction with 'holy shit'"
        },
        {
            "id": "var_en_slang_02",
            "text": "That presentation was drop-dead gorgeous and insanely informative.",
            "language": "english",
            "category": "POSITIVE_LEXICAL_AMBIGUITY",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Colloquial superlative 'drop-dead gorgeous'"
        },
        {
            "id": "var_en_slang_03",
            "text": "Dude you absolutely murdered that bassline, pure fire!",
            "language": "english",
            "category": "POSITIVE_LEXICAL_AMBIGUITY",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Musical compliment 'murdered that bassline'"
        },
        {
            "id": "var_en_slang_04",
            "text": "This spicy ramen is so good it's criminal.",
            "language": "english",
            "category": "POSITIVE_LEXICAL_AMBIGUITY",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Metaphorical use of 'criminal'"
        },

        # ── 4. Variations: Direct, Indirect, & Indic Threats ───────────────
        {
            "id": "var_threat_01",
            "text": "We will find you and beat you until you can't walk again.",
            "language": "english",
            "category": "DIRECT_THREAT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
            "description": "Direct physical assault threat"
        },
        {
            "id": "var_threat_02",
            "text": "Watch your back when you leave your house tomorrow morning.",
            "language": "english",
            "category": "INDIRECT_THREAT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
            "description": "Ominous stalking / ambush warning"
        },
        {
            "id": "var_threat_03",
            "text": "If you ever post here again, your family will pay the ultimate price.",
            "language": "english",
            "category": "CONDITIONAL_THREAT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
            "description": "Conditional threat targeting user's family"
        },
        {
            "id": "var_threat_04",
            "text": "तुझे जान से मार डालूंगा कुत्ते, तेरा नामोनिशान मिटा दूंगा।",
            "language": "hindi",
            "category": "INDIC_THREAT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
            "description": "Explicit death threat in Devanagari Hindi"
        },
        {
            "id": "var_threat_05",
            "text": "तुझे घर में घुसकर मारूंगा अगर दोबारा मेरे रास्ते में आया तो।",
            "language": "hindi",
            "category": "INDIC_THREAT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
            "description": "Conditional home invasion threat in Devanagari Hindi"
        },
        {
            "id": "var_threat_06",
            "text": "Raste me mil kabhi, goli maar ke jaan se khatam kar dunga.",
            "language": "hinglish",
            "category": "INDIC_THREAT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
            "description": "Explicit shooting threat in Romanized Hinglish"
        },

        # ── 5. Variations: Insults & Hate Speech ────────────────────────────
        {
            "id": "var_insult_01",
            "text": "Shut up you repulsive scumbag, nobody asked for your garbage opinion.",
            "language": "english",
            "category": "GENUINE_INSULT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
            "description": "Direct vulgar insult"
        },
        {
            "id": "var_insult_02",
            "text": "तू एक नंबर का झूठा, चोर और बेईमान इंसान है।",
            "language": "hindi",
            "category": "GENUINE_INSULT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
            "description": "Devanagari Hindi insult calling someone a thief and liar"
        },
        {
            "id": "var_insult_03",
            "text": "Tu sabse bada chutiya hai behenchod, bilkul chup baith.",
            "language": "hinglish",
            "category": "GENUINE_INSULT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
            "description": "Vulgar Hinglish insult with explicit slurs"
        },
        {
            "id": "var_hate_01",
            "text": "People of your religion are parasitic filth and need to be cleansed.",
            "language": "english",
            "category": "IDENTITY_HATE",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
            "description": "Religious dehumanization and genocidal trope"
        },
        {
            "id": "var_hate_02",
            "text": "इन मुल्लों और देशद्रोहियों को मार भगाओ, यह देश इनका नहीं है।",
            "language": "hindi",
            "category": "IDENTITY_HATE",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
            "description": "Communal hate speech against religious minority in Devanagari"
        },

        # ── 6. Semantic Minimal Contrasts ──────────────────────────────────
        {
            "id": "contrast_kill_01_safe",
            "text": "You killed it bro",
            "language": "english",
            "category": "MINIMAL_CONTRAST_SAFE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Minimal contrast: praise using kill"
        },
        {
            "id": "contrast_kill_01_threat",
            "text": "I will kill you tonight",
            "language": "english",
            "category": "MINIMAL_CONTRAST_THREAT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
            "description": "Minimal contrast: literal threat using kill"
        },
        {
            "id": "contrast_sick_01_safe",
            "text": "This guitar solo is sick",
            "language": "english",
            "category": "MINIMAL_CONTRAST_SAFE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Minimal contrast: slang praise using sick"
        },
        {
            "id": "contrast_sick_01_insult",
            "text": "You are sick in the head",
            "language": "english",
            "category": "MINIMAL_CONTRAST_INSULT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
            "description": "Minimal contrast: mental health insult using sick"
        },
        {
            "id": "contrast_fuck_01_safe",
            "text": "Fuck yeah, we passed!",
            "language": "english",
            "category": "MINIMAL_CONTRAST_SAFE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Minimal contrast: benign celebratory profanity"
        },
        {
            "id": "contrast_fuck_01_insult",
            "text": "Fuck you, idiot",
            "language": "english",
            "category": "MINIMAL_CONTRAST_INSULT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
            "description": "Minimal contrast: targeted attack using fuck"
        },
        {
            "id": "contrast_bawaal_01_safe",
            "text": "Bhai kya bawaal kaam kiya!",
            "language": "hinglish",
            "category": "MINIMAL_CONTRAST_SAFE",
            "expected": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            "description": "Minimal contrast: Hinglish slang praise using bawaal"
        },
        {
            "id": "contrast_bawaal_01_insult",
            "text": "Tu ek bawaal hai saale",
            "language": "hinglish",
            "category": "MINIMAL_CONTRAST_INSULT",
            "expected": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
            "description": "Minimal contrast: Hinglish insult using bawaal as a pejorative"
        }
    ]

    # Also load the existing 60 samples from multilingual_eval_phase5c.json to preserve strict evaluation linkage
    eval_phase5c_path = PROJECT_ROOT / "evaluation_results" / "multilingual_eval_phase5c.json"
    if eval_phase5c_path.exists():
        with open(eval_phase5c_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        for s in eval_data.get("samples", []):
            cats = set(s.get("expected_categories", []))
            exp_vec = {
                "toxic": 1 if s.get("expected_toxic", False) or "toxic" in cats else 0,
                "severe_toxic": 1 if "severe_toxic" in cats else 0,
                "obscene": 1 if "obscene" in cats else 0,
                "threat": 1 if "threat" in cats else 0,
                "insult": 1 if "insult" in cats else 0,
                "identity_hate": 1 if "identity_hate" in cats else 0,
            }
            suite.append({
                "id": f"phase5c_{s['id']}",
                "text": s["text"],
                "language": s["language"],
                "category": f"PHASE5C_{s.get('sample_type', 'GENERAL').upper()}",
                "expected": exp_vec,
                "description": f"Curated Phase 5C item ({s.get('sample_type', '')})"
            })

    return suite


def evaluate_baseline():
    logger.info("Loading MuRIL model from checkpoint...")
    model, tokenizer, labels, thresholds, device = load_muril_model()
    model.eval()

    verify_model_architecture(model)

    suite = build_regression_test_suite()
    print(f"\nTotal test samples in diagnostic regression suite: {len(suite)}")

    results = []
    print("\n" + "=" * 65)
    print("RUNNING BASELINE INFERENCE ON DIAGNOSTIC REGRESSION SUITE")
    print("=" * 65)

    for item in suite:
        text = item["text"]
        expected = item["expected"]

        # Run forward pass through model
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.inference_mode():
            logits_tensor = model(input_ids, attention_mask).squeeze(0)
            probs_tensor = torch.sigmoid(logits_tensor)

        logits = logits_tensor.cpu().numpy()
        probs = probs_tensor.cpu().numpy()

        pred_labels = []
        is_correct = True
        error_type = "CORRECT"

        pred_dict = {}
        for idx, lbl in enumerate(labels):
            p_val = float(probs[idx])
            th_val = float(thresholds.get(lbl, 0.5))
            pred_dict[lbl] = p_val
            pred_bool = 1 if p_val >= th_val else 0
            if pred_bool == 1:
                pred_labels.append(lbl)

            exp_val = expected.get(lbl, 0)
            if pred_bool != exp_val:
                is_correct = False

        # Classify specific error type
        is_safe_expected = (sum(expected.values()) == 0)
        is_safe_pred = (len(pred_labels) == 0)

        if is_safe_expected and not is_safe_pred:
            if "bawaal" in text.lower() or "धांसू" in text or "gadar" in text.lower():
                error_type = "FALSE_POSITIVE_INDIAN_PRAISE"
            elif "sick" in text.lower() or "killed" in text.lower() or "deadly" in text.lower():
                error_type = "FALSE_POSITIVE_COLLOQUIAL_SLANG"
            elif any(w in text.lower() for w in ["fuck", "shit", "bc", "saale"]):
                error_type = "FALSE_POSITIVE_BENIGN_PROFANITY"
            else:
                error_type = "FALSE_POSITIVE_OTHER"
        elif not is_safe_expected and is_safe_pred:
            if expected.get("threat", 0) == 1:
                error_type = "FALSE_NEGATIVE_THREAT"
            elif expected.get("insult", 0) == 1:
                error_type = "FALSE_NEGATIVE_INSULT"
            else:
                error_type = "FALSE_NEGATIVE_OTHER"
        elif not is_correct:
            error_type = "LABEL_MISMATCH"

        row = {
            "id": item["id"],
            "text": text,
            "language": item["language"],
            "category": item["category"],
            "expected_labels": ",".join([lbl for lbl, v in expected.items() if v == 1]) or "SAFE",
            "predicted_labels": ",".join(pred_labels) or "SAFE",
            "is_correct": is_correct,
            "error_category": error_type,
            # Raw logits
            "logit_toxic": round(float(logits[0]), 4),
            "logit_severe_toxic": round(float(logits[1]), 4),
            "logit_obscene": round(float(logits[2]), 4),
            "logit_threat": round(float(logits[3]), 4),
            "logit_insult": round(float(logits[4]), 4),
            "logit_identity_hate": round(float(logits[5]), 4),
            # Sigmoid probabilities
            "prob_toxic": round(float(probs[0]), 4),
            "prob_severe_toxic": round(float(probs[1]), 4),
            "prob_obscene": round(float(probs[2]), 4),
            "prob_threat": round(float(probs[3]), 4),
            "prob_insult": round(float(probs[4]), 4),
            "prob_identity_hate": round(float(probs[5]), 4),
            # Thresholds
            "th_toxic": thresholds.get("toxic", 0.45),
            "th_severe_toxic": thresholds.get("severe_toxic", 0.62),
            "th_obscene": thresholds.get("obscene", 0.65),
            "th_threat": thresholds.get("threat", 0.59),
            "th_insult": thresholds.get("insult", 0.61),
            "th_identity_hate": thresholds.get("identity_hate", 0.60),
        }
        results.append(row)

    df_results = pd.DataFrame(results)
    out_csv = ARTIFACTS_DIR / "baseline_error_analysis.csv"
    df_results.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n--> Successfully saved {len(df_results)} diagnostic records to: {out_csv}")

    # Print analysis of the 6 canonical failure modes
    print("\n" + "=" * 65)
    print("ANALYSIS OF THE 6 CANONICAL FAILURE MODES UNDER BASELINE:")
    print("=" * 65)
    canonicals = df_results[df_results["id"].str.startswith("canonical_")]
    for _, r in canonicals.iterrows():
        status = "PASSED" if r["is_correct"] else f"FAILED ({r['error_category']})"
        print(f"\n[{r['id']}] \"{r['text']}\"")
        print(f"  Expected:  {r['expected_labels']}")
        print(f"  Predicted: {r['predicted_labels']} -> {status}")
        print(f"  Probs: toxic={r['prob_toxic']} (th={r['th_toxic']}), obscene={r['prob_obscene']} (th={r['th_obscene']}), threat={r['prob_threat']} (th={r['th_threat']}), insult={r['prob_insult']} (th={r['th_insult']})")
        print(f"  Logits: toxic={r['logit_toxic']}, obscene={r['logit_obscene']}, threat={r['logit_threat']}, insult={r['logit_insult']}")

    return df_results


if __name__ == "__main__":
    evaluate_baseline()
