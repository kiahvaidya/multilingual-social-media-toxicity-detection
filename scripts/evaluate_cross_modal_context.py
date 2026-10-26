"""
Model C: Cross-Modal Context and Semantic Contrast Evaluation Script.

Evaluates:
  1. The 7 Semantic Contrast Pairs:
     - Tests Model A (Production), Model B (Meme Text), Model C (Multimodal Architecture)
     - Measures toxic score separation (Harmful - Benign)
  2. The Dedicated Cross-Modal Context Evaluation Set:
     - Category A: Text looks toxic, Image is benign (Contextual Inversion)
     - Category B: Text looks benign, Image is harmful (Emergent Toxicity)
     - Category C: Text and Image agree on harm (Explicit Harm)
     - Category D: Text and Image agree on benign (Normal Harmless Meme)
     - Compares Model B (text-only) vs Model C (multimodal) to measure modality conflict resolution.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.model import MuRILForMultiLabelToxicity, DEFAULT_LABELS
from scripts.train_meme_multimodal_baseline import MultimodalMemeToxicityModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalCrossModal")

EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_c"
MODEL_A_CHECKPOINT = _PROJECT_ROOT / "muril_toxicity_model.pt"
MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
MODEL_C_CHECKPOINT = EXP_DIR / "muril_model_c.pt"
CROSS_MODAL_RESULTS_OUT = EXP_DIR / "cross_modal_context_results.json"

MULTI3HATE_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "multi3hate"
MULTI3HATE_IMAGES_BASE = MULTI3HATE_DIR / "Multi3Hate_repo" / "data" / "memes"

FROZEN_THRESHOLDS = {
    "toxic": 0.45,
    "severe_toxic": 0.62,
    "obscene": 0.65,
    "threat": 0.59,
    "insult": 0.61,
    "identity_hate": 0.60,
}

# The identical 7 Semantic Contrast Pairs evaluated for Model B
SEMANTIC_CONTRAST_PAIRS = [
    {
        "pair_id": 1,
        "category": "English Profanity: Benign Exclamation vs Personal Attack",
        "benign": "Fuck yeah! We finally passed all the unit tests!",
        "harmful": "You're a fucking idiot and I hope you get hurt.",
    },
    {
        "pair_id": 2,
        "category": "English Metaphor: Praise vs Literal Threat",
        "benign": "Bro, you totally killed it on stage tonight!",
        "harmful": "I am going to kill you tonight.",
    },
    {
        "pair_id": 3,
        "category": "English Slang: Admiration vs Degradation",
        "benign": "This movie was sick!",
        "harmful": "You're sick and disgusting.",
    },
    {
        "pair_id": 4,
        "category": "Hinglish Slang: Informal Praise vs Explicit Abuse",
        "benign": "Bhai kya bawaal cheez banayi hai yaar maza aa gaya",
        "harmful": "Bhai tu ek number ka chutiya hai chup baith",
    },
    {
        "pair_id": 5,
        "category": "Hindi Vernacular: Praise vs Demeaning Abuse",
        "benign": "यह काम वास्तव में बहुत धांसू और प्रेरणादायक है",
        "harmful": "तू बिल्कुल नीच और घटिया इंसान है",
    },
    {
        "pair_id": 6,
        "category": "Obfuscation: Benign Leetspeak vs Camouflaged Toxicity",
        "benign": "Nice sh!rt bro, looks super c00l and fresh",
        "harmful": "You are an absolute f.u.c.k.i.n.g b!tch",
    },
    {
        "pair_id": 7,
        "category": "Social Dynamics: Peer Banter vs Directed Threat",
        "benign": "Abe chup kar saale, tera dimag kharab ho gaya hai kya",
        "harmful": "Ghar se bahar nikal, aaj teri jaan le lunga",
    },
]


def load_cross_modal_evaluation_set() -> List[Dict[str, Any]]:
    """Loads verified, real instances from Multi3Hate representing the 4 cross-modal phenomena."""
    # Find matching image files in Multi3Hate repository
    def find_img(lang: str, mid: int) -> str:
        matches = list((MULTI3HATE_IMAGES_BASE / lang).rglob(f"{mid}.jpg"))
        if not matches:
            raise FileNotFoundError(f"Image for meme {mid} not found under {lang}")
        return str(matches[0])

    dataset = [
        # --- Category A: TEXT LOOKS TOXIC, IMAGE IS BENIGN (Contextual Inversion) ---
        {
            "category": "Category A: Text looks toxic, Image is benign (Contextual Inversion)",
            "sample_id": "cross_modal_A1_mid115",
            "meme_id": 115,
            "text": "I'm pretty damn happy about football",
            "image_path": find_img("en", 115),
            "template_name": "troll asian (smiling meme face)",
            "text_prima_facie": "TOXIC_OR_VULGAR (contains 'damn')",
            "visual_context": "BENIGN (smiling comic face)",
            "ground_truth_label": 0,  # SAFE / NON-HATE
            "expected_behavior": "Visual context should invert/suppress false positive toxicity.",
        },
        {
            "category": "Category A: Text looks toxic, Image is benign (Contextual Inversion)",
            "sample_id": "cross_modal_A2_mid282",
            "meme_id": 282,
            "text": "Freeze! You're under arrest For producing non accessible material",
            "image_path": find_img("en", 282),
            "template_name": "policeman",
            "text_prima_facie": "TOXIC_OR_THREAT ('under arrest', 'freeze')",
            "visual_context": "BENIGN (humorous office accessibility meme)",
            "ground_truth_label": 0,  # SAFE / NON-HATE
            "expected_behavior": "Visual context should prevent false alarm on authoritative arrest text.",
        },

        # --- Category B: TEXT LOOKS BENIGN, IMAGE IS HARMFUL (Emergent Toxicity) ---
        {
            "category": "Category B: Text looks benign, Image is harmful (Emergent Toxicity)",
            "sample_id": "cross_modal_B1_mid37",
            "meme_id": 37,
            "text": "SEE THIS HAT THRIFT STORE",
            "image_path": find_img("en", 37),
            "template_name": "advicejew (antisemitic caricature)",
            "text_prima_facie": "BENIGN ('See this hat thrift store')",
            "visual_context": "HARMFUL (antisemitic caricature trope)",
            "ground_truth_label": 1,  # HATE
            "expected_behavior": "Visual information MUST elevate toxicity that text-only models miss.",
        },
        {
            "category": "Category B: Text looks benign, Image is harmful (Emergent Toxicity)",
            "sample_id": "cross_modal_B2_mid141",
            "meme_id": 141,
            "text": "wants to defend 'white tradition' lives in NY and eats at mcdonalds",
            "image_path": find_img("en", 141),
            "template_name": "white power problems",
            "text_prima_facie": "BENIGN ('eats at mcdonalds')",
            "visual_context": "HARMFUL (white supremacist trope template)",
            "ground_truth_label": 1,  # HATE
            "expected_behavior": "Visual grounding identifies hate template that text-only model underpredicts.",
        },

        # --- Category C: TEXT AND IMAGE AGREE ON HARM (Explicit Harm) ---
        {
            "category": "Category C: Text and Image agree on harm (Explicit Harm)",
            "sample_id": "cross_modal_C1_mid89",
            "meme_id": 89,
            "text": "i hate black people thats why i got a job with a gun",
            "image_path": find_img("en", 89),
            "template_name": "strict policeman",
            "text_prima_facie": "HARMFUL (explicit racial hate)",
            "visual_context": "HARMFUL (hostile firearm imagery)",
            "ground_truth_label": 1,  # HATE
            "expected_behavior": "Both modalities reinforce high toxic score.",
        },
        {
            "category": "Category C: Text and Image agree on harm (Explicit Harm)",
            "sample_id": "cross_modal_C2_mid96",
            "meme_id": 96,
            "text": "i hate men! GOD, i NEED A GUY'S COCK RIGHT NOW...",
            "image_path": find_img("en", 96),
            "template_name": "feminist cunt",
            "text_prima_facie": "HARMFUL (misandry + crude obscenity)",
            "visual_context": "HARMFUL (hostile caricature)",
            "ground_truth_label": 1,  # HATE
            "expected_behavior": "Both modalities detect explicit abuse.",
        },

        # --- Category D: TEXT AND IMAGE AGREE ON BENIGN MEANING (Harmless Meme) ---
        {
            "category": "Category D: Text and Image agree on benign (Harmless Meme)",
            "sample_id": "cross_modal_D1_mid174",
            "meme_id": 174,
            "text": "showed this blog to my non-german friends they don't get it",
            "image_path": find_img("en", 174),
            "template_name": "germany pls",
            "text_prima_facie": "BENIGN (ordinary blogging meme)",
            "visual_context": "BENIGN (harmless cartoon)",
            "ground_truth_label": 0,  # SAFE
            "expected_behavior": "Both modalities predict low toxicity.",
        },
        {
            "category": "Category D: Text and Image agree on benign (Harmless Meme)",
            "sample_id": "cross_modal_D2_mid265",
            "meme_id": 265,
            "text": "Quality men believe in equality of women",
            "image_path": find_img("en", 265),
            "template_name": "privilege abusing white couple",
            "text_prima_facie": "BENIGN (pro-equality quote)",
            "visual_context": "BENIGN (conversational meme)",
            "ground_truth_label": 0,  # SAFE
            "expected_behavior": "Both modalities predict low toxicity.",
        },
    ]
    return dataset


def run_evaluations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load Model A
    model_a = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_a.load_state_dict(torch.load(MODEL_A_CHECKPOINT, map_location=device, weights_only=True))
    model_a.to(device).eval()

    # Load Model B
    model_b = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_b.load_state_dict(torch.load(MODEL_B_CHECKPOINT, map_location=device, weights_only=True))
    model_b.to(device).eval()

    # Load Model C
    model_c = MultimodalMemeToxicityModel(num_labels=6)
    model_c.load_state_dict(torch.load(MODEL_C_CHECKPOINT, map_location=device, weights_only=True))
    model_c.to(device).eval()

    toxic_idx = DEFAULT_LABELS.index("toxic")
    threshold = FROZEN_THRESHOLDS["toxic"]

    def score_text(model, text: str) -> float:
        enc = tokenizer(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            return float(torch.sigmoid(logits[0, toxic_idx]).cpu().item())

    # ==============================================================
    # 1. Evaluate the 7 Semantic Contrast Pairs
    # ==============================================================
    logger.info("Evaluating 7 Semantic Contrast Pairs across Model A, Model B, and Model C...")
    semantic_results = []

    for pair in SEMANTIC_CONTRAST_PAIRS:
        score_a_benign = score_text(model_a, pair["benign"])
        score_a_harmful = score_text(model_a, pair["harmful"])
        sep_a = score_a_harmful - score_a_benign

        score_b_benign = score_text(model_b, pair["benign"])
        score_b_harmful = score_text(model_b, pair["harmful"])
        sep_b = score_b_harmful - score_b_benign

        # Model C text branch evaluation
        enc_benign = tokenizer(pair["benign"], truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        enc_harmful = tokenizer(pair["harmful"], truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits_c_benign = model_c(input_ids=enc_benign["input_ids"].to(device), attention_mask=enc_benign["attention_mask"].to(device), mode="text_only")
            score_c_benign = float(torch.sigmoid(logits_c_benign[0, toxic_idx]).cpu().item())

            logits_c_harmful = model_c(input_ids=enc_harmful["input_ids"].to(device), attention_mask=enc_harmful["attention_mask"].to(device), mode="text_only")
            score_c_harmful = float(torch.sigmoid(logits_c_harmful[0, toxic_idx]).cpu().item())
        sep_c = score_c_harmful - score_c_benign

        semantic_results.append({
            "pair_id": pair["pair_id"],
            "category": pair["category"],
            "benign_text": pair["benign"],
            "harmful_text": pair["harmful"],
            "model_a": {
                "benign_score": round(score_a_benign, 4),
                "harmful_score": round(score_a_harmful, 4),
                "separation": round(sep_a, 4),
            },
            "model_b": {
                "benign_score": round(score_b_benign, 4),
                "harmful_score": round(score_b_harmful, 4),
                "separation": round(sep_b, 4),
            },
            "model_c": {
                "benign_score": round(score_c_benign, 4),
                "harmful_score": round(score_c_harmful, 4),
                "separation": round(sep_c, 4),
            },
            "gain_b_vs_a": round(sep_b - sep_a, 4),
            "gain_c_vs_a": round(sep_c - sep_a, 4),
        })

    # ==============================================================
    # 2. Evaluate Dedicated Cross-Modal Context Evaluation Set
    # ==============================================================
    logger.info("Evaluating Dedicated Cross-Modal Context Evaluation Set...")
    cross_modal_items = load_cross_modal_evaluation_set()
    cross_modal_results = []

    for item in cross_modal_items:
        text = item["text"]
        img_path = item["image_path"]

        # Text encoding
        enc = tokenizer(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Image processing
        img = Image.open(img_path).convert("RGB")
        img_inputs = vision_processor(images=img, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(device)

        with torch.no_grad():
            # Model A (Text Only)
            logits_a = model_a(input_ids=input_ids, attention_mask=attention_mask)
            score_a = float(torch.sigmoid(logits_a[0, toxic_idx]).cpu().item())

            # Model B (Text Only)
            logits_b = model_b(input_ids=input_ids, attention_mask=attention_mask)
            score_b = float(torch.sigmoid(logits_b[0, toxic_idx]).cpu().item())

            # Model C (Multimodal)
            logits_c = model_c(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, mode="multimodal")
            score_c = float(torch.sigmoid(logits_c[0, toxic_idx]).cpu().item())

            # Image Only Baseline
            logits_img = model_c(pixel_values=pixel_values, mode="vision_only")
            score_img = float(torch.sigmoid(logits_img[0, toxic_idx]).cpu().item())

        gt = item["ground_truth_label"]
        pred_b = int(score_b >= threshold)
        pred_c = int(score_c >= threshold)
        correct_b = bool(pred_b == gt)
        correct_c = bool(pred_c == gt)

        cross_modal_results.append({
            "sample_id": item["sample_id"],
            "category": item["category"],
            "meme_id": item["meme_id"],
            "text": text,
            "template_name": item["template_name"],
            "ground_truth_hate": gt,
            "text_prima_facie": item["text_prima_facie"],
            "visual_context": item["visual_context"],
            "expected_behavior": item["expected_behavior"],
            "scores": {
                "model_a": round(score_a, 4),
                "model_b_text_only": round(score_b, 4),
                "model_c_image_only": round(score_img, 4),
                "model_c_multimodal": round(score_c, 4),
            },
            "predictions": {
                "model_b": pred_b,
                "model_c": pred_c,
            },
            "correctness": {
                "model_b_correct": correct_b,
                "model_c_correct": correct_c,
            },
            "conflict_resolved_by_c": bool(not correct_b and correct_c),
        })

    # Save output
    output = {
        "semantic_contrast_pairs": semantic_results,
        "cross_modal_context_set": cross_modal_results,
    }

    with open(CROSS_MODAL_RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Evaluation complete! Results saved to %s", CROSS_MODAL_RESULTS_OUT)

    print("\n" + "=" * 70)
    print("7 SEMANTIC CONTRAST PAIRS: SCORE SEPARATION SUMMARY")
    for r in semantic_results:
        print(f"Pair {r['pair_id']}: {r['category']}")
        print(f"  Model A Sep: {r['model_a']['separation']:+.4f} | Model B Sep: {r['model_b']['separation']:+.4f} | Model C Sep: {r['model_c']['separation']:+.4f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("CROSS-MODAL CONTEXT EVALUATION: MODALITY CONFLICT RESOLUTION")
    for r in cross_modal_results:
        print(f"[{r['sample_id']}] {r['category'][:40]}...")
        print(f"  Text: '{r['text']}' | Template: {r['template_name']}")
        print(f"  GT: {r['ground_truth_hate']} | Score B (Text): {r['scores']['model_b_text_only']:.4f} | Score C (Multi): {r['scores']['model_c_multimodal']:.4f} | Resolved: {r['conflict_resolved_by_c']}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    run_evaluations()
