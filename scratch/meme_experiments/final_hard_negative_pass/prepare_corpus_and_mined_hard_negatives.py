"""
Step 1: Dataset Construction and Hard-Negative Mining for Final Pass.
Extracts and categorizes training and validation samples with hard negatives.
Mines training false positives produced by the candidate model on the training split ONLY.
Precomputes and caches Layer-7 hidden states (h7), attention masks, and image embeddings.
Zero access to Multi3Hate.
"""

import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_hard_negative_pass"
EXP_DIR.mkdir(parents=True, exist_ok=True)
SRC_EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_improvement"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(42)
    device = torch.device("cpu")
    print("==================================================================")
    print("  STEP 1: DATASET MANIFEST & HARD-NEGATIVE MINING (TRAIN ONLY)")
    print("==================================================================")

    # 1. Load Existing Training and Validation Embeddings & Metadata
    print("\nLoading training and validation data partitions from model_improvement...")
    train_data = torch.load(SRC_EXP_DIR / "unified_train_embeddings.pt", weights_only=False)
    val_data = torch.load(SRC_EXP_DIR / "unified_val_embeddings.pt", weights_only=False)

    train_records = train_data["records"]
    train_t_emb = train_data["text_embeddings"]
    train_v_emb = train_data["image_embeddings"]
    train_y = train_data["labels"]

    val_records = val_data["records"]
    val_t_emb = val_data["text_embeddings"]
    val_v_emb = val_data["image_embeddings"]
    val_y = val_data["labels"]

    print(f"Loaded {len(train_records)} train records and {len(val_records)} val records.")

    # 2. Mine Model-Driven False Positives on TRAINING SPLIT ONLY
    print("\nMining model-driven false positives on TRAINING split using candidate checkpoint...")
    sys.path.insert(0, str(_PROJECT_ROOT))
    from scratch.meme_experiments.model_improvement.train_production_candidate import ProductionCandidateMultimodalClassifier

    cand_model = ProductionCandidateMultimodalClassifier(
        text_dim=768, image_dim=512, proj_dim=256, modality_dropout=0.0, classifier_dropout=0.0
    )
    cand_ckpt = torch.load(SRC_EXP_DIR / "best_model.pt", map_location=device, weights_only=False)
    cand_model.load_state_dict(cand_ckpt)
    cand_model.eval()

    with torch.no_grad():
        train_logits = cand_model(train_t_emb, train_v_emb)
        train_probs = torch.sigmoid(train_logits).numpy()

    # Mined FPs: ground truth == 0 and prob >= 0.40 on TRAINING data
    mined_fp_indices = set()
    for idx in range(len(train_records)):
        if float(train_y[idx]) == 0.0 and float(train_probs[idx]) >= 0.40:
            mined_fp_indices.add(idx)

    print(f"Mined {len(mined_fp_indices)} training false positives (p >= 0.40 on y=0).")

    # 3. Categorize Hard Negatives and Assign Sample Weights (Section 4 & 8)
    # Categories:
    # A. hindi_vulgar_non_abusive (HASOC vulgar but non-abusive) -> weight 2.0
    # B. hindi_sarcasm_banter (HASOC sarcastic but non-abusive) -> weight 2.0
    # C. hinglish_banter_non_toxic (TRAC non-aggressive banter) -> weight 2.0
    # D. hard_negative_pridemm_humor_banter (PrideMM humor=1 non-hate memes) -> weight 2.0
    # E. model_mined_hard_negative (training false positive) -> weight 2.0
    # F. clean_conversational / standard negative -> weight 1.0
    # Positives (indic_targeted_abuse, pridemm hate) -> weight 1.0

    train_weights = []
    hard_neg_categories = []
    train_texts = []

    cat_counts = {}

    for idx, r in enumerate(train_records):
        y_val = float(train_y[idx])
        raw_ctx = r.get("context_type", "clean_conversational")
        src = r.get("source", "")
        txt = r.get("text", "")
        train_texts.append(txt)

        is_fp = (idx in mined_fp_indices)

        if y_val == 1.0:
            assigned_cat = raw_ctx # "indic_targeted_abuse" or "standard_meme_hate"
            weight = 1.0
        else: # y_val == 0.0 (Negative)
            if is_fp:
                assigned_cat = "model_mined_hard_negative"
                weight = 2.0
            elif raw_ctx == "benign_profanity":
                assigned_cat = "hindi_vulgar_non_abusive"
                weight = 2.0
            elif raw_ctx == "sarcasm_banter":
                assigned_cat = "hindi_sarcasm_banter"
                weight = 2.0
            elif raw_ctx == "hinglish_banter_clean":
                assigned_cat = "hinglish_banter_non_toxic"
                weight = 2.0
            elif raw_ctx == "humor_banter":
                assigned_cat = "hard_negative_pridemm_humor_banter"
                weight = 2.0
            else:
                assigned_cat = "clean_conversational"
                weight = 1.0

        r["assigned_hard_negative_category"] = assigned_cat
        r["sample_weight"] = weight
        train_weights.append(weight)
        hard_neg_categories.append(assigned_cat)

        cat_counts[assigned_cat] = cat_counts.get(assigned_cat, 0) + 1

    print("\nTraining Split Hard-Negative Breakdown:")
    for c, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {c}: {count}")

    # Process validation records
    val_texts = [r.get("text", "") for r in val_records]
    val_weights = [1.0 for _ in val_records]

    # Save dataset_manifest.json
    manifest_data = {
        "manifest_title": "Final Hard-Negative & Multilingual Multimodal Training Corpus",
        "total_train_samples": len(train_records),
        "total_val_samples": len(val_records),
        "held_out_evaluation_dataset": {
            "name": "Multi3Hate",
            "samples": 600,
            "slices": {"english": 300, "hindi": 300},
            "status": "STRICTLY_HELD_OUT",
            "access_during_training": "ZERO_ACCESS"
        },
        "train_sources": {
            "pridemm_english_paired": int(sum(1 for r in train_records if r.get("source") == "pridemm")),
            "hasoc_2025_hindi_ocr": int(sum(1 for r in train_records if r.get("source") == "hasoc_2025")),
            "trac_hinglish": int(sum(1 for r in train_records if r.get("source") == "trac"))
        },
        "val_sources": {
            "pridemm_english_paired": int(sum(1 for r in val_records if r.get("source") == "pridemm")),
            "hasoc_2025_hindi_ocr": int(sum(1 for r in val_records if r.get("source") == "hasoc_2025")),
            "trac_hinglish": int(sum(1 for r in val_records if r.get("source") == "trac"))
        },
        "hard_negative_composition_train": cat_counts,
        "sample_weighting_policy": {
            "normal_samples": 1.0,
            "hard_negatives": 2.0
        }
    }
    with open(EXP_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\nSaved dataset_manifest.json to {EXP_DIR / 'dataset_manifest.json'}")

    # 4. Precompute and Cache Layer-7 Hidden States (h7) and Attention Masks
    print("\nPrecomputing and caching Layer 0..7 outputs (h7) using google/muril-base-cased...")
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    muril_full = AutoModel.from_pretrained("google/muril-base-cased")
    muril_full.eval()

    def extract_h7_and_masks(texts, max_len=64, batch_size=64):
        all_h7 = []
        all_ext_masks = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(chunk, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
            with torch.no_grad():
                embs = muril_full.embeddings(enc["input_ids"], token_type_ids=enc["token_type_ids"])
                ext_mask = muril_full.get_extended_attention_mask(enc["attention_mask"], enc["input_ids"].shape)
                h = embs
                for layer in muril_full.encoder.layer[:8]:
                    h = layer(h, attention_mask=ext_mask)[0]
                all_h7.append(h.cpu())
                all_ext_masks.append(ext_mask.cpu())
            if (i // batch_size) % 20 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} samples...")
        return torch.cat(all_h7, dim=0), torch.cat(all_ext_masks, dim=0)

    print("\nCaching training features...")
    train_h7, train_ext_masks = extract_h7_and_masks(train_texts, max_len=64, batch_size=64)
    torch.save({
        "h7": train_h7,
        "extended_attention_mask": train_ext_masks,
        "image_embeddings": train_v_emb,
        "labels": train_y,
        "sample_weights": torch.tensor(train_weights, dtype=torch.float32),
        "records": train_records
    }, EXP_DIR / "cached_train_features.pt")
    print(f"Saved cached_train_features.pt ({train_h7.shape})")

    print("\nCaching validation features...")
    val_h7, val_ext_masks = extract_h7_and_masks(val_texts, max_len=64, batch_size=64)
    torch.save({
        "h7": val_h7,
        "extended_attention_mask": val_ext_masks,
        "image_embeddings": val_v_emb,
        "labels": val_y,
        "sample_weights": torch.tensor(val_weights, dtype=torch.float32),
        "records": val_records
    }, EXP_DIR / "cached_val_features.pt")
    print(f"Saved cached_val_features.pt ({val_h7.shape})")

    print("\nStep 1 finished successfully!")

if __name__ == "__main__":
    main()
