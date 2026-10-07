"""
Build unified training and validation dataset for the single production-candidate pass.
Integrates:
1. PrideMM multimodal hate supervision (4,328 train, 228 val)
2. HASOC 2025 Hindi meme OCR supervision (991 train, 150 val)
3. TRAC Hinglish banter and abuse supervision (800 train, 200 val)
Multi3Hate (600 memes) remains STRICTLY HELD OUT.
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
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_improvement"
EXP_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Building Unified Training & Multilingual Validation Dataset on {device} ===")

    # 1. Load PrideMM Features
    print("Loading precomputed PrideMM embeddings...")
    pridemm_train = torch.load(CACHE_DIR / "pridemm_train_embeddings.pt", weights_only=False)
    pridemm_val = torch.load(CACHE_DIR / "pridemm_val_embeddings.pt", weights_only=False)

    p_train_t = pridemm_train["text_embeddings"] # (4328, 768)
    p_train_v = pridemm_train["image_embeddings"] # (4328, 512)
    p_train_y = pridemm_train["labels"].float()   # (4328,)
    p_train_items = pridemm_train["items"]

    p_val_t = pridemm_val["text_embeddings"] # (228, 768)
    p_val_v = pridemm_val["image_embeddings"] # (228, 512)
    p_val_y = pridemm_val["labels"].float()   # (228,)
    p_val_items = pridemm_val["items"]

    print(f"PrideMM Train: {p_train_t.shape[0]} | Val: {p_val_t.shape[0]}")

    # Tag PrideMM items with metadata
    train_records = []
    for idx, it in enumerate(p_train_items):
        train_records.append({
            "source": "pridemm",
            "language": "english",
            "text": it.get("text", ""),
            "label": float(p_train_y[idx]),
            "context_type": "humor_banter" if it.get("humor") == 1 else "standard_meme",
            "is_paired": True
        })

    val_records = []
    for idx, it in enumerate(p_val_items):
        val_records.append({
            "source": "pridemm",
            "language": "english",
            "text": it.get("text", ""),
            "label": float(p_val_y[idx]),
            "context_type": "humor_banter" if it.get("humor") == 1 else "standard_meme",
            "is_paired": True
        })

    # 2. Process HASOC 2025 Hindi Meme OCR
    print("\nProcessing HASOC 2025 Hindi Meme OCR data...")
    hasoc_path = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "hasoc_meme_2025" / "Hindi Train HASOC 2025 - Updated.csv"
    df_hasoc = pd.read_csv(hasoc_path)
    df_hasoc = df_hasoc.dropna(subset=["OCR"]).copy()
    df_hasoc["OCR"] = df_hasoc["OCR"].astype(str).str.strip()
    df_hasoc = df_hasoc[df_hasoc["OCR"].str.len() > 3].reset_index(drop=True)

    # Label definition for HASOC:
    # Abuse == 'Abusive' -> 1.0 (Hate / Toxic Abuse)
    # Non-abusive with vulgarity/sarcasm -> 0.0 (Benign profanity / Sarcasm)
    hasoc_labels = []
    hasoc_types = []
    for _, row in df_hasoc.iterrows():
        is_abuse = (str(row.get("Abuse", "")).strip().lower() == "abusive")
        is_vulgar = (str(row.get("Vulgar", "")).strip().lower() == "vulgar")
        is_sarcastic = (str(row.get("Sarcasm", "")).strip().lower() == "sarcastic")
        
        if is_abuse:
            hasoc_labels.append(1.0)
            hasoc_types.append("indic_targeted_abuse")
        else:
            hasoc_labels.append(0.0)
            if is_vulgar:
                hasoc_types.append("benign_profanity")
            elif is_sarcastic:
                hasoc_types.append("sarcasm_banter")
            else:
                hasoc_types.append("clean_conversational")

    df_hasoc["target_label"] = hasoc_labels
    df_hasoc["context_type"] = hasoc_types

    # Split HASOC into 85% train, 15% validation
    h_train_df, h_val_df = train_test_split(
        df_hasoc, test_size=0.15, random_state=42, stratify=df_hasoc["target_label"]
    )
    print(f"HASOC 2025: {len(h_train_df)} train, {len(h_val_df)} val (Positive rate: {df_hasoc['target_label'].mean():.2%})")

    # 3. Process TRAC Hinglish Banter & Aggression
    print("\nProcessing TRAC Hinglish Social Media data...")
    trac_path = _PROJECT_ROOT / "dataset" / "trac_hinglish_raw.csv"
    df_trac = pd.read_csv(trac_path)
    df_trac = df_trac.dropna(subset=["Text"]).copy()
    df_trac["Text"] = df_trac["Text"].astype(str).str.strip()

    # Sample 500 Non-Aggressive (0) and 500 Aggressive (1)
    df_trac_nonagg = df_trac[df_trac["Aggression-level"] == 0].sample(n=500, random_state=42)
    df_trac_agg = df_trac[df_trac["Aggression-level"] == 1].sample(n=500, random_state=42)
    df_trac_sampled = pd.concat([df_trac_nonagg, df_trac_agg], ignore_index=True)

    trac_labels = []
    trac_types = []
    for _, row in df_trac_sampled.iterrows():
        lvl = int(row["Aggression-level"])
        trac_labels.append(float(lvl))
        trac_types.append("indic_targeted_abuse" if lvl == 1 else "hinglish_banter_clean")

    df_trac_sampled["target_label"] = trac_labels
    df_trac_sampled["context_type"] = trac_types

    t_train_df, t_val_df = train_test_split(
        df_trac_sampled, test_size=0.20, random_state=42, stratify=df_trac_sampled["target_label"]
    )
    print(f"TRAC Hinglish: {len(t_train_df)} train, {len(t_val_df)} val")

    # 4. Extract MuRIL Text Embeddings for Indic samples
    print("\nExtracting MuRIL 768-dim embeddings for Indic samples...")
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    muril = AutoModel.from_pretrained("google/muril-base-cased")
    muril.to(device)
    muril.eval()

    def extract_muril_tensors(texts):
        embs = []
        bs = 64
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            inputs = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = muril(**inputs)
                cls_tensors = out.last_hidden_state[:, 0, :].cpu()
                embs.append(cls_tensors)
        return torch.cat(embs, dim=0)

    h_train_texts = h_train_df["OCR"].tolist()
    h_val_texts = h_val_df["OCR"].tolist()
    t_train_texts = t_train_df["Text"].tolist()
    t_val_texts = t_val_df["Text"].tolist()

    h_train_embs = extract_muril_tensors(h_train_texts)
    h_val_embs = extract_muril_tensors(h_val_texts)
    t_train_embs = extract_muril_tensors(t_train_texts)
    t_val_embs = extract_muril_tensors(t_val_texts)

    # 5. Assemble Combined Tensors
    # For Indic text samples without image, visual embedding is the zero vector (512-dim)
    h_train_v = torch.zeros(len(h_train_texts), 512)
    h_val_v = torch.zeros(len(h_val_texts), 512)
    t_train_v = torch.zeros(len(t_train_texts), 512)
    t_val_v = torch.zeros(len(t_val_texts), 512)

    h_train_y = torch.tensor(h_train_df["target_label"].values, dtype=torch.float32)
    h_val_y = torch.tensor(h_val_df["target_label"].values, dtype=torch.float32)
    t_train_y = torch.tensor(t_train_df["target_label"].values, dtype=torch.float32)
    t_val_y = torch.tensor(t_val_df["target_label"].values, dtype=torch.float32)

    # Concatenate all training partitions
    all_train_t = torch.cat([p_train_t, h_train_embs, t_train_embs], dim=0)
    all_train_v = torch.cat([p_train_v, h_train_v, t_train_v], dim=0)
    all_train_y = torch.cat([p_train_y, h_train_y, t_train_y], dim=0)

    # Concatenate all validation partitions
    all_val_t = torch.cat([p_val_t, h_val_embs, t_val_embs], dim=0)
    all_val_v = torch.cat([p_val_v, h_val_v, t_val_v], dim=0)
    all_val_y = torch.cat([p_val_y, h_val_y, t_val_y], dim=0)

    # Update metadata records
    for i, r in h_train_df.iterrows():
        train_records.append({
            "source": "hasoc_2025",
            "language": "hindi",
            "text": r["OCR"],
            "label": float(r["target_label"]),
            "context_type": r["context_type"],
            "is_paired": False
        })
    for i, r in t_train_df.iterrows():
        train_records.append({
            "source": "trac_hinglish",
            "language": "hinglish",
            "text": r["Text"],
            "label": float(r["target_label"]),
            "context_type": r["context_type"],
            "is_paired": False
        })

    for i, r in h_val_df.iterrows():
        val_records.append({
            "source": "hasoc_2025",
            "language": "hindi",
            "text": r["OCR"],
            "label": float(r["target_label"]),
            "context_type": r["context_type"],
            "is_paired": False
        })
    for i, r in t_val_df.iterrows():
        val_records.append({
            "source": "trac_hinglish",
            "language": "hinglish",
            "text": r["Text"],
            "label": float(r["target_label"]),
            "context_type": r["context_type"],
            "is_paired": False
        })

    print(f"\nFinal Unified Training Set: {all_train_t.shape[0]} samples")
    print(f"  - English (PrideMM): {p_train_t.shape[0]} ({p_train_t.shape[0]/all_train_t.shape[0]:.1%})")
    print(f"  - Hindi Meme OCR (HASOC): {h_train_embs.shape[0]} ({h_train_embs.shape[0]/all_train_t.shape[0]:.1%})")
    print(f"  - Hinglish Banter/Abuse (TRAC): {t_train_embs.shape[0]} ({t_train_embs.shape[0]/all_train_t.shape[0]:.1%})")
    print(f"  - Positive Rate: {all_train_y.mean():.2%}")

    print(f"\nFinal Unified Validation Set: {all_val_t.shape[0]} samples")
    print(f"  - English (PrideMM): {p_val_t.shape[0]}")
    print(f"  - Hindi (HASOC): {h_val_embs.shape[0]}")
    print(f"  - Hinglish (TRAC): {t_val_embs.shape[0]}")
    print(f"  - Positive Rate: {all_val_y.mean():.2%}")

    # Save processed tensors
    torch.save({
        "text_embeddings": all_train_t,
        "image_embeddings": all_train_v,
        "labels": all_train_y,
        "records": train_records
    }, EXP_DIR / "unified_train_embeddings.pt")

    torch.save({
        "text_embeddings": all_val_t,
        "image_embeddings": all_val_v,
        "labels": all_val_y,
        "records": val_records
    }, EXP_DIR / "unified_val_embeddings.pt")

    # Manifest metadata
    manifest = {
        "dataset_title": "Production Candidate Unified Multimodal & Indic Toxicity Training Corpus",
        "total_train_samples": all_train_t.shape[0],
        "total_val_samples": all_val_t.shape[0],
        "held_out_evaluation_dataset": {
            "name": "Multi3Hate",
            "samples": 600,
            "slices": {"english": 300, "hindi": 300},
            "status": "STRICTLY_HELD_OUT",
            "leakage_guarantee": "ZERO samples used in training or validation"
        },
        "train_partition_breakdown": {
            "pridemm_english_paired_memes": len(p_train_items),
            "hasoc_2025_hindi_meme_ocr": len(h_train_df),
            "trac_hinglish_social_media": len(t_train_df)
        },
        "val_partition_breakdown": {
            "pridemm_english_paired_memes": len(p_val_items),
            "hasoc_2025_hindi_meme_ocr": len(h_val_df),
            "trac_hinglish_social_media": len(t_val_df)
        },
        "context_categories_represented": [
            "multimodal_meme_hate",
            "benign_profanity",
            "sarcasm_banter",
            "indic_targeted_abuse",
            "clean_conversational"
        ]
    }

    with open(EXP_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest and tensors successfully exported to {EXP_DIR}")

if __name__ == "__main__":
    main()
