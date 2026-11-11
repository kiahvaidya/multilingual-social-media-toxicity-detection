"""
Audit and Normalization Script for Benchmark Datasets
1. Inspects schemas and label distributions for all raw datasets in data/raw/
2. Checks for cross-dataset duplicate overlap and train/test leakage
3. Implements dataset-specific label adapters conforming to Phase 3 unified schema:
   {
       "text": "...",
       "language": "hi" | "en" | "hi-en",
       "source": "<DATASET_NAME>",
       "task": "toxicity_six_label" | "offensive_hasoc" | "aggression_trac",
       "labels": {...}
   }
4. Saves processed datasets separately under data/processed/ without modifying raw files.
"""

import sys
import os
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def hash_text(text: str) -> str:
    """Normalize whitespace and compute MD5 hash for exact duplication checking."""
    cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


# ==============================================================
# Phase 2 Validation: Schema & Distribution Audit
# ==============================================================
def audit_raw_datasets() -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 2: RAW DATASET SCHEMA & DISTRIBUTION AUDIT")
    print("=" * 60)
    audit_report = {}

    # 1. Jigsaw Toxic Comment
    jigsaw_train_p = RAW_DIR / "jigsaw_toxic_comment" / "train.csv"
    jigsaw_test_p = RAW_DIR / "jigsaw_toxic_comment" / "test.csv"
    jigsaw_test_lbl_p = RAW_DIR / "jigsaw_toxic_comment" / "test_labels.csv"

    if jigsaw_train_p.exists():
        df_j_train = pd.read_csv(jigsaw_train_p)
        df_j_test_lbl = pd.read_csv(jigsaw_test_lbl_p) if jigsaw_test_lbl_p.exists() else pd.DataFrame()
        valid_test_mask = (df_j_test_lbl["toxic"] != -1) if not df_j_test_lbl.empty else []
        df_j_test_scored = df_j_test_lbl[valid_test_mask] if not df_j_test_lbl.empty else pd.DataFrame()

        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        jigsaw_stats = {
            "total_train_rows": len(df_j_train),
            "columns": list(df_j_train.columns),
            "total_scored_test_rows": len(df_j_test_scored),
            "class_frequencies_train": {
                col: int(df_j_train[col].sum()) for col in labels
            },
            "class_percentages_train": {
                col: round(float(df_j_train[col].mean()) * 100, 3) for col in labels
            }
        }
        audit_report["jigsaw_toxic_comment"] = jigsaw_stats
        print("[Jigsaw] Train samples:", f"{len(df_j_train):,}")
        print("  Frequencies:", jigsaw_stats["class_frequencies_train"])
        print("  Percentages:", jigsaw_stats["class_percentages_train"])

    # 2. HASOC 2021
    hasoc_hi_p = RAW_DIR / "hasoc_2021" / "hindi_2021.csv"
    hasoc_en_p = RAW_DIR / "hasoc_2021" / "english_2021.csv"
    if hasoc_hi_p.exists():
        df_h_hi = pd.read_csv(hasoc_hi_p)
        df_h_en = pd.read_csv(hasoc_en_p) if hasoc_en_p.exists() else pd.DataFrame()

        hasoc_stats = {
            "hindi_rows": len(df_h_hi),
            "hindi_columns": list(df_h_hi.columns),
            "hindi_task1_distribution": df_h_hi["task_1"].value_counts().to_dict(),
            "hindi_task2_distribution": df_h_hi["task_2"].value_counts().to_dict(),
            "english_rows": len(df_h_en) if not df_h_en.empty else 0,
            "english_task1_distribution": df_h_en["task_1"].value_counts().to_dict() if not df_h_en.empty else {}
        }
        audit_report["hasoc_2021"] = hasoc_stats
        print("\n[HASOC 2021]")
        print("  Hindi Task 1 (HOF/NOT):", hasoc_stats["hindi_task1_distribution"])
        print("  Hindi Task 2 (HATE/OFFN/PRFN/NONE):", hasoc_stats["hindi_task2_distribution"])
        if not df_h_en.empty:
            print("  English Task 1:", hasoc_stats["english_task1_distribution"])

    # 3. TRAC-1
    trac_hi_p = RAW_DIR / "trac_1" / "agr_hi_train.csv"
    trac_en_p = RAW_DIR / "trac_1" / "agr_en_train.csv"
    if trac_hi_p.exists():
        df_t_hi = pd.read_csv(trac_hi_p, header=None, names=["id", "text", "label"])
        df_t_en = pd.read_csv(trac_en_p, header=None, names=["id", "text", "label"]) if trac_en_p.exists() else pd.DataFrame()

        trac_stats = {
            "hindi_train_rows": len(df_t_hi),
            "hindi_label_distribution": df_t_hi["label"].value_counts().to_dict(),
            "english_train_rows": len(df_t_en) if not df_t_en.empty else 0,
            "english_label_distribution": df_t_en["label"].value_counts().to_dict() if not df_t_en.empty else {}
        }
        audit_report["trac_1"] = trac_stats
        print("\n[TRAC-1 Aggression]")
        print("  Hindi Distribution (NAG/CAG/OAG):", trac_stats["hindi_label_distribution"])
        if not df_t_en.empty:
            print("  English Distribution:", trac_stats["english_label_distribution"])

    # 4. Civil Comments
    civil_train_p = RAW_DIR / "civil_comments" / "civil_comments_train_sample.csv"
    civil_val_p = RAW_DIR / "civil_comments" / "civil_comments_validation.csv"
    if civil_train_p.exists():
        df_c_train = pd.read_csv(civil_train_p)
        df_c_val = pd.read_csv(civil_val_p) if civil_val_p.exists() else pd.DataFrame()

        civil_stats = {
            "train_sample_rows": len(df_c_train),
            "columns": list(df_c_train.columns),
            "train_sample_threat_count": int((df_c_train["threat"] >= 0.5).sum()),
            "train_sample_toxic_count": int((df_c_train["toxicity"] >= 0.5).sum()),
            "validation_rows": len(df_c_val) if not df_c_val.empty else 0
        }
        audit_report["civil_comments"] = civil_stats
        print("\n[Civil Comments]")
        print(f"  Train Sample: {len(df_c_train):,} rows | Threats (>=0.5): {civil_stats['train_sample_threat_count']:,} | Toxic (>=0.5): {civil_stats['train_sample_toxic_count']:,}")
        if not df_c_val.empty:
            print(f"  Validation Set: {len(df_c_val):,} rows")

    return audit_report


# ==============================================================
# Cross-Dataset Leakage & Duplicate Detection
# ==============================================================
def check_leakage_and_duplicates() -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("CROSS-DATASET LEAKAGE & DUPLICATE AUDIT")
    print("=" * 60)

    hashes_by_dataset = {}

    # 1. Jigsaw train & test
    j_train = pd.read_csv(RAW_DIR / "jigsaw_toxic_comment" / "train.csv")
    j_train_hashes = set(j_train["comment_text"].dropna().map(hash_text))
    hashes_by_dataset["jigsaw_train"] = j_train_hashes

    j_test = pd.read_csv(RAW_DIR / "jigsaw_toxic_comment" / "test.csv")
    j_test_hashes = set(j_test["comment_text"].dropna().map(hash_text))
    hashes_by_dataset["jigsaw_test"] = j_test_hashes

    # 2. HASOC 2021 Hindi & English
    h_hi = pd.read_csv(RAW_DIR / "hasoc_2021" / "hindi_2021.csv")
    h_hi_hashes = set(h_hi["text"].dropna().map(hash_text))
    hashes_by_dataset["hasoc_hindi"] = h_hi_hashes

    h_en = pd.read_csv(RAW_DIR / "hasoc_2021" / "english_2021.csv")
    h_en_hashes = set(h_en["text"].dropna().map(hash_text))
    hashes_by_dataset["hasoc_english"] = h_en_hashes

    # 3. TRAC-1 Hindi & English
    t_hi = pd.read_csv(RAW_DIR / "trac_1" / "agr_hi_train.csv", header=None, names=["id", "text", "label"])
    t_hi_hashes = set(t_hi["text"].dropna().map(hash_text))
    hashes_by_dataset["trac_hindi_train"] = t_hi_hashes

    t_en = pd.read_csv(RAW_DIR / "trac_1" / "agr_en_train.csv", header=None, names=["id", "text", "label"])
    t_en_hashes = set(t_en["text"].dropna().map(hash_text))
    hashes_by_dataset["trac_english_train"] = t_en_hashes

    # 4. Civil Comments Train Sample
    c_train = pd.read_csv(RAW_DIR / "civil_comments" / "civil_comments_train_sample.csv")
    c_train_hashes = set(c_train["text"].dropna().map(hash_text))
    hashes_by_dataset["civil_comments_train"] = c_train_hashes

    leakage_matrix = {}
    for name_a, set_a in hashes_by_dataset.items():
        leakage_matrix[name_a] = {}
        for name_b, set_b in hashes_by_dataset.items():
            if name_a == name_b:
                continue
            overlap = len(set_a.intersection(set_b))
            if overlap > 0:
                leakage_matrix[name_a][name_b] = overlap
                print(f"  [Overlap Detected] {name_a} vs {name_b}: {overlap:,} duplicate text matches")

    if not any(leakage_matrix.values()):
        print("  --> Zero cross-dataset leakage detected across all acquired splits!")

    return leakage_matrix


# ==============================================================
# Phase 3: Unified Data Normalization & Adapters
# ==============================================================
def normalize_and_save_datasets():
    print("\n" + "=" * 60)
    print("PHASE 3: UNIFIED DATA NORMALIZATION & EXPORT")
    print(f"Target Directory: {PROCESSED_DIR}")
    print("=" * 60)

    # ── Adapter 1: Jigsaw Toxic Comment ───────────────────────────
    print("\n[Normalizing Jigsaw Toxic Comment]...")
    df_j = pd.read_csv(RAW_DIR / "jigsaw_toxic_comment" / "train.csv")
    jigsaw_records = []
    labels_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    for _, row in df_j.iterrows():
        text = str(row["comment_text"]).strip()
        if not text:
            continue
        jigsaw_records.append({
            "text": text,
            "language": "en",
            "source": "JIGSAW_TOXIC_COMMENT",
            "task": "toxicity_six_label",
            "labels": {col: int(row[col]) for col in labels_cols}
        })

    out_j = PROCESSED_DIR / "jigsaw_six_label_train.jsonl"
    with open(out_j, "w", encoding="utf-8") as f:
        for r in jigsaw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  --> Saved {len(jigsaw_records):,} unified records to {out_j.name}")

    # ── Adapter 2: HASOC 2021 Hindi & English ─────────────────────
    print("\n[Normalizing HASOC 2021 Subtask 1]...")
    hasoc_records = []

    # Hindi
    df_h_hi = pd.read_csv(RAW_DIR / "hasoc_2021" / "hindi_2021.csv")
    for _, row in df_h_hi.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        hasoc_records.append({
            "text": text,
            "language": "hi",
            "source": "HASOC_2021_HINDI",
            "task": "offensive_hasoc",
            "labels": {
                "hof": 1 if str(row["task_1"]).strip().upper() == "HOF" else 0,
                "subtask2_category": str(row["task_2"]).strip().upper()
            }
        })

    # English
    df_h_en = pd.read_csv(RAW_DIR / "hasoc_2021" / "english_2021.csv")
    for _, row in df_h_en.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        hasoc_records.append({
            "text": text,
            "language": "en",
            "source": "HASOC_2021_ENGLISH",
            "task": "offensive_hasoc",
            "labels": {
                "hof": 1 if str(row["task_1"]).strip().upper() == "HOF" else 0,
                "subtask2_category": str(row["task_2"]).strip().upper()
            }
        })

    out_h = PROCESSED_DIR / "hasoc_2021_unified.jsonl"
    with open(out_h, "w", encoding="utf-8") as f:
        for r in hasoc_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  --> Saved {len(hasoc_records):,} unified records to {out_h.name}")

    # ── Adapter 3: TRAC-1 Aggression ──────────────────────────────
    print("\n[Normalizing TRAC-1 Aggression]...")
    trac_records = []

    # Hindi
    df_t_hi = pd.read_csv(RAW_DIR / "trac_1" / "agr_hi_train.csv", header=None, names=["id", "text", "label"])
    for _, row in df_t_hi.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        trac_records.append({
            "text": text,
            "language": "hi",
            "source": "TRAC_1_HINDI",
            "task": "aggression_trac",
            "labels": {
                "aggression_level": str(row["label"]).strip().upper(),
                "is_aggressive": 1 if str(row["label"]).strip().upper() in ["OAG", "CAG"] else 0
            }
        })

    # English
    df_t_en = pd.read_csv(RAW_DIR / "trac_1" / "agr_en_train.csv", header=None, names=["id", "text", "label"])
    for _, row in df_t_en.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        trac_records.append({
            "text": text,
            "language": "en",
            "source": "TRAC_1_ENGLISH",
            "task": "aggression_trac",
            "labels": {
                "aggression_level": str(row["label"]).strip().upper(),
                "is_aggressive": 1 if str(row["label"]).strip().upper() in ["OAG", "CAG"] else 0
            }
        })

    out_t = PROCESSED_DIR / "trac_1_unified.jsonl"
    with open(out_t, "w", encoding="utf-8") as f:
        for r in trac_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  --> Saved {len(trac_records):,} unified records to {out_t.name}")

    # ── Adapter 4: Civil Comments Supplementary Sample ───────────
    print("\n[Normalizing Civil Comments Supplementary Sample]...")
    df_c = pd.read_csv(RAW_DIR / "civil_comments" / "civil_comments_train_sample.csv")
    civil_records = []

    for _, row in df_c.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        # Binarize using official 0.5 threshold for multi-label supervision
        civil_records.append({
            "text": text,
            "language": "en",
            "source": "CIVIL_COMMENTS_SAMPLE",
            "task": "toxicity_six_label",
            "labels": {
                "toxic": 1 if float(row.get("toxicity", 0.0)) >= 0.5 else 0,
                "severe_toxic": 1 if float(row.get("severe_toxicity", 0.0)) >= 0.5 else 0,
                "obscene": 1 if float(row.get("obscene", 0.0)) >= 0.5 else 0,
                "threat": 1 if float(row.get("threat", 0.0)) >= 0.5 else 0,
                "insult": 1 if float(row.get("insult", 0.0)) >= 0.5 else 0,
                "identity_hate": 1 if float(row.get("identity_attack", 0.0)) >= 0.5 else 0
            }
        })

    out_c = PROCESSED_DIR / "civil_comments_six_label_sample.jsonl"
    with open(out_c, "w", encoding="utf-8") as f:
        for r in civil_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  --> Saved {len(civil_records):,} unified records to {out_c.name}")


def main():
    raw_audit = audit_raw_datasets()
    leakage = check_leakage_and_duplicates()
    normalize_and_save_datasets()

    # Save full audit & normalization report
    report = {
        "audit_timestamp": "2026-10-27",
        "raw_datasets": raw_audit,
        "cross_dataset_leakage": leakage,
        "processed_artifacts": [
            "data/processed/jigsaw_six_label_train.jsonl",
            "data/processed/hasoc_2021_unified.jsonl",
            "data/processed/trac_1_unified.jsonl",
            "data/processed/civil_comments_six_label_sample.jsonl"
        ]
    }

    report_p = PROCESSED_DIR / "DATASET_AUDIT_AND_NORMALIZATION_REPORT.json"
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("AUDIT & NORMALIZATION COMPLETE!")
    print(f"Detailed Report: {report_p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
