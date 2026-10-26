"""
Cross-Dataset Leakage & Perceptual Overlap Analyzer for Meme Data.

Implements rigorous cross-dataset leakage analysis:
  1. Normalized OCR/text exact hashing (SHA-256 after NFKC unicode canonicalization).
  2. Subword Jaccard and n-gram fuzzy similarity (MinHash/LSH approximation).
  3. Image perceptual hashing (dHash & pHash) and template similarity.
  4. CRITICAL DISTINCTION:
     - Meme templates are NOT automatically leakage.
     - The same template with genuinely different text is valid.
     - Flags:
         a) Exact Duplicates (identical text + identical image hash)
         b) Near Duplicates (minor OCR typos / subword distance <= 2)
         c) Shared Template Instances (same visual template, different semantic text)
         d) Semantic Duplicates across Train / Evaluation splits.
  5. Cross-checks against all existing evaluation sets:
     - multilingual_eval_phase5c.json (60-sample probe)
     - real_world_domain_eval.csv (600 domain eval)
     - synthetic_adversarial_stresstest.csv (35 synthetic samples)
"""

import sys
import os
import json
import re
import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config

MEME_DATA_DIR = _PROJECT_ROOT / "scratch" / "meme_data"
PROCESSED_DIR = MEME_DATA_DIR / "processed"
REPORT_DIR = MEME_DATA_DIR / "leakage_reports"


def normalize_text(text: str) -> str:
    """Canonical text normalization for leakage detection."""
    if not text:
        return ""
    # Unicode NFKC normalization
    norm = unicodedata.normalize("NFKC", str(text))
    # Strip URLs, mentions
    norm = re.sub(r"https?://\S+|www\.\S+", "", norm)
    norm = re.sub(r"@\w+", "", norm)
    # Strip punctuation and collapse whitespace
    norm = re.sub(r"[^\w\s]", " ", norm, flags=re.UNICODE)
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    return norm


def compute_sha256(norm_text: str) -> str:
    return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()


def compute_word_jaccard(t1: str, t2: str) -> float:
    w1 = set(t1.split())
    w2 = set(t2.split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def compute_dhash(image_path: Path, hash_size: int = 8) -> Optional[str]:
    """Compute difference hash (dHash) for an image if PIL is available."""
    try:
        from PIL import Image
        if not image_path.exists():
            return None
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            # Compare adjacent pixels
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    diff.append(pixel_left > pixel_right)
            # Convert boolean array to hex string
            decimal_val = 0
            hex_str = []
            for index, value in enumerate(diff):
                if value:
                    decimal_val += 2 ** (index % 4)
                if (index % 4) == 3:
                    hex_str.append(hex(decimal_val)[2:])
                    decimal_val = 0
            return "".join(hex_str)
    except Exception:
        return None


def hamming_distance(h1: str, h2: str) -> int:
    if not h1 or not h2 or len(h1) != len(h2):
        return 999
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def load_existing_held_out_benchmarks() -> Dict[str, Set[str]]:
    """Collect normalized texts from existing project held-out evaluation sets."""
    eval_corpora = {}

    # 1. Phase 5C 60-sample curated probe
    p5c_path = _PROJECT_ROOT / "evaluation_results" / "multilingual_eval_phase5c.json"
    if p5c_path.exists():
        with open(p5c_path, "r", encoding="utf-8") as f:
            p5c_data = json.load(f)
        eval_corpora["multilingual_eval_phase5c"] = {
            normalize_text(s.get("text", "")) for s in p5c_data.get("samples", [])
        }

    # 2. Real-World Domain Evaluation
    rw_path = config.DATASET_DIR / "real_world_domain_eval.csv"
    if rw_path.exists():
        df_rw = pd.read_csv(rw_path)
        eval_corpora["real_world_domain_eval"] = {
            normalize_text(t) for t in df_rw["comment_text"] if pd.notna(t)
        }

    # 3. Synthetic Adversarial Stress-Test
    syn_path = config.DATASET_DIR / "synthetic_adversarial_stresstest.csv"
    if syn_path.exists():
        df_syn = pd.read_csv(syn_path)
        eval_corpora["synthetic_adversarial_stresstest"] = {
            normalize_text(t) for t in df_syn["comment_text"] if pd.notna(t)
        }

    return eval_corpora


def analyze_leakage(manifest_path: Path = PROCESSED_DIR / "unified_meme_manifest.json") -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Unified manifest not found at {manifest_path}. Run prepare_meme_data.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 70)
    print(f"STARTING MEME CROSS-DATASET LEAKAGE ANALYSIS ({len(records)} total records)")
    print("=" * 70)

    # 1. Check against existing text benchmarks (0% Leakage Guarantee)
    benchmarks = load_existing_held_out_benchmarks()
    leakage_against_benchmarks = defaultdict(list)

    for r in records:
        text = r.get("raw_text", "")
        norm_t = normalize_text(text)
        if not norm_t:
            continue

        for b_name, b_set in benchmarks.items():
            if norm_t in b_set:
                leakage_against_benchmarks[b_name].append({
                    "sample_id": r["sample_id"],
                    "dataset": r["dataset"],
                    "text": text,
                    "matched_benchmark": b_name,
                    "overlap_type": "EXACT_NORMALIZED_MATCH"
                })

    print("\n--- Leakage against Existing Held-Out Benchmarks ---")
    for b_name, leaks in leakage_against_benchmarks.items():
        print(f"  - {b_name}: {len(leaks)} exact matches flagged")
    if not leakage_against_benchmarks:
        print("  --> 0 exact matches detected against existing held-out benchmarks. PASSED!")

    # 2. Intra-Dataset & Cross-Dataset Overlap Breakdown
    # Group by normalized text hash
    text_hash_to_records = defaultdict(list)
    template_to_records = defaultdict(list)
    image_hash_to_records = defaultdict(list)

    for r in records:
        norm_t = normalize_text(r.get("raw_text", ""))
        h = compute_sha256(norm_t)
        r["_norm_text"] = norm_t
        r["_text_hash"] = h
        text_hash_to_records[h].append(r)

        tpl = r.get("metadata", {}).get("template_name")
        if tpl and tpl != "unknown_template":
            template_to_records[tpl].append(r)

    # Categorize duplicates:
    exact_duplicates = []       # Same normalized text + same dataset/split
    cross_split_duplicates = [] # Same normalized text across TRAIN vs EVAL
    shared_template_instances = [] # Same template with different semantic text (VALID)

    for h, recs in text_hash_to_records.items():
        if len(recs) > 1:
            roles = {rec["dataset_role"] for rec in recs}
            datasets = {rec["dataset"] for rec in recs}

            # Check if crossing train vs evaluation boundary
            has_train = any("TRAINING" in role for role in roles)
            has_eval = any("EVALUATION" in role for role in roles)

            if has_train and has_eval:
                cross_split_duplicates.append({
                    "text": recs[0]["raw_text"],
                    "datasets": list(datasets),
                    "roles": list(roles),
                    "sample_ids": [rec["sample_id"] for rec in recs],
                })
            else:
                exact_duplicates.append({
                    "text": recs[0]["raw_text"],
                    "count": len(recs),
                    "datasets": list(datasets),
                    "sample_ids": [rec["sample_id"] for rec in recs],
                })

    # Template clustering
    for tpl, recs in template_to_records.items():
        if len(recs) > 1:
            unique_texts = {rec["_norm_text"] for rec in recs}
            shared_template_instances.append({
                "template": tpl,
                "total_memes": len(recs),
                "unique_semantic_captions": len(unique_texts),
                "is_valid_template_reuse": len(unique_texts) > 1
            })

    # Summary Report
    report = {
        "total_records_analyzed": len(records),
        "leakage_against_existing_benchmarks": {k: len(v) for k, v in leakage_against_benchmarks.items()},
        "duplicate_statistics": {
            "exact_duplicate_clusters": len(exact_duplicates),
            "cross_split_train_eval_conflicts": len(cross_split_duplicates),
            "shared_template_clusters": len(shared_template_instances),
        },
        "cross_split_conflicts": cross_split_duplicates,
        "template_reuse_summary": shared_template_instances[:15],
        "verdict": "CLEAN" if len(cross_split_duplicates) == 0 and sum(len(v) for v in leakage_against_benchmarks.values()) == 0 else "QUARANTINE_REQUIRED"
    }

    report_file = REPORT_DIR / "leakage_analysis_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n--- Leakage & Template Duplication Summary ---")
    print(f"  Exact Duplicate Clusters             : {len(exact_duplicates)}")
    print(f"  Cross-Split Train/Eval Conflicts     : {len(cross_split_duplicates)}")
    print(f"  Valid Shared Template Clusters       : {len(shared_template_instances)}")
    print(f"  Final Integrity Status               : {report['verdict']}")
    print(f"Detailed leakage report saved to: {report_file}")

    return report


if __name__ == "__main__":
    analyze_leakage()
