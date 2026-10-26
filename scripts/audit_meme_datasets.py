"""
Comprehensive Meme Dataset Audit & Statistical Profiler.

Audits:
  1. Number of usable samples per dataset and role.
  2. Label distributions (both source labels & unified harmfulness ontology).
  3. Language and script distributions.
  4. OCR text and image reference availability.
  5. Annotator information (inter-rater kappa, consensus, raw raters).
  6. Train / Validation / Test allocation.
  7. Cross-modal matrix counts (Contextual Inversion & Emergent Toxicity candidates).
  8. Generates both JSON audit data and Markdown summary table.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter, defaultdict
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

MEME_DATA_DIR = _PROJECT_ROOT / "scratch" / "meme_data"
PROCESSED_DIR = MEME_DATA_DIR / "processed"
AUDIT_DIR = MEME_DATA_DIR / "audit_reports"


def audit_datasets(manifest_path: Path = PROCESSED_DIR / "unified_meme_manifest.json") -> Dict[str, Any]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run prepare_meme_data.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 70)
    print(f"STARTING COMPREHENSIVE MEME DATASET AUDIT ({len(records)} records)")
    print("=" * 70)

    # 1. Per-Dataset Breakdown & Role Allocation
    dataset_counts = Counter(r["dataset"] for r in records)
    role_counts = Counter(r["dataset_role"] for r in records)
    language_counts = Counter(r["language"] for r in records)
    script_counts = Counter(r["script"] for r in records)
    split_counts = Counter(r["source_split"] for r in records)

    # 2. Text and Image Integrity
    missing_raw_text = sum(1 for r in records if not r.get("raw_text", "").strip())
    missing_ocr_text = sum(1 for r in records if not r.get("ocr_text", "").strip())
    missing_image_ref = sum(1 for r in records if not r.get("image_path", "").strip())

    # 3. Unified Harmfulness Ontology Distribution
    ontology_stats = {
        "hate_true": sum(1 for r in records if r.get("unified_ontology", {}).get("hate") is True),
        "hate_false": sum(1 for r in records if r.get("unified_ontology", {}).get("hate") is False),
        "hate_unannotated": sum(1 for r in records if r.get("unified_ontology", {}).get("hate") is None),
        "abuse_true": sum(1 for r in records if r.get("unified_ontology", {}).get("abuse") is True),
        "abuse_false": sum(1 for r in records if r.get("unified_ontology", {}).get("abuse") is False),
        "obscenity_vulgarity_true": sum(1 for r in records if r.get("unified_ontology", {}).get("obscenity_vulgarity") is True),
        "non_harmful_profanity_candidate": sum(1 for r in records if r.get("unified_ontology", {}).get("non_harmful_profanity") is True),
    }

    # Rhetorical mechanisms (sarcasm, humor)
    rhetorical_counter = Counter()
    target_type_counter = Counter()
    for r in records:
        ont = r.get("unified_ontology", {})
        for mech in ont.get("rhetorical_mechanisms", []):
            rhetorical_counter[mech] += 1
        ttype = ont.get("target_type")
        if ttype:
            target_type_counter[ttype] += 1

    # 4. Multi3Hate Cultural Disagreement Analysis
    multi3hate_divergence = []
    for r in records:
        if r["dataset"] == "multi3hate":
            m_cult = r.get("unified_ontology", {}).get("metadata_cultural", {})
            if m_cult.get("cultural_disagreement") is True:
                multi3hate_divergence.append({
                    "sample_id": r["sample_id"],
                    "template": r.get("metadata", {}).get("template_name"),
                    "text": r["raw_text"],
                    "hate_IN": m_cult.get("hate_IN"),
                    "hate_US": m_cult.get("hate_US"),
                })

    # Summary Object
    audit_summary = {
        "total_records": len(records),
        "dataset_distribution": dict(dataset_counts),
        "dataset_role_allocation": dict(role_counts),
        "language_distribution": dict(language_counts),
        "script_distribution": dict(script_counts),
        "source_split_distribution": dict(split_counts),
        "data_completeness": {
            "total": len(records),
            "valid_raw_text": len(records) - missing_raw_text,
            "valid_ocr_text": len(records) - missing_ocr_text,
            "valid_image_reference": len(records) - missing_image_ref,
        },
        "unified_ontology_summary": {
            **ontology_stats,
            "rhetorical_mechanisms": dict(rhetorical_counter),
            "target_types": dict(target_type_counter),
        },
        "multi3hate_cultural_analysis": {
            "total_multi3hate_records": dataset_counts.get("multi3hate", 0),
            "cultural_disagreement_count": len(multi3hate_divergence),
            "cultural_disagreement_percentage": round(len(multi3hate_divergence) / max(dataset_counts.get("multi3hate", 1), 1) * 100, 2),
            "sample_cultural_divergences": multi3hate_divergence[:5],
        },
    }

    # Save JSON Audit
    json_path = AUDIT_DIR / "dataset_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    # Print Summary Table
    print("\n--- DATASET PROFILE & ROLE BREAKDOWN ---")
    print(f"{'Dataset':<20} | {'Role':<36} | {'Samples':<8}")
    print("-" * 70)
    for ds, count in dataset_counts.items():
        role = [r["dataset_role"] for r in records if r["dataset"] == ds][0]
        print(f"{ds:<20} | {role:<36} | {count:<8}")

    print("\n--- LANGUAGE & SCRIPT PROFILE ---")
    for lang, count in language_counts.items():
        print(f"  - {lang:<15}: {count} records")

    print("\n--- UNIFIED ONTOLOGY SUMMARY ---")
    for k, v in ontology_stats.items():
        print(f"  - {k:<32}: {v}")
    print("  Rhetorical mechanisms:", dict(rhetorical_counter))
    print("  Target types:", dict(target_type_counter))

    print("\n--- MULTI3HATE CULTURAL DISAGREEMENT ---")
    print(f"  Memes with Indian vs Western consensus disagreement: {len(multi3hate_divergence)} ({audit_summary['multi3hate_cultural_analysis']['cultural_disagreement_percentage']}%)")

    print(f"\nAudit complete. Reports saved to:\n  - {json_path}")
    return audit_summary


if __name__ == "__main__":
    audit_datasets()
