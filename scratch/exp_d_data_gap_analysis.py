import re
import json
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_experiment_d():
    print("=== EXPERIMENT D: TRAINING DATA GAP ANALYSIS ===")
    
    jigsaw_path = _PROJECT_ROOT / "dataset" / "train.csv"
    trac_path = _PROJECT_ROOT / "dataset" / "trac_hinglish_raw.csv"
    
    # 1. Jigsaw Training Data Inspection
    print(f"Loading Jigsaw data from {jigsaw_path}...")
    df_jigsaw = pd.read_csv(jigsaw_path)
    total_jigsaw = len(df_jigsaw)
    print(f"Total Jigsaw rows: {total_jigsaw}")
    
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    jigsaw_label_counts = {lbl: int(df_jigsaw[lbl].sum()) for lbl in labels}
    print("Jigsaw label positive counts:", jigsaw_label_counts)
    
    # Check rare label sample size used in Phase 4B
    pos_mask = (
        (df_jigsaw["severe_toxic"] == 1) |
        (df_jigsaw["threat"] == 1) |
        (df_jigsaw["identity_hate"] == 1)
    )
    df_jpos = df_jigsaw[pos_mask]
    print(f"Jigsaw rare-positives count (|severe_toxic|threat|identity_hate): {len(df_jpos)}")
    
    # Check benign profanity in Jigsaw
    benign_profanity_patterns = [
        r"\bfuck yeah\b", r"\bholy shit\b", r"\bfucking awesome\b", r"\bfucking amazing\b",
        r"\bfucking love\b", r"\bfucking good\b", r"\bfucking great\b", r"\bbadass\b", r"\bkickass\b"
    ]
    praise_metaphor_patterns = [
        r"\bkilled it\b", r"\bsick movie\b", r"\btotally sick\b", r"\bsick beat\b", r"\bdrop dead gorgeous\b"
    ]
    
    jigsaw_texts = df_jigsaw["comment_text"].astype(str).str.lower()
    
    benign_prof_matches = df_jigsaw[jigsaw_texts.str.contains("|".join(benign_profanity_patterns), regex=True, na=False)]
    praise_metaphor_matches = df_jigsaw[jigsaw_texts.str.contains("|".join(praise_metaphor_patterns), regex=True, na=False)]
    
    print(f"Jigsaw rows with benign profanity phrases: {len(benign_prof_matches)}")
    if len(benign_prof_matches) > 0:
        print("  Breakdown of benign profanity rows by toxic label:")
        print(benign_prof_matches["toxic"].value_counts().to_dict())
        print("  Breakdown by severe_toxic label:")
        print(benign_prof_matches["severe_toxic"].value_counts().to_dict())
        print("  Breakdown by obscene label:")
        print(benign_prof_matches["obscene"].value_counts().to_dict())
        
    print(f"Jigsaw rows with praise metaphor phrases: {len(praise_metaphor_matches)}")
    if len(praise_metaphor_matches) > 0:
        print("  Breakdown of praise metaphors by toxic label:")
        print(praise_metaphor_matches["toxic"].value_counts().to_dict())

    # 2. TRAC Dataset Inspection
    print(f"\nLoading TRAC data from {trac_path}...")
    df_trac = pd.read_csv(trac_path)
    print(f"Total TRAC raw rows: {len(df_trac)}")
    print("TRAC columns:", list(df_trac.columns))
    
    # TRAC label column
    target_col = "Aggression-level" if "Aggression-level" in df_trac.columns else ("Sub-task A" if "Sub-task A" in df_trac.columns else df_trac.columns[-1])
    print(f"TRAC target column '{target_col}' value counts:")
    print(df_trac[target_col].value_counts().to_dict())
    
    # Check if TRAC has multi-label annotations for severe_toxic, threat, insult, obscene, identity_hate
    trac_has_multilabel = any(c in df_trac.columns for c in ["severe_toxic", "threat", "insult", "obscene", "identity_hate"])
    print("Does TRAC contain multi-label columns (severe_toxic, threat, insult, etc.)?:", trac_has_multilabel)
    
    # Check Indic praise phrases in TRAC
    indic_praise_patterns = [
        "bawaal", "dhansu", "dhaansu", "धांसू", "बवाल", "gadar", "गदर",
        "dil jeet", "mazza", "maza", "superb", "shandar", "zabardast", "shandaar"
    ]
    text_col = "Text" if "Text" in df_trac.columns else df_trac.columns[0]
    trac_texts = df_trac[text_col].astype(str).str.lower()
    indic_praise_matches = df_trac[trac_texts.str.contains("|".join(indic_praise_patterns), regex=True, na=False)]
    print(f"TRAC rows matching Indic praise/enthusiastic markers: {len(indic_praise_matches)}")
    if len(indic_praise_matches) > 0:
        print(f"  Distribution in TRAC of praise marker rows ({target_col}):")
        print(indic_praise_matches[target_col].value_counts().to_dict())

    # 3. Phase 4B Assembled Training Split Stats
    print("\n--- Phase 4B Training Assembly Protocol Stats ---")
    pos_samples = len(df_jpos)
    neg_samples = 2000
    trac_samples = 2000
    total_train_universe = pos_samples + neg_samples + trac_samples
    print(f"Total assembled pool in Phase 4B: {total_train_universe}")
    print(f"  Jigsaw Positive Over-sample: {pos_samples} ({pos_samples/total_train_universe*100:.1f}%)")
    print(f"  Jigsaw Negative Sample: {neg_samples} ({neg_samples/total_train_universe*100:.1f}%)")
    print(f"  TRAC Indic Sample (toxic-only annotated, 5 heads masked 0.0): {trac_samples} ({trac_samples/total_train_universe*100:.1f}%)")
    
    report = {
        "jigsaw_total_rows": total_jigsaw,
        "jigsaw_label_positives": jigsaw_label_counts,
        "jigsaw_rare_positives_selected": pos_samples,
        "jigsaw_benign_profanity_matches": len(benign_prof_matches),
        "jigsaw_benign_profanity_toxic_ratio": benign_prof_matches["toxic"].value_counts().to_dict() if len(benign_prof_matches) > 0 else {},
        "jigsaw_praise_metaphor_matches": len(praise_metaphor_matches),
        "trac_total_rows": len(df_trac),
        "trac_multi_label_available": trac_has_multilabel,
        "trac_praise_matches": len(indic_praise_matches),
        "phase4b_train_assembly": {
            "jigsaw_positives": pos_samples,
            "jigsaw_negatives": neg_samples,
            "trac_samples": trac_samples,
            "total": total_train_universe,
            "indic_masked_heads_count": 5
        }
    }
    
    out_file = _PROJECT_ROOT / "scratch" / "exp_d_data_gap_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Experiment D completed. Output saved to:", out_file)

if __name__ == "__main__":
    run_experiment_d()
