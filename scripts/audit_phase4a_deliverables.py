"""
Phase 4A Audit Script: Dataset Inventory, Label Mapping, Duplicate Leakage Checks,
Pos-Weight Calculation, and Hardware Benchmark Specification.
"""

import sys
import io
import hashlib
import urllib.request
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import re

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase4AAudit")

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def detect_language(text):
    text_str = str(text)
    devanagari = len(re.findall(r'[\u0900-\u097F]', text_str))
    latin = len(re.findall(r'[a-zA-Z]', text_str))
    
    if devanagari > 3:
        return "hindi_devanagari"
    elif latin > 3:
        hinglish_markers = [
            "hai", "hoon", "bhai", "kya", "mera", "teri", "wale", "karo", "teree",
            "chutiya", "gand", "desh", "modi", "bjp", "log", "madarchod", "behenchod",
            "bhosdike", "harami", "sab", "apne", "raat", "bhi", "bain"
        ]
        words = set(re.findall(r'\w+', text_str.lower()))
        if any(w in words for w in hinglish_markers):
            return "hinglish"
        else:
            return "english"
    else:
        return "other"


def compute_sha256(text):
    clean = re.sub(r'\s+', ' ', str(text).strip().lower())
    return hashlib.sha256(clean.encode('utf-8')).hexdigest()


def audit_datasets():
    logger.info("--- 1. Auditing Jigsaw Dataset ---")
    jigsaw_path = config.DATASET_DIR / "train.csv"
    df_jigsaw = pd.read_csv(jigsaw_path)
    logger.info("Jigsaw Train shape: %s", df_jigsaw.shape)
    
    logger.info("--- 2. Auditing TRAC / HASOC Multilingual Dataset ---")
    url = "https://raw.githubusercontent.com/hmMythreya/hatespeechdetection/main/Hinglish.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        df_trac = pd.read_csv(io.StringIO(resp.read().decode('utf-8', errors='ignore')))
    logger.info("TRAC/HASOC Raw shape: %s", df_trac.shape)
    
    df_trac["language_cat"] = df_trac["Text"].apply(detect_language)
    logger.info("TRAC Language Breakdown:\n%s", df_trac["language_cat"].value_counts())
    
    logger.info("--- 3. Auditing Real-World Evaluation Held-Out Set ---")
    real_eval_path = config.DATASET_DIR / "real_world_domain_eval.csv"
    df_real_eval = pd.read_csv(real_eval_path)
    logger.info("Real-World Domain Eval shape: %s", df_real_eval.shape)
    
    # Check for duplicates & data leakage
    logger.info("--- 4. Data Leakage & Duplicate Audit ---")
    jigsaw_hashes = set(df_jigsaw["comment_text"].apply(compute_sha256))
    trac_hashes = set(df_trac["Text"].apply(compute_sha256))
    eval_hashes = set(df_real_eval["comment_text"].apply(compute_sha256))
    
    leakage_trac_eval = trac_hashes.intersection(eval_hashes)
    leakage_jigsaw_eval = jigsaw_hashes.intersection(eval_hashes)
    leakage_jigsaw_trac = jigsaw_hashes.intersection(trac_hashes)
    
    logger.info("Leakage between TRAC & Real-World Eval set: %d duplicates found.", len(leakage_trac_eval))
    logger.info("Leakage between Jigsaw & Real-World Eval set: %d duplicates found.", len(leakage_jigsaw_eval))
    logger.info("Leakage between Jigsaw & TRAC: %d duplicates found.", len(leakage_jigsaw_trac))
    
    # Calculate pos_weight for BCEWithLogitsLoss strictly on Jigsaw Training data
    logger.info("--- 5. Pos_Weight Calculation (Calculated on Jigsaw Train Data Only) ---")
    pos_weights = {}
    n_samples = len(df_jigsaw)
    for col in LABEL_COLS:
        pos_count = df_jigsaw[col].sum()
        neg_count = n_samples - pos_count
        weight = neg_count / max(pos_count, 1)
        pos_weights[col] = round(float(weight), 4)
        logger.info("  Label '%s': Pos Count = %d / %d (pos_weight = %.4f)", col, pos_count, n_samples, weight)
        
    audit_summary = {
        "jigsaw_samples": len(df_jigsaw),
        "trac_total_samples": len(df_trac),
        "trac_hinglish_samples": int((df_trac["language_cat"] == "hinglish").sum()),
        "trac_hindi_devanagari_samples": int((df_trac["language_cat"] == "hindi_devanagari").sum()),
        "trac_english_samples": int((df_trac["language_cat"] == "english").sum()),
        "real_world_eval_samples": len(df_real_eval),
        "leakage_trac_eval": len(leakage_trac_eval),
        "leakage_jigsaw_eval": len(leakage_jigsaw_eval),
        "leakage_jigsaw_trac": len(leakage_jigsaw_trac),
        "pos_weights_calculated": pos_weights
    }
    
    out_file = config.PROJECT_ROOT / "evaluation_results" / "phase4a_audit_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
        
    logger.info("Audit Summary saved to %s", out_file)


if __name__ == "__main__":
    audit_datasets()
