"""
Script to prepare:
  1. Real-World Domain Evaluation Dataset (dataset/real_world_domain_eval.csv)
  2. Synthetic Adversarial Stress-Test Dataset (dataset/synthetic_adversarial_stresstest.csv)
"""

import sys
import io
import json
import logging
import urllib.request
from pathlib import Path
import pandas as pd
import re

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PrepareDomainData")

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
            "bhosdike", "harami", "sab", "apne", "wale", "wale", "raat", "bhi", "bain"
        ]
        words = set(re.findall(r'\w+', text_str.lower()))
        if any(w in words for w in hinglish_markers):
            return "hinglish"
        else:
            return "english"
    else:
        return "other"


def fetch_and_build_real_world_dataset():
    logger.info("Fetching real-world social media dataset from public repository...")
    url = "https://raw.githubusercontent.com/hmMythreya/hatespeechdetection/main/Hinglish.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as resp:
        df = pd.read_csv(io.StringIO(resp.read().decode('utf-8', errors='ignore')))

    logger.info("Loaded %d raw real-world social media comments.", len(df))
    
    # Categorize language/script
    df["language_cat"] = df["Text"].apply(detect_language)
    
    # Remove 'other'
    df = df[df["language_cat"].isin(["english", "hinglish", "hindi_devanagari"])].copy()
    
    # Map aggression level to multi-label toxicity targets
    # Aggression-level: 0 (non-aggressive/clean), 1 (aggressive/toxic)
    # We also inspect keywords to assign severe_toxic, obscene, threat, insult, identity_hate accurately
    
    records = []
    
    # Select high-quality balanced samples per language category
    grouped = df.groupby("language_cat")
    
    for lang, group in grouped:
        # Take a balanced sample of toxic and non-toxic
        sample_group = pd.concat([
            group[group["Aggression-level"] == 1].head(100),
            group[group["Aggression-level"] == 0].head(100)
        ])
        
        for _, row in sample_group.iterrows():
            text = str(row["Text"]).strip()
            agg = int(row["Aggression-level"])
            
            text_lower = text.lower()
            
            # Label heuristics based on text content & aggression label
            is_toxic = 1 if agg == 1 else 0
            
            obscene_words = ["chutiya", "madarchod", "behenchod", "bhosdike", "gand", "fuck", "bitch", "loda", "lode", "choot"]
            is_obscene = 1 if (is_toxic and any(w in text_lower for w in obscene_words)) else 0
            
            insult_words = ["idiot", "chutiya", "gandua", "loser", "dog", "kutta", "gatter", "कीड़े", "कीड़ा", "गटर"]
            is_insult = 1 if (is_toxic and any(w in text_lower for w in insult_words)) else 0
            
            threat_words = ["kill", "maar", "maroon", "dekh lunga", "sperm", "jaan se", "throat", "goli"]
            is_threat = 1 if (is_toxic and any(w in text_lower for w in threat_words)) else 0
            
            identity_words = ["porki", "mulla", "bjp", "upa", "musalman", "hindu", "sikh", "koti", "chamar", "mandir", "masjid"]
            is_identity = 1 if (is_toxic and any(w in text_lower for w in identity_words)) else 0
            
            is_severe = 1 if (is_toxic and (is_obscene + is_threat + is_identity + is_insult >= 2)) else 0
            
            records.append({
                "comment_text": text,
                "language_cat": lang,
                "toxic": is_toxic,
                "severe_toxic": is_severe,
                "obscene": is_obscene,
                "threat": is_threat,
                "insult": is_insult,
                "identity_hate": is_identity,
            })
            
    real_df = pd.DataFrame(records)
    out_path = config.DATASET_DIR / "real_world_domain_eval.csv"
    real_df.to_csv(out_path, index=False)
    logger.info("Saved Real-World Evaluation Dataset to %s (Total: %d samples)", out_path, len(real_df))
    logger.info("Category breakdown:\n%s", real_df["language_cat"].value_counts())
    return real_df


def build_synthetic_adversarial_dataset():
    logger.info("Building Synthetic Adversarial Stress-Test Dataset...")
    
    synthetic_samples = [
        # Adversarial / Leetspeak / Obfuscated Slurs (Toxic)
        ("u r a f**king 1diot and a total l0ser", 1, 0, 1, 0, 1, 0),
        ("stfu u b.c madarch0d go away", 1, 1, 1, 0, 1, 0),
        ("i will k!ll u and ur family u @sshole", 1, 1, 1, 1, 1, 0),
        ("screw u b!tch stay out of this", 1, 0, 1, 0, 1, 0),
        ("b.h.o.s.d.i.k.e chup kar lodu", 1, 1, 1, 0, 1, 0),
        ("heeeate all u dirty m*llahs go back", 1, 1, 0, 0, 1, 1),
        ("u r a g@ndu bastard", 1, 0, 1, 0, 1, 0),
        ("goli mar dunga b.c. tereko", 1, 1, 1, 1, 1, 0),
        ("m*therf*cker shut ur mouth", 1, 1, 1, 0, 1, 0),
        ("you stupid k.u.t.t.a get out", 1, 0, 0, 0, 1, 0),
        
        # Char Deduplication & Repeated letters (Toxic)
        ("chuuuuutiya ho tum bilkul", 1, 0, 1, 0, 1, 0),
        ("baaaaaastard I will destr0y u", 1, 0, 1, 1, 1, 0),
        ("fuuuuuuck off u piece of trash", 1, 0, 1, 0, 1, 0),
        ("idiiiiiot don't comment here", 1, 0, 0, 0, 1, 0),
        ("bhosdiiiiwale chup reh", 1, 0, 1, 0, 1, 0),
        
        # Implicit / Evasion Stress Tests (Toxic)
        ("go back to your country you don't belong here", 1, 0, 0, 0, 1, 1),
        ("people of your religion are all terrorists", 1, 1, 0, 0, 1, 1),
        ("your kind should be wiped out from this earth", 1, 1, 0, 1, 1, 1),
        ("tere jaiso ko sadak pe peetna chahiye", 1, 1, 0, 1, 1, 0),
        ("sabhi mulla ko desh se nikalo", 1, 1, 0, 0, 1, 1),
        
        # Non-Toxic Stress Tests (Clean Controls with Trigger Words)
        ("I love the movie Bastards of War it is amazing", 0, 0, 0, 0, 0, 0),
        ("The word bitch means a female dog in zoology", 0, 0, 0, 0, 0, 0),
        ("Kill the process using taskkill command on windows", 0, 0, 0, 0, 0, 0),
        ("Bhai kya chal raha hai aaj kal sab badhiya", 0, 0, 0, 0, 0, 0),
        ("Modi ji and Rahul Gandhi both gave speeches today", 0, 0, 0, 0, 0, 0),
        ("Cricket match mein India ne bahut accha khela", 0, 0, 0, 0, 0, 0),
        ("Good morning everyone hope you have a great day", 0, 0, 0, 0, 0, 0),
        ("This dish tastes so good I really love Indian food", 0, 0, 0, 0, 0, 0),
        ("Aapka bahut bahut dhanyawad ji", 0, 0, 0, 0, 0, 0),
        ("Thank you for sharing this video helpful information", 0, 0, 0, 0, 0, 0),
        
        # Hinglish Adversarial
        ("teri m@a ki choot b.c", 1, 1, 1, 0, 1, 0),
        ("tatti content hai yr ye toh", 1, 0, 1, 0, 1, 0),
        ("nalla creator h ye ek number ka", 1, 0, 0, 0, 1, 0),
        ("abey saale chup chap baith", 1, 0, 0, 0, 1, 0),
        ("laude lag gaye bhai aaj toh exam me", 0, 0, 1, 0, 0, 0), # Conversational profanity (non-hate)
    ]
    
    rows = []
    for text, tox, sev, obs, thr, ins, idh in synthetic_samples:
        rows.append({
            "comment_text": text,
            "language_cat": "slang_adversarial",
            "toxic": tox,
            "severe_toxic": sev,
            "obscene": obs,
            "threat": thr,
            "insult": ins,
            "identity_hate": idh,
        })
        
    syn_df = pd.DataFrame(rows)
    out_path = config.DATASET_DIR / "synthetic_adversarial_stresstest.csv"
    syn_df.to_csv(out_path, index=False)
    logger.info("Saved Synthetic Adversarial Stress-Test Dataset to %s (Total: %d samples)", out_path, len(syn_df))
    return syn_df


def main():
    fetch_and_build_real_world_dataset()
    build_synthetic_adversarial_dataset()
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
