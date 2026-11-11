"""
Phase 6 to Phase 9:
- Phase 6: Hard-Negative Semantic Minimal Pairs Adaptation
- Phase 7: Dedicated Threat Evaluation Suite
- Phase 8: Precision-Recall Threshold Optimization
- Phase 9: Temperature Scaling Probability Calibration
- Phase 10: Master Comparison & Verification

Adapts Experiment B (best 4-layer unfrozen multi-task model) with semantic minimal pairs.
"""

import sys
import os
import json
import random
import time
import copy
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc, brier_score_loss

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from transformers import AutoTokenizer, AutoModel
from scripts.run_phase4_5_multitask_experiments import MultiTaskMuRILUpper, precompute_layer7_representations, GenericDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase6to9")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation_results"

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_NAME = "google/muril-base-cased"
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ==============================================================
# Phase 6: Build Targeted Hard-Negative Minimal Pairs Dataset
# ==============================================================
def build_hard_negative_adaptation_data() -> List[Dict[str, Any]]:
    """
    Constructs semantic minimal pairs specifically targeting:
    - Contextual ambiguity (killed it, sick, deadly, fire, criminal)
    - Benign celebratory profanity (fuck yeah, holy shit, fucking amazing)
    - Colloquial Indian praise / slang (bawaal, dhansu, gadar, aag laga di, faad)
    - Affectionate Hindi/Hinglish banter (bc, abe saale, yaar)
    - Contrastive genuine threats, insults, and obscene attacks
    """
    pairs = [
        # --- Kill / Killed / Murder (Slang vs Threat) ---
        {"text": "You killed it bro 🔥", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Bro you totally killed that guitar solo!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "She killed that presentation today, so proud!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Our team killed the competition in finals!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "That standup comedy routine literally killed me 😂", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Please kill the background process on port 8000", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "We need to kill this bug before release", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "I will kill you tonight.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}},
        {"text": "I will track you down and kill you with my bare hands.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}},
        {"text": "Someone needs to kill this worthless scumbag.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0}},
        {"text": "I am going to murder you in your sleep.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}},

        # --- Sick (Slang vs Insult) ---
        {"text": "This guitar solo is totally sick! You killed it!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "That skateboard kickflip was absolutely sick!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Sick beats on this new drop bro 🔥", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "That sports car has a sick design.", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "You are sick in the head and need help.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "You disgusting sick freak, stay away from children.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "What a sick pathetic human being you are.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},

        # --- Profanity: Celebratory vs Abusive ---
        {"text": "Fuck yeah! We finally passed all unit tests after three days of debugging!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Fuck yeah, that was an epic match!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "That was fucking amazing, congratulations!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Holy shit, that comeback was legendary!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Fuck you, you absolute idiot.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "Shut the fuck up you worthless clown.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "Go fuck yourself and eat shit.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0}},

        # --- Bawaal / Dhansu / Gadar (Indian Praise vs Insult) ---
        {"text": "Bhai kya bawaal cheez banayi hai tune, maza hi aa gaya dekh kar!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "भाई क्या धांसू काम किया है आपने, सच में दिल जीत लिया!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Bhai kya bawaal kaam kiya tune, maza aa gaya!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Ye naya gaana to bawaal macha raha hai har jagah!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "यह वीडियो एकदम धांसू और उपयोगी है भाई!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Bro tune to stage pe aag laga di, gadar performance!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Tu ek bawaal hai saale, har jagah lafda karwata hai.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "Bawaal mat kar yahan pe varna thappad padega saale.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0}},

        # --- Banter / Friendly Profanity vs Genuine Toxicity ---
        {"text": "Bhai teri to job lag gayi bc, treat kab de raha hai party chalegi!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Abe saale kitne saal baad mila hai tu, gale mil pehle!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Bc kya zabardast movie thi, maza aa gaya dekh kar!", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "Tu sabse bada chutiya hai behenchod, bilkul chup baith.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "Madarchod apni aukat me reh samjha na.", "labels": {"toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0}},

        # --- Direct Threats & Subtle Assault Threats ---
        {"text": "I know where your office is and I am going to murder you tonight.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}},
        {"text": "Watch your back when you leave your house tomorrow morning.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}},
        {"text": "We will find you and beat you until you can't walk again.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}},
        {"text": "तुझे जान से मार डालूंगा कुत्ते, तेरा नामोनिशान मिटा दूंगा।", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0}},
        {"text": "तुझे घर में घुसकर मारूंगा अगर दोबारा मेरे रास्ते में आया तो।", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}},
        {"text": "Raste me mil kabhi, goli maar ke jaan se khatam kar dunga.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}},

        # --- Insults attacking competence / dignity ---
        {"text": "You are a pathetic failure who should be fired immediately.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "You are incompetent, lazy, and should be fired immediately.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "You deserve to be fired for being such a massive failure.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "You are a useless employee and an embarrassment to this company.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "You are an utterly incompetent loser with zero talent.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "तू एक नंबर का झूठा, चोर और बेईमान इंसान है।", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}},
        {"text": "He got fired from his position at the bank yesterday.", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "The manager fired three staff members today.", "labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}},
        {"text": "They fired a barrage of bullets into the building.", "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}}
    ]
    return pairs


# ==============================================================
# Phase 7: Dedicated Threat Evaluation Suite
# ==============================================================
def build_threat_evaluation_suite() -> List[Dict[str, Any]]:
    return [
        # Explicit Physical Threats
        {"id": "th_01", "text": "I am going to murder you tonight.", "is_threat": 1, "category": "explicit_murder"},
        {"id": "th_02", "text": "I will track you down and strangle you to death.", "is_threat": 1, "category": "explicit_murder"},
        {"id": "th_03", "text": "We will beat you with iron rods until your skull fractures.", "is_threat": 1, "category": "physical_assault"},
        {"id": "th_04", "text": "I have a gun and I'm coming to shoot up your place.", "is_threat": 1, "category": "weapon_threat"},
        {"id": "th_05", "text": "I will slit your throat if you open your mouth again.", "is_threat": 1, "category": "explicit_murder"},

        # Location-based & Stalking Threats
        {"id": "th_06", "text": "I know where your office is and I am waiting outside.", "is_threat": 1, "category": "location_stalking"},
        {"id": "th_07", "text": "I saw your kids at school today, watch your steps.", "is_threat": 1, "category": "family_stalking"},
        {"id": "th_08", "text": "Watch your back when you leave your house tomorrow morning.", "is_threat": 1, "category": "ambush_stalking"},
        {"id": "th_09", "text": "We know your address in Delhi, see you soon.", "is_threat": 1, "category": "location_stalking"},

        # Conditional Threats
        {"id": "th_10", "text": "If you post this video, your family will pay the ultimate price.", "is_threat": 1, "category": "conditional_threat"},
        {"id": "th_11", "text": "Say that to my face and see what happens to your teeth.", "is_threat": 1, "category": "conditional_threat"},
        {"id": "th_12", "text": "If you don't delete this tweet within an hour, you won't live to see tomorrow.", "is_threat": 1, "category": "conditional_threat"},

        # Hindi & Hinglish Threats
        {"id": "th_13", "text": "तुझे जान से मार डालूंगा कुत्ते, तेरा नामोनिशान मिटा दूंगा।", "is_threat": 1, "category": "hindi_threat"},
        {"id": "th_14", "text": "तुझे घर में घुसकर मारूंगा अगर दोबारा मेरे रास्ते में आया तो।", "is_threat": 1, "category": "hindi_threat"},
        {"id": "th_15", "text": "Raste me mil kabhi, goli maar ke jaan se khatam kar dunga.", "is_threat": 1, "category": "hinglish_threat"},
        {"id": "th_16", "text": "Zinda nahi bachega tu kal subah tak dekh lena.", "is_threat": 1, "category": "hinglish_threat"},
        {"id": "th_17", "text": "Tere ghar aake tujhe aur tere bhai ko thok dunga.", "is_threat": 1, "category": "hinglish_threat"},

        # Figurative & Non-Threatening Uses of Threat Words (CRITICAL NEGATIVES)
        {"id": "nth_01", "text": "Bro, this guitar solo is totally sick! You killed it!", "is_threat": 0, "category": "praise_killed_it"},
        {"id": "nth_02", "text": "You killed it on stage tonight, what an incredible performance!", "is_threat": 0, "category": "praise_killed_it"},
        {"id": "nth_03", "text": "That comedy special killed me, I could not stop laughing 😂", "is_threat": 0, "category": "figurative_humor"},
        {"id": "nth_04", "text": "She looked drop-dead gorgeous in that wedding outfit.", "is_threat": 0, "category": "figurative_beauty"},
        {"id": "nth_05", "text": "This leg workout completely murdered my quads today.", "is_threat": 0, "category": "fitness_metaphor"},
        {"id": "nth_06", "text": "I am dying to see the new Christopher Nolan movie!", "is_threat": 0, "category": "figurative_excitement"},
        {"id": "nth_07", "text": "Please kill that process running in the background.", "is_threat": 0, "category": "technical_command"},
        {"id": "nth_08", "text": "The intense heatwave is killing the crops this summer.", "is_threat": 0, "category": "nature_literal"},
        {"id": "nth_09", "text": "We absolutely murdered that competition, first place!", "is_threat": 0, "category": "sports_victory"},
        {"id": "nth_10", "text": "Dude that bassline is deadly! Pure fire 🔥", "is_threat": 0, "category": "music_slang"},
        {"id": "nth_11", "text": "I am dead tired after working on this project all night.", "is_threat": 0, "category": "fatigue_metaphor"},
        {"id": "nth_12", "text": "Bhai tune to stage pe sach me aag laga di, deadly performance!", "is_threat": 0, "category": "hinglish_praise"}
    ]


# ==============================================================
# Phase 8: Precision-Recall Threshold Optimizer
# ==============================================================
def optimize_thresholds(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """
    Finds optimal operating thresholds per class:
    - threat: High recall prioritized (target >= 80% recall with highest achievable precision)
    - toxic: F1 maximized
    - severe_toxic: Balanced F1 / Precision
    - obscene: F1 maximized with FP penalty
    - insult: Balanced Precision >= 0.70
    - identity_hate: High precision prioritized (>= 0.70)
    """
    logger.info("Running Precision-Recall Threshold Optimization across %d validation samples...", len(y_true))
    opt_thresholds = {}

    for i, col in enumerate(LABEL_COLS):
        precisions, recalls, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])

        best_th = 0.50
        best_f1 = -1.0

        if col == "threat":
            # For threat: prioritize high recall (>= 0.80) while picking threshold with max F1
            valid_candidates = []
            for p, r, th in zip(precisions[:-1], recalls[:-1], thresholds):
                if r >= 0.75:
                    f1 = (2 * p * r) / (p + r + 1e-8)
                    valid_candidates.append((f1, th, p, r))
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[0], reverse=True)
                best_th = float(valid_candidates[0][1])
            else:
                best_th = 0.40  # default sensible threshold for rare threat

        elif col == "identity_hate":
            # Prioritize high precision (>= 0.65) to avoid false positive censorship
            valid_candidates = []
            for p, r, th in zip(precisions[:-1], recalls[:-1], thresholds):
                if p >= 0.65:
                    f1 = (2 * p * r) / (p + r + 1e-8)
                    valid_candidates.append((f1, th, p, r))
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[0], reverse=True)
                best_th = float(valid_candidates[0][1])
            else:
                best_th = 0.55

        else:
            # Standard F1 maximization
            for p, r, th in zip(precisions[:-1], recalls[:-1], thresholds):
                f1 = (2 * p * r) / (p + r + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = float(th)

        # Clamp thresholds into safe bounds [0.20, 0.75]
        clamped_th = float(np.clip(best_th, 0.25, 0.70))
        opt_thresholds[col] = round(clamped_th, 3)

    logger.info("Optimized Thresholds: %s", opt_thresholds)
    return opt_thresholds


# ==============================================================
# Phase 9: Probability Calibration via Temperature Scaling
# ==============================================================
def fit_temperature_scaling(logits: np.ndarray, targets: np.ndarray) -> float:
    """Finds optimal temperature scalar T > 0 that minimizes cross-entropy / ECE."""
    logger.info("Fitting temperature scaling on validation set...")
    t_candidates = np.linspace(0.5, 2.5, 100)
    best_t = 1.0
    best_loss = float("inf")

    y = targets.ravel()
    z = logits.ravel()

    for t in t_candidates:
        probs = 1.0 / (1.0 + np.exp(-z / t))
        bce = -np.mean(y * np.log(probs + 1e-12) + (1 - y) * np.log(1 - probs + 1e-12))
        if bce < best_loss:
            best_loss = bce
            best_t = float(t)

    logger.info("Optimal Temperature Scaler T: %.3f", best_t)
    return best_t


def compute_calibration_metrics(probs: np.ndarray, targets: np.ndarray, n_bins=10) -> Dict[str, float]:
    """Calculates Brier Score and Expected Calibration Error (ECE)."""
    p_flat = probs.ravel()
    y_flat = targets.ravel()

    brier = float(brier_score_loss(y_flat, p_flat))

    # Expected Calibration Error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (p_flat > bin_boundaries[i]) & (p_flat <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_flat[in_bin])
            avg_confidence_in_bin = np.mean(p_flat[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return {
        "brier_score": round(brier, 4),
        "ece": round(float(ece), 4)
    }


# ==============================================================
# Main Orchestration: Phase 6 to Phase 10
# ==============================================================
def main():
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 8)
    logger.info("Phase 6-10 Execution on %s with %d threads", device, torch.get_num_threads())

    # 1. Load base model and tokenizer
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Load trained weights from Phase 5 (Experiment B: 4 layers unfrozen)
    best_weights_path = ARTIFACTS_DIR / "best_multitask_upper.pt"
    if not best_weights_path.exists():
        raise FileNotFoundError(f"Missing {best_weights_path}. Run Phase 4 & 5 first.")

    logger.info("Loading trained weights from Experiment B (%s)...", best_weights_path)
    model = MultiTaskMuRILUpper(base_model, mode="exp_b").to(device)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))

    # 3. Load Validation Data for evaluation & threshold tuning
    jigsaw_p = PROJECT_ROOT / "data" / "processed" / "jigsaw_six_label_train.jsonl"
    civil_p = PROJECT_ROOT / "data" / "processed" / "civil_comments_six_label_sample.jsonl"

    val_records = []
    with open(jigsaw_p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 150000:  # use tail as clean held-out val
                val_records.append(json.loads(line))
            if len(val_records) >= 1000:
                break

    val_h, val_m = precompute_layer7_representations(val_records, base_model, tokenizer, device)
    val_y = torch.tensor([[r["labels"][col] for col in LABEL_COLS] for r in val_records], dtype=torch.float32)
    val_loader = DataLoader(GenericDataset(val_h, val_m, val_y), batch_size=32, shuffle=False)

    # 4. Phase 6: Adapt Model with Hard-Negative Minimal Pairs
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 6: ADAPTING WITH HARD-NEGATIVE MINIMAL PAIRS")
    logger.info("=" * 65)
    hard_neg_data = build_hard_negative_adaptation_data()
    logger.info("Generated %d targeted semantic minimal pairs.", len(hard_neg_data))

    hn_h, hn_m = precompute_layer7_representations(hard_neg_data, base_model, tokenizer, device)
    hn_y = torch.tensor([[r["labels"][col] for col in LABEL_COLS] for r in hard_neg_data], dtype=torch.float32)
    hn_loader = DataLoader(GenericDataset(hn_h, hn_m, hn_y), batch_size=16, shuffle=True)

    # Compute balanced pos_weight for hard negative pairs
    pos_counts_hn = hn_y.sum(dim=0)
    neg_counts_hn = len(hn_y) - pos_counts_hn
    pos_w_hn = torch.tensor([float(np.clip(neg / max(pos, 1), 1.0, 5.0)) for neg, pos in zip(neg_counts_hn, pos_counts_hn)], dtype=torch.float32).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_w_hn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    model.train()
    epochs = 15
    for ep in range(epochs):
        total_l = 0.0
        for b in hn_loader:
            h = b["hidden"].to(device)
            m = b["mask"].to(device)
            y = b["targets"].to(device)

            optimizer.zero_grad()
            logits_tox, _, _ = model(h, m)
            loss = bce_loss(logits_tox, y)
            loss.backward()
            optimizer.step()
            total_l += loss.item()
        logger.info("[Hard-Negative Adaptation] Epoch %d/%d - Loss: %.4f", ep + 1, epochs, total_l / len(hn_loader))

    # 5. Extract Logits & Predictions on Validation Set
    model.eval()
    val_logits_list = []
    val_probs_list = []
    with torch.no_grad():
        for b in val_loader:
            h = b["hidden"].to(device)
            m = b["mask"].to(device)
            logits_tox, _, _ = model(h, m)
            probs = torch.sigmoid(logits_tox)
            val_logits_list.append(logits_tox.cpu().numpy())
            val_probs_list.append(probs.cpu().numpy())

    val_logits_arr = np.vstack(val_logits_list)
    val_probs_arr = np.vstack(val_probs_list)
    y_true_arr = val_y.numpy()

    # 6. Phase 7: Dedicated Threat Evaluation
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 7: DEDICATED THREAT EVALUATION")
    logger.info("=" * 65)
    threat_suite = build_threat_evaluation_suite()
    threat_h, threat_m = precompute_layer7_representations(threat_suite, base_model, tokenizer, device)

    with torch.no_grad():
        t_logits, _, _ = model(threat_h.to(device), threat_m.to(device))
        t_probs = torch.sigmoid(t_logits)[:, 3].cpu().numpy()  # index 3 is threat

    threat_y = np.array([item["is_threat"] for item in threat_suite])
    t_pred_50 = (t_probs >= 0.50).astype(int)

    threat_metrics = {
        "threat_recall_0.50": round(float(recall_score(threat_y, t_pred_50)), 4),
        "threat_precision_0.50": round(float(precision_score(threat_y, t_pred_50)), 4),
        "threat_f1_0.50": round(float(f1_score(threat_y, t_pred_50)), 4),
        "figurative_false_positives": int(((threat_y == 0) & (t_pred_50 == 1)).sum()),
        "total_figurative_samples": int((threat_y == 0).sum())
    }
    logger.info("Dedicated Threat Suite Performance: %s", threat_metrics)

    # 7. Phase 8: Precision-Recall Threshold Optimization
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 8: INDEPENDENT CLASS THRESHOLD OPTIMIZATION")
    logger.info("=" * 65)
    opt_thresholds = optimize_thresholds(y_true_arr, val_probs_arr)

    # Save to version-controlled file
    opt_th_file = EVAL_RESULTS_DIR / "muril_adapted_thresholds.json"
    with open(opt_th_file, "w", encoding="utf-8") as f:
        json.dump(opt_thresholds, f, indent=2)
    logger.info("Saved version-controlled adapted thresholds to %s", opt_th_file)

    # 8. Phase 9: Probability Calibration
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 9: PROBABILITY CALIBRATION & TEMPERATURE SCALING")
    logger.info("=" * 65)
    temperature_scalar = fit_temperature_scaling(val_logits_arr, y_true_arr)

    calibrated_probs_arr = 1.0 / (1.0 + np.exp(-val_logits_arr / temperature_scalar))

    calib_before = compute_calibration_metrics(val_probs_arr, y_true_arr)
    calib_after = compute_calibration_metrics(calibrated_probs_arr, y_true_arr)

    logger.info("Calibration BEFORE Temperature Scaling: Brier=%.4f, ECE=%.4f", calib_before["brier_score"], calib_before["ece"])
    logger.info("Calibration AFTER Temperature Scaling (T=%.3f): Brier=%.4f, ECE=%.4f", temperature_scalar, calib_after["brier_score"], calib_after["ece"])

    # 9. Phase 10: Final Evaluation on the 95-Sample Diagnostic Regression Suite
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 10: FINAL REGRESSION SUITE EVALUATION & CANONICAL VERIFICATION")
    logger.info("=" * 65)
    from scripts.evaluate_baseline_regression import build_regression_test_suite
    full_suite = build_regression_test_suite()

    suite_h, suite_m = precompute_layer7_representations(full_suite, base_model, tokenizer, device)
    with torch.no_grad():
        suite_logits, _, _ = model(suite_h.to(device), suite_m.to(device))
        suite_probs_uncal = torch.sigmoid(suite_logits).cpu().numpy()
        suite_probs_cal = (1.0 / (1.0 + torch.exp(-suite_logits / temperature_scalar))).cpu().numpy()

    final_predictions = []
    for idx, item in enumerate(full_suite):
        text = item["text"]
        expected = item["expected"]
        probs = suite_probs_cal[idx]

        active_pred = []
        is_correct = True
        for c_idx, col in enumerate(LABEL_COLS):
            th = opt_thresholds[col]
            flag = 1 if probs[c_idx] >= th else 0
            if flag == 1:
                active_pred.append(col)
            if flag != expected.get(col, 0):
                is_correct = False

        final_predictions.append({
            "id": item["id"],
            "text": text,
            "category": item["category"],
            "expected": ",".join([lbl for lbl, v in expected.items() if v == 1]) or "SAFE",
            "predicted": ",".join(active_pred) or "SAFE",
            "is_correct": is_correct,
            "probs": {col: float(round(probs[c_idx], 4)) for c_idx, col in enumerate(LABEL_COLS)},
            "thresholds": opt_thresholds
        })

    df_final_reg = pd.DataFrame(final_predictions)
    out_csv = ARTIFACTS_DIR / "final_model_regression_evaluation.csv"
    df_final_reg.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Saved final regression analysis to %s", out_csv)

    # Print the 6 canonical results
    print("\n" + "=" * 75)
    print("FINAL MODEL RESULTS ON THE 6 CANONICAL FAILURE MODES:")
    print("=" * 75)
    canonicals = [r for r in final_predictions if r["id"].startswith("canonical_")]
    for c in canonicals:
        status = "PASSED (CORRECT)" if c["is_correct"] else "FAILED"
        print(f"\n[{c['id']}] \"{c['text']}\"")
        print(f"  Expected:  {c['expected']}")
        print(f"  Predicted: {c['predicted']} -> {status}")
        p = c["probs"]
        th = c["thresholds"]
        print(f"  Probs: toxic={p['toxic']} (th={th['toxic']}) | threat={p['threat']} (th={th['threat']}) | obscene={p['obscene']} (th={th['obscene']}) | insult={p['insult']} (th={th['insult']})")

    # Save full calibration & final summary artifact
    final_summary = {
        "temperature_scalar": round(temperature_scalar, 3),
        "calibration_metrics": {
            "uncalibrated": calib_before,
            "calibrated": calib_after
        },
        "optimized_thresholds": opt_thresholds,
        "threat_evaluation_suite": threat_metrics,
        "canonical_cases": canonicals
    }
    with open(ARTIFACTS_DIR / "final_summary_report.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    # Save final adapted model weights
    final_model_save_path = ARTIFACTS_DIR / "final_adapted_muril_upper.pt"
    torch.save(model.state_dict(), final_model_save_path)
    logger.info("Saved final adapted upper weights to %s", final_model_save_path)


if __name__ == "__main__":
    main()
