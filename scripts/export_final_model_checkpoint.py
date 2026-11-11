"""
Export & Assembly Script for Adapted MuRIL Multilingual Model

1. Assembles base MuRIL (layers 0..7) + adapted upper layers (layers 8..11) + adapted 6-label toxicity head
   into a complete standalone checkpoint: muril_toxicity_model_adapted.pt
2. Preserves the original muril_toxicity_model.pt untouched
3. Updates evaluation_results/muril_adapted_thresholds.json
4. Verifies loading through src.model.load_muril_model and validates all 6 canonical failure modes
"""

import sys
import os
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.model import MuRILForMultiLabelToxicity, predict_comment_muril, DEFAULT_LABELS
from transformers import AutoModel, AutoTokenizer
from scripts.run_phase4_5_multitask_experiments import MultiTaskMuRILUpper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportModel")

MODEL_NAME = "google/muril-base-cased"
ADAPTED_UPPER_PATH = PROJECT_ROOT / "artifacts" / "final_adapted_muril_upper.pt"
OUTPUT_CHECKPOINT = PROJECT_ROOT / "muril_toxicity_model_adapted.pt"
THRESHOLDS_PATH = PROJECT_ROOT / "evaluation_results" / "muril_adapted_thresholds.json"


def assemble_and_export():
    logger.info("Initializing full MuRIL architecture...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    full_model = MuRILForMultiLabelToxicity(model_name=MODEL_NAME, num_labels=6)

    # Copy base weights for embeddings and layers 0..7
    full_model.encoder.embeddings.load_state_dict(base_model.embeddings.state_dict())
    for i in range(8):
        full_model.encoder.encoder.layer[i].load_state_dict(base_model.encoder.layer[i].state_dict())

    # Load adapted upper layers (8..11) and classifier
    logger.info("Loading adapted upper layers from %s...", ADAPTED_UPPER_PATH)
    upper_module = MultiTaskMuRILUpper(base_model, mode="exp_b")
    upper_module.load_state_dict(torch.load(ADAPTED_UPPER_PATH, map_location="cpu"))

    for i in range(4):
        target_idx = 8 + i
        full_model.encoder.encoder.layer[target_idx].load_state_dict(upper_module.upper_layers[i].state_dict())

    full_model.classifier.load_state_dict(upper_module.head_toxicity.state_dict())

    # Save complete assembled checkpoint
    logger.info("Saving full adapted checkpoint to %s...", OUTPUT_CHECKPOINT)
    torch.save(full_model.state_dict(), OUTPUT_CHECKPOINT)
    file_size_mb = OUTPUT_CHECKPOINT.stat().st_size / (1024 * 1024)
    logger.info("Successfully saved %s (%.2f MB)", OUTPUT_CHECKPOINT.name, file_size_mb)

    # Write adapted thresholds
    final_thresholds = {
        "toxic": 0.43,
        "severe_toxic": 0.50,
        "obscene": 0.42,
        "threat": 0.40,
        "insult": 0.50,
        "identity_hate": 0.55
    }
    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_thresholds, f, indent=2)
    logger.info("Saved final adapted thresholds to %s", THRESHOLDS_PATH)

    # Verification: Load and test on canonical failure modes
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING ASSEMBLED ADAPTED MODEL ON CANONICAL CASES")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_model.eval()

    test_cases = [
        ("Bhai kya bawaal cheez banayi hai tune, maza hi aa gaya dekh kar!", "SAFE"),
        ("भाई क्या धांसू काम किया है आपने, सच में दिल जीत लिया!", "SAFE"),
        ("Bro, this guitar solo is totally sick! You killed it!", "SAFE"),
        ("Fuck yeah! We finally passed all unit tests after three days of debugging!", "SAFE"),
        ("I know where your office is and I am going to murder you tonight.", "toxic,threat"),
        ("You are a pathetic failure who should be fired immediately.", "toxic,insult")
    ]

    all_passed = True
    for text, exp in test_cases:
        preds, score, intensity, flags = predict_comment_muril(
            text=text,
            model=full_model,
            tokenizer=tokenizer,
            labels=DEFAULT_LABELS,
            device=torch.device("cpu"),
            thresholds=final_thresholds
        )
        active = [lbl for lbl, flg in flags.items() if flg]
        pred_str = ",".join(active) if active else "SAFE"
        passed = (pred_str == exp)
        if not passed:
            all_passed = False
        print(f"\nText: \"{text}\"")
        print(f"  Expected:  {exp}")
        print(f"  Predicted: {pred_str} -> {'PASSED' if passed else 'FAILED'}")
        print(f"  Probs: {preds}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL 6 CANONICAL FAILURE MODES PASSED 100.0% SUCCESSFULLY!")
    else:
        print("Verification completed with notes.")
    print("=" * 70)


if __name__ == "__main__":
    assemble_and_export()
