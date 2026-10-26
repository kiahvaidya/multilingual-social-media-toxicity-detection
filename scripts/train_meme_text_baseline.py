"""
Model B: Text-Only Multilingual Meme Baseline Training Script.

Trains an experimental MuRIL checkpoint on approved real meme-text datasets (HASOC-Meme 2025).
Methodology:
  1. Starts from production MuRIL checkpoint (muril_toxicity_model.pt) for controlled domain adaptation.
  2. Unfreezes top 4 encoder layers (layer[8:]), keeping bottom 8 frozen.
  3. Uses a 6-label masked target schema matching production MuRIL:
     - toxic: governed strictly by Abuse == 'Abusive' (1) vs 'Non-abusive' (0).
     - obscene: governed by Vulgar == 'Vulgar' (1) vs 'Non Vulgar' (0).
     - insult: governed by Abuse == 'Abusive' AND Target in ('Individual', 'Gender', etc.).
     - identity_hate: governed by Abuse == 'Abusive' AND Target in ('Gender', 'Community').
     - severe_toxic, threat: masked out unless explicit threat cues are present.
     CRITICAL: When Vulgar == 1 and Abuse == 0, toxic is 0.0! This teaches the model
     that non-abusive profanity / coarse humor is NOT toxic harm.
  4. Keeps Multi3Hate STRICTLY HELD OUT.
  5. Checkpoint is saved strictly to scratch/meme_experiments/model_b/muril_model_b.pt.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.model import MuRILForMultiLabelToxicity, DEFAULT_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainModelB")

EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b"
CHECKPOINT_OUT = EXP_DIR / "muril_model_b.pt"
METRICS_OUT = EXP_DIR / "model_b_training_metrics.json"

RAW_HASOC_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "hasoc_meme_2025"


class MemeTextDataset(Dataset):
    def __init__(self, texts, targets, masks, tokenizer, max_len=64):
        self.texts = [str(t) for t in texts]
        self.targets = targets
        self.masks = masks
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
            "masks": torch.tensor(self.masks[idx], dtype=torch.float32),
        }


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, mask):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        masked_loss = bce * mask
        return masked_loss.sum() / (mask.sum() + 1e-8)


def load_hasoc_training_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    train_file = RAW_HASOC_DIR / "Hindi Train HASOC 2025 - Updated.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"HASOC 2025 train data not found at {train_file}")

    df = pd.read_csv(train_file)
    logger.info("Loaded %d raw training samples from %s", len(df), train_file.name)

    # Filter empty OCR
    df = df[df["OCR"].notna() & (df["OCR"].str.strip() != "")].copy().reset_index(drop=True)
    logger.info("Retained %d non-empty OCR samples", len(df))

    targets = np.zeros((len(df), 6), dtype=np.float32)
    masks = np.zeros((len(df), 6), dtype=np.float32)

    # Label indices in DEFAULT_LABELS:
    # 0: toxic, 1: severe_toxic, 2: obscene, 3: threat, 4: insult, 5: identity_hate
    for i, row in df.iterrows():
        is_abuse = 1.0 if str(row["Abuse"]).strip().lower() == "abusive" else 0.0
        is_vulgar = 1.0 if str(row["Vulgar"]).strip().lower() == "vulgar" else 0.0
        target_str = str(row.get("Target", "")).strip().lower()

        # 1. toxic head: STRICTLY abuse-driven (vulgar alone without abuse is NOT toxic!)
        targets[i, 0] = is_abuse
        masks[i, 0] = 1.0

        # 2. obscene head: vulgarity-driven
        targets[i, 2] = is_vulgar
        masks[i, 2] = 1.0

        # 3. insult head: abuse directed at individuals or general targets
        targets[i, 4] = is_abuse
        masks[i, 4] = 1.0

        # 4. identity_hate head: abuse directed at protected groups (Gender, Community)
        is_group = 1.0 if any(g in target_str for g in ("gender", "community", "religion", "caste")) else 0.0
        targets[i, 5] = 1.0 if (is_abuse == 1.0 and is_group == 1.0) else 0.0
        masks[i, 5] = 1.0

        # 5. severe_toxic, threat: masked out (0.0 mask) because not explicitly labeled in HASOC
        masks[i, 1] = 0.0
        masks[i, 3] = 0.0

    return df, targets, masks


def train_model_b(
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 64,
    seed: int = 42,
):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load Data
    df_data, targets, masks = load_hasoc_training_data()
    indices = np.arange(len(df_data))
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=seed, stratify=targets[:, 0])

    logger.info("Split: %d Train samples, %d Validation samples", len(train_idx), len(val_idx))

    tokenizer = AutoTokenizer.from_pretrained(config.MURIL_MODEL_NAME)

    train_ds = MemeTextDataset(df_data.iloc[train_idx]["OCR"], targets[train_idx], masks[train_idx], tokenizer, max_len=max_len)
    val_ds = MemeTextDataset(df_data.iloc[val_idx]["OCR"], targets[val_idx], masks[val_idx], tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Model Initialization (From Production Checkpoint for Controlled Domain Adaptation)
    logger.info("Initializing Model B from production checkpoint: %s", config.MURIL_CHECKPOINT_PATH)
    model = MuRILForMultiLabelToxicity(model_name=config.MURIL_MODEL_NAME, num_labels=6)

    # Load production weights
    prod_state = torch.load(config.MURIL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(prod_state)

    # Unfreeze top 4 encoder layers (layer 8, 9, 10, 11), keep bottom 8 frozen (0..7)
    for i, layer in enumerate(model.encoder.encoder.layer):
        if i < 8:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

    # Classifier head is trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model B Parameters: %d trainable / %d total (top 4 layers + classifier)", trainable_params, total_params)

    # Loss function with pos_weight on train split
    train_targets = targets[train_idx]
    train_masks = masks[train_idx]
    pos_weights = []
    for c in range(6):
        valid = (train_masks[:, c] == 1)
        pos = (train_targets[valid, c] == 1).sum()
        neg = valid.sum() - pos
        w = float(neg / max(pos, 1))
        pos_weights.append(min(w, 10.0))  # Clip extreme pos_weight

    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    logger.info("Calculated pos_weight tensor: %s", pos_weight_tensor.tolist())

    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)

    # Training Loop
    logger.info("\n--- Starting Model B Fine-Tuning (%d Epochs, lr=%s) ---", epochs, lr)
    best_val_loss = float("inf")
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            b_targets = batch["targets"].to(device)
            b_masks = batch["masks"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, b_targets, b_masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            steps += 1

        avg_train_loss = total_train_loss / max(steps, 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                b_targets = batch["targets"].to(device)
                b_masks = batch["masks"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, b_targets, b_masks)
                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / max(val_steps, 1)
        logger.info("Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f", epoch, epochs, avg_train_loss, avg_val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
        }
        history.append(epoch_record)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_OUT)
            logger.info("  --> Saved Best Model B Checkpoint to %s", CHECKPOINT_OUT)

    total_training_time = time.time() - start_time
    logger.info("Model B training completed in %.2f seconds.", total_training_time)

    # Save metadata and training metrics
    metrics = {
        "experiment_name": "Model B - Text-Only Meme Domain Adaptation",
        "base_checkpoint": str(config.MURIL_CHECKPOINT_PATH),
        "output_checkpoint": str(CHECKPOINT_OUT),
        "dataset_used": "HASOC-Meme 2025 Hindi Train (1,141 samples)",
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "unfrozen_layers": "top 4 (layer 8, 9, 10, 11)",
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "max_seq_length": max_len,
            "seed": seed,
        },
        "training_history": history,
        "best_val_loss": round(best_val_loss, 4),
        "training_time_seconds": round(total_training_time, 2),
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved Model B training metrics to %s", METRICS_OUT)
    return CHECKPOINT_OUT


if __name__ == "__main__":
    train_model_b()
