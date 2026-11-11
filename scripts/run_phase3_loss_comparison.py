"""
Phase 3: Single-Task Baseline Loss Comparison

Establishes a clean single-task baseline using the existing six-label Jigsaw formulation
augmented with the acquired CC0 Civil Comments sample.
DOES NOT mix HASOC or TRAC labels.

Compares:
  A. Current loss (Masked BCE with raw pos_weight)
  B. Class-weighted BCEWithLogitsLoss (Smoothed inverse frequency weighting)
  C. Multi-Label Focal Loss (gamma=2.0, alpha=0.25)

Evaluates on held-out validation data:
  - Micro F1, Macro F1, Per-class F1
  - Precision and Recall for every class
  - Threat Recall, Threat Precision, False Positive Rate (FPR), PR-AUC
  - Evaluates on the diagnostic regression suite (95 samples)
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
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.model import load_muril_model, DEFAULT_LABELS, MuRILForMultiLabelToxicity
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3LossComparison")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
# 1. Dataset Preparation: Jigsaw + Civil Comments Sample
# ==============================================================
def load_and_split_six_label_data(train_size=4000, val_size=1000):
    logger.info("Loading processed Jigsaw and Civil Comments data...")
    jigsaw_p = PROJECT_ROOT / "data" / "processed" / "jigsaw_six_label_train.jsonl"
    civil_p = PROJECT_ROOT / "data" / "processed" / "civil_comments_six_label_sample.jsonl"

    all_records = []
    with open(jigsaw_p, "r", encoding="utf-8") as f:
        for line in f:
            all_records.append(json.loads(line))

    with open(civil_p, "r", encoding="utf-8") as f:
        for line in f:
            all_records.append(json.loads(line))

    # Stratified collection
    threats = [r for r in all_records if r["labels"]["threat"] == 1]
    severe_or_id = [r for r in all_records if r["labels"]["threat"] == 0 and (r["labels"]["severe_toxic"] == 1 or r["labels"]["identity_hate"] == 1)]
    toxic_or_insult = [r for r in all_records if r["labels"]["threat"] == 0 and r["labels"]["severe_toxic"] == 0 and r["labels"]["identity_hate"] == 0 and (r["labels"]["toxic"] == 1 or r["labels"]["insult"] == 1 or r["labels"]["obscene"] == 1)]
    clean_negatives = [r for r in all_records if sum(r["labels"].values()) == 0]

    random.shuffle(threats)
    random.shuffle(severe_or_id)
    random.shuffle(toxic_or_insult)
    random.shuffle(clean_negatives)

    # Reserve validation partition (1,000 samples)
    val_threats = threats[:150]
    val_severe_id = severe_or_id[:250]
    val_toxic = toxic_or_insult[:250]
    val_clean = clean_negatives[:350]
    val_data = val_threats + val_severe_id + val_toxic + val_clean
    random.shuffle(val_data)

    # Training partition (4,000 samples)
    train_threats = threats[150:150 + 600]
    train_severe_id = severe_or_id[250:250 + 1000]
    train_toxic = toxic_or_insult[250:250 + 1000]
    train_clean = clean_negatives[350:350 + 1400]
    train_data = train_threats + train_severe_id + train_toxic + train_clean
    random.shuffle(train_data)

    logger.info("Train Set: %d samples (Threats: %d, Clean: %d)", len(train_data), sum(r["labels"]["threat"] for r in train_data), sum(1 for r in train_data if sum(r["labels"].values()) == 0))
    logger.info("Val Set:   %d samples (Threats: %d, Clean: %d)", len(val_data), sum(r["labels"]["threat"] for r in val_data), sum(1 for r in val_data if sum(r["labels"].values()) == 0))

    return train_data, val_data


# ==============================================================
# 2. Fast Exact Cached Prefix Extraction for Layers 0..10
# ==============================================================
def precompute_hidden_states(data: List[Dict[str, Any]], base_model: AutoModel, tokenizer: AutoTokenizer, device: torch.device, batch_size=64, freeze_until=11):
    logger.info("Precomputing static hidden states through frozen layers 0..%d (%d items)...", freeze_until - 1, len(data))
    base_model.eval()
    all_h = []
    all_masks = []
    all_targets = []

    texts = [r["text"] for r in data]
    targets = [[r["labels"][col] for col in LABEL_COLS] for r in data]

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_y = targets[i:i + batch_size]

            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=64,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            emb = base_model.embeddings(input_ids=input_ids)
            ext_mask = base_model.get_extended_attention_mask(attention_mask, input_ids.shape)

            h = emb
            for layer_idx in range(freeze_until):
                h = base_model.encoder.layer[layer_idx](h, attention_mask=ext_mask)[0]

            all_h.append(h.cpu())
            all_masks.append(ext_mask.cpu())
            all_targets.append(torch.tensor(batch_y, dtype=torch.float32))

    total_h = torch.cat(all_h, dim=0)
    total_masks = torch.cat(all_masks, dim=0)
    total_targets = torch.cat(all_targets, dim=0)
    logger.info("Precomputation finished in %.2fs. Shape: %s", time.time() - t0, list(total_h.shape))
    return total_h, total_masks, total_targets


class CachedHiddenDataset(Dataset):
    def __init__(self, h_tensor, mask_tensor, target_tensor):
        self.h = h_tensor
        self.mask = mask_tensor
        self.targets = target_tensor

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "hidden": self.h[idx],
            "mask": self.mask[idx],
            "targets": self.targets[idx]
        }


# ==============================================================
# 3. Model Architecture with Trainable Upper Layer & Classifier
# ==============================================================
class MuRILUpperLayers(nn.Module):
    def __init__(self, full_model: AutoModel, num_labels=6, unfreeze_from=11):
        super().__init__()
        self.upper_layers = nn.ModuleList([
            copy.deepcopy(full_model.encoder.layer[i]) for i in range(unfreeze_from, 12)
        ])
        for layer in self.upper_layers:
            for p in layer.parameters():
                p.requires_grad = True

        self.classifier = nn.Linear(full_model.config.hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = hidden_states
        for layer in self.upper_layers:
            h = layer(h, attention_mask=attention_mask)[0]
        pooled = h[:, 0, :]
        return self.classifier(pooled)


# ==============================================================
# 4. Loss Functions: A, B, C
# ==============================================================
class LossA_CurrentBCE(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class LossB_SmoothedWeightedBCE(nn.Module):
    def __init__(self, smoothed_pos_weight: torch.Tensor):
        super().__init__()
        self.pos_weight = smoothed_pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class LossC_MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma)
        loss = focal_weight * bce_loss
        return loss.mean()


# ==============================================================
# 5. Evaluation Function
# ==============================================================
def evaluate_cached_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            h = batch["hidden"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].numpy()

            logits = model(h, mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets)

    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    y_pred = (y_probs >= 0.5).astype(int)

    per_class = {}
    f1_list = []
    for i, col in enumerate(LABEL_COLS):
        p = float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        r = float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1 = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1_list.append(f1)

        precisions, recalls, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        pr_auc = float(auc(recalls, precisions)) if len(recalls) > 1 else 0.0

        fp = int(((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum())
        tn = int(((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum())
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        per_class[col] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "pr_auc": round(pr_auc, 4),
            "fpr": round(fpr, 4)
        }

    macro_f1 = float(np.mean(f1_list))
    micro_f1 = float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0))

    clean_mask = (y_true.sum(axis=1) == 0)
    clean_fps = (y_pred[clean_mask].sum(axis=1) > 0).sum() if clean_mask.sum() > 0 else 0
    clean_fpr = float(clean_fps / clean_mask.sum()) if clean_mask.sum() > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "clean_fpr": round(clean_fpr, 4),
        "threat_recall": per_class["threat"]["recall"],
        "threat_precision": per_class["threat"]["precision"],
        "threat_f1": per_class["threat"]["f1"],
        "threat_pr_auc": per_class["threat"]["pr_auc"],
        "per_class": per_class
    }


def evaluate_on_regression_suite(model: nn.Module, base_model: AutoModel, tokenizer: AutoTokenizer, device: torch.device):
    """Evaluate against the diagnostic regression suite of 95 failure modes and variations."""
    from scripts.evaluate_baseline_regression import build_regression_test_suite
    suite = build_regression_test_suite()

    model.eval()
    base_model.eval()
    results = []

    with torch.no_grad():
        for item in suite:
            text = item["text"]
            enc = tokenizer(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            emb = base_model.embeddings(input_ids=input_ids)
            ext_mask = base_model.get_extended_attention_mask(attention_mask, input_ids.shape)

            h = emb
            for idx in range(11):
                h = base_model.encoder.layer[idx](h, attention_mask=ext_mask)[0]

            logits = model(h, ext_mask).squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy()

            results.append({
                "id": item["id"],
                "text": text,
                "expected": item["expected"],
                "probs": {lbl: float(round(probs[i], 4)) for i, lbl in enumerate(LABEL_COLS)}
            })

    return results


def main():
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 8)
    logger.info("Phase 3 Execution on %s with %d threads", device, torch.get_num_threads())

    train_data, val_data = load_and_split_six_label_data(train_size=4000, val_size=1000)

    # Weights
    y_train = np.array([[r["labels"][col] for col in LABEL_COLS] for r in train_data])
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    raw_pos_weights = [float(neg / max(pos, 1)) for neg, pos in zip(neg_counts, pos_counts)]
    smoothed_pos_weights = [float(np.clip(np.sqrt(neg / max(pos, 1)), 1.0, 10.0)) for neg, pos in zip(neg_counts, pos_counts)]

    logger.info("Raw pos_weight:      %s", [round(w, 2) for w in raw_pos_weights])
    logger.info("Smoothed pos_weight: %s", [round(w, 2) for w in smoothed_pos_weights])

    logger.info("Loading MuRIL backbone from %s...", MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Precompute frozen hidden states (Layer 10 output)
    train_h, train_m, train_y = precompute_hidden_states(train_data, base_model, tokenizer, device, batch_size=64, freeze_until=11)
    val_h, val_m, val_y = precompute_hidden_states(val_data, base_model, tokenizer, device, batch_size=64, freeze_until=11)

    train_ds = CachedHiddenDataset(train_h, train_m, train_y)
    val_ds = CachedHiddenDataset(val_h, val_m, val_y)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    tensor_raw_pw = torch.tensor(raw_pos_weights, dtype=torch.float32).to(device)
    tensor_smooth_pw = torch.tensor(smoothed_pos_weights, dtype=torch.float32).to(device)

    loss_configs = [
        ("Loss_A_Current_PosWeight_BCE", LossA_CurrentBCE(pos_weight=tensor_raw_pw)),
        ("Loss_B_Smoothed_ClassWeighted_BCE", LossB_SmoothedWeightedBCE(smoothed_pos_weight=tensor_smooth_pw)),
        ("Loss_C_MultiLabel_FocalLoss", LossC_MultiLabelFocalLoss(alpha=0.25, gamma=2.0)),
    ]

    all_results = {}
    regression_evals = {}
    best_models = {}

    for name, crit in loss_configs:
        logger.info("\n" + "=" * 60)
        logger.info("Starting Training for: %s", name)
        logger.info("=" * 60)
        set_seed()
        model = MuRILUpperLayers(base_model, num_labels=6, unfreeze_from=11).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

        epochs = 3
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            t0 = time.time()
            for batch in train_loader:
                h = batch["hidden"].to(device)
                m = batch["mask"].to(device)
                y = batch["targets"].to(device)

                optimizer.zero_grad()
                logits = model(h, m)
                loss = crit(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info("[%s] Epoch %d/%d - Avg Loss: %.4f (%.2fs)", name, epoch + 1, epochs, avg_loss, time.time() - t0)

        metrics = evaluate_cached_model(model, val_loader, device)
        all_results[name] = metrics
        best_models[name] = model

        # Evaluate on the 6 canonical failure modes
        reg_preds = evaluate_on_regression_suite(model, base_model, tokenizer, device)
        canonicals = [r for r in reg_preds if r["id"].startswith("canonical_")]
        regression_evals[name] = canonicals

    print("\n" + "=" * 80)
    print("PHASE 3 LOSS COMPARISON SUMMARY TABLE")
    print("=" * 80)
    headers = ["Loss Configuration", "Macro F1", "Micro F1", "Threat Recall", "Threat Prec", "Threat F1", "Clean FPR"]
    print(f"{headers[0]:<35} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]:<13} | {headers[4]:<11} | {headers[5]:<9} | {headers[6]:<9}")
    print("-" * 105)

    for name, res in all_results.items():
        print(f"{name:<35} | {res['macro_f1']:<8.4f} | {res['micro_f1']:<8.4f} | {res['threat_recall']:<13.4f} | {res['threat_precision']:<11.4f} | {res['threat_f1']:<9.4f} | {res['clean_fpr']:<9.4f}")

    print("\n" + "=" * 80)
    print("PREDICTIONS ON 6 CANONICAL FAILURE MODES UNDER EACH LOSS CONFIGURATION:")
    print("=" * 80)
    for name, items in regression_evals.items():
        print(f"\n--- {name} ---")
        for item in items:
            probs = item["probs"]
            print(f"[{item['id']}] \"{item['text'][:60]}...\"")
            print(f"   toxic={probs['toxic']} | threat={probs['threat']} | obscene={probs['obscene']} | insult={probs['insult']}")

    # Save artifact
    out_p = ARTIFACTS_DIR / "phase3_loss_comparison.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            "comparison_metrics": all_results,
            "canonical_failure_predictions": regression_evals
        }, f, indent=2)
    logger.info("Saved Phase 3 report to %s", out_p)


if __name__ == "__main__":
    main()
