"""
Phase 4 & Phase 5: Multi-Task Domain Adaptation & Controlled Fine-Tuning

Architecture:
Shared MuRIL Encoder (google/muril-base-cased)
 |
 +-- Head 1: Jigsaw/Civil 6-label Toxicity Head (nn.Linear(768, 6))
 |
 +-- Head 2: HASOC Offensive Head (nn.Linear(768, 1))
 |
 +-- Head 3: TRAC Aggression Head (nn.Linear(768, 3))

Loss Formulation:
  L_total = L_toxicity + lambda_hasoc * L_hasoc + lambda_trac * L_trac
  (lambda_hasoc = 0.2, lambda_trac = 0.2)

Controlled Fine-Tuning Experiments:
  - Experiment A: 1 layer unfrozen (Layer 11) - Baseline Freezing
  - Experiment B: 4 layers unfrozen (Layers 8..11)
  - Experiment C: LoRA on upper transformer layers (Layers 8..11 Attention Projections, r=8, alpha=16)

Evaluates on:
  - Held-out 6-label validation data (1,000 samples)
  - Held-out HASOC validation data (500 samples)
  - Held-out TRAC validation data (500 samples)
  - Dedicated Diagnostic Regression Suite (95 samples including 6 canonical cases)
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
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc, accuracy_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase4_5MultiTask")

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
# 1. Multi-Task Data Loading & Preparation
# ==============================================================
def load_all_multitask_data():
    logger.info("Loading Jigsaw + Civil Comments processed records...")
    jigsaw_p = PROJECT_ROOT / "data" / "processed" / "jigsaw_six_label_train.jsonl"
    civil_p = PROJECT_ROOT / "data" / "processed" / "civil_comments_six_label_sample.jsonl"
    hasoc_p = PROJECT_ROOT / "data" / "processed" / "hasoc_2021_unified.jsonl"
    trac_p = PROJECT_ROOT / "data" / "processed" / "trac_1_unified.jsonl"

    # 1. Toxicity Data (Jigsaw + Civil)
    tox_records = []
    with open(jigsaw_p, "r", encoding="utf-8") as f:
        for line in f:
            tox_records.append(json.loads(line))
    with open(civil_p, "r", encoding="utf-8") as f:
        for line in f:
            tox_records.append(json.loads(line))

    threats = [r for r in tox_records if r["labels"]["threat"] == 1]
    severe_id = [r for r in tox_records if r["labels"]["threat"] == 0 and (r["labels"]["severe_toxic"] == 1 or r["labels"]["identity_hate"] == 1)]
    toxic_insult = [r for r in tox_records if r["labels"]["threat"] == 0 and r["labels"]["severe_toxic"] == 0 and r["labels"]["identity_hate"] == 0 and (r["labels"]["toxic"] == 1 or r["labels"]["insult"] == 1 or r["labels"]["obscene"] == 1)]
    clean_neg = [r for r in tox_records if sum(r["labels"].values()) == 0]

    random.shuffle(threats)
    random.shuffle(severe_id)
    random.shuffle(toxic_insult)
    random.shuffle(clean_neg)

    val_tox = threats[:150] + severe_id[:250] + toxic_insult[:250] + clean_neg[:350]
    train_tox = threats[150:750] + severe_id[250:1250] + toxic_insult[250:1250] + clean_neg[350:1750]
    random.shuffle(val_tox)
    random.shuffle(train_tox)

    # 2. HASOC 2021 Data (Hindi + English)
    hasoc_records = []
    with open(hasoc_p, "r", encoding="utf-8") as f:
        for line in f:
            hasoc_records.append(json.loads(line))
    random.shuffle(hasoc_records)
    val_hasoc = hasoc_records[:500]
    train_hasoc = hasoc_records[500:2500]

    # 3. TRAC-1 Data (Hindi + English)
    trac_records = []
    with open(trac_p, "r", encoding="utf-8") as f:
        for line in f:
            trac_records.append(json.loads(line))
    random.shuffle(trac_records)
    val_trac = trac_records[:500]
    train_trac = trac_records[500:2500]

    logger.info("Loaded Data Splits:")
    logger.info("  Toxicity:  Train=%d, Val=%d", len(train_tox), len(val_tox))
    logger.info("  HASOC:     Train=%d, Val=%d", len(train_hasoc), len(val_hasoc))
    logger.info("  TRAC:      Train=%d, Val=%d", len(train_trac), len(val_trac))

    return {
        "train_tox": train_tox, "val_tox": val_tox,
        "train_hasoc": train_hasoc, "val_hasoc": val_hasoc,
        "train_trac": train_trac, "val_trac": val_trac
    }


# ==============================================================
# 2. Precomputation of Hidden States at Layer 7 (Frozen Prefix 0..7)
# ==============================================================
def precompute_layer7_representations(data_list, base_model, tokenizer, device, batch_size=64):
    """Passes text through Embeddings + Layers 0..7 (all 8 layers strictly frozen)."""
    base_model.eval()
    all_h = []
    all_masks = []
    texts = [r["text"] for r in data_list]

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
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
            for idx in range(8):  # Layers 0 through 7
                h = base_model.encoder.layer[idx](h, attention_mask=ext_mask)[0]

            all_h.append(h.cpu())
            all_masks.append(ext_mask.cpu())

    return torch.cat(all_h, dim=0), torch.cat(all_masks, dim=0)


# ==============================================================
# 3. LoRA Module Implementation
# ==============================================================
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.scaling = lora_alpha / r
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(r, in_features) * (1.0 / np.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        for p in self.original_linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def apply_lora_to_layer(layer: nn.Module, r: int = 8, alpha: float = 16.0):
    """Applies LoRA to query and value projections of a BertLayer."""
    for p in layer.parameters():
        p.requires_grad = False
    layer.attention.self.query = LoRALinear(layer.attention.self.query, r=r, lora_alpha=alpha)
    layer.attention.self.value = LoRALinear(layer.attention.self.value, r=r, lora_alpha=alpha)
    return layer


# ==============================================================
# 4. Multi-Task Upper Architecture (Layers 8..11 + 3 Task Heads)
# ==============================================================
class MultiTaskMuRILUpper(nn.Module):
    def __init__(self, base_model: AutoModel, mode: str = "exp_a"):
        """
        mode:
          - 'exp_a': Layers 8..10 frozen, only Layer 11 trainable
          - 'exp_b': Layers 8..11 fully trainable
          - 'exp_c': LoRA applied to Layers 8..11 (query and value)
        """
        super().__init__()
        self.mode = mode
        hidden_size = base_model.config.hidden_size

        layers = [copy.deepcopy(base_model.encoder.layer[i]) for i in range(8, 12)]

        if mode == "exp_a":
            # Freeze layers 8, 9, 10
            for i in range(3):
                for p in layers[i].parameters():
                    p.requires_grad = False
            # Train Layer 11
            for p in layers[3].parameters():
                p.requires_grad = True

        elif mode == "exp_b":
            # Train all 4 layers (8..11)
            for layer in layers:
                for p in layer.parameters():
                    p.requires_grad = True

        elif mode == "exp_c":
            # LoRA on layers 8..11
            layers = [apply_lora_to_layer(layer, r=8, alpha=16.0) for layer in layers]

        self.upper_layers = nn.ModuleList(layers)

        # Task Heads
        self.head_toxicity = nn.Linear(hidden_size, 6)
        self.head_hasoc = nn.Linear(hidden_size, 1)
        self.head_trac = nn.Linear(hidden_size, 3)

        nn.init.xavier_uniform_(self.head_toxicity.weight)
        nn.init.zeros_(self.head_toxicity.bias)
        nn.init.xavier_uniform_(self.head_hasoc.weight)
        nn.init.zeros_(self.head_hasoc.bias)
        nn.init.xavier_uniform_(self.head_trac.weight)
        nn.init.zeros_(self.head_trac.bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = hidden_states
        for layer in self.upper_layers:
            h = layer(h, attention_mask=attention_mask)[0]

        pooled = h[:, 0, :]
        logits_tox = self.head_toxicity(pooled)
        logits_hasoc = self.head_hasoc(pooled).squeeze(-1)
        logits_trac = self.head_trac(pooled)

        return logits_tox, logits_hasoc, logits_trac


# ==============================================================
# 5. Multi-Task Training Loop
# ==============================================================
def train_multitask_epoch(model, opt, tox_loader, hasoc_loader, trac_loader, pos_weight_tensor, device, lambda_hasoc=0.2, lambda_trac=0.2):
    model.train()
    bce_tox = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    bce_hasoc = nn.BCEWithLogitsLoss()
    ce_trac = nn.CrossEntropyLoss()

    hasoc_iter = iter(hasoc_loader)
    trac_iter = iter(trac_loader)

    total_loss = 0.0
    step_count = 0

    for tox_batch in tox_loader:
        # 1. Toxicity Step
        h_tox = tox_batch["hidden"].to(device)
        m_tox = tox_batch["mask"].to(device)
        y_tox = tox_batch["targets"].to(device)

        opt.zero_grad()
        logits_tox, _, _ = model(h_tox, m_tox)
        l_tox = bce_tox(logits_tox, y_tox)

        # 2. HASOC Step
        try:
            hasoc_batch = next(hasoc_iter)
        except StopIteration:
            hasoc_iter = iter(hasoc_loader)
            hasoc_batch = next(hasoc_iter)

        h_hasoc = hasoc_batch["hidden"].to(device)
        m_hasoc = hasoc_batch["mask"].to(device)
        y_hasoc = hasoc_batch["targets"].to(device)
        _, logits_hasoc, _ = model(h_hasoc, m_hasoc)
        l_hasoc = bce_hasoc(logits_hasoc, y_hasoc)

        # 3. TRAC Step
        try:
            trac_batch = next(trac_iter)
        except StopIteration:
            trac_iter = iter(trac_loader)
            trac_batch = next(trac_iter)

        h_trac = trac_batch["hidden"].to(device)
        m_trac = trac_batch["mask"].to(device)
        y_trac = trac_batch["targets"].to(device)
        _, _, logits_trac = model(h_trac, m_trac)
        l_trac = ce_trac(logits_trac, y_trac)

        loss = l_tox + (lambda_hasoc * l_hasoc) + (lambda_trac * l_trac)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total_loss += loss.item()
        step_count += 1

    return total_loss / max(step_count, 1)


# ==============================================================
# 6. Evaluation Functions
# ==============================================================
def evaluate_toxicity(model, val_loader, device):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            h = batch["hidden"].to(device)
            m = batch["mask"].to(device)
            y = batch["targets"].numpy()

            logits_tox, _, _ = model(h, m)
            probs = torch.sigmoid(logits_tox).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y)

    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    y_pred = (y_probs >= 0.5).astype(int)

    f1_list = []
    per_class = {}
    for i, col in enumerate(LABEL_COLS):
        p = float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        r = float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1 = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1_list.append(f1)
        per_class[col] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    macro_f1 = float(np.mean(f1_list))
    micro_f1 = float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0))

    clean_mask = (y_true.sum(axis=1) == 0)
    clean_fps = (y_pred[clean_mask].sum(axis=1) > 0).sum() if clean_mask.sum() > 0 else 0
    clean_fpr = float(clean_fps / clean_mask.sum()) if clean_mask.sum() > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "threat_recall": per_class["threat"]["recall"],
        "threat_precision": per_class["threat"]["precision"],
        "threat_f1": per_class["threat"]["f1"],
        "clean_fpr": round(clean_fpr, 4),
        "per_class": per_class
    }


def evaluate_auxiliary(model, hasoc_loader, trac_loader, device):
    model.eval()
    # HASOC
    hasoc_preds = []
    hasoc_trues = []
    with torch.no_grad():
        for batch in hasoc_loader:
            h = batch["hidden"].to(device)
            m = batch["mask"].to(device)
            y = batch["targets"].numpy()
            _, logits_hasoc, _ = model(h, m)
            probs = torch.sigmoid(logits_hasoc).cpu().numpy()
            hasoc_preds.extend((probs >= 0.5).astype(int))
            hasoc_trues.extend(y)
    hasoc_f1 = float(f1_score(hasoc_trues, hasoc_preds, zero_division=0))

    # TRAC
    trac_preds = []
    trac_trues = []
    with torch.no_grad():
        for batch in trac_loader:
            h = batch["hidden"].to(device)
            m = batch["mask"].to(device)
            y = batch["targets"].numpy()
            _, _, logits_trac = model(h, m)
            preds = torch.argmax(logits_trac, dim=-1).cpu().numpy()
            trac_preds.extend(preds)
            trac_trues.extend(y)
    trac_acc = float(accuracy_score(trac_trues, trac_preds))
    trac_macro_f1 = float(f1_score(trac_trues, trac_preds, average="macro", zero_division=0))

    return {
        "hasoc_f1": round(hasoc_f1, 4),
        "trac_acc": round(trac_acc, 4),
        "trac_macro_f1": round(trac_macro_f1, 4)
    }


def evaluate_regression(model, base_model, tokenizer, device):
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
            for idx in range(8):  # Layers 0..7
                h = base_model.encoder.layer[idx](h, attention_mask=ext_mask)[0]

            logits_tox, _, _ = model(h, ext_mask)
            probs = torch.sigmoid(logits_tox).squeeze(0).cpu().numpy()

            results.append({
                "id": item["id"],
                "text": text,
                "expected": item["expected"],
                "probs": {lbl: float(round(probs[i], 4)) for i, lbl in enumerate(LABEL_COLS)}
            })

    return results


# ==============================================================
# 7. Main Execution: Experiments A, B, C
# ==============================================================
class GenericDataset(Dataset):
    def __init__(self, h, m, y):
        self.h = h
        self.m = m
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"hidden": self.h[idx], "mask": self.m[idx], "targets": self.y[idx]}


def main():
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 8)
    logger.info("Running Phase 4 & 5 Multi-Task Pipeline on CPU with %d threads", torch.get_num_threads())

    splits = load_all_multitask_data()

    logger.info("Loading MuRIL base encoder...")
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. Precompute Layer 7 representations for all splits
    logger.info("\n--- Precomputing Layer 7 representations ---")
    tox_tr_h, tox_tr_m = precompute_layer7_representations(splits["train_tox"], base_model, tokenizer, device)
    tox_val_h, tox_val_m = precompute_layer7_representations(splits["val_tox"], base_model, tokenizer, device)

    hasoc_tr_h, hasoc_tr_m = precompute_layer7_representations(splits["train_hasoc"], base_model, tokenizer, device)
    hasoc_val_h, hasoc_val_m = precompute_layer7_representations(splits["val_hasoc"], base_model, tokenizer, device)

    trac_tr_h, trac_tr_m = precompute_layer7_representations(splits["train_trac"], base_model, tokenizer, device)
    trac_val_h, trac_val_m = precompute_layer7_representations(splits["val_trac"], base_model, tokenizer, device)

    # Targets
    tox_tr_y = torch.tensor([[r["labels"][col] for col in LABEL_COLS] for r in splits["train_tox"]], dtype=torch.float32)
    tox_val_y = torch.tensor([[r["labels"][col] for col in LABEL_COLS] for r in splits["val_tox"]], dtype=torch.float32)

    hasoc_tr_y = torch.tensor([r["labels"]["hof"] for r in splits["train_hasoc"]], dtype=torch.float32)
    hasoc_val_y = torch.tensor([r["labels"]["hof"] for r in splits["val_hasoc"]], dtype=torch.float32)

    trac_map = {"NAG": 0, "CAG": 1, "OAG": 2}
    trac_tr_y = torch.tensor([trac_map.get(r["labels"]["aggression_level"], 0) for r in splits["train_trac"]], dtype=torch.long)
    trac_val_y = torch.tensor([trac_map.get(r["labels"]["aggression_level"], 0) for r in splits["val_trac"]], dtype=torch.long)

    # Calculate pos_weights for toxicity
    pos_counts = tox_tr_y.sum(dim=0).numpy()
    neg_counts = len(tox_tr_y) - pos_counts
    raw_pw = [float(neg / max(pos, 1)) for neg, pos in zip(neg_counts, pos_counts)]
    pos_weight_tensor = torch.tensor(raw_pw, dtype=torch.float32).to(device)

    # DataLoaders
    tox_tr_loader = DataLoader(GenericDataset(tox_tr_h, tox_tr_m, tox_tr_y), batch_size=32, shuffle=True)
    tox_val_loader = DataLoader(GenericDataset(tox_val_h, tox_val_m, tox_val_y), batch_size=32, shuffle=False)

    hasoc_tr_loader = DataLoader(GenericDataset(hasoc_tr_h, hasoc_tr_m, hasoc_tr_y), batch_size=32, shuffle=True)
    hasoc_val_loader = DataLoader(GenericDataset(hasoc_val_h, hasoc_val_m, hasoc_val_y), batch_size=32, shuffle=False)

    trac_tr_loader = DataLoader(GenericDataset(trac_tr_h, trac_tr_m, trac_tr_y), batch_size=32, shuffle=True)
    trac_val_loader = DataLoader(GenericDataset(trac_val_h, trac_val_m, trac_val_y), batch_size=32, shuffle=False)

    experiments = [
        ("Experiment_A_1Layer_Frozen", "exp_a"),
        ("Experiment_B_4Layers_Unfrozen", "exp_b"),
        ("Experiment_C_LoRA_UpperLayers", "exp_c")
    ]

    results = {}
    trained_models = {}

    for exp_name, mode in experiments:
        logger.info("\n" + "=" * 65)
        logger.info("STARTING %s (Mode: %s)", exp_name, mode)
        logger.info("=" * 65)
        set_seed()

        model = MultiTaskMuRILUpper(base_model, mode=mode).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in trainable_params)
        logger.info("Trainable parameters in %s: %d", exp_name, param_count)

        optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=0.01)

        epochs = 3
        for epoch in range(epochs):
            t0 = time.time()
            loss = train_multitask_epoch(
                model, optimizer, tox_tr_loader, hasoc_tr_loader, trac_tr_loader,
                pos_weight_tensor, device, lambda_hasoc=0.2, lambda_trac=0.2
            )
            logger.info("[%s] Epoch %d/%d - Avg Multi-Task Loss: %.4f (%.2fs)", exp_name, epoch + 1, epochs, loss, time.time() - t0)

        # Evaluations
        tox_metrics = evaluate_toxicity(model, tox_val_loader, device)
        aux_metrics = evaluate_auxiliary(model, hasoc_val_loader, trac_val_loader, device)
        reg_preds = evaluate_regression(model, base_model, tokenizer, device)
        canonicals = [r for r in reg_preds if r["id"].startswith("canonical_")]

        results[exp_name] = {
            "mode": mode,
            "trainable_parameters": param_count,
            "toxicity_metrics": tox_metrics,
            "auxiliary_metrics": aux_metrics,
            "canonical_failure_predictions": canonicals
        }
        trained_models[exp_name] = model

    # Save summary report
    out_p = ARTIFACTS_DIR / "phase4_5_experiments.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print("PHASE 4 & 5 MULTI-TASK & FINE-TUNING COMPARISON TABLE")
    print("=" * 90)
    headers = ["Experiment", "Trainable Params", "Macro F1", "Micro F1", "Threat Recall", "Threat F1", "HASOC F1", "TRAC Macro F1", "Clean FPR"]
    print(f"{headers[0]:<32} | {headers[1]:<16} | {headers[2]:<8} | {headers[3]:<8} | {headers[4]:<13} | {headers[5]:<9} | {headers[6]:<8} | {headers[7]:<13} | {headers[8]:<9}")
    print("-" * 130)

    for exp_name, res in results.items():
        tm = res["toxicity_metrics"]
        am = res["auxiliary_metrics"]
        print(f"{exp_name:<32} | {res['trainable_parameters']:<16,} | {tm['macro_f1']:<8.4f} | {tm['micro_f1']:<8.4f} | {tm['threat_recall']:<13.4f} | {tm['threat_f1']:<9.4f} | {am['hasoc_f1']:<8.4f} | {am['trac_macro_f1']:<13.4f} | {tm['clean_fpr']:<9.4f}")

    print("\n" + "=" * 90)
    print("6 CANONICAL CASES UNDER EACH EXPERIMENT:")
    print("=" * 90)
    for exp_name, res in results.items():
        print(f"\n--- {exp_name} ---")
        for c in res["canonical_failure_predictions"]:
            p = c["probs"]
            print(f"[{c['id']}] \"{c['text'][:55]}...\"")
            print(f"   toxic={p['toxic']} | threat={p['threat']} | obscene={p['obscene']} | insult={p['insult']}")

    # Save the best model parameters for downstream Phase 6 hard-negative adaptation
    # Select best model based on validation Macro F1 and Threat Recall
    best_exp = max(results.keys(), key=lambda k: results[k]["toxicity_metrics"]["macro_f1"] + results[k]["toxicity_metrics"]["threat_recall"])
    logger.info("Best configuration selected: %s", best_exp)
    torch.save(trained_models[best_exp].state_dict(), ARTIFACTS_DIR / "best_multitask_upper.pt")
    logger.info("Saved best upper weights to %s", ARTIFACTS_DIR / "best_multitask_upper.pt")


if __name__ == "__main__":
    main()
