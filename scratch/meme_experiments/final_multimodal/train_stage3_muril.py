import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal"
DATA_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm"
EXP_DIR.mkdir(parents=True, exist_ok=True)

from scratch.meme_experiments.final_multimodal.run_stage1_search import AdvancedMultimodalClassifier, compute_metrics, set_seed

class PrideMMTextDataset(Dataset):
    def __init__(self, texts, image_embeddings, labels):
        self.texts = texts
        self.image_embeddings = image_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "image_emb": self.image_embeddings[idx],
            "label": self.labels[idx]
        }

class Stage3MultimodalModel(nn.Module):
    def __init__(self, muril_model_name="google/muril-base-cased", proj_dim=256, 
                 fusion_type="interaction", modality_dropout=0.15, classifier_dropout=0.2):
        super().__init__()
        self.muril = AutoModel.from_pretrained(muril_model_name)
        # Freeze embeddings and bottom 10 layers, unfreeze top 2 layers (layers 10, 11)
        for p in self.muril.parameters():
            p.requires_grad = False
        for p in self.muril.encoder.layer[-2:].parameters():
            p.requires_grad = True
            
        self.fusion_head = AdvancedMultimodalClassifier(
            text_dim=768,
            image_dim=512,
            proj_dim=proj_dim,
            fusion_type=fusion_type,
            modality_dropout=modality_dropout,
            classifier_dropout=classifier_dropout
        )

    def forward(self, input_ids, attention_mask, image_emb, force_drop_modality=None):
        outputs = self.muril(input_ids=input_ids, attention_mask=attention_mask)
        t_emb = outputs.last_hidden_state[:, 0, :] # CLS
        logits = self.fusion_head(t_emb, image_emb, force_drop_modality=force_drop_modality)
        return logits

def run_stage3():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stage 3 (Top-2 MuRIL fine-tuning) on {device}...")

    # Load precomputed data
    train_data = torch.load(DATA_DIR / "pridemm_train_embeddings.pt", weights_only=False)
    val_data = torch.load(DATA_DIR / "pridemm_val_embeddings.pt", weights_only=False)

    train_texts = [it["text"] for it in train_data["items"]]
    val_texts = [it["text"] for it in val_data["items"]]

    tok = AutoTokenizer.from_pretrained("google/muril-base-cased")

    def collate_fn(batch):
        texts = [b["text"] for b in batch]
        img_embs = torch.stack([b["image_emb"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "image_emb": img_embs,
            "labels": labels
        }

    train_ds = PrideMMTextDataset(train_texts, train_data["image_embeddings"], train_data["labels"])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    val_enc = tok(val_texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
    val_ids = val_enc["input_ids"].to(device)
    val_mask = val_enc["attention_mask"].to(device)
    val_img = val_data["image_embeddings"].to(device)
    val_y = val_data["labels"].numpy()

    model = Stage3MultimodalModel(
        proj_dim=256,
        fusion_type="interaction",
        modality_dropout=0.15,
        classifier_dropout=0.2
    ).to(device)

    # Differential learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.muril.encoder.layer[-2:].parameters(), "lr": 1.5e-5, "weight_decay": 1e-3},
        {"params": model.fusion_head.parameters(), "lr": 2e-4, "weight_decay": 1e-2}
    ])
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_state = None
    best_metrics = None
    patience = 3
    no_improve = 0
    history = []

    for epoch in range(1, 5):
        model.train()
        train_loss = 0.0
        step = 0
        for batch in train_loader:
            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_img = batch["image_emb"].to(device)
            b_y = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask, b_img)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
            step += 1
            if step % 40 == 0:
                print(f"  Epoch {epoch} Step {step}/{len(train_loader)} - loss: {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_ids, val_mask, val_img)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)

        m = compute_metrics(val_y, val_preds, val_probs)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} | val_loss={m['f1']:.4f} | val_F1={m['f1']:.4f} | val_AUROC={m['auroc']:.4f} | val_FPR={m['fpr']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, **m})

        if m["f1"] > best_val_f1:
            best_val_f1 = m["f1"]
            best_metrics = m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Evaluate best state on validation ablation and shuffling
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_logits = model(val_ids, val_mask, val_img)
        p_probs = torch.sigmoid(p_logits).cpu().numpy()
        val_paired = compute_metrics(val_y, (p_probs >= 0.5).astype(int), p_probs)

        t_logits = model(val_ids, val_mask, val_img, force_drop_modality="image")
        t_probs = torch.sigmoid(t_logits).cpu().numpy()
        val_text = compute_metrics(val_y, (t_probs >= 0.5).astype(int), t_probs)

        v_logits = model(val_ids, val_mask, val_img, force_drop_modality="text")
        v_probs = torch.sigmoid(v_logits).cpu().numpy()
        val_image = compute_metrics(val_y, (v_probs >= 0.5).astype(int), v_probs)

        shuff_idx = torch.randperm(val_img.size(0))
        s_logits = model(val_ids, val_mask, val_img[shuff_idx])
        s_probs = torch.sigmoid(s_logits).cpu().numpy()
        val_shuff = compute_metrics(val_y, (s_probs >= 0.5).astype(int), s_probs)

    res = {
        "candidate": "Stage3_Top2MuRIL_FineTuned",
        "best_epoch": len(history) - no_improve,
        "val_paired": val_paired,
        "val_text_only": val_text,
        "val_image_only": val_image,
        "val_shuffled_img": val_shuff,
        "shuffled_auroc_drop": round(val_paired["auroc"] - val_shuff["auroc"], 4),
        "history": history
    }

    with open(EXP_DIR / "stage3_muril_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    # Save model state
    torch.save(best_state, EXP_DIR / "stage3_model.pt")
    print("\nStage 3 summary:", json.dumps(res, indent=2))

if __name__ == "__main__":
    run_stage3()
