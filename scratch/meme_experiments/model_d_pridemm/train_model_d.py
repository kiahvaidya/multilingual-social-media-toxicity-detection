import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

OUT_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ModelDFusionMLP(nn.Module):
    def __init__(self, in_features=1280, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Model D on {device}...")

    # Load precomputed embeddings
    train_data = torch.load(OUT_DIR / "pridemm_train_embeddings.pt", weights_only=False)
    val_data = torch.load(OUT_DIR / "pridemm_val_embeddings.pt", weights_only=False)

    X_train = torch.cat([train_data["text_embeddings"], train_data["image_embeddings"]], dim=-1)
    y_train = train_data["labels"]

    X_val = torch.cat([val_data["text_embeddings"], val_data["image_embeddings"]], dim=-1)
    y_val = val_data["labels"]

    print(f"X_train shape: {X_train.shape}, positive ratio: {float(y_train.mean()):.4f}")
    print(f"X_val shape: {X_val.shape}, positive ratio: {float(y_val.mean()):.4f}")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = ModelDFusionMLP(in_features=1280, hidden_dim=256, dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    history = []

    for epoch in range(1, 61):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)

        train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                val_probs.extend(probs.tolist())
                val_preds.extend(preds.tolist())
                val_targets.extend(by.cpu().numpy().tolist())

        val_loss = val_loss / len(val_ds)
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)

        v_acc = accuracy_score(val_targets, val_preds)
        v_prec = precision_score(val_targets, val_preds, zero_division=0)
        v_rec = recall_score(val_targets, val_preds, zero_division=0)
        v_f1 = f1_score(val_targets, val_preds, zero_division=0)
        v_auroc = roc_auc_score(val_targets, val_probs)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(v_acc, 4),
            "val_precision": round(v_prec, 4),
            "val_recall": round(v_rec, 4),
            "val_f1": round(v_f1, 4),
            "val_auroc": round(v_auroc, 4)
        }
        history.append(epoch_stats)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_F1={v_f1:.4f} | val_AUROC={v_auroc:.4f}")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_f1": v_f1,
                "best_val_auroc": v_auroc,
                "config": {
                    "in_features": 1280,
                    "hidden_dim": 256,
                    "dropout": 0.2,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "seed": 42
                }
            }
            torch.save(checkpoint, OUT_DIR / "model_d_fusion.pt")
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val F1: {v_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val F1: {best_val_f1:.4f}.")
                break

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training stage 1 finished successfully!")

if __name__ == "__main__":
    train()
