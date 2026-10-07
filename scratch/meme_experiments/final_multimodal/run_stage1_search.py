import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal"
DATA_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm"
EXP_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred, y_probs):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    acc = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
    
    try:
        auroc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auroc = 0.5

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

class AdvancedMultimodalClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, proj_dim=256, fusion_type="interaction", 
                 modality_dropout=0.2, classifier_dropout=0.2):
        super().__init__()
        self.fusion_type = fusion_type
        self.modality_dropout = modality_dropout
        
        # Projections into common dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        
        if fusion_type == "interaction":
            fused_dim = proj_dim * 4  # [t, v, t*v, abs(t-v)]
        elif fusion_type == "concat":
            fused_dim = proj_dim * 2  # [t, v]
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 1)
        )

    def forward(self, text_emb, image_emb, force_drop_modality=None):
        # L2-normalize defensively
        t_norm = F.normalize(text_emb, p=2, dim=-1)
        v_norm = F.normalize(image_emb, p=2, dim=-1)
        
        t_p = self.text_proj(t_norm)
        v_p = self.image_proj(v_norm)
        
        # Modality dropout during training
        if self.training and self.modality_dropout > 0:
            batch_size = text_emb.size(0)
            rand_vals = torch.rand(batch_size, 1, device=text_emb.device)
            # drop image with prob modality_dropout/2
            drop_img_mask = (rand_vals < (self.modality_dropout / 2.0)).float()
            # drop text with prob modality_dropout/2
            drop_txt_mask = ((rand_vals >= (self.modality_dropout / 2.0)) & (rand_vals < self.modality_dropout)).float()
            
            t_p = t_p * (1.0 - drop_txt_mask)
            v_p = v_p * (1.0 - drop_img_mask)
            
        # Inference-time forced dropout for ablation
        if force_drop_modality == "text":
            t_p = torch.zeros_like(t_p)
        elif force_drop_modality == "image":
            v_p = torch.zeros_like(v_p)
            
        if self.fusion_type == "interaction":
            prod = t_p * v_p
            diff = torch.abs(t_p - v_p)
            fused = torch.cat([t_p, v_p, prod, diff], dim=-1)
        else:
            fused = torch.cat([t_p, v_p], dim=-1)
            
        fused = self.fusion_norm(fused)
        logits = self.classifier(fused).squeeze(-1)
        return logits

def train_and_eval_candidate(cfg, train_loader, val_data, device):
    set_seed(42)
    model = AdvancedMultimodalClassifier(
        text_dim=768,
        image_dim=512,
        proj_dim=cfg.get("proj_dim", 256),
        fusion_type=cfg["fusion_type"],
        modality_dropout=cfg["modality_dropout"],
        classifier_dropout=cfg["classifier_dropout"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    
    val_t = val_data["text_embeddings"].to(device)
    val_v = val_data["image_embeddings"].to(device)
    val_y = val_data["labels"].numpy()
    
    best_val_f1 = -1.0
    best_metrics = None
    best_state = None
    patience = 12
    epochs_no_improve = 0
    
    history = []
    
    for epoch in range(1, 35):
        model.train()
        train_loss = 0.0
        for b_t, b_v, b_y in train_loader:
            b_t, b_v, b_y = b_t.to(device), b_v.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_t, b_v)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_t, val_v)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            
        m = compute_metrics(val_y, val_preds, val_probs)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_f1": m["f1"],
            "val_auroc": m["auroc"]
        })
        
        if m["f1"] > best_val_f1:
            best_val_f1 = m["f1"]
            best_metrics = m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    # Load best model for full validation diagnosis
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # 1. Normal Paired
        p_logits = model(val_t, val_v)
        p_probs = torch.sigmoid(p_logits).cpu().numpy()
        val_paired = compute_metrics(val_y, (p_probs >= 0.5).astype(int), p_probs)
        
        # 2. Text Only (zero image)
        t_logits = model(val_t, val_v, force_drop_modality="image")
        t_probs = torch.sigmoid(t_logits).cpu().numpy()
        val_text_only = compute_metrics(val_y, (t_probs >= 0.5).astype(int), t_probs)
        
        # 3. Image Only (zero text)
        v_logits = model(val_t, val_v, force_drop_modality="text")
        v_probs = torch.sigmoid(v_logits).cpu().numpy()
        val_image_only = compute_metrics(val_y, (v_probs >= 0.5).astype(int), v_probs)
        
        # 4. Shuffled Image
        shuff_idx = torch.randperm(val_v.size(0))
        s_logits = model(val_t, val_v[shuff_idx])
        s_probs = torch.sigmoid(s_logits).cpu().numpy()
        val_shuffled_img = compute_metrics(val_y, (s_probs >= 0.5).astype(int), s_probs)
        
    auroc_drop = round(val_paired["auroc"] - val_shuffled_img["auroc"], 4)
    f1_drop = round(val_paired["f1"] - val_shuffled_img["f1"], 4)
    
    return {
        "cfg": cfg,
        "best_epoch": len(history) - epochs_no_improve,
        "val_paired": val_paired,
        "val_text_only": val_text_only,
        "val_image_only": val_image_only,
        "val_shuffled_img": val_shuffled_img,
        "shuffled_auroc_drop": auroc_drop,
        "shuffled_f1_drop": f1_drop,
        "model_state": best_state
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading cached embeddings from {DATA_DIR} on {device}...")
    
    train_data = torch.load(DATA_DIR / "pridemm_train_embeddings.pt", weights_only=False)
    val_data = torch.load(DATA_DIR / "pridemm_val_embeddings.pt", weights_only=False)
    
    train_ds = TensorDataset(
        train_data["text_embeddings"],
        train_data["image_embeddings"],
        train_data["labels"]
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    search_space = [
        {"name": "M1_concat_baseline", "fusion_type": "concat", "modality_dropout": 0.0, "classifier_dropout": 0.2, "lr": 3e-4},
        {"name": "M2_interact_nodrop", "fusion_type": "interaction", "modality_dropout": 0.0, "classifier_dropout": 0.2, "lr": 3e-4},
        {"name": "M3_interact_drop15", "fusion_type": "interaction", "modality_dropout": 0.15, "classifier_dropout": 0.2, "lr": 3e-4},
        {"name": "M4_interact_drop25", "fusion_type": "interaction", "modality_dropout": 0.25, "classifier_dropout": 0.2, "lr": 3e-4},
        {"name": "M5_interact_drop20_lr1e4", "fusion_type": "interaction", "modality_dropout": 0.20, "classifier_dropout": 0.2, "lr": 1e-4},
        {"name": "M6_interact_drop20_drop3", "fusion_type": "interaction", "modality_dropout": 0.20, "classifier_dropout": 0.3, "lr": 3e-4},
        {"name": "M7_interact_drop20_drop1", "fusion_type": "interaction", "modality_dropout": 0.20, "classifier_dropout": 0.1, "lr": 3e-4},
    ]
    
    results = []
    print("\nStarting Compact Architecture & Hyperparameter Search...")
    for idx, cfg in enumerate(search_space, 1):
        print(f"\n--- Candidate {idx}/{len(search_space)}: {cfg['name']} ---")
        res = train_and_eval_candidate(cfg, train_loader, val_data, device)
        results.append(res)
        print(f"  Val Paired F1: {res['val_paired']['f1']} | AUROC: {res['val_paired']['auroc']}")
        print(f"  Val Text-Only F1: {res['val_text_only']['f1']} | Recall: {res['val_text_only']['recall']}")
        print(f"  Val Image-Only F1: {res['val_image_only']['f1']} | Recall: {res['val_image_only']['recall']}")
        print(f"  Shuffled Image AUROC Drop: {res['shuffled_auroc_drop']} (Paired: {res['val_paired']['auroc']} -> Shuff: {res['val_shuffled_img']['auroc']})")
        
    # Save search results summary
    summary = []
    for r in results:
        summary.append({
            "name": r["cfg"]["name"],
            "fusion": r["cfg"]["fusion_type"],
            "modality_dropout": r["cfg"]["modality_dropout"],
            "classifier_dropout": r["cfg"]["classifier_dropout"],
            "lr": r["cfg"]["lr"],
            "val_f1": r["val_paired"]["f1"],
            "val_auroc": r["val_paired"]["auroc"],
            "val_fpr": r["val_paired"]["fpr"],
            "val_text_recall": r["val_text_only"]["recall"],
            "val_image_recall": r["val_image_only"]["recall"],
            "shuffled_auroc_drop": r["shuffled_auroc_drop"]
        })
        
    with open(EXP_DIR / "stage1_search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n=== SEARCH SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
