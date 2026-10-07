import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal"
DATA_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm"
MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
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
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    acc = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
    pos_rate = float(np.sum(y_pred == 1) / len(y_pred)) if len(y_pred) > 0 else 0.0
    
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
        "fnr": round(fnr, 4),
        "pred_pos": int(np.sum(y_pred == 1)),
        "pred_neg": int(np.sum(y_pred == 0)),
        "positive_prediction_rate": round(pos_rate, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

class FinalMultimodalClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, proj_dim=256, modality_dropout=0.15, classifier_dropout=0.2):
        super().__init__()
        self.modality_dropout = modality_dropout
        
        # Latent Projections
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
        
        # Interaction features: [t, v, t*v, abs(t-v)] -> proj_dim * 4
        fused_dim = proj_dim * 4
        self.fusion_norm = nn.LayerNorm(fused_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 1)
        )

    def forward(self, text_emb, image_emb, force_drop_modality=None):
        t_norm = F.normalize(text_emb, p=2, dim=-1)
        v_norm = F.normalize(image_emb, p=2, dim=-1)
        
        t_p = self.text_proj(t_norm)
        v_p = self.image_proj(v_norm)
        
        # Modality dropout during training
        if self.training and self.modality_dropout > 0:
            batch_size = text_emb.size(0)
            rand_vals = torch.rand(batch_size, 1, device=text_emb.device)
            drop_img_mask = (rand_vals < (self.modality_dropout / 2.0)).float()
            drop_txt_mask = ((rand_vals >= (self.modality_dropout / 2.0)) & (rand_vals < self.modality_dropout)).float()
            t_p = t_p * (1.0 - drop_txt_mask)
            v_p = v_p * (1.0 - drop_img_mask)
            
        if force_drop_modality == "text":
            t_p = torch.zeros_like(t_p)
        elif force_drop_modality == "image":
            v_p = torch.zeros_like(v_p)
            
        prod = t_p * v_p
        diff = torch.abs(t_p - v_p)
        fused = torch.cat([t_p, v_p, prod, diff], dim=-1)
        fused = self.fusion_norm(fused)
        logits = self.classifier(fused).squeeze(-1)
        return logits

def train_and_evaluate():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== FINAL MULTIMODAL MODEL OPTIMIZATION on {device} ===")

    # 1. Load Precomputed Embeddings
    print("Loading precomputed feature tensors...")
    train_data = torch.load(DATA_DIR / "pridemm_train_embeddings.pt", weights_only=False)
    val_data = torch.load(DATA_DIR / "pridemm_val_embeddings.pt", weights_only=False)
    test_data = torch.load(DATA_DIR / "pridemm_test_embeddings.pt", weights_only=False)
    m3h_data = torch.load(DATA_DIR / "multi3hate_embeddings.pt", weights_only=False)

    train_ds = TensorDataset(train_data["text_embeddings"], train_data["image_embeddings"], train_data["labels"])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    val_t = val_data["text_embeddings"].to(device)
    val_v = val_data["image_embeddings"].to(device)
    val_y = val_data["labels"].numpy()

    # 2. Train Model
    config = {
        "architecture": "CrossModal_Interaction_Fusion",
        "text_encoder": "MuRIL (Frozen 768-dim CLS)",
        "image_encoder": "CLIP ViT-B/32 (Frozen 512-dim)",
        "projection_dim": 256,
        "interaction_features": "[proj_text, proj_img, proj_text*proj_img, abs(proj_text-proj_img)]",
        "modality_dropout": 0.15,
        "classifier_dropout": 0.2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "batch_size": 32,
        "random_seed": 42
    }

    model = FinalMultimodalClassifier(
        text_dim=768,
        image_dim=512,
        proj_dim=256,
        modality_dropout=0.15,
        classifier_dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience = 12
    epochs_no_improve = 0
    history = []

    print("\n--- Training Model ---")
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

        model.eval()
        with torch.no_grad():
            v_logits = model(val_t, val_v)
            v_probs = torch.sigmoid(v_logits).cpu().numpy()
            v_preds = (v_probs >= 0.5).astype(int)

        m = compute_metrics(val_y, v_preds, v_probs)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_f1": m["f1"],
            "val_auroc": m["auroc"],
            "val_fpr": m["fpr"]
        })

        if m["f1"] > best_val_f1:
            best_val_f1 = m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
            print(f"  Epoch {epoch:02d}: train_loss={train_loss:.4f} | Val F1={m['f1']:.4f} | Val AUROC={m['auroc']:.4f} | Val FPR={m['fpr']:.4f}  [BEST]")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Best was epoch {best_epoch}.")
                break

    # Save Best Model Checkpoint
    torch.save(best_state, EXP_DIR / "best_model.pt")
    with open(EXP_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(best_state)
    model.eval()

    # 3. Decision Threshold Optimization on Validation Set
    print("\n--- Optimizing Decision Threshold on PrideMM Validation ---")
    with torch.no_grad():
        val_logits = model(val_t, val_v)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    best_thresh = 0.50
    best_thresh_f1 = -1.0
    threshold_sweep = []

    for th in np.arange(0.20, 0.81, 0.01):
        th = round(float(th), 2)
        preds_th = (val_probs >= th).astype(int)
        m_th = compute_metrics(val_y, preds_th, val_probs)
        threshold_sweep.append({"threshold": th, "f1": m_th["f1"], "fpr": m_th["fpr"], "precision": m_th["precision"], "recall": m_th["recall"]})
        # Optimize F1 while preventing excessive FPR
        if m_th["f1"] > best_thresh_f1 and m_th["fpr"] <= 0.35:
            best_thresh_f1 = m_th["f1"]
            best_thresh = th

    print(f"Selected Optimal Threshold on Validation: {best_thresh} (Val F1: {best_thresh_f1})")
    config["optimal_threshold"] = best_thresh
    with open(EXP_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    val_metrics_at_opt = compute_metrics(val_y, (val_probs >= best_thresh).astype(int), val_probs)
    with open(EXP_DIR / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics_at_opt, f, indent=2)

    # 4. Untouched PrideMM Test Evaluation (N=507)
    print(f"\n--- Evaluating on Untouched PrideMM Test (N=507) at T={best_thresh} ---")
    test_t = test_data["text_embeddings"].to(device)
    test_v = test_data["image_embeddings"].to(device)
    test_y = test_data["labels"].numpy()

    with torch.no_grad():
        test_logits = model(test_t, test_v)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= best_thresh).astype(int)

    pridemm_test_metrics = compute_metrics(test_y, test_preds, test_probs)
    with open(EXP_DIR / "pridemm_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(pridemm_test_metrics, f, indent=2)
    print("PrideMM Test Metrics:", json.dumps(pridemm_test_metrics, indent=2))

    # 5. Untouched Multi3Hate Evaluation (N=600)
    print(f"\n--- Evaluating on Held-Out Multi3Hate (N=600) at T={best_thresh} ---")
    m3h_t = m3h_data["text_embeddings"].to(device)
    m3h_v = m3h_data["image_embeddings"].to(device)
    m3h_y_in = np.array([it["IN"] for it in m3h_data["items"]], dtype=int)
    m3h_y_us = np.array([it["US"] for it in m3h_data["items"]], dtype=int)
    m3h_y_de = np.array([it["DE"] for it in m3h_data["items"]], dtype=int)
    m3h_y_mx = np.array([it["MX"] for it in m3h_data["items"]], dtype=int)
    m3h_y_cn = np.array([it["CN"] for it in m3h_data["items"]], dtype=int)
    m3h_langs = np.array([it["language"] for it in m3h_data["items"]])

    with torch.no_grad():
        m3h_logits = model(m3h_t, m3h_v)
        m3h_probs = torch.sigmoid(m3h_logits).cpu().numpy()
        m3h_preds = (m3h_probs >= best_thresh).astype(int)

    m3h_overall = compute_metrics(m3h_y_in, m3h_preds, m3h_probs)
    
    # Slices
    mask_en = (m3h_langs == "english")
    mask_hi = (m3h_langs == "hindi")
    m3h_en = compute_metrics(m3h_y_in[mask_en], m3h_preds[mask_en], m3h_probs[mask_en])
    m3h_hi = compute_metrics(m3h_y_in[mask_hi], m3h_preds[mask_hi], m3h_probs[mask_hi])

    # Consensus
    m3h_us = compute_metrics(m3h_y_us, m3h_preds, m3h_probs)
    m3h_de = compute_metrics(m3h_y_de, m3h_preds, m3h_probs)
    m3h_mx = compute_metrics(m3h_y_mx, m3h_preds, m3h_probs)
    m3h_cn = compute_metrics(m3h_y_cn, m3h_preds, m3h_probs)

    multi3hate_all = {
        "overall_in": m3h_overall,
        "english": m3h_en,
        "hindi": m3h_hi,
        "us_consensus": m3h_us,
        "de_consensus": m3h_de,
        "mx_consensus": m3h_mx,
        "cn_consensus": m3h_cn
    }
    with open(EXP_DIR / "multi3hate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(multi3hate_all, f, indent=2)
    print("Multi3Hate Overall:", json.dumps(m3h_overall, indent=2))
    print("Multi3Hate English:", json.dumps(m3h_en, indent=2))
    print("Multi3Hate Hindi:", json.dumps(m3h_hi, indent=2))

    # 6. Modality Ablation Analysis (PrideMM Test & Multi3Hate)
    print("\n--- Modality Ablation Analysis ---")
    with torch.no_grad():
        # PrideMM Test
        pride_txt_probs = torch.sigmoid(model(test_t, test_v, force_drop_modality="image")).cpu().numpy()
        pride_img_probs = torch.sigmoid(model(test_t, test_v, force_drop_modality="text")).cpu().numpy()
        pride_abl_txt = compute_metrics(test_y, (pride_txt_probs >= best_thresh).astype(int), pride_txt_probs)
        pride_abl_img = compute_metrics(test_y, (pride_img_probs >= best_thresh).astype(int), pride_img_probs)

        # Multi3Hate
        m3h_txt_probs = torch.sigmoid(model(m3h_t, m3h_v, force_drop_modality="image")).cpu().numpy()
        m3h_img_probs = torch.sigmoid(model(m3h_t, m3h_v, force_drop_modality="text")).cpu().numpy()
        m3h_abl_txt = compute_metrics(m3h_y_in, (m3h_txt_probs >= best_thresh).astype(int), m3h_txt_probs)
        m3h_abl_img = compute_metrics(m3h_y_in, (m3h_img_probs >= best_thresh).astype(int), m3h_img_probs)

    ablation_dict = {
        "pridemm_test": {
            "both_modalities": pridemm_test_metrics,
            "text_only_zero_img": pride_abl_txt,
            "image_only_zero_txt": pride_abl_img
        },
        "multi3hate": {
            "both_modalities": m3h_overall,
            "text_only_zero_img": m3h_abl_txt,
            "image_only_zero_txt": m3h_abl_img
        }
    }
    with open(EXP_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_dict, f, indent=2)

    # 7. Anti-Cheating & Modality Shuffling Sanity Checks
    print("\n--- Anti-Cheating Modality Shuffling Sanity Checks ---")
    with torch.no_grad():
        # Shuffled Images on PrideMM Test
        s_img_idx_pride = torch.randperm(test_v.size(0))
        shuff_img_pride_probs = torch.sigmoid(model(test_t, test_v[s_img_idx_pride])).cpu().numpy()
        m_shuff_img_pride = compute_metrics(test_y, (shuff_img_pride_probs >= best_thresh).astype(int), shuff_img_pride_probs)

        # Shuffled Text on PrideMM Test
        s_txt_idx_pride = torch.randperm(test_t.size(0))
        shuff_txt_pride_probs = torch.sigmoid(model(test_t[s_txt_idx_pride], test_v)).cpu().numpy()
        m_shuff_txt_pride = compute_metrics(test_y, (shuff_txt_pride_probs >= best_thresh).astype(int), shuff_txt_pride_probs)

        # Shuffled Images on Multi3Hate
        s_img_idx_m3h = torch.randperm(m3h_v.size(0))
        shuff_img_m3h_probs = torch.sigmoid(model(m3h_t, m3h_v[s_img_idx_m3h])).cpu().numpy()
        m_shuff_img_m3h = compute_metrics(m3h_y_in, (shuff_img_m3h_probs >= best_thresh).astype(int), shuff_img_m3h_probs)

        # Shuffled Text on Multi3Hate
        s_txt_idx_m3h = torch.randperm(m3h_t.size(0))
        shuff_txt_m3h_probs = torch.sigmoid(model(m3h_t[s_txt_idx_m3h], m3h_v)).cpu().numpy()
        m_shuff_txt_m3h = compute_metrics(m3h_y_in, (shuff_txt_m3h_probs >= best_thresh).astype(int), shuff_txt_m3h_probs)

    sanity_dict = {
        "pridemm_test": {
            "unshuffled_paired": pridemm_test_metrics,
            "shuffled_image": m_shuff_img_pride,
            "shuffled_text": m_shuff_txt_pride,
            "image_shuffle_f1_delta": round(m_shuff_img_pride["f1"] - pridemm_test_metrics["f1"], 4),
            "image_shuffle_auroc_delta": round(m_shuff_img_pride["auroc"] - pridemm_test_metrics["auroc"], 4)
        },
        "multi3hate": {
            "unshuffled_paired": m3h_overall,
            "shuffled_image": m_shuff_img_m3h,
            "shuffled_text": m_shuff_txt_m3h,
            "image_shuffle_f1_delta": round(m_shuff_img_m3h["f1"] - m3h_overall["f1"], 4),
            "image_shuffle_auroc_delta": round(m_shuff_img_m3h["auroc"] - m3h_overall["auroc"], 4)
        }
    }
    with open(EXP_DIR / "shuffle_sanity_results.json", "w", encoding="utf-8") as f:
        json.dump(sanity_dict, f, indent=2)

    # 8. Predictions CSV Export
    print("\n--- Exporting Predictions CSV ---")
    m3h_df = pd.read_csv(DATA_DIR / "multi3hate_predictions.csv")
    predictions_df = pd.DataFrame({
        "sample_id": [it["sample_id"] for it in m3h_data["items"]],
        "meme_id": [it["meme_id"] for it in m3h_data["items"]],
        "language": [it["language"] for it in m3h_data["items"]],
        "text": [it["text"] for it in m3h_data["items"]],
        "ground_truth_IN": m3h_y_in,
        "model_b_toxic_score": m3h_df["model_b_toxic_prob"].values,
        "model_d_old_score": m3h_df["model_d_prob"].values,
        "final_multimodal_prob": m3h_probs,
        "final_multimodal_pred": m3h_preds
    })
    predictions_df.to_csv(EXP_DIR / "predictions.csv", index=False)

    # 9. Extract Case Studies
    print("\n--- Extracting Cross-Modal Case Studies ---")
    # A. Emergent Hate (text score low < 0.35, GT=1, multimodal score >= best_thresh)
    emergent = predictions_df[(predictions_df["model_b_toxic_score"] < 0.35) & (predictions_df["ground_truth_IN"] == 1) & (predictions_df["final_multimodal_pred"] == 1)]
    # B. Contextual Inversion (text score high >= 0.45, GT=0, multimodal clears it: pred == 0)
    inversion = predictions_df[(predictions_df["model_b_toxic_score"] >= 0.45) & (predictions_df["ground_truth_IN"] == 0) & (predictions_df["final_multimodal_pred"] == 0)]
    # C. Explicit Hate (both high, GT=1)
    explicit = predictions_df[(predictions_df["model_b_toxic_score"] >= 0.50) & (predictions_df["ground_truth_IN"] == 1) & (predictions_df["final_multimodal_pred"] == 1)]
    # D. Failure Mode 1: False Positive (GT=0, but pred=1)
    fp_cases = predictions_df[(predictions_df["ground_truth_IN"] == 0) & (predictions_df["final_multimodal_pred"] == 1)]
    # E. Failure Mode 2: False Negative (GT=1, but pred=0)
    fn_cases = predictions_df[(predictions_df["ground_truth_IN"] == 1) & (predictions_df["final_multimodal_pred"] == 0)]

    def format_cases(sub_df, n=2):
        c_list = []
        for _, r in sub_df.head(n).iterrows():
            c_list.append({
                "meme_id": int(r["meme_id"]),
                "language": str(r["language"]),
                "text": str(r["text"]).encode("ascii", "backslashreplace").decode("ascii")[:120],
                "ground_truth": int(r["ground_truth_IN"]),
                "model_b_text_score": round(float(r["model_b_toxic_score"]), 4),
                "final_multimodal_score": round(float(r["final_multimodal_prob"]), 4),
                "prediction": int(r["final_multimodal_pred"])
            })
        return c_list

    case_studies = {
        "emergent_hate": {
            "description": "Text appears benign (Model B toxic score < 0.35) but multimodal context detects hate (GT=1, pred=1)",
            "count": len(emergent),
            "samples": format_cases(emergent, 3)
        },
        "contextual_inversion": {
            "description": "Text looks toxic (Model B >= 0.45) but multimodal context clears it as benign (GT=0, pred=0)",
            "count": len(inversion),
            "samples": format_cases(inversion, 3)
        },
        "explicit_hate": {
            "description": "Both text and multimodal indicate hate (GT=1, pred=1)",
            "count": len(explicit),
            "samples": format_cases(explicit, 3)
        },
        "multimodal_failure_false_positive": {
            "description": "Benign meme falsely flagged as hateful (GT=0, pred=1)",
            "count": len(fp_cases),
            "samples": format_cases(fp_cases, 3)
        },
        "multimodal_failure_false_negative": {
            "description": "Hateful meme missed by multimodal model (GT=1, pred=0)",
            "count": len(fn_cases),
            "samples": format_cases(fn_cases, 3)
        }
    }
    with open(EXP_DIR / "case_studies.json", "w", encoding="utf-8") as f:
        json.dump(case_studies, f, indent=2)

    print("\nAll evaluation artifacts generated successfully!")

if __name__ == "__main__":
    train_and_evaluate()
