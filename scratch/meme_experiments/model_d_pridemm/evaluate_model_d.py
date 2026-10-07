import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import MuRILForMultiLabelToxicity
from scratch.meme_experiments.model_d_pridemm.train_model_d import ModelDFusionMLP

OUT_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm"
MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"

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

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}...")

    # Load Model D Checkpoint
    ckpt_path = OUT_DIR / "model_d_fusion.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_d = ModelDFusionMLP(in_features=1280, hidden_dim=256, dropout=0.2).to(device)
    model_d.load_state_dict(ckpt["model_state_dict"])
    model_d.eval()

    # Load precomputed embeddings
    pridemm_test_data = torch.load(OUT_DIR / "pridemm_test_embeddings.pt", weights_only=False)
    m3h_data = torch.load(OUT_DIR / "multi3hate_embeddings.pt", weights_only=False)

    # 1. EVALUATION ON PRIDEMM TEST SET (N=507)
    print("\n=== PRIDEMM TEST EVALUATION (N=507) ===")
    X_pridemm_test = torch.cat([pridemm_test_data["text_embeddings"], pridemm_test_data["image_embeddings"]], dim=-1).to(device)
    y_pridemm_test = pridemm_test_data["labels"].numpy()

    with torch.no_grad():
        logits_d_pridemm = model_d(X_pridemm_test)
        probs_d_pridemm = torch.sigmoid(logits_d_pridemm).cpu().numpy()
        preds_d_pridemm = (probs_d_pridemm >= 0.5).astype(int)

    metrics_d_pridemm = compute_metrics(y_pridemm_test, preds_d_pridemm, probs_d_pridemm)
    print("Model D on PrideMM Test:", json.dumps(metrics_d_pridemm, indent=2))

    # Also evaluate Model B text-only on PrideMM test text
    print("\nRunning Model B on PrideMM Test text...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/muril-base-cased")
    model_b = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6).to(device)
    model_b.load_state_dict(torch.load(MODEL_B_CHECKPOINT, map_location=device, weights_only=True))
    model_b.eval()

    pridemm_texts = [it["text"] for it in pridemm_test_data["items"]]
    b_probs_list = []
    b_hate_probs_list = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(pridemm_texts), batch_size):
            batch_txt = pridemm_texts[i : i + batch_size]
            enc = tok(batch_txt, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
            out_logits = model_b(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs_b = torch.sigmoid(out_logits).cpu().numpy()
            b_probs_list.extend(probs_b[:, 0].tolist()) # toxic head (head 0)
            b_hate_probs_list.extend(probs_b[:, 5].tolist()) # identity_hate head (head 5)

    probs_b_toxic = np.array(b_probs_list)
    probs_b_hate = np.array(b_hate_probs_list)
    
    # Model B baseline 1: toxic >= 0.45
    preds_b_toxic = (probs_b_toxic >= 0.45).astype(int)
    metrics_b_toxic = compute_metrics(y_pridemm_test, preds_b_toxic, probs_b_toxic)

    # Model B baseline 2: identity_hate >= 0.60
    preds_b_hate = (probs_b_hate >= 0.60).astype(int)
    metrics_b_hate = compute_metrics(y_pridemm_test, preds_b_hate, probs_b_hate)

    print("Model B (toxic >= 0.45) on PrideMM Test:", json.dumps(metrics_b_toxic, indent=2))
    print("Model B (identity_hate >= 0.60) on PrideMM Test:", json.dumps(metrics_b_hate, indent=2))

    # Save PrideMM predictions CSV
    df_pridemm_preds = pd.DataFrame({
        "sample_id": pridemm_test_data["sample_ids"],
        "text": pridemm_texts,
        "ground_truth": y_pridemm_test,
        "model_b_toxic_prob": probs_b_toxic,
        "model_b_hate_prob": probs_b_hate,
        "model_d_prob": probs_d_pridemm,
        "model_d_pred": preds_d_pridemm
    })
    df_pridemm_preds.to_csv(OUT_DIR / "pridemm_test_predictions.csv", index=False)

    # 2. EVALUATION ON HELD-OUT MULTI3HATE (N=600)
    print("\n=== MULTI3HATE EVALUATION (N=600) ===")
    X_m3h = torch.cat([m3h_data["text_embeddings"], m3h_data["image_embeddings"]], dim=-1).to(device)
    y_m3h_in = np.array([it["IN"] for it in m3h_data["items"]], dtype=int)
    y_m3h_us = np.array([it["US"] for it in m3h_data["items"]], dtype=int)
    y_m3h_de = np.array([it["DE"] for it in m3h_data["items"]], dtype=int)
    y_m3h_mx = np.array([it["MX"] for it in m3h_data["items"]], dtype=int)
    y_m3h_cn = np.array([it["CN"] for it in m3h_data["items"]], dtype=int)
    langs_m3h = np.array([it["language"] for it in m3h_data["items"]])

    with torch.no_grad():
        logits_d_m3h = model_d(X_m3h)
        probs_d_m3h = torch.sigmoid(logits_d_m3h).cpu().numpy()
        preds_d_m3h = (probs_d_m3h >= 0.5).astype(int)

    metrics_d_m3h_in = compute_metrics(y_m3h_in, preds_d_m3h, probs_d_m3h)
    print("Model D on Multi3Hate (IN Ground Truth):", json.dumps(metrics_d_m3h_in, indent=2))

    # Slices: English vs Hindi
    mask_en = (langs_m3h == "english")
    mask_hi = (langs_m3h == "hindi")
    metrics_en = compute_metrics(y_m3h_in[mask_en], preds_d_m3h[mask_en], probs_d_m3h[mask_en])
    metrics_hi = compute_metrics(y_m3h_in[mask_hi], preds_d_m3h[mask_hi], probs_d_m3h[mask_hi])

    # Cultural consensus targets
    metrics_us = compute_metrics(y_m3h_us, preds_d_m3h, probs_d_m3h)
    metrics_de = compute_metrics(y_m3h_de, preds_d_m3h, probs_d_m3h)
    metrics_mx = compute_metrics(y_m3h_mx, preds_d_m3h, probs_d_m3h)
    metrics_cn = compute_metrics(y_m3h_cn, preds_d_m3h, probs_d_m3h)

    # Multi3Hate Model B baseline inference
    m3h_texts = [it["text"] for it in m3h_data["items"]]
    m3h_b_toxic = []
    m3h_b_hate = []
    with torch.no_grad():
        for i in range(0, len(m3h_texts), batch_size):
            batch_txt = m3h_texts[i : i + batch_size]
            enc = tok(batch_txt, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
            out_logits = model_b(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs_b = torch.sigmoid(out_logits).cpu().numpy()
            m3h_b_toxic.extend(probs_b[:, 0].tolist())
            m3h_b_hate.extend(probs_b[:, 5].tolist())

    probs_m3h_b_toxic = np.array(m3h_b_toxic)
    probs_m3h_b_hate = np.array(m3h_b_hate)
    preds_m3h_b_toxic = (probs_m3h_b_toxic >= 0.45).astype(int)
    preds_m3h_b_hate = (probs_m3h_b_hate >= 0.60).astype(int)

    metrics_m3h_b_toxic = compute_metrics(y_m3h_in, preds_m3h_b_toxic, probs_m3h_b_toxic)
    metrics_m3h_b_hate = compute_metrics(y_m3h_in, preds_m3h_b_hate, probs_m3h_b_hate)

    # Save Multi3Hate predictions CSV
    df_m3h_preds = pd.DataFrame({
        "sample_id": m3h_data["sample_ids"],
        "meme_id": [it["meme_id"] for it in m3h_data["items"]],
        "language": langs_m3h,
        "text": m3h_texts,
        "IN": y_m3h_in,
        "US": y_m3h_us,
        "DE": y_m3h_de,
        "MX": y_m3h_mx,
        "CN": y_m3h_cn,
        "model_b_toxic_prob": probs_m3h_b_toxic,
        "model_b_hate_prob": probs_m3h_b_hate,
        "model_d_prob": probs_d_m3h,
        "model_d_pred": preds_d_m3h
    })
    df_m3h_preds.to_csv(OUT_DIR / "multi3hate_predictions.csv", index=False)

    # 3. ABLATION ANALYSIS
    print("\n=== MODALITY ABLATION ANALYSIS ===")
    # A. Text Only (zero out image embeddings)
    X_text_only = torch.cat([pridemm_test_data["text_embeddings"], torch.zeros_like(pridemm_test_data["image_embeddings"])], dim=-1).to(device)
    with torch.no_grad():
        p_to = torch.sigmoid(model_d(X_text_only)).cpu().numpy()
        metrics_pridemm_text_only = compute_metrics(y_pridemm_test, (p_to >= 0.5).astype(int), p_to)

    # B. Image Only (zero out text embeddings)
    X_img_only = torch.cat([torch.zeros_like(pridemm_test_data["text_embeddings"]), pridemm_test_data["image_embeddings"]], dim=-1).to(device)
    with torch.no_grad():
        p_io = torch.sigmoid(model_d(X_img_only)).cpu().numpy()
        metrics_pridemm_img_only = compute_metrics(y_pridemm_test, (p_io >= 0.5).astype(int), p_io)

    # Multi3Hate Ablations
    X_m3h_text_only = torch.cat([m3h_data["text_embeddings"], torch.zeros_like(m3h_data["image_embeddings"])], dim=-1).to(device)
    with torch.no_grad():
        p_m3h_to = torch.sigmoid(model_d(X_m3h_text_only)).cpu().numpy()
        metrics_m3h_text_only = compute_metrics(y_m3h_in, (p_m3h_to >= 0.5).astype(int), p_m3h_to)

    X_m3h_img_only = torch.cat([torch.zeros_like(m3h_data["text_embeddings"]), m3h_data["image_embeddings"]], dim=-1).to(device)
    with torch.no_grad():
        p_m3h_io = torch.sigmoid(model_d(X_m3h_img_only)).cpu().numpy()
        metrics_m3h_img_only = compute_metrics(y_m3h_in, (p_m3h_io >= 0.5).astype(int), p_m3h_io)

    ablation_results = {
        "pridemm_test": {
            "both_modalities": metrics_d_pridemm,
            "text_only_zero_img": metrics_pridemm_text_only,
            "image_only_zero_text": metrics_pridemm_img_only
        },
        "multi3hate": {
            "both_modalities": metrics_d_m3h_in,
            "text_only_zero_img": metrics_m3h_text_only,
            "image_only_zero_text": metrics_m3h_img_only
        }
    }
    with open(OUT_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # 4. ANTI-CHEATING / SHUFFLED MODALITY SANITY CHECKS
    print("\n=== SHUFFLED MODALITY SANITY CHECKS ===")
    np.random.seed(42)
    # PrideMM Shuffled
    shuffled_img_idx = np.random.permutation(len(pridemm_test_data["image_embeddings"]))
    X_shuffled_img = torch.cat([pridemm_test_data["text_embeddings"], pridemm_test_data["image_embeddings"][shuffled_img_idx]], dim=-1).to(device)
    with torch.no_grad():
        p_shuff_img = torch.sigmoid(model_d(X_shuffled_img)).cpu().numpy()
        metrics_shuff_img = compute_metrics(y_pridemm_test, (p_shuff_img >= 0.5).astype(int), p_shuff_img)

    shuffled_txt_idx = np.random.permutation(len(pridemm_test_data["text_embeddings"]))
    X_shuffled_txt = torch.cat([pridemm_test_data["text_embeddings"][shuffled_txt_idx], pridemm_test_data["image_embeddings"]], dim=-1).to(device)
    with torch.no_grad():
        p_shuff_txt = torch.sigmoid(model_d(X_shuffled_txt)).cpu().numpy()
        metrics_shuff_txt = compute_metrics(y_pridemm_test, (p_shuff_txt >= 0.5).astype(int), p_shuff_txt)

    # Multi3Hate Shuffled
    m3h_shuff_img_idx = np.random.permutation(len(m3h_data["image_embeddings"]))
    X_m3h_shuff_img = torch.cat([m3h_data["text_embeddings"], m3h_data["image_embeddings"][m3h_shuff_img_idx]], dim=-1).to(device)
    with torch.no_grad():
        p_m3h_shuff_img = torch.sigmoid(model_d(X_m3h_shuff_img)).cpu().numpy()
        metrics_m3h_shuff_img = compute_metrics(y_m3h_in, (p_m3h_shuff_img >= 0.5).astype(int), p_m3h_shuff_img)

    sanity_results = {
        "pridemm_test": {
            "unshuffled_paired": metrics_d_pridemm,
            "shuffled_image": metrics_shuff_img,
            "shuffled_text": metrics_shuff_txt,
            "image_shuffle_f1_delta": round(metrics_shuff_img["f1"] - metrics_d_pridemm["f1"], 4),
            "text_shuffle_f1_delta": round(metrics_shuff_txt["f1"] - metrics_d_pridemm["f1"], 4)
        },
        "multi3hate": {
            "unshuffled_paired": metrics_d_m3h_in,
            "shuffled_image": metrics_m3h_shuff_img,
            "image_shuffle_f1_delta": round(metrics_m3h_shuff_img["f1"] - metrics_d_m3h_in["f1"], 4)
        }
    }
    with open(OUT_DIR / "shuffled_sanity_results.json", "w") as f:
        json.dump(sanity_results, f, indent=2)

    # Compile all metrics into master metrics.json
    all_metrics = {
        "pridemm_test": {
            "model_d": metrics_d_pridemm,
            "model_b_toxic_head": metrics_b_toxic,
            "model_b_identity_hate_head": metrics_b_hate
        },
        "multi3hate": {
            "model_d": {
                "overall_in": metrics_d_m3h_in,
                "english": metrics_en,
                "hindi": metrics_hi,
                "us_consensus": metrics_us,
                "de_consensus": metrics_de,
                "mx_consensus": metrics_mx,
                "cn_consensus": metrics_cn
            },
            "model_b_toxic_head": metrics_m3h_b_toxic,
            "model_b_identity_hate_head": metrics_m3h_b_hate
        },
        "ablation": ablation_results,
        "sanity": sanity_results
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nEvaluation pipeline completed successfully! Saved all artifacts to:", OUT_DIR)

if __name__ == "__main__":
    run_evaluation()
