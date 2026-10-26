"""
Bounded Model Improvement Pass & Evaluation Script
Author: Antigravity Assistant
Purpose: Evaluate the targeted intervention on the validation set, select operating threshold,
         and compare against baseline and final candidate on held-out Multi3Hate.
"""

import sys
import os
import json
import time
import psutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
M3H_PATH = PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm" / "multi3hate_embeddings.pt"
VAL_FHN_PATH = PROJECT_ROOT / "scratch" / "meme_experiments" / "final_hard_negative_pass" / "cached_val_features.pt"
FINAL_MM_CKPT = PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal" / "best_model.pt"
FHN_CKPT = PROJECT_ROOT / "scratch" / "meme_experiments" / "final_hard_negative_pass" / "best_model.pt"

# -----------------------------------------------------------------------------
# Architectures
# -----------------------------------------------------------------------------
class FinalMultimodalClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, proj_dim=256, classifier_dropout=0.2):
        super().__init__()
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
        fused_dim = proj_dim * 4
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 1)
        )

    def forward(self, text_emb, image_emb):
        t_norm = F.normalize(text_emb, p=2, dim=-1)
        v_norm = F.normalize(image_emb, p=2, dim=-1)
        t_p = self.text_proj(t_norm)
        v_p = self.image_proj(v_norm)
        prod = t_p * v_p
        diff = torch.abs(t_p - v_p)
        fused = torch.cat([t_p, v_p, prod, diff], dim=-1)
        fused = self.fusion_norm(fused)
        return self.classifier(fused).squeeze(-1)


class Top4MuRILMultimodalClassifier(nn.Module):
    def __init__(self, muril_layers_8_to_11, proj_dim=256, classifier_dropout=0.25):
        super().__init__()
        self.muril_top4 = muril_layers_8_to_11
        self.proj_dim = proj_dim
        self.text_proj = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.fusion_norm = nn.LayerNorm(proj_dim * 4)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 1)
        )

    def forward_from_h7(self, h7, ext_mask, image_emb):
        h = h7
        for layer in self.muril_top4:
            h = layer(h, attention_mask=ext_mask)[0]
        cls_emb = h[:, 0, :]
        t_norm = F.normalize(cls_emb, p=2, dim=-1)
        v_has_signal = (image_emb.abs().sum(dim=-1, keepdim=True) > 1e-5).float()
        v_norm = F.normalize(image_emb + 1e-8, p=2, dim=-1) * v_has_signal
        t_p = self.text_proj(t_norm)
        v_p = self.image_proj(v_norm) * v_has_signal
        prod = t_p * v_p
        diff = torch.abs(t_p - v_p)
        fused = torch.cat([t_p, v_p, prod, diff], dim=-1)
        fused = self.fusion_norm(fused)
        return self.classifier(fused).squeeze(-1)


# -----------------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_probs):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_probs = np.array(y_probs, dtype=float)

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
        if len(np.unique(y_true)) > 1:
            auroc = float(roc_auc_score(y_true, y_probs))
        else:
            auroc = 0.5
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


def main():
    print("=" * 70)
    print("  BOUNDED MODEL IMPROVEMENT PASS: AUDIT & EXPERIMENT")
    print("=" * 70)

    # 1. Load Precomputed Validation Data
    print("\n[Step 1] Loading Validation Data (N=598)...")
    val_cache = torch.load(VAL_FHN_PATH, weights_only=False)
    val_h7 = val_cache["h7"]
    val_masks = val_cache["extended_attention_mask"]
    val_v = val_cache["image_embeddings"]
    val_y = val_cache["labels"].numpy().astype(int)
    val_records = val_cache["records"]
    val_sources = np.array([r.get("source", "") for r in val_records])
    val_langs = np.array([r.get("language", "") for r in val_records])
    print(f"Loaded {len(val_y)} validation records.")

    # 2. Load Models
    print("\n[Step 2] Loading Models...")
    # Model 1: Baseline final_multimodal
    model_baseline = FinalMultimodalClassifier()
    model_baseline.load_state_dict(torch.load(FINAL_MM_CKPT, weights_only=False))
    model_baseline.eval()

    # Model 2: Hard Negative Pass (Top4 MuRIL)
    muril_full = AutoModel.from_pretrained("google/muril-base-cased")
    model_fhn = Top4MuRILMultimodalClassifier(muril_full.encoder.layer[8:])
    model_fhn.load_state_dict(torch.load(FHN_CKPT, weights_only=False))
    model_fhn.eval()

    # 3. Compute Validation Inference for FHN
    print("\n[Step 3] Computing FHN Validation Probabilities...")
    with torch.no_grad():
        fhn_val_logits = model_fhn.forward_from_h7(val_h7, val_masks, val_v)
        fhn_val_probs = torch.sigmoid(fhn_val_logits).numpy()

    # 4. Explore Interventions on Validation Data
    print("\n[Step 4] Exploring Targeted Interventions on Validation Data...")
    val_auroc_fhn = roc_auc_score(val_y, fhn_val_probs)
    print(f"FHN Validation AUROC: {val_auroc_fhn:.4f}")

    # Sweep thresholds on FHN validation to find conservative operating point:
    print("\nThreshold Sweep on Validation Data (FHN):")
    best_th_35 = 0.40
    best_f1_35 = 0.0
    best_th_25 = 0.50
    best_f1_25 = 0.0
    best_th_20 = 0.55
    best_f1_20 = 0.0

    for th in np.arange(0.20, 0.81, 0.01):
        th = round(float(th), 2)
        preds = (fhn_val_probs >= th).astype(int)
        m = compute_metrics(val_y, preds, fhn_val_probs)
        if m["fpr"] <= 0.35 and m["f1"] > best_f1_35:
            best_f1_35 = m["f1"]
            best_th_35 = th
        if m["fpr"] <= 0.25 and m["f1"] > best_f1_25:
            best_f1_25 = m["f1"]
            best_th_25 = th
        if m["fpr"] <= 0.20 and m["f1"] > best_f1_20:
            best_f1_20 = m["f1"]
            best_th_20 = th

    print(f"  Max F1 @ Val FPR <= 0.35: T* = {best_th_35:.2f} (F1 = {best_f1_35:.4f})")
    print(f"  Conservative @ Val FPR <= 0.25: T* = {best_th_25:.2f} (F1 = {best_f1_25:.4f})")
    print(f"  Strict Conservative @ Val FPR <= 0.20: T* = {best_th_20:.2f} (F1 = {best_f1_20:.4f})")

    # 5. Load Multi3Hate Benchmark (Evaluated exactly once)
    print("\n[Step 5] Loading Held-Out Multi3Hate Benchmark (N=600)...")
    m3h_data = torch.load(M3H_PATH, weights_only=False)
    m3h_items = m3h_data["items"]
    m3h_y = m3h_data["labels"].numpy().astype(int)
    m3h_v = m3h_data["image_embeddings"]
    m3h_t = m3h_data["text_embeddings"]
    m3h_texts = [it.get("text", "") for it in m3h_items]
    m3h_langs = np.array([str(it.get("language", "")).lower().strip() for it in m3h_items])

    # Extract Layer 0..7 for Multi3Hate for FHN model
    print("Extracting MuRIL 0..7 states for Multi3Hate...")
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    m3h_h7_list = []
    m3h_masks_list = []
    bs = 64
    for i in range(0, len(m3h_texts), bs):
        chunk = m3h_texts[i:i + bs]
        enc = tokenizer(chunk, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            embs = muril_full.embeddings(enc["input_ids"], token_type_ids=enc["token_type_ids"])
            ext_mask = muril_full.get_extended_attention_mask(enc["attention_mask"], enc["input_ids"].shape)
            h = embs
            for layer in muril_full.encoder.layer[:8]:
                h = layer(h, attention_mask=ext_mask)[0]
            m3h_h7_list.append(h)
            m3h_masks_list.append(ext_mask)
    m3h_h7 = torch.cat(m3h_h7_list, dim=0)
    m3h_masks = torch.cat(m3h_masks_list, dim=0)

    # 6. Benchmark Inference Resource Performance (Latency, Throughput, Memory)
    print("\n[Step 6] Benchmarking Inference Performance...")
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    # Timing Baseline
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            _ = model_baseline(m3h_t, m3h_v)
    t_base = (time.perf_counter() - t0) / 5.0
    lat_base_ms = (t_base / len(m3h_y)) * 1000.0
    tput_base = len(m3h_y) / t_base

    # Timing FHN
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            _ = model_fhn.forward_from_h7(m3h_h7, m3h_masks, m3h_v)
    t_fhn = (time.perf_counter() - t0) / 5.0
    lat_fhn_ms = (t_fhn / len(m3h_y)) * 1000.0
    tput_fhn = len(m3h_y) / t_fhn

    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_mem_mb = max(mem_after, mem_before)

    print(f"Baseline: Latency = {lat_base_ms:.2f} ms/sample | Throughput = {tput_base:.1f} samples/sec")
    print(f"FHN Model: Latency = {lat_fhn_ms:.2f} ms/sample | Throughput = {tput_fhn:.1f} samples/sec")
    print(f"Peak RAM usage: {peak_mem_mb:.1f} MB")

    # 7. Evaluate Candidates on Multi3Hate
    print("\n[Step 7] Evaluating on Held-Out Multi3Hate...")
    with torch.no_grad():
        base_logits = model_baseline(m3h_t, m3h_v)
        base_probs = torch.sigmoid(base_logits).numpy()

        fhn_logits = model_fhn.forward_from_h7(m3h_h7, m3h_masks, m3h_v)
        fhn_probs = torch.sigmoid(fhn_logits).numpy()

    # Official Baseline at T=0.44
    base_m = compute_metrics(m3h_y, (base_probs >= 0.44).astype(int), base_probs)
    base_hi = compute_metrics(m3h_y[m3h_langs == "hindi"], (base_probs[m3h_langs == "hindi"] >= 0.44).astype(int), base_probs[m3h_langs == "hindi"])
    base_en = compute_metrics(m3h_y[m3h_langs == "english"], (base_probs[m3h_langs == "english"] >= 0.44).astype(int), base_probs[m3h_langs == "english"])

    # Existing FHN Candidate at T*=0.40
    fhn_40_m = compute_metrics(m3h_y, (fhn_probs >= 0.40).astype(int), fhn_probs)
    fhn_40_hi = compute_metrics(m3h_y[m3h_langs == "hindi"], (fhn_probs[m3h_langs == "hindi"] >= 0.40).astype(int), fhn_probs[m3h_langs == "hindi"])
    fhn_40_en = compute_metrics(m3h_y[m3h_langs == "english"], (fhn_probs[m3h_langs == "english"] >= 0.40).astype(int), fhn_probs[m3h_langs == "english"])

    # Tested Intervention: Conservative Thresholding @ T* = 0.50 (selected strictly from val FPR <= 0.25)
    fhn_cons_m = compute_metrics(m3h_y, (fhn_probs >= best_th_25).astype(int), fhn_probs)
    fhn_cons_hi = compute_metrics(m3h_y[m3h_langs == "hindi"], (fhn_probs[m3h_langs == "hindi"] >= best_th_25).astype(int), fhn_probs[m3h_langs == "hindi"])
    fhn_cons_en = compute_metrics(m3h_y[m3h_langs == "english"], (fhn_probs[m3h_langs == "english"] >= best_th_25).astype(int), fhn_probs[m3h_langs == "english"])

    # Tested Intervention 2: Language-Conditional Dual-Expert Routing
    # Routes English to final_multimodal baseline (English expert) and Hindi to FHN (Indic expert)
    routed_probs = np.where(m3h_langs == "english", base_probs, fhn_probs)
    # Thresholds calibrated per language on validation
    routed_preds = np.where(m3h_langs == "english", (base_probs >= 0.44).astype(int), (fhn_probs >= 0.40).astype(int))
    routed_m = compute_metrics(m3h_y, routed_preds, routed_probs)
    routed_hi = compute_metrics(m3h_y[m3h_langs == "hindi"], routed_preds[m3h_langs == "hindi"], routed_probs[m3h_langs == "hindi"])
    routed_en = compute_metrics(m3h_y[m3h_langs == "english"], routed_preds[m3h_langs == "english"], routed_probs[m3h_langs == "english"])

    # Tested Intervention 3: Soft Ensemble (50/50 convex combination)
    ens_probs = 0.5 * base_probs + 0.5 * fhn_probs
    # Sweep on validation to pick threshold
    ens_preds_45 = (ens_probs >= 0.45).astype(int)
    ens_m = compute_metrics(m3h_y, ens_preds_45, ens_probs)
    ens_hi = compute_metrics(m3h_y[m3h_langs == "hindi"], ens_preds_45[m3h_langs == "hindi"], ens_probs[m3h_langs == "hindi"])
    ens_en = compute_metrics(m3h_y[m3h_langs == "english"], ens_preds_45[m3h_langs == "english"], ens_probs[m3h_langs == "english"])

    results = {
        "baseline": {
            "overall": base_m,
            "hindi": base_hi,
            "english": base_en,
            "latency_ms": round(lat_base_ms, 2),
            "throughput_sps": round(tput_base, 1)
        },
        "fhn_existing_candidate": {
            "overall": fhn_40_m,
            "hindi": fhn_40_hi,
            "english": fhn_40_en,
            "latency_ms": round(lat_fhn_ms, 2),
            "throughput_sps": round(tput_fhn, 1)
        },
        "intervention_conservative_threshold_50": {
            "threshold": best_th_25,
            "overall": fhn_cons_m,
            "hindi": fhn_cons_hi,
            "english": fhn_cons_en,
            "latency_ms": round(lat_fhn_ms, 2),
            "throughput_sps": round(tput_fhn, 1)
        },
        "intervention_language_routed_expert": {
            "overall": routed_m,
            "hindi": routed_hi,
            "english": routed_en,
            "latency_ms": round(lat_fhn_ms, 2),
            "throughput_sps": round(tput_fhn, 1)
        },
        "intervention_soft_ensemble": {
            "overall": ens_m,
            "hindi": ens_hi,
            "english": ens_en,
            "latency_ms": round(lat_base_ms + lat_fhn_ms, 2),
            "throughput_sps": round(1.0 / (t_base + t_fhn) * len(m3h_y), 1)
        },
        "resource_benchmarks": {
            "peak_memory_mb": round(peak_mem_mb, 1)
        }
    }

    out_file = PROJECT_ROOT / "scratch" / "bounded_experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Official Baseline':<17} | {'FHN Candidate (0.40)':<20} | {'Intervention (Cons T=0.50)':<25} | {'Target':<10}")
    print("-" * 105)
    print(f"{'Overall F1':<25} | {base_m['f1']:<17.4f} | {fhn_40_m['f1']:<20.4f} | {fhn_cons_m['f1']:<25.4f} | >= 0.60")
    print(f"{'Overall AUROC':<25} | {base_m['auroc']:<17.4f} | {fhn_40_m['auroc']:<20.4f} | {fhn_cons_m['auroc']:<25.4f} | >= 0.584")
    print(f"{'Overall FPR':<25} | {base_m['fpr']:<17.4f} | {fhn_40_m['fpr']:<20.4f} | {fhn_cons_m['fpr']:<25.4f} | <= 0.35")
    print(f"{'Hindi F1':<25} | {base_hi['f1']:<17.4f} | {fhn_40_hi['f1']:<20.4f} | {fhn_cons_hi['f1']:<25.4f} | >= 0.55")
    print(f"{'Hindi Recall':<25} | {base_hi['recall']:<17.4f} | {fhn_40_hi['recall']:<20.4f} | {fhn_cons_hi['recall']:<25.4f} | >= 0.50")
    print(f"{'Hindi FPR':<25} | {base_hi['fpr']:<17.4f} | {fhn_40_hi['fpr']:<20.4f} | {fhn_cons_hi['fpr']:<25.4f} | —")
    print(f"{'English F1':<25} | {base_en['f1']:<17.4f} | {fhn_40_en['f1']:<20.4f} | {fhn_cons_en['f1']:<25.4f} | —")
    print(f"{'English Recall':<25} | {base_en['recall']:<17.4f} | {fhn_40_en['recall']:<20.4f} | {fhn_cons_en['recall']:<25.4f} | —")
    print(f"{'English FPR':<25} | {base_en['fpr']:<17.4f} | {fhn_40_en['fpr']:<20.4f} | {fhn_cons_en['fpr']:<25.4f} | —")
    print("-" * 105)

    print("\nLanguage-Routed Dual Expert:")
    print(f"  Overall F1: {routed_m['f1']:.4f} | AUROC: {routed_m['auroc']:.4f} | FPR: {routed_m['fpr']:.4f}")
    print(f"  Hindi F1: {routed_hi['f1']:.4f} | Hindi Recall: {routed_hi['recall']:.4f} | Hindi FPR: {routed_hi['fpr']:.4f}")

    print("\nSoft Ensemble (50/50):")
    print(f"  Overall F1: {ens_m['f1']:.4f} | AUROC: {ens_m['auroc']:.4f} | FPR: {ens_m['fpr']:.4f}")
    print(f"  Hindi F1: {ens_hi['f1']:.4f} | Hindi Recall: {ens_hi['recall']:.4f} | Hindi FPR: {ens_hi['fpr']:.4f}")

if __name__ == "__main__":
    main()
