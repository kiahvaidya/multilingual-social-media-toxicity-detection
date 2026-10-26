"""
Forensic Diagnosis Script for Multimodal Toxicity Detection Models (Model C and Model C-2).
Performs:
1. Weight inspection & gradient/parameter norm analysis
2. Embedding magnitude, distribution, and scale comparison (MuRIL vs CLIP)
3. Controlled ablations on EXISTING checkpoints:
   A. Text-only
   B. Image-only
   C. Text + Image with text zeroed
   D. Text + Image with image zeroed
   E. Shuffled images with correct text
   F. Shuffled text with correct image
   G. Same text with different images (sensitivity to visual modality)
   H. Same image with different text (sensitivity to text modality)
4. Logit distributions across partitions (benign, hateful, contextual inversion, emergent toxicity)
5. Cross-modal context set detailed predictions
6. Multi3Hate annotation & ontology alignment analysis
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoProcessor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import MuRILForMultiLabelToxicity, DEFAULT_LABELS
from scripts.train_meme_multimodal_baseline import MultimodalMemeToxicityModel
from scripts.train_meme_multimodal_calibrated import CalibratedMultimodalMemeModel

EXP_C_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_c"
EXP_C2_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_c2"
MODEL_A_PATH = _PROJECT_ROOT / "muril_toxicity_model.pt"
MODEL_B_PATH = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
MODEL_C_PATH = EXP_C_DIR / "muril_model_c.pt"
MODEL_C2_PATH = EXP_C2_DIR / "muril_model_c2.pt"
METRICS_C2_PATH = EXP_C2_DIR / "model_c2_training_metrics.json"

MULTI3HATE_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "multi3hate"
MULTI3HATE_IMAGES_BASE = MULTI3HATE_DIR / "Multi3Hate_repo" / "data" / "memes"

OUT_JSON = _PROJECT_ROOT / "scratch" / "forensic_diagnosis_results.json"


def compute_metrics(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pred_pos": int(np.sum(preds == 1)),
        "pred_neg": int(np.sum(preds == 0)),
        "score_stats": {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "p25": round(float(np.percentile(scores, 25)), 4),
            "median": round(float(np.median(scores)), 4),
            "p75": round(float(np.percentile(scores, 75)), 4),
        }
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running forensic diagnosis on device: {device}")

    # Load Multi3Hate data
    annot_csv = MULTI3HATE_DIR / "final_annotations.csv"
    hi_csv = MULTI3HATE_DIR / "hi_captions.csv"
    en_csv = MULTI3HATE_DIR / "en_captions.csv"

    df_annot = pd.read_csv(annot_csv)
    df_hi = pd.read_csv(hi_csv)
    df_en = pd.read_csv(en_csv)
    hi_map = {r["Meme ID"]: r for _, r in df_hi.iterrows()}
    en_map = {r["Meme ID"]: r for _, r in df_en.iterrows()}

    instances = []
    for _, row in df_annot.iterrows():
        mid = row["Meme ID"]
        hate_in = int(row["IN"]) if pd.notna(row.get("IN")) else None
        hate_us = int(row["US"]) if pd.notna(row.get("US")) else None

        # Hindi
        hi_row = hi_map.get(mid, {})
        hi_text = str(hi_row.get("Translation", "")).replace("<sep>", " ").strip()
        hi_matches = list((MULTI3HATE_IMAGES_BASE / "hi").rglob(f"{mid}.jpg"))
        if hi_text and hi_matches and hi_matches[0].exists():
            instances.append({
                "sample_id": f"multi3hate_hi_{mid}",
                "meme_id": mid,
                "language": "hindi",
                "text": hi_text,
                "image_path": str(hi_matches[0]),
                "template_name": str(hi_row.get("Template Name", "")),
                "hate_IN": hate_in,
                "hate_US": hate_us,
            })

        # English
        en_row = en_map.get(mid, {})
        en_text = str(en_row.get("Original (English)", "")).replace("<sep>", " ").strip()
        en_matches = list((MULTI3HATE_IMAGES_BASE / "en").rglob(f"{mid}.jpg"))
        if en_text and en_matches and en_matches[0].exists():
            instances.append({
                "sample_id": f"multi3hate_en_{mid}",
                "meme_id": mid,
                "language": "english",
                "text": en_text,
                "image_path": str(en_matches[0]),
                "template_name": str(en_row.get("Template Name", "")),
                "hate_IN": hate_in,
                "hate_US": hate_us,
            })

    print(f"Loaded {len(instances)} Multi3Hate instances.")

    # Load Tokenizer & Vision Processor
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load Model B, Model C, Model C-2
    model_b = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=device, weights_only=True))
    model_b.to(device).eval()

    model_c = MultimodalMemeToxicityModel(num_labels=6)
    model_c.load_state_dict(torch.load(MODEL_C_PATH, map_location=device, weights_only=True))
    model_c.to(device).eval()

    model_c2 = CalibratedMultimodalMemeModel(num_labels=6)
    model_c2.load_state_dict(torch.load(MODEL_C2_PATH, map_location=device, weights_only=True))
    model_c2.to(device).eval()

    with open(METRICS_C2_PATH, "r", encoding="utf-8") as f:
        meta_c2 = json.load(f)
    platt_a = meta_c2["calibration"]["platt_a"]
    platt_b = meta_c2["calibration"]["platt_b"]
    val_thresh = meta_c2["thresholds"]["validation_selected_threshold"]

    # 1. Inspect weight statistics
    print("\n--- Weight & Parameter Inspection ---")
    def inspect_weights(model, name):
        stats = {}
        for p_name, p in model.named_parameters():
            if any(k in p_name for k in ["proj", "mlp", "gate", "classifier"]):
                data = p.data.cpu().numpy()
                stats[p_name] = {
                    "shape": list(data.shape),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "norm": float(np.linalg.norm(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                }
        return stats

    c_weights = inspect_weights(model_c, "Model C")
    c2_weights = inspect_weights(model_c2, "Model C-2")

    # 2. Extract embeddings and evaluate on Multi3Hate
    print("Extracting representations and running ablations across 600 samples...")
    y_in = np.array([inst["hate_IN"] for inst in instances])
    y_us = np.array([inst["hate_US"] for inst in instances])

    # Pre-extract all embeddings for fast ablation
    all_text_inputs = []
    all_pixel_inputs = []
    all_h_text_c = []
    all_h_vision_c = []
    all_h_text_c2 = []
    all_h_vision_c2 = []

    scores_b = []
    scores_c_multi = []
    scores_c_img = []
    scores_c2_raw = []
    scores_c2_cal = []
    gates_c2 = []

    with torch.no_grad():
        for i, inst in enumerate(instances):
            text = inst["text"]
            img_path = inst["image_path"]

            enc = tokenizer(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            img = Image.open(img_path).convert("RGB")
            img_inputs = vision_processor(images=img, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(device)

            # Model B
            logits_b = model_b(input_ids=input_ids, attention_mask=attention_mask)
            sb = float(torch.sigmoid(logits_b[0, 0]).cpu().item())
            scores_b.append(sb)

            # Model C embeddings & logits
            ht_c = model_c.extract_text_embedding(input_ids, attention_mask)
            hv_c = model_c.extract_vision_embedding(pixel_values)
            all_h_text_c.append(ht_c)
            all_h_vision_c.append(hv_c)

            fused_c = model_c.fusion_mlp(torch.cat([ht_c, hv_c], dim=-1))
            sc = float(torch.sigmoid(model_c.classifier(fused_c)[0, 0]).cpu().item())
            scores_c_multi.append(sc)

            sc_img = float(torch.sigmoid(model_c.vision_only_classifier(hv_c)[0, 0]).cpu().item())
            scores_c_img.append(sc_img)

            # Model C-2 embeddings & logits
            ht_c2 = model_c2.extract_text_embedding(input_ids, attention_mask)
            hv_c2 = model_c2.extract_vision_embedding(pixel_values)
            all_h_text_c2.append(ht_c2)
            all_h_vision_c2.append(hv_c2)

            logits_c2, g_val = model_c2(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, mode="gated", return_gate=True)
            sc2_raw = float(torch.sigmoid(logits_c2[0, 0]).cpu().item())
            scores_c2_raw.append(sc2_raw)
            gates_c2.append(float(g_val.cpu().item()))

            # Platt calibrate
            eps = 1e-7
            raw_logit = np.log((sc2_raw + eps) / (1.0 - sc2_raw + eps))
            sc2_cal = float(1.0 / (1.0 + np.exp(-(platt_a * raw_logit + platt_b))))
            scores_c2_cal.append(sc2_cal)

    scores_b = np.array(scores_b)
    scores_c_multi = np.array(scores_c_multi)
    scores_c_img = np.array(scores_c_img)
    scores_c2_raw = np.array(scores_c2_raw)
    scores_c2_cal = np.array(scores_c2_cal)
    gates_c2 = np.array(gates_c2)

    # Convert embedding lists to tensors
    h_text_c_tensor = torch.cat(all_h_text_c, dim=0)       # (600, 512)
    h_vision_c_tensor = torch.cat(all_h_vision_c, dim=0)   # (600, 512)
    h_text_c2_tensor = torch.cat(all_h_text_c2, dim=0)     # (600, 512)
    h_vision_c2_tensor = torch.cat(all_h_vision_c2, dim=0) # (600, 512)

    print("\n--- Representation Magnitude & Cosine Analysis ---")
    norm_ht_c = torch.norm(h_text_c_tensor, dim=-1).mean().item()
    norm_hv_c = torch.norm(h_vision_c_tensor, dim=-1).mean().item()
    cos_c = torch.nn.functional.cosine_similarity(h_text_c_tensor, h_vision_c_tensor, dim=-1).mean().item()

    norm_ht_c2 = torch.norm(h_text_c2_tensor, dim=-1).mean().item()
    norm_hv_c2 = torch.norm(h_vision_c2_tensor, dim=-1).mean().item()
    cos_c2 = torch.nn.functional.cosine_similarity(h_text_c2_tensor, h_vision_c2_tensor, dim=-1).mean().item()

    embedding_analysis = {
        "model_c": {
            "text_embedding_mean_norm": round(norm_ht_c, 4),
            "vision_embedding_mean_norm": round(norm_hv_c, 4),
            "norm_ratio_vision_to_text": round(norm_hv_c / (norm_ht_c + 1e-8), 4),
            "mean_cosine_similarity": round(cos_c, 4),
        },
        "model_c2": {
            "text_embedding_mean_norm": round(norm_ht_c2, 4),
            "vision_embedding_mean_norm": round(norm_hv_c2, 4),
            "norm_ratio_vision_to_text": round(norm_hv_c2 / (norm_ht_c2 + 1e-8), 4),
            "mean_cosine_similarity": round(cos_c2, 4),
        }
    }
    print("Embedding Stats:", json.dumps(embedding_analysis, indent=2))

    # Now Controlled Ablations on Model C-2 and Model C
    print("\n--- Running Controlled Ablations A-H ---")
    ablations = {}

    # A. Text-only (Model B)
    ablations["A_text_only_model_b"] = compute_metrics(y_in, scores_b, 0.45)

    # B. Image-only (Model C vision-only head vs Model C2 vision-only head)
    # Model C2 vision only:
    with torch.no_grad():
        logits_img_c2 = model_c2.vision_classifier(h_vision_c2_tensor)[:, 0].cpu().numpy()
        scores_img_c2 = 1.0 / (1.0 + np.exp(-logits_img_c2))
    ablations["B_image_only_c2"] = compute_metrics(y_in, scores_img_c2, 0.45)
    ablations["B_image_only_c"] = compute_metrics(y_in, scores_c_img, 0.45)

    # Function to run forward on Model C-2 gated head with arbitrary h_text and h_vision
    def run_c2_gated(ht, hv):
        with torch.no_grad():
            h_paired = torch.cat([ht, hv], dim=-1)
            gate = model_c2.gate_network(h_paired)
            h_fused = gate * ht + (1.0 - gate) * hv
            logits = model_c2.gated_classifier(h_fused)[:, 0].cpu().numpy()
            raw_s = 1.0 / (1.0 + np.exp(-logits))
            eps = 1e-7
            raw_logit = np.log((raw_s + eps) / (1.0 - raw_s + eps))
            cal_s = 1.0 / (1.0 + np.exp(-(platt_a * raw_logit + platt_b)))
            return raw_s, cal_s, gate.mean(dim=-1).cpu().numpy()

    # Function to run forward on Model C with arbitrary h_text and h_vision
    def run_c_multi(ht, hv):
        with torch.no_grad():
            h_paired = torch.cat([ht, hv], dim=-1)
            fused = model_c.fusion_mlp(h_paired)
            logits = model_c.classifier(fused)[:, 0].cpu().numpy()
            return 1.0 / (1.0 + np.exp(-logits))

    # C. Text + Image with text zeroed (h_text = 0)
    zero_text_c2 = torch.zeros_like(h_text_c2_tensor)
    raw_s_c, cal_s_c, _ = run_c2_gated(zero_text_c2, h_vision_c2_tensor)
    ablations["C_text_zeroed_c2_frozen"] = compute_metrics(y_in, cal_s_c, 0.45)
    ablations["C_text_zeroed_c2_val"] = compute_metrics(y_in, cal_s_c, val_thresh)
    zero_text_c = torch.zeros_like(h_text_c_tensor)
    scores_c_text_zeroed = run_c_multi(zero_text_c, h_vision_c_tensor)
    ablations["C_text_zeroed_c_frozen"] = compute_metrics(y_in, scores_c_text_zeroed, 0.45)

    # D. Text + Image with image zeroed (h_vision = 0)
    zero_vis_c2 = torch.zeros_like(h_vision_c2_tensor)
    raw_s_d, cal_s_d, _ = run_c2_gated(h_text_c2_tensor, zero_vis_c2)
    ablations["D_image_zeroed_c2_frozen"] = compute_metrics(y_in, cal_s_d, 0.45)
    ablations["D_image_zeroed_c2_val"] = compute_metrics(y_in, cal_s_d, val_thresh)
    zero_vis_c = torch.zeros_like(h_vision_c_tensor)
    scores_c_img_zeroed = run_c_multi(h_text_c_tensor, zero_vis_c)
    ablations["D_image_zeroed_c_frozen"] = compute_metrics(y_in, scores_c_img_zeroed, 0.45)

    # E. Shuffled images paired with correct text (breaks cross-modal binding)
    np.random.seed(42)
    shuffled_img_idx = np.random.permutation(len(instances))
    shuffled_hv_c2 = h_vision_c2_tensor[shuffled_img_idx]
    raw_s_e, cal_s_e, _ = run_c2_gated(h_text_c2_tensor, shuffled_hv_c2)
    ablations["E_shuffled_images_c2_val"] = compute_metrics(y_in, cal_s_e, val_thresh)
    shuffled_hv_c = h_vision_c_tensor[shuffled_img_idx]
    scores_c_shuffled_img = run_c_multi(h_text_c_tensor, shuffled_hv_c)
    ablations["E_shuffled_images_c_frozen"] = compute_metrics(y_in, scores_c_shuffled_img, 0.45)

    # F. Shuffled text paired with correct image
    shuffled_txt_idx = np.random.permutation(len(instances))
    shuffled_ht_c2 = h_text_c2_tensor[shuffled_txt_idx]
    raw_s_f, cal_s_f, _ = run_c2_gated(shuffled_ht_c2, h_vision_c2_tensor)
    ablations["F_shuffled_text_c2_val"] = compute_metrics(y_in, cal_s_f, val_thresh)
    shuffled_ht_c = h_text_c_tensor[shuffled_txt_idx]
    scores_c_shuffled_txt = run_c_multi(shuffled_ht_c, h_vision_c_tensor)
    ablations["F_shuffled_text_c_frozen"] = compute_metrics(y_in, scores_c_shuffled_txt, 0.45)

    # G. Same text with different images: Sensitivity of score to image change
    # For each sample i, pair text[i] with 5 different random images, compute std of score
    stds_g_c2 = []
    stds_g_c = []
    for i in range(min(100, len(instances))):
        sample_ht_c2 = h_text_c2_tensor[i:i+1].expand(10, -1)
        rand_idx = np.random.choice(len(instances), 10, replace=False)
        rand_hv_c2 = h_vision_c2_tensor[rand_idx]
        _, cal_s_g, _ = run_c2_gated(sample_ht_c2, rand_hv_c2)
        stds_g_c2.append(float(np.std(cal_s_g)))

        sample_ht_c = h_text_c_tensor[i:i+1].expand(10, -1)
        rand_hv_c = h_vision_c_tensor[rand_idx]
        sc_g = run_c_multi(sample_ht_c, rand_hv_c)
        stds_g_c.append(float(np.std(sc_g)))

    ablations["G_same_text_diff_images"] = {
        "model_c2_calibrated_score_std_across_images": round(float(np.mean(stds_g_c2)), 5),
        "model_c_score_std_across_images": round(float(np.mean(stds_g_c)), 5),
        "interpretation": "Variation in prediction when text is fixed but image is altered.",
    }

    # H. Same image with different text: Sensitivity of score to text change
    stds_h_c2 = []
    stds_h_c = []
    for i in range(min(100, len(instances))):
        sample_hv_c2 = h_vision_c2_tensor[i:i+1].expand(10, -1)
        rand_idx = np.random.choice(len(instances), 10, replace=False)
        rand_ht_c2 = h_text_c2_tensor[rand_idx]
        _, cal_s_h, _ = run_c2_gated(rand_ht_c2, sample_hv_c2)
        stds_h_c2.append(float(np.std(cal_s_h)))

        sample_hv_c = h_vision_c_tensor[i:i+1].expand(10, -1)
        rand_ht_c = h_text_c_tensor[rand_idx]
        sc_h = run_c_multi(rand_ht_c, sample_hv_c)
        stds_h_c.append(float(np.std(sc_h)))

    ablations["H_same_image_diff_texts"] = {
        "model_c2_calibrated_score_std_across_texts": round(float(np.mean(stds_h_c2)), 5),
        "model_c_score_std_across_texts": round(float(np.mean(stds_h_c)), 5),
        "interpretation": "Variation in prediction when image is fixed but text is altered.",
    }

    # 3. Logit distributions across partitions
    print("\n--- Partition Logit Analysis ---")
    benign_mask = (y_in == 0)
    hateful_mask = (y_in == 1)

    partition_stats = {
        "benign_memes_count": int(np.sum(benign_mask)),
        "hateful_memes_count": int(np.sum(hateful_mask)),
        "model_c_scores": {
            "benign": {
                "mean": round(float(np.mean(scores_c_multi[benign_mask])), 4),
                "std": round(float(np.std(scores_c_multi[benign_mask])), 4),
                "min": round(float(np.min(scores_c_multi[benign_mask])), 4),
                "max": round(float(np.max(scores_c_multi[benign_mask])), 4),
            },
            "hateful": {
                "mean": round(float(np.mean(scores_c_multi[hateful_mask])), 4),
                "std": round(float(np.std(scores_c_multi[hateful_mask])), 4),
                "min": round(float(np.min(scores_c_multi[hateful_mask])), 4),
                "max": round(float(np.max(scores_c_multi[hateful_mask])), 4),
            },
        },
        "model_c2_raw_scores": {
            "benign": {
                "mean": round(float(np.mean(scores_c2_raw[benign_mask])), 4),
                "std": round(float(np.std(scores_c2_raw[benign_mask])), 4),
                "min": round(float(np.min(scores_c2_raw[benign_mask])), 4),
                "max": round(float(np.max(scores_c2_raw[benign_mask])), 4),
            },
            "hateful": {
                "mean": round(float(np.mean(scores_c2_raw[hateful_mask])), 4),
                "std": round(float(np.std(scores_c2_raw[hateful_mask])), 4),
                "min": round(float(np.min(scores_c2_raw[hateful_mask])), 4),
                "max": round(float(np.max(scores_c2_raw[hateful_mask])), 4),
            },
        },
        "model_c2_calibrated_scores": {
            "benign": {
                "mean": round(float(np.mean(scores_c2_cal[benign_mask])), 4),
                "std": round(float(np.std(scores_c2_cal[benign_mask])), 4),
                "min": round(float(np.min(scores_c2_cal[benign_mask])), 4),
                "max": round(float(np.max(scores_c2_cal[benign_mask])), 4),
            },
            "hateful": {
                "mean": round(float(np.mean(scores_c2_cal[hateful_mask])), 4),
                "std": round(float(np.std(scores_c2_cal[hateful_mask])), 4),
                "min": round(float(np.min(scores_c2_cal[hateful_mask])), 4),
                "max": round(float(np.max(scores_c2_cal[hateful_mask])), 4),
            },
        },
        "model_c2_gates": {
            "benign": {
                "mean": round(float(np.mean(gates_c2[benign_mask])), 4),
                "std": round(float(np.std(gates_c2[benign_mask])), 4),
            },
            "hateful": {
                "mean": round(float(np.mean(gates_c2[hateful_mask])), 4),
                "std": round(float(np.std(gates_c2[hateful_mask])), 4),
            },
        }
    }

    # 4. Cross-Modal Context Cases Table
    print("\n--- Detailed Cross-Modal Cases ---")
    context_cases = [
        {"mid": 115, "cat": "Contextual Inversion", "desc": "damn happy about football (smiling meme)", "gt": 0},
        {"mid": 282, "cat": "Contextual Inversion", "desc": "under arrest for accessibility (office satire)", "gt": 0},
        {"mid": 37, "cat": "Emergent Toxicity", "desc": "see this hat thrift store (advicejew trope)", "gt": 1},
        {"mid": 141, "cat": "Emergent Toxicity", "desc": "wants white tradition eats mcdonalds (white power)", "gt": 1},
        {"mid": 89, "cat": "Explicit Harm", "desc": "hate black people got gun (strict policeman)", "gt": 1},
        {"mid": 96, "cat": "Explicit Harm", "desc": "hate men need cock (feminist caricature)", "gt": 1},
        {"mid": 174, "cat": "Benign Meme", "desc": "blog non-german friends (harmless cartoon)", "gt": 0},
        {"mid": 265, "cat": "Benign Meme", "desc": "quality men believe equality (couple meme)", "gt": 0},
    ]

    case_rows = []
    for c_case in context_cases:
        target_mid = c_case["mid"]
        # Find matching instance
        matched = [inst for inst in instances if inst["meme_id"] == target_mid and inst["language"] == "english"]
        if not matched:
            matched = [inst for inst in instances if inst["meme_id"] == target_mid]
        inst = matched[0]

        enc = tokenizer(inst["text"], truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        img = Image.open(inst["image_path"]).convert("RGB")
        pixel_values = vision_processor(images=img, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            # Text only (Model B)
            sb = float(torch.sigmoid(model_b(input_ids, attention_mask)[0, 0]).cpu().item())
            # Image only (Model C)
            s_img_c = float(torch.sigmoid(model_c(pixel_values=pixel_values, mode="vision_only")[0, 0]).cpu().item())
            # Model C
            sc = float(torch.sigmoid(model_c(input_ids, attention_mask, pixel_values, mode="multimodal")[0, 0]).cpu().item())
            # Model C-2
            lc2, g_val = model_c2(input_ids, attention_mask, pixel_values, mode="gated", return_gate=True)
            sc2_raw = float(torch.sigmoid(lc2[0, 0]).cpu().item())
            eps = 1e-7
            raw_logit = np.log((sc2_raw + eps) / (1.0 - sc2_raw + eps))
            sc2_cal = float(1.0 / (1.0 + np.exp(-(platt_a * raw_logit + platt_b))))

        case_rows.append({
            "case_id": f"mid_{target_mid}",
            "category": c_case["cat"],
            "description": c_case["desc"],
            "ground_truth": c_case["gt"],
            "text_only_b_score": round(sb, 4),
            "text_only_b_pred": int(sb >= 0.45),
            "image_only_c_score": round(s_img_c, 4),
            "image_only_c_pred": int(s_img_c >= 0.45),
            "model_c_score": round(sc, 4),
            "model_c_pred": int(sc >= 0.45),
            "model_c2_cal_score": round(sc2_cal, 4),
            "model_c2_pred_val": int(sc2_cal >= val_thresh),
            "gate_value": round(float(g_val.cpu().item()), 4),
        })

    # Save all forensic results
    full_forensic_report = {
        "c_weights": c_weights,
        "c2_weights": c2_weights,
        "embedding_analysis": embedding_analysis,
        "ablations": ablations,
        "partition_stats": partition_stats,
        "cross_modal_cases": case_rows,
        "platt_params": {"a": platt_a, "b": platt_b, "val_thresh": val_thresh},
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_forensic_report, f, indent=2)

    print(f"\nForensic results successfully written to {OUT_JSON}")


if __name__ == "__main__":
    main()
