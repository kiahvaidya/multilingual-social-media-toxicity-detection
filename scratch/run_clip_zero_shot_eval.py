"""
Execution of Zero-Shot CLIP ViT-B/32 and Model B Late Fusion Evaluation on Held-Out Multi3Hate.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from sklearn.metrics import roc_auc_score, average_precision_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import MuRILForMultiLabelToxicity

OUT_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "clip_zero_shot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
MULTI3HATE_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "multi3hate"
MULTI3HATE_IMAGES_BASE = MULTI3HATE_DIR / "Multi3Hate_repo" / "data" / "memes"

PROMPT_PAIRS = [
    {
        "id": 1,
        "name": "Pair 1 (hateful vs benign)",
        "pos": "a hateful meme",
        "neg": "a benign meme",
    },
    {
        "id": 2,
        "name": "Pair 2 (hate speech vs without hate speech)",
        "pos": "a meme containing hate speech",
        "neg": "a meme without hate speech",
    },
    {
        "id": 3,
        "name": "Pair 3 (attacking protected group vs not)",
        "pos": "a meme attacking a protected group",
        "neg": "a meme that does not attack a protected group",
    },
    {
        "id": 4,
        "name": "Pair 4 (discriminatory vs non-discriminatory)",
        "pos": "a discriminatory meme",
        "neg": "a non-discriminatory meme",
    },
    {
        "id": 5,
        "name": "Pair 5 (expressing prejudice vs neutral)",
        "pos": "a meme expressing prejudice",
        "neg": "a neutral meme",
    },
    {
        "id": 6,
        "name": "Pair 6 (group targeting vs without group hostility)",
        "pos": "a meme targeting a racial, religious, gender, or national group",
        "neg": "a meme without group-targeted hostility",
    },
    {
        "id": 7,
        "name": "Pair 7 (offensive hate vs harmless)",
        "pos": "an offensive hate meme",
        "neg": "a harmless meme",
    },
    {
        "id": 8,
        "name": "Pair 8 (hostility toward demographic vs no hostility)",
        "pos": "a meme expressing hostility toward a demographic group",
        "neg": "a meme expressing no hostility toward a demographic group",
    },
]

KNOWN_TROPE_MIDS = [113, 205, 166, 50, 81, 220]


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "accuracy": round(acc, 4),
        "pred_pos": int(np.sum(y_pred == 1)),
        "pred_neg": int(np.sum(y_pred == 0)),
        "total": len(y_true),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing CLIP Zero-Shot + Model B Late Fusion on device: {device}")

    # ==========================================
    # 1. LOAD DATASET & VERIFY COUNTS
    # ==========================================
    annot_csv = MULTI3HATE_DIR / "final_annotations.csv"
    hi_csv = MULTI3HATE_DIR / "hi_captions.csv"
    en_csv = MULTI3HATE_DIR / "en_captions.csv"

    df_annot = pd.read_csv(annot_csv)
    df_hi = pd.read_csv(hi_csv)
    df_en = pd.read_csv(en_csv)

    pos_in = int((df_annot["IN"] == 1).sum())
    neg_in = int((df_annot["IN"] == 0).sum())
    print(f"Multi3Hate final_annotations unique memes - IN=1: {pos_in}, IN=0: {neg_in}")
    if pos_in != 180 or neg_in != 120:
        raise ValueError(f"Ground truth unique meme counts differ: expected 180/120, got {pos_in}/{neg_in}")

    hi_map = {r["Meme ID"]: r for _, r in df_hi.iterrows()}
    en_map = {r["Meme ID"]: r for _, r in df_en.iterrows()}

    instances = []
    for _, row in df_annot.iterrows():
        mid = int(row["Meme ID"])
        gt_in = int(row["IN"])
        gt_us = int(row["US"])
        gt_de = int(row["DE"])
        gt_mx = int(row["MX"])
        gt_cn = int(row["CN"])

        # English
        e_row = en_map[mid]
        e_txt = str(e_row["Original (English)"]).replace("<sep>", " ").strip()
        e_matches = list((MULTI3HATE_IMAGES_BASE / "en").rglob(f"{mid}.jpg"))
        if not e_matches:
            raise FileNotFoundError(f"Missing English image for Meme ID {mid}")

        instances.append({
            "sample_id": f"multi3hate_en_{mid}",
            "meme_id": mid,
            "language": "english",
            "text": e_txt,
            "image_path": str(e_matches[0]),
            "image_filename": e_matches[0].name,
            "IN": gt_in,
            "US": gt_us,
            "DE": gt_de,
            "MX": gt_mx,
            "CN": gt_cn,
        })

        # Hindi
        h_row = hi_map[mid]
        h_txt = str(h_row["Translation"]).replace("<sep>", " ").strip()
        h_matches = list((MULTI3HATE_IMAGES_BASE / "hi").rglob(f"{mid}.jpg"))
        if not h_matches:
            raise FileNotFoundError(f"Missing Hindi image for Meme ID {mid}")

        instances.append({
            "sample_id": f"multi3hate_hi_{mid}",
            "meme_id": mid,
            "language": "hindi",
            "text": h_txt,
            "image_path": str(h_matches[0]),
            "image_filename": h_matches[0].name,
            "IN": gt_in,
            "US": gt_us,
            "DE": gt_de,
            "MX": gt_mx,
            "CN": gt_cn,
        })

    total_instances = len(instances)
    total_pos = sum(1 for inst in instances if inst["IN"] == 1)
    total_neg = sum(1 for inst in instances if inst["IN"] == 0)
    print(f"Total Paired Instances: {total_instances}, Positive (IN=1): {total_pos}, Negative (IN=0): {total_neg}")
    if total_pos != 360 or total_neg != 240:
        raise ValueError(f"Expected 360 positive and 240 negative instances, got {total_pos} and {total_neg}")

    y_in = np.array([inst["IN"] for inst in instances])
    y_us = np.array([inst["US"] for inst in instances])
    y_de = np.array([inst["DE"] for inst in instances])
    y_mx = np.array([inst["MX"] for inst in instances])
    y_cn = np.array([inst["CN"] for inst in instances])
    langs = np.array([inst["language"] for inst in instances])

    # ==========================================
    # 2. MODEL B INFERENCE
    # ==========================================
    print("Running Model B inference...")
    model_b = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_b.load_state_dict(torch.load(MODEL_B_CHECKPOINT, map_location=device, weights_only=True))
    model_b.to(device).eval()

    muril_tok = AutoTokenizer.from_pretrained("google/muril-base-cased")

    p_text_list = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(instances), batch_size):
            batch_insts = instances[i : i + batch_size]
            texts = [inst["text"] for inst in batch_insts]
            enc = muril_tok(texts, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            logits = model_b(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs = torch.sigmoid(logits[:, 0]).cpu().numpy()
            p_text_list.extend(probs.tolist())

    p_text = np.array(p_text_list)
    for idx, inst in enumerate(instances):
        inst["p_text"] = float(p_text[idx])

    # Baseline Model B Metrics
    pred_b = (p_text >= 0.45).astype(int)
    model_b_metrics = compute_binary_metrics(y_in, pred_b)
    print("Model B Baseline (toxic >= 0.45):", model_b_metrics)

    # ==========================================
    # 3. ZERO-SHOT CLIP INFERENCE
    # ==========================================
    print("Loading CLIP ViT-B/32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Encode all 8 prompt pairs
    # For each pair: positive text & negative text
    prompt_text_embeds = {}
    with torch.no_grad():
        for p in PROMPT_PAIRS:
            pid = p["id"]
            texts = [p["pos"], p["neg"]]
            inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            prompt_text_embeds[pid] = {
                "pos": text_features[0],
                "neg": text_features[1],
            }

    # Extract all image embeddings (caching by unique image path)
    unique_img_paths = list(set(inst["image_path"] for inst in instances))
    print(f"Extracting CLIP visual embeddings for {len(unique_img_paths)} unique images...")
    img_embed_cache = {}
    with torch.no_grad():
        for i in range(0, len(unique_img_paths), 16):
            batch_paths = unique_img_paths[i : i + 16]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            img_inputs = clip_processor(images=images, return_tensors="pt").to(device)
            img_features = clip_model.get_image_features(**img_inputs)
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
            for path, feat in zip(batch_paths, img_features):
                img_embed_cache[path] = feat

    # Compute cosine similarities and softmax scores for each instance
    # score = exp(sim_pos) / (exp(sim_pos) + exp(sim_neg))
    clip_scores_by_pair = {p["id"]: [] for p in PROMPT_PAIRS}

    for inst in instances:
        img_feat = img_embed_cache[inst["image_path"]]  # (512,)
        for p in PROMPT_PAIRS:
            pid = p["id"]
            pos_feat = prompt_text_embeds[pid]["pos"]
            neg_feat = prompt_text_embeds[pid]["neg"]

            sim_pos = float(torch.dot(img_feat, pos_feat).cpu().item())
            sim_neg = float(torch.dot(img_feat, neg_feat).cpu().item())

            # score = exp(sim_pos) / (exp(sim_pos) + exp(sim_neg))
            # numerically stable: sigmoid(sim_pos - sim_neg)
            score = float(torch.sigmoid(torch.tensor(sim_pos - sim_neg)).item())
            clip_scores_by_pair[pid].append(score)
            inst[f"clip_score_p{pid}"] = score

    for pid in clip_scores_by_pair:
        clip_scores_by_pair[pid] = np.array(clip_scores_by_pair[pid])

    # Ensemble: mean score across all 8 prompt pairs
    all_scores_matrix = np.column_stack([clip_scores_by_pair[p["id"]] for p in PROMPT_PAIRS])  # (600, 8)
    mean_ensemble_scores = np.mean(all_scores_matrix, axis=1)
    median_ensemble_scores = np.median(all_scores_matrix, axis=1)
    std_ensemble_scores = np.std(all_scores_matrix, axis=1)

    for idx, inst in enumerate(instances):
        inst["clip_mean_score"] = float(mean_ensemble_scores[idx])
        inst["clip_median_score"] = float(median_ensemble_scores[idx])
        inst["clip_std_score"] = float(std_ensemble_scores[idx])

    # ==========================================
    # PART 2 — CLIP RESULTS FOR EACH PROMPT PAIR
    # ==========================================
    clip_pair_results = []
    in_neg_mask = (y_in == 0)
    in_pos_mask = (y_in == 1)

    for p in PROMPT_PAIRS:
        pid = p["id"]
        scores = clip_scores_by_pair[pid]
        preds = (scores >= 0.50).astype(int)

        auroc = float(roc_auc_score(y_in, scores))
        ap = float(average_precision_score(y_in, scores))
        b_metrics = compute_binary_metrics(y_in, preds)

        mean_neg = float(np.mean(scores[in_neg_mask]))
        mean_pos = float(np.mean(scores[in_pos_mask]))
        sep = mean_pos - mean_neg

        res = {
            "pair_id": pid,
            "pair_name": p["name"],
            "positive_prompt": p["pos"],
            "negative_prompt": p["neg"],
            "auroc": round(auroc, 4),
            "ap": round(ap, 4),
            "precision_at_0.50": b_metrics["precision"],
            "recall_at_0.50": b_metrics["recall"],
            "f1_at_0.50": b_metrics["f1"],
            "fpr": b_metrics["fpr"],
            "fnr": b_metrics["fnr"],
            "tp": b_metrics["tp"],
            "fp": b_metrics["fp"],
            "tn": b_metrics["tn"],
            "fn": b_metrics["fn"],
            "score_mean_in_0": round(mean_neg, 4),
            "score_mean_in_1": round(mean_pos, 4),
            "mean_separation": round(sep, 4),
        }
        clip_pair_results.append(res)

    # ==========================================
    # PART 3 — PROMPT ROBUSTNESS & ENSEMBLE
    # ==========================================
    corr_df = pd.DataFrame(all_scores_matrix, columns=[f"P{p['id']}" for p in PROMPT_PAIRS]).corr()
    corr_dict = {col: {k: round(float(v), 4) for k, v in corr_df[col].to_dict().items()} for col in corr_df.columns}

    ens_preds = (mean_ensemble_scores >= 0.50).astype(int)
    ens_auroc = float(roc_auc_score(y_in, mean_ensemble_scores))
    ens_ap = float(average_precision_score(y_in, mean_ensemble_scores))
    ens_b_metrics = compute_binary_metrics(y_in, ens_preds)

    prompt_robustness = {
        "mean_score_across_all_prompts": round(float(np.mean(all_scores_matrix)), 4),
        "median_score_across_all_prompts": round(float(np.median(all_scores_matrix)), 4),
        "std_across_prompts_per_sample_mean": round(float(np.mean(std_ensemble_scores)), 4),
        "pairwise_correlation": corr_dict,
        "mean_ensemble_eval_at_0.50": {
            "auroc": round(ens_auroc, 4),
            "ap": round(ens_ap, 4),
            "precision": ens_b_metrics["precision"],
            "recall": ens_b_metrics["recall"],
            "f1": ens_b_metrics["f1"],
            "fpr": ens_b_metrics["fpr"],
            "fnr": ens_b_metrics["fnr"],
            "tp": ens_b_metrics["tp"],
            "fp": ens_b_metrics["fp"],
            "tn": ens_b_metrics["tn"],
            "fn": ens_b_metrics["fn"],
            "pred_pos": ens_b_metrics["pred_pos"],
            "pred_neg": ens_b_metrics["pred_neg"],
        },
    }

    # ==========================================
    # PARTS 5 & 6 — LATE FUSION EVALUATION
    # ==========================================
    # Rule A: p_text >= 0.45
    # Rule B: clip_score >= 0.50
    # Rule C: p_text >= 0.45 OR clip_score >= 0.50
    fusion_evaluations = []

    # Include all 8 pairs and the mean ensemble
    fusion_configs = [(p["id"], p["name"], clip_scores_by_pair[p["id"]]) for p in PROMPT_PAIRS]
    fusion_configs.append(("Ensemble", "Mean Score Ensemble", mean_ensemble_scores))

    for cid, cname, cscores in fusion_configs:
        pred_clip = (cscores >= 0.50).astype(int)
        pred_or = ((p_text >= 0.45) | (cscores >= 0.50)).astype(int)

        m_clip = compute_binary_metrics(y_in, pred_clip)
        m_or = compute_binary_metrics(y_in, pred_or)

        delta_prec = m_or["precision"] - model_b_metrics["precision"]
        delta_rec = m_or["recall"] - model_b_metrics["recall"]
        delta_f1 = m_or["f1"] - model_b_metrics["f1"]
        delta_fpr = m_or["fpr"] - model_b_metrics["fpr"]

        fusion_evaluations.append({
            "config_id": str(cid),
            "config_name": cname,
            "rule_b_clip_metrics": m_clip,
            "rule_c_or_metrics": m_or,
            "deltas_vs_model_b": {
                "delta_precision": round(delta_prec, 4),
                "delta_recall": round(delta_rec, 4),
                "delta_f1": round(delta_f1, 4),
                "delta_fpr": round(delta_fpr, 4),
            },
        })

    # ==========================================
    # PART 7 — VALIDATION OF VISUAL COMPLEMENTARITY
    # ==========================================
    # Using the Mean Ensemble CLIP score (and Pair 1 for reference)
    def analyze_complementarity(clip_s):
        p_c = (clip_s >= 0.50).astype(int)
        p_b = (p_text >= 0.45).astype(int)

        cat_a = (y_in == 1) & (p_b == 0) & (p_c == 1)  # Model B FN, CLIP caught
        cat_b = (y_in == 1) & (p_b == 1) & (p_c == 1)  # Both TP
        cat_c = (y_in == 0) & (p_b == 0) & (p_c == 0)  # Both TN
        cat_d = (y_in == 0) & (p_b == 0) & (p_c == 1)  # Model B TN, CLIP caused FP

        def format_cat(mask):
            idx_list = np.where(mask)[0]
            examples = []
            for i in idx_list[:10]:
                inst = instances[i]
                examples.append({
                    "meme_id": inst["meme_id"],
                    "language": inst["language"],
                    "text": inst["text"][:100],
                    "image_filename": inst["image_filename"],
                    "ground_truth_IN": inst["IN"],
                    "p_text": round(float(inst["p_text"]), 4),
                    "clip_score": round(float(clip_s[i]), 4),
                })
            return {
                "count": int(np.sum(mask)),
                "percentage_of_total": round(float(np.sum(mask) / len(y_in) * 100), 2),
                "percentage_of_subset": round(
                    float(np.sum(mask) / (np.sum(y_in == 1) if "FN" in str(mask) or "TP" in str(mask) else np.sum(y_in == 0)) * 100), 2
                ),
                "sample_examples": examples,
            }

        return {
            "A_model_b_FN_clip_positive": format_cat(cat_a),
            "B_both_true_positive": format_cat(cat_b),
            "C_both_true_negative": format_cat(cat_c),
            "D_model_b_TN_clip_false_positive": format_cat(cat_d),
        }

    visual_complementarity = {
        "using_mean_ensemble": analyze_complementarity(mean_ensemble_scores),
        "using_pair_1": analyze_complementarity(clip_scores_by_pair[1]),
    }

    # ==========================================
    # PART 8 — KNOWN VISUAL-TROPE CASES
    # ==========================================
    trope_case_rows = []
    for mid in KNOWN_TROPE_MIDS:
        matched = [inst for inst in instances if inst["meme_id"] == mid and inst["language"] == "english"]
        if not matched:
            matched = [inst for inst in instances if inst["meme_id"] == mid]
        inst = matched[0]
        i = instances.index(inst)

        p_b_val = float(p_text[i])
        flag_b = int(p_b_val >= 0.45)

        pair_scores = {f"P{p['id']}": round(float(clip_scores_by_pair[p['id']][i]), 4) for p in PROMPT_PAIRS}
        ens_score = round(float(mean_ensemble_scores[i]), 4)
        flag_clip_ens = int(ens_score >= 0.50)
        flag_or = int(flag_b or flag_clip_ens)

        trope_case_rows.append({
            "meme_id": mid,
            "description": inst["text"][:80],
            "ground_truth_IN": inst["IN"],
            "model_b_toxic_prob": round(p_b_val, 4),
            "model_b_flagged": flag_b,
            "clip_scores_by_prompt_pair": pair_scores,
            "clip_mean_score": ens_score,
            "clip_flagged_at_0.50": flag_clip_ens,
            "or_fusion_flagged": flag_or,
        })

    # ==========================================
    # PART 9 — LANGUAGE SLICES
    # ==========================================
    en_mask = (langs == "english")
    hi_mask = (langs == "hindi")

    def eval_slice(mask):
        sub_y = y_in[mask]
        sub_pt = p_text[mask]
        sub_clip = mean_ensemble_scores[mask]

        p_b = (sub_pt >= 0.45).astype(int)
        p_c = (sub_clip >= 0.50).astype(int)
        p_or = (p_b | p_c).astype(int)

        return {
            "model_b": compute_binary_metrics(sub_y, p_b),
            "clip_ensemble": compute_binary_metrics(sub_y, p_c),
            "or_fusion": compute_binary_metrics(sub_y, p_or),
        }

    language_slices = {
        "english": eval_slice(en_mask),
        "hindi": eval_slice(hi_mask),
    }

    # ==========================================
    # PART 10 — CULTURAL SLICES (AUROC & AP)
    # ==========================================
    cultural_slices = {}
    country_targets = [("IN", y_in), ("US", y_us), ("DE", y_de), ("MX", y_mx), ("CN", y_cn)]

    for cname, c_y in country_targets:
        cultural_slices[cname] = {
            "model_b_toxic": {
                "auroc": round(float(roc_auc_score(c_y, p_text)), 4),
                "ap": round(float(average_precision_score(c_y, p_text)), 4),
            },
            "clip_ensemble": {
                "auroc": round(float(roc_auc_score(c_y, mean_ensemble_scores)), 4),
                "ap": round(float(average_precision_score(c_y, mean_ensemble_scores)), 4),
            },
            "late_fusion_or_max_score": {
                "auroc": round(float(roc_auc_score(c_y, np.maximum(p_text, mean_ensemble_scores))), 4),
                "ap": round(float(average_precision_score(c_y, np.maximum(p_text, mean_ensemble_scores))), 4),
            },
        }

    # ==========================================
    # PART 11 — THRESHOLD ANALYSIS (DESCRIPTIVE SWEEP)
    # ==========================================
    thresholds_to_sweep = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    sweep_results = []

    for t in thresholds_to_sweep:
        # Standalone CLIP
        p_c_sweep = (mean_ensemble_scores >= t).astype(int)
        m_c = compute_binary_metrics(y_in, p_c_sweep)

        # OR fusion with frozen Model B at 0.45
        p_or_sweep = ((p_text >= 0.45) | (mean_ensemble_scores >= t)).astype(int)
        m_or = compute_binary_metrics(y_in, p_or_sweep)

        sweep_results.append({
            "threshold": t,
            "clip_standalone": {
                "precision": m_c["precision"],
                "recall": m_c["recall"],
                "f1": m_c["f1"],
                "fpr": m_c["fpr"],
                "fnr": m_c["fnr"],
                "tp": m_c["tp"],
                "fp": m_c["fp"],
                "tn": m_c["tn"],
                "fn": m_c["fn"],
            },
            "or_fusion_with_model_b_at_0.45": {
                "precision": m_or["precision"],
                "recall": m_or["recall"],
                "f1": m_or["f1"],
                "fpr": m_or["fpr"],
                "fnr": m_or["fnr"],
                "tp": m_or["tp"],
                "fp": m_or["fp"],
                "tn": m_or["tn"],
                "fn": m_or["fn"],
            },
        })

    # ==========================================
    # PART 12 — WRITE ALL OUTPUT FILES
    # ==========================================
    # 1. results.json
    full_json = {
        "benchmark": "Multi3Hate Held-Out Zero-Shot CLIP and Model B Late Fusion Evaluation",
        "sample_counts": {
            "total_instances": total_instances,
            "positive_IN": total_pos,
            "negative_IN": total_neg,
        },
        "model_b_baseline": model_b_metrics,
        "clip_prompt_pairs": clip_pair_results,
        "prompt_robustness": prompt_robustness,
        "fusion_evaluations": fusion_evaluations,
        "visual_complementarity": visual_complementarity,
        "known_trope_cases": trope_case_rows,
        "language_slices": language_slices,
        "cultural_slices": cultural_slices,
        "descriptive_threshold_sweep": {
            "label": "HELD-OUT DESCRIPTIVE THRESHOLD SWEEP — NOT USED FOR MODEL SELECTION",
            "sweep": sweep_results,
        },
    }
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2)

    # 2. results.csv (CLIP prompt pairs)
    df_clip_results = pd.DataFrame(clip_pair_results)
    df_clip_results.to_csv(OUT_DIR / "results.csv", index=False)

    # 3. fusion_results.csv
    fusion_rows = []
    for fe in fusion_evaluations:
        row = {
            "config_id": fe["config_id"],
            "config_name": fe["config_name"],
            "clip_precision": fe["rule_b_clip_metrics"]["precision"],
            "clip_recall": fe["rule_b_clip_metrics"]["recall"],
            "clip_f1": fe["rule_b_clip_metrics"]["f1"],
            "clip_fpr": fe["rule_b_clip_metrics"]["fpr"],
            "or_precision": fe["rule_c_or_metrics"]["precision"],
            "or_recall": fe["rule_c_or_metrics"]["recall"],
            "or_f1": fe["rule_c_or_metrics"]["f1"],
            "or_fpr": fe["rule_c_or_metrics"]["fpr"],
            "delta_precision": fe["deltas_vs_model_b"]["delta_precision"],
            "delta_recall": fe["deltas_vs_model_b"]["delta_recall"],
            "delta_f1": fe["deltas_vs_model_b"]["delta_f1"],
            "delta_fpr": fe["deltas_vs_model_b"]["delta_fpr"],
        }
        fusion_rows.append(row)
    pd.DataFrame(fusion_rows).to_csv(OUT_DIR / "fusion_results.csv", index=False)

    # 4. error_analysis.csv
    error_rows = []
    for inst in instances:
        i = instances.index(inst)
        error_rows.append({
            "sample_id": inst["sample_id"],
            "meme_id": inst["meme_id"],
            "language": inst["language"],
            "text": inst["text"],
            "image_filename": inst["image_filename"],
            "ground_truth_IN": inst["IN"],
            "p_text": round(float(inst["p_text"]), 4),
            "pred_model_b": int(inst["p_text"] >= 0.45),
            "clip_mean_score": round(float(inst["clip_mean_score"]), 4),
            "pred_clip_0.50": int(inst["clip_mean_score"] >= 0.50),
            "pred_or_fusion": int((inst["p_text"] >= 0.45) | (inst["clip_mean_score"] >= 0.50)),
        })
    pd.DataFrame(error_rows).to_csv(OUT_DIR / "error_analysis.csv", index=False)

    # 5. README.md
    readme_content = f"""# Zero-Shot CLIP ViT-B/32 & Model B Late Fusion Evaluation

**Artifact Directory**: `scratch/meme_experiments/clip_zero_shot/`  
**Dataset**: Held-Out Multi3Hate ($N = 600$ paired instances: 300 English, 300 Hindi)  
**Ground Truth**: Indian Consensus (`IN == 1` positive: 360, `IN == 0` negative: 240)  

### Key Findings
1. **Model B Baseline**: Precision = {model_b_metrics['precision']}, Recall = {model_b_metrics['recall']}, F1 = {model_b_metrics['f1']}, FPR = {model_b_metrics['fpr']}
2. **Zero-Shot CLIP Mean Ensemble (at 0.50)**: AUROC = {prompt_robustness['mean_ensemble_eval_at_0.50']['auroc']}, AP = {prompt_robustness['mean_ensemble_eval_at_0.50']['ap']}, Precision = {prompt_robustness['mean_ensemble_eval_at_0.50']['precision']}, Recall = {prompt_robustness['mean_ensemble_eval_at_0.50']['recall']}, FPR = {prompt_robustness['mean_ensemble_eval_at_0.50']['fpr']}
3. **OR Late Fusion (`p_text >= 0.45 OR clip >= 0.50`)**:
   - Precision: {fusion_evaluations[-1]['rule_c_or_metrics']['precision']} (Δ {fusion_evaluations[-1]['deltas_vs_model_b']['delta_precision']})
   - Recall: {fusion_evaluations[-1]['rule_c_or_metrics']['recall']} (Δ {fusion_evaluations[-1]['deltas_vs_model_b']['delta_recall']})
   - F1: {fusion_evaluations[-1]['rule_c_or_metrics']['f1']} (Δ {fusion_evaluations[-1]['deltas_vs_model_b']['delta_f1']})
   - FPR: {fusion_evaluations[-1]['rule_c_or_metrics']['fpr']} (Δ {fusion_evaluations[-1]['deltas_vs_model_b']['delta_fpr']})

All artifacts generated automatically without model retraining or threshold tuning.
"""
    with open(OUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("All artifacts successfully created in scratch/meme_experiments/clip_zero_shot/")


if __name__ == "__main__":
    main()
