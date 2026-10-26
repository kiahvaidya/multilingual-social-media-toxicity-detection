"""
Model B Ontology-Alignment Evaluation on Held-Out Multi3Hate.
Performs:
1. Model B checkpoint validation & architecture verification (6 heads).
2. Evaluation of 3 rules:
   - Head 0: toxic only (tau = 0.45)
   - Head 5: identity_hate only (tau = 0.60)
   - Combined: toxic >= 0.45 OR identity_hate >= 0.60
3. Language slices: English (N=300) vs Hindi (N=300).
4. Cultural slices: IN, US, DE, MX, CN consensus labels.
5. Score distributions across all 6 heads for IN == 0 vs IN == 1.
6. Semantic error analysis: Top 20 false positives and top 20 false negatives for each rule.
7. Checkpoint target audit for Head 5 training on HASOC-Meme 2025.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import MuRILForMultiLabelToxicity, DEFAULT_LABELS
from src.config import config

MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
MULTI3HATE_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "multi3hate"
MULTI3HATE_IMAGES_BASE = MULTI3HATE_DIR / "Multi3Hate_repo" / "data" / "memes"
HASOC_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "hasoc_meme_2025"

OUT_JSON = _PROJECT_ROOT / "scratch" / "model_b_ontology_eval_results.json"

FROZEN_THRESHOLDS = config.FROZEN_MURIL_THRESHOLDS
# {"toxic": 0.45, "severe_toxic": 0.62, "obscene": 0.65, "threat": 0.59, "insult": 0.61, "identity_hate": 0.60}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # ==========================================
    # 1. MODEL CHECK
    # ==========================================
    if not MODEL_B_CHECKPOINT.exists():
        raise FileNotFoundError(f"Model B checkpoint not found at {MODEL_B_CHECKPOINT}")

    state_dict = torch.load(MODEL_B_CHECKPOINT, map_location="cpu", weights_only=True)
    classifier_weight = state_dict.get("classifier.weight")
    classifier_bias = state_dict.get("classifier.bias")

    model_check = {
        "checkpoint_path": str(MODEL_B_CHECKPOINT),
        "classifier_weight_shape": list(classifier_weight.shape) if classifier_weight is not None else None,
        "classifier_bias_shape": list(classifier_bias.shape) if classifier_bias is not None else None,
        "num_output_neurons": classifier_weight.shape[0] if classifier_weight is not None else None,
        "labels": DEFAULT_LABELS,
        "frozen_thresholds_used": {
            "toxic": FROZEN_THRESHOLDS["toxic"],
            "identity_hate": FROZEN_THRESHOLDS["identity_hate"],
        },
    }
    print("Model check:", json.dumps(model_check, indent=2))

    # Load Model B architecture
    model_b = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_b.load_state_dict(state_dict)
    model_b.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    # ==========================================
    # 2. LOAD MULTI3HATE
    # ==========================================
    annot_csv = MULTI3HATE_DIR / "final_annotations.csv"
    hi_csv = MULTI3HATE_DIR / "hi_captions.csv"
    en_csv = MULTI3HATE_DIR / "en_captions.csv"

    df_annot = pd.read_csv(annot_csv)
    df_hi = pd.read_csv(hi_csv)
    df_en = pd.read_csv(en_csv)

    # Check unique meme count and consensus IN counts
    pos_count_in = int((df_annot["IN"] == 1).sum())
    neg_count_in = int((df_annot["IN"] == 0).sum())
    print(f"Multi3Hate final_annotations Meme counts - Positives (IN=1): {pos_count_in}, Negatives (IN=0): {neg_count_in}")
    if pos_count_in != 180 or neg_count_in != 120:
        raise ValueError(f"Discrepancy in Multi3Hate ground truth counts: {pos_count_in} vs 180, {neg_count_in} vs 120")

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
        e_img_path = str(e_matches[0]) if e_matches else ""
        e_img_fname = e_matches[0].name if e_matches else ""

        instances.append({
            "sample_id": f"multi3hate_en_{mid}",
            "meme_id": mid,
            "language": "english",
            "text": e_txt,
            "image_path": e_img_path,
            "image_filename": e_img_fname,
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
        h_img_path = str(h_matches[0]) if h_matches else ""
        h_img_fname = h_matches[0].name if h_matches else ""

        instances.append({
            "sample_id": f"multi3hate_hi_{mid}",
            "meme_id": mid,
            "language": "hindi",
            "text": h_txt,
            "image_path": h_img_path,
            "image_filename": h_img_fname,
            "IN": gt_in,
            "US": gt_us,
            "DE": gt_de,
            "MX": gt_mx,
            "CN": gt_cn,
        })

    print(f"Loaded {len(instances)} paired instances (300 English, 300 Hindi).")

    # ==========================================
    # 3. RUN MODEL B INFERENCE
    # ==========================================
    print("Running Model B inference across all 600 instances...")
    all_probs = []  # shape: (600, 6)

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(instances), batch_size):
            batch_insts = instances[i : i + batch_size]
            texts = [inst["text"] for inst in batch_insts]
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=64,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            logits = model_b(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)  # (600, 6)

    # Attach probabilities to instances
    for idx, inst in enumerate(instances):
        inst["p_toxic"] = float(all_probs[idx, 0])
        inst["p_severe_toxic"] = float(all_probs[idx, 1])
        inst["p_obscene"] = float(all_probs[idx, 2])
        inst["p_threat"] = float(all_probs[idx, 3])
        inst["p_insult"] = float(all_probs[idx, 4])
        inst["p_identity_hate"] = float(all_probs[idx, 5])

    # Labels and thresholds
    p_toxic = all_probs[:, 0]
    p_identity_hate = all_probs[:, 5]
    y_in = np.array([inst["IN"] for inst in instances])
    y_us = np.array([inst["US"] for inst in instances])
    y_de = np.array([inst["DE"] for inst in instances])
    y_mx = np.array([inst["MX"] for inst in instances])
    y_cn = np.array([inst["CN"] for inst in instances])
    langs = np.array([inst["language"] for inst in instances])

    tau_toxic = FROZEN_THRESHOLDS["toxic"]          # 0.45
    tau_idhate = FROZEN_THRESHOLDS["identity_hate"]  # 0.60

    # Predictions
    pred_toxic = (p_toxic >= tau_toxic).astype(int)
    pred_idhate = (p_identity_hate >= tau_idhate).astype(int)
    pred_combined = ((p_toxic >= tau_toxic) | (p_identity_hate >= tau_idhate)).astype(int)

    for idx, inst in enumerate(instances):
        inst["pred_toxic"] = int(pred_toxic[idx])
        inst["pred_idhate"] = int(pred_idhate[idx])
        inst["pred_combined"] = int(pred_combined[idx])

    # ==========================================
    # 4. OVERALL RESULTS ON INDIAN CONSENSUS
    # ==========================================
    results_in = {
        "A_toxic_only": compute_metrics(y_in, pred_toxic),
        "B_identity_hate_only": compute_metrics(y_in, pred_idhate),
        "C_toxic_or_identity_hate": compute_metrics(y_in, pred_combined),
    }

    # ==========================================
    # 5. LANGUAGE SLICES
    # ==========================================
    en_mask = (langs == "english")
    hi_mask = (langs == "hindi")

    results_language = {
        "english": {
            "A_toxic_only": compute_metrics(y_in[en_mask], pred_toxic[en_mask]),
            "B_identity_hate_only": compute_metrics(y_in[en_mask], pred_idhate[en_mask]),
            "C_toxic_or_identity_hate": compute_metrics(y_in[en_mask], pred_combined[en_mask]),
        },
        "hindi": {
            "A_toxic_only": compute_metrics(y_in[hi_mask], pred_toxic[hi_mask]),
            "B_identity_hate_only": compute_metrics(y_in[hi_mask], pred_idhate[hi_mask]),
            "C_toxic_or_identity_hate": compute_metrics(y_in[hi_mask], pred_combined[hi_mask]),
        },
    }

    # ==========================================
    # 6. CULTURAL SLICES
    # ==========================================
    results_cultural = {
        "IN": {
            "toxic_only": compute_metrics(y_in, pred_toxic),
            "identity_hate_only": compute_metrics(y_in, pred_idhate),
            "combined": compute_metrics(y_in, pred_combined),
        },
        "US": {
            "toxic_only": compute_metrics(y_us, pred_toxic),
            "identity_hate_only": compute_metrics(y_us, pred_idhate),
            "combined": compute_metrics(y_us, pred_combined),
        },
        "DE": {
            "toxic_only": compute_metrics(y_de, pred_toxic),
            "identity_hate_only": compute_metrics(y_de, pred_idhate),
            "combined": compute_metrics(y_de, pred_combined),
        },
        "MX": {
            "toxic_only": compute_metrics(y_mx, pred_toxic),
            "identity_hate_only": compute_metrics(y_mx, pred_idhate),
            "combined": compute_metrics(y_mx, pred_combined),
        },
        "CN": {
            "toxic_only": compute_metrics(y_cn, pred_toxic),
            "identity_hate_only": compute_metrics(y_cn, pred_idhate),
            "combined": compute_metrics(y_cn, pred_combined),
        },
    }

    # ==========================================
    # 7. SCORE DISTRIBUTIONS FOR ALL 6 HEADS
    # ==========================================
    in_neg_mask = (y_in == 0)
    in_pos_mask = (y_in == 1)

    score_distributions = {}
    for h_idx, h_name in enumerate(DEFAULT_LABELS):
        h_probs = all_probs[:, h_idx]
        neg_vals = h_probs[in_neg_mask]
        pos_vals = h_probs[in_pos_mask]

        score_distributions[h_name] = {
            "head_index": h_idx,
            "IN_0_negative": {
                "count": len(neg_vals),
                "mean": round(float(np.mean(neg_vals)), 4),
                "std": round(float(np.std(neg_vals)), 4),
                "min": round(float(np.min(neg_vals)), 4),
                "max": round(float(np.max(neg_vals)), 4),
            },
            "IN_1_positive": {
                "count": len(pos_vals),
                "mean": round(float(np.mean(pos_vals)), 4),
                "std": round(float(np.std(pos_vals)), 4),
                "min": round(float(np.min(pos_vals)), 4),
                "max": round(float(np.max(pos_vals)), 4),
            },
            "mean_separation": round(float(np.mean(pos_vals) - np.mean(neg_vals)), 4),
        }

    # ==========================================
    # 8. SEMANTIC ERROR ANALYSIS (Top 20 FP & FN)
    # ==========================================
    def get_errors(y_true, y_pred, ranking_scores):
        df_eval = pd.DataFrame(instances)
        df_eval["y_true"] = y_true
        df_eval["y_pred"] = y_pred
        df_eval["rank_score"] = ranking_scores

        fps = df_eval[(df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)].sort_values(
            by="rank_score", ascending=False
        ).head(20)

        fns = df_eval[(df_eval["y_true"] == 1) & (df_eval["y_pred"] == 0)].sort_values(
            by="rank_score", ascending=True
        ).head(20)

        def format_rows(sub_df):
            rows = []
            for _, r in sub_df.iterrows():
                rows.append({
                    "meme_id": int(r["meme_id"]),
                    "language": str(r["language"]),
                    "text": str(r["text"]),
                    "ground_truth_IN": int(r["IN"]),
                    "toxic_probability": round(float(r["p_toxic"]), 4),
                    "identity_hate_probability": round(float(r["p_identity_hate"]), 4),
                    "predicted_result": int(r["y_pred"]),
                    "image_filename": str(r["image_filename"]),
                })
            return rows

        return {
            "top_20_false_positives": format_rows(fps),
            "top_20_false_negatives": format_rows(fns),
        }

    error_analysis = {
        "A_toxic_only": get_errors(y_in, pred_toxic, p_toxic),
        "B_identity_hate_only": get_errors(y_in, pred_idhate, p_identity_hate),
        "C_toxic_or_identity_hate": get_errors(
            y_in, pred_combined, np.maximum(p_toxic, p_identity_hate)
        ),
    }

    # ==========================================
    # 9. CHECKPOINT TARGET AUDIT FOR HEAD 5
    # ==========================================
    hasoc_train_csv = HASOC_DIR / "Hindi Train HASOC 2025 - Updated.csv"
    df_hasoc = pd.read_csv(hasoc_train_csv)
    df_hasoc = df_hasoc[df_hasoc["OCR"].notna() & (df_hasoc["OCR"].str.strip() != "")].reset_index(drop=True)

    is_abuse = (df_hasoc["Abuse"].str.strip().str.lower() == "abusive")
    target_str = df_hasoc["Target"].astype(str).str.strip().str.lower()

    # Exact group conditions in train_meme_text_baseline.py:
    # is_group = any(g in target_str for g in ("gender", "community", "religion", "caste"))
    is_group = target_str.apply(lambda s: any(g in s for g in ("gender", "community", "religion", "caste")))
    is_identity_hate_target = is_abuse & is_group

    total_hasoc = len(df_hasoc)
    pos_idhate_total = int(is_identity_hate_target.sum())
    neg_idhate_total = total_hasoc - pos_idhate_total

    # In train split (80% seed 42 stratified by targets[:, 0])
    from sklearn.model_selection import train_test_split
    indices = np.arange(total_hasoc)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=is_abuse.values.astype(int)
    )

    pos_idhate_train = int(is_identity_hate_target.iloc[train_idx].sum())
    neg_idhate_train = len(train_idx) - pos_idhate_train
    pos_idhate_val = int(is_identity_hate_target.iloc[val_idx].sum())
    neg_idhate_val = len(val_idx) - pos_idhate_val

    # Value counts of Target when Abuse == Abusive
    target_dist_abusive = df_hasoc[is_abuse]["Target"].value_counts(dropna=False).to_dict()

    head5_training_audit = {
        "dataset_file": str(hasoc_train_csv.name),
        "total_valid_samples": total_hasoc,
        "exact_source_fields": ["Abuse", "Target"],
        "exact_group_values_checked": ["gender", "community", "religion", "caste"],
        "target_condition": "Abuse == 'abusive' AND any(g in Target.lower() for g in ['gender', 'community', 'religion', 'caste'])",
        "target_distribution_when_abusive": {str(k): int(v) for k, v in target_dist_abusive.items()},
        "positive_count_total": pos_idhate_total,
        "negative_count_total": neg_idhate_total,
        "positive_count_train_split": pos_idhate_train,
        "negative_count_train_split": neg_idhate_train,
        "positive_count_val_split": pos_idhate_val,
        "negative_count_val_split": neg_idhate_val,
        "was_supervised_in_mask": True,
        "mask_value": 1.0,
        "pos_weight_in_loss": round(float(neg_idhate_train / max(pos_idhate_train, 1)), 3),
    }

    # ==========================================
    # SAVE ALL RESULTS
    # ==========================================
    output_data = {
        "model_check": model_check,
        "results_in_consensus": results_in,
        "results_language": results_language,
        "results_cultural": results_cultural,
        "score_distributions": score_distributions,
        "head5_training_audit": head5_training_audit,
        "error_analysis_summary": {
            "toxic_fp_count": len(error_analysis["A_toxic_only"]["top_20_false_positives"]),
            "toxic_fn_count": len(error_analysis["A_toxic_only"]["top_20_false_negatives"]),
            "idhate_fp_count": len(error_analysis["B_identity_hate_only"]["top_20_false_positives"]),
            "idhate_fn_count": len(error_analysis["B_identity_hate_only"]["top_20_false_negatives"]),
            "combined_fp_count": len(error_analysis["C_toxic_or_identity_hate"]["top_20_false_positives"]),
            "combined_fn_count": len(error_analysis["C_toxic_or_identity_hate"]["top_20_false_negatives"]),
        },
        "top_errors": error_analysis,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nAll Model B ontology evaluation results saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
