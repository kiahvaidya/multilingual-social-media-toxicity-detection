import sys
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def run_experiment_a():
    npz_path = _PROJECT_ROOT / "evaluation_results" / "muril_val_reproducibility_data.npz"
    data = np.load(npz_path)
    
    val_probs = data["val_probs"]      # (699, 6)
    val_targets = data["val_targets"]  # (699, 6)
    val_masks = data["val_masks"]      # (699, 6)
    
    test_probs = data["test_probs"]    # (699, 6)
    test_targets = data["test_targets"]
    test_masks = data["test_masks"]
    
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    frozen_thresholds = [0.45, 0.62, 0.65, 0.59, 0.61, 0.60]
    
    results = {}
    
    # Also test mapping for clean text floor (prob ~0.30 - 0.40)
    probe_clean_probs = np.array([0.28, 0.30, 0.32, 0.35, 0.40, 0.50, 0.60, 0.65, 0.68])
    
    for i, (label, th_uncal) in enumerate(zip(labels, frozen_thresholds)):
        # Validation split: only valid masked samples
        val_m = (val_masks[:, i] == 1)
        y_val = val_targets[val_m, i]
        p_val = val_probs[val_m, i]
        # Invert sigmoid to recover logits for Platt scaling
        p_val_clipped = np.clip(p_val, 1e-6, 1 - 1e-6)
        logit_val = np.log(p_val_clipped / (1 - p_val_clipped)).reshape(-1, 1)
        
        # Test split: only valid masked samples
        test_m = (test_masks[:, i] == 1)
        y_test = test_targets[test_m, i]
        p_test = test_probs[test_m, i]
        p_test_clipped = np.clip(p_test, 1e-6, 1 - 1e-6)
        logit_test = np.log(p_test_clipped / (1 - p_test_clipped)).reshape(-1, 1)
        
        # 1. Uncalibrated baseline on TEST set
        brier_uncal = float(brier_score_loss(y_test, p_test))
        ece_uncal = compute_ece(y_test, p_test)
        pred_uncal = (p_test >= th_uncal).astype(int)
        
        tp = int(np.sum((y_test == 1) & (pred_uncal == 1)))
        fp = int(np.sum((y_test == 0) & (pred_uncal == 1)))
        fn = int(np.sum((y_test == 1) & (pred_uncal == 0)))
        tn = int(np.sum((y_test == 0) & (pred_uncal == 0)))
        
        prec_uncal = float(precision_score(y_test, pred_uncal, zero_division=0))
        rec_uncal = float(recall_score(y_test, pred_uncal, zero_division=0))
        f1_uncal = float(f1_score(y_test, pred_uncal, zero_division=0))
        fpr_uncal = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr_uncal = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        # 2. Fit Platt Scaling (Logistic Regression on validation logits)
        platt = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
        platt.fit(logit_val, y_val)
        p_test_platt = platt.predict_proba(logit_test)[:, 1]
        brier_platt = float(brier_score_loss(y_test, p_test_platt))
        ece_platt = compute_ece(y_test, p_test_platt)
        
        # Map probe clean probs through Platt
        probe_logits = np.log(np.clip(probe_clean_probs, 1e-6, 1 - 1e-6) / (1 - np.clip(probe_clean_probs, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        probe_platt_mapped = platt.predict_proba(probe_logits)[:, 1].tolist()
        
        # Platt predictions at standard 0.5 threshold
        pred_platt_05 = (p_test_platt >= 0.5).astype(int)
        prec_platt = float(precision_score(y_test, pred_platt_05, zero_division=0))
        rec_platt = float(recall_score(y_test, pred_platt_05, zero_division=0))
        f1_platt = float(f1_score(y_test, pred_platt_05, zero_division=0))
        fp_p = int(np.sum((y_test == 0) & (pred_platt_05 == 1)))
        fn_p = int(np.sum((y_test == 1) & (pred_platt_05 == 0)))
        tp_p = int(np.sum((y_test == 1) & (pred_platt_05 == 1)))
        tn_p = int(np.sum((y_test == 0) & (pred_platt_05 == 0)))
        fpr_platt = float(fp_p / (fp_p + tn_p)) if (fp_p + tn_p) > 0 else 0.0
        fnr_platt = float(fn_p / (fn_p + tp_p)) if (fn_p + tp_p) > 0 else 0.0
        
        # 3. Fit Isotonic Regression on validation probabilities
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_val, y_val)
        p_test_iso = iso.predict(p_test)
        brier_iso = float(brier_score_loss(y_test, p_test_iso))
        ece_iso = compute_ece(y_test, p_test_iso)
        
        pred_iso_05 = (p_test_iso >= 0.5).astype(int)
        prec_iso = float(precision_score(y_test, pred_iso_05, zero_division=0))
        rec_iso = float(recall_score(y_test, pred_iso_05, zero_division=0))
        f1_iso = float(f1_score(y_test, pred_iso_05, zero_division=0))
        fp_i = int(np.sum((y_test == 0) & (pred_iso_05 == 1)))
        fn_i = int(np.sum((y_test == 1) & (pred_iso_05 == 0)))
        tp_i = int(np.sum((y_test == 1) & (pred_iso_05 == 1)))
        tn_i = int(np.sum((y_test == 0) & (pred_iso_05 == 0)))
        fpr_iso = float(fp_i / (fp_i + tn_i)) if (fp_i + tn_i) > 0 else 0.0
        fnr_iso = float(fn_i / (fn_i + tp_i)) if (fn_i + tp_i) > 0 else 0.0
        probe_iso_mapped = iso.predict(probe_clean_probs).tolist()

        results[label] = {
            "uncalibrated": {
                "threshold": th_uncal,
                "brier_score": round(brier_uncal, 4),
                "ece": round(ece_uncal, 4),
                "precision": round(prec_uncal, 4),
                "recall": round(rec_uncal, 4),
                "f1": round(f1_uncal, 4),
                "fpr": round(fpr_uncal, 4),
                "fnr": round(fnr_uncal, 4),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            },
            "platt_scaling": {
                "slope": round(float(platt.coef_[0][0]), 4),
                "intercept": round(float(platt.intercept_[0]), 4),
                "brier_score": round(brier_platt, 4),
                "ece": round(ece_platt, 4),
                "precision_at_05": round(prec_platt, 4),
                "recall_at_05": round(rec_platt, 4),
                "f1_at_05": round(f1_platt, 4),
                "fpr_at_05": round(fpr_platt, 4),
                "fnr_at_05": round(fnr_platt, 4),
                "clean_prob_mapping": {str(round(k, 2)): round(v, 4) for k, v in zip(probe_clean_probs, probe_platt_mapped)}
            },
            "isotonic": {
                "brier_score": round(brier_iso, 4),
                "ece": round(ece_iso, 4),
                "precision_at_05": round(prec_iso, 4),
                "recall_at_05": round(rec_iso, 4),
                "f1_at_05": round(f1_iso, 4),
                "fpr_at_05": round(fpr_iso, 4),
                "fnr_at_05": round(fnr_iso, 4),
                "clean_prob_mapping": {str(round(k, 2)): round(v, 4) for k, v in zip(probe_clean_probs, probe_iso_mapped)}
            }
        }
        
    out_file = _PROJECT_ROOT / "scratch" / "exp_a_calibration_results.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Calibration experiment completed successfully. Output saved to:", out_file)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_experiment_a()
