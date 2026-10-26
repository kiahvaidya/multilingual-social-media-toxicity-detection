import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

def run_experiment_c():
    npz_path = _PROJECT_ROOT / "evaluation_results" / "muril_val_reproducibility_data.npz"
    data = np.load(npz_path)
    
    val_probs = data["val_probs"]      # (699, 6)
    val_targets = data["val_targets"]  # (699, 6)
    val_masks = data["val_masks"]      # (699, 6)
    
    test_probs = data["test_probs"]    # (699, 6)
    test_targets = data["test_targets"]
    test_masks = data["test_masks"]
    
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    threshold_grid = np.arange(0.30, 0.701, 0.02)
    
    results = {}
    
    for i, label in enumerate(labels):
        test_m = (test_masks[:, i] == 1)
        y_true = test_targets[test_m, i]
        scores = test_probs[test_m, i]
        
        curve = []
        for th in threshold_grid:
            th = round(float(th), 2)
            y_pred = (scores >= th).astype(int)
            
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            
            curve.append({
                "threshold": th,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4),
                "fnr": round(fnr, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            })
            
        # Find threshold where FPR <= 0.05, FPR <= 0.10, and best F1
        best_f1_pt = max(curve, key=lambda x: x["f1"])
        low_fpr_10 = next((pt for pt in reversed(curve) if pt["fpr"] <= 0.10), curve[-1])
        low_fpr_05 = next((pt for pt in reversed(curve) if pt["fpr"] <= 0.05), curve[-1])
        
        results[label] = {
            "total_valid_test_samples": int(len(scores)),
            "positive_test_samples": int(np.sum(y_true)),
            "negative_test_samples": int(len(scores) - np.sum(y_true)),
            "best_f1_operating_point": best_f1_pt,
            "low_fpr_10_operating_point": low_fpr_10,
            "low_fpr_05_operating_point": low_fpr_05,
            "operating_curve": curve
        }
        
    out_file = _PROJECT_ROOT / "scratch" / "exp_c_threshold_sensitivity_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Experiment C completed. Output saved to:", out_file)

if __name__ == "__main__":
    run_experiment_c()
