import json
from pathlib import Path

with open('scratch/exp_c_threshold_sensitivity_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for lbl, info in d.items():
    pos = info['positive_test_samples']
    tot = info['total_valid_test_samples']
    pct = (pos / tot) * 100
    print(f"\n=== {lbl.upper()} (Prevalence: {pos}/{tot} = {pct:.1f}%) ===")
    for pt in info['operating_curve']:
        th = pt['threshold']
        if th in [0.44, 0.46, 0.50, 0.54, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68]:
            print(f"  Thresh: {th:0.2f} | Prec: {pt['precision']:0.4f} | Rec: {pt['recall']:0.4f} | F1: {pt['f1']:0.4f} | FPR: {pt['fpr']:0.4f} | FNR: {pt['fnr']:0.4f} | FP: {pt['fp']} | TP: {pt['tp']}")
