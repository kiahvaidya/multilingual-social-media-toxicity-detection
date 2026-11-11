import os
import json
import glob

def summarize_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return f"ERROR: {e}"

print("=== 1. EVALUATION RESULTS SUMMARY ===")
eval_dir = "evaluation_results"
for f in sorted(os.listdir(eval_dir)):
    if f.endswith('.json'):
        p = os.path.join(eval_dir, f)
        data = summarize_json(p)
        print(f"\n--- {f} ---")
        if isinstance(data, dict):
            # Print top-level keys and selected metrics
            keys = list(data.keys())
            print(f"Keys: {keys[:15]}")
            # print snippet
            snippet = {k: data[k] for k in keys[:6] if not isinstance(data[k], (list, dict)) or len(str(data[k])) < 200}
            print(f"Sample: {snippet}")
        elif isinstance(data, list):
            print(f"List with {len(data)} items. First item: {data[0] if data else None}")

print("\n=== 2. DATASET MANIFESTS & AUDITS ===")
for p in ["data/raw/ACQUISITION_MANIFEST.json", "data/processed/DATASET_AUDIT_AND_NORMALIZATION_REPORT.json"]:
    if os.path.exists(p):
        print(f"\n--- {p} ---")
        data = summarize_json(p)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float, str, bool)):
                    print(f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  {k}: dict with keys {list(v.keys())[:10]}")
                elif isinstance(v, list):
                    print(f"  {k}: list with {len(v)} items")

print("\n=== 3. MEME EXPERIMENTS REPORTS & METRICS ===")
meme_files = [
    "scratch/meme_experiments/final_multimodal/FINAL_REPORT.md",
    "scratch/meme_experiments/final_multimodal/config.json",
    "scratch/meme_experiments/final_multimodal/validation_metrics.json",
    "scratch/meme_experiments/final_multimodal/multi3hate_metrics.json",
    "scratch/meme_experiments/final_multimodal/pridemm_test_metrics.json",
    "scratch/meme_experiments/final_multimodal/ablation_results.json",
    "scratch/meme_experiments/final_hard_negative_pass/FINAL_REPORT.md",
    "scratch/meme_experiments/final_hard_negative_pass/validation_metrics.json",
    "scratch/meme_experiments/model_d_pridemm/FINAL_EXPERIMENT_REPORT.md",
    "scratch/meme_experiments/model_d_pridemm/metrics.json",
]
for p in meme_files:
    if os.path.exists(p):
        print(f"\n--- {p} ---")
        if p.endswith('.json'):
            d = summarize_json(p)
            print(json.dumps(d, indent=2)[:500])
        else:
            with open(p, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print("".join(lines[:35]))

