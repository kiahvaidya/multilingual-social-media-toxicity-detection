import json
import os

files_to_dump = [
    ("muril_phase4b", "evaluation_results/muril_phase4b_evaluation_results.json"),
    ("muril_phase4c", "evaluation_results/muril_phase4c_threshold_error_analysis.json"),
    ("phase3a", "evaluation_results/evaluation_results_phase3a.json"),
    ("phase3b", "evaluation_results/domain_evaluation_results_phase3b.json"),
    ("phase6a", "evaluation_results/phase6a_experiment_results.json"),
    ("benchmark", "evaluation_results/performance_benchmark.json"),
    ("benchmark_5c", "evaluation_results/benchmark_phase5c.json"),
    ("dataset_audit_5c", "evaluation_results/dataset_audit_phase5c.json"),
    ("data_norm_report", "data/processed/DATASET_AUDIT_AND_NORMALIZATION_REPORT.json"),
    ("final_mm_val", "scratch/meme_experiments/final_multimodal/validation_metrics.json"),
    ("final_mm_m3h", "scratch/meme_experiments/final_multimodal/multi3hate_metrics.json"),
    ("final_mm_pridemm", "scratch/meme_experiments/final_multimodal/pridemm_test_metrics.json"),
    ("final_mm_ablation", "scratch/meme_experiments/final_multimodal/ablation_results.json"),
    ("final_mm_shuffle", "scratch/meme_experiments/final_multimodal/shuffle_sanity_results.json"),
    ("final_mm_config", "scratch/meme_experiments/final_multimodal/config.json"),
    ("hard_neg_lang", "scratch/meme_experiments/final_hard_negative_pass/per_language_metrics.json"),
]

for label, path in files_to_dump:
    print(f"\n==================== {label} ({path}) ====================")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                d = json.load(f)
                s = json.dumps(d, indent=2)
                if len(s) > 2500:
                    print(s[:2500] + "\n... [TRUNCATED] ...")
                else:
                    print(s)
            except Exception as e:
                print("Error loading json:", e)
    else:
        print("FILE NOT FOUND")

