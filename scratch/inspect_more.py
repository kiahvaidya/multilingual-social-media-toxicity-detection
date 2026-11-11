import json

def read_and_print(path, keys=None):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f"\n===== {path} =====")
    if keys:
        for k in keys:
            if k in d:
                print(f"--- {k} ---")
                print(json.dumps(d[k], indent=2))
    else:
        print(json.dumps(d, indent=2))

read_and_print("evaluation_results/phase6a_experiment_results.json")
read_and_print("evaluation_results/benchmark_phase5c.json", ["metadata", "scales"])
read_and_print("data/raw/ACQUISITION_MANIFEST.json")
read_and_print("evaluation_results/muril_phase4c_threshold_error_analysis.json", ["untouched_jigsaw_test_evaluation", "final_comparison_table", "threat_failure_analysis"])

