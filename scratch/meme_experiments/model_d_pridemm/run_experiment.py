import os
import sys
import subprocess
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm"

def run_step(cmd, desc):
    print(f"\n{'='*60}\n>>> STARTING: {desc}\n{'='*60}")
    t0 = time.time()
    res = subprocess.run([sys.executable, str(cmd)], cwd=str(_PROJECT_ROOT), check=True)
    el = time.time() - t0
    print(f"\n>>> FINISHED: {desc} in {el:.1f}s\n{'='*60}\n")

if __name__ == "__main__":
    t_start = time.time()
    run_step(EXP_DIR / "extract_features.py", "STEP 1: Feature Extraction (MuRIL & CLIP)")
    run_step(EXP_DIR / "train_model_d.py", "STEP 2: Model D MLP Fusion Training")
    run_step(EXP_DIR / "evaluate_model_d.py", "STEP 3: Model D & Baseline Evaluation")
    print(f"\n*** COMPLETE MODEL D PIPELINE FINISHED IN {(time.time() - t_start)/60:.2f} MINUTES ***\n")
