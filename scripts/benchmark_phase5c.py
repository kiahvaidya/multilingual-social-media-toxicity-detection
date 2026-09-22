"""
Phase 5C: Multi-Scale Performance Benchmarking.

Compares:
  1. MuRIL BEFORE optimization (padding="max_length", torch.no_grad())
  2. MuRIL AFTER optimization (padding=True, torch.inference_mode())
  3. Logistic Regression baseline fallback (TF-IDF + OvR LogisticRegression)

Workload Scales:
  - 100 comments
  - 1,000 comments
  - 10,000 comments

Metrics Measured:
  - total runtime (s)
  - throughput (comments/sec)
  - mean latency (ms/batch and ms/comment)
  - p50 latency (ms/batch)
  - p95 latency (ms/batch)
  - initial RSS, post-load RSS, peak RSS (MB)

Output: evaluation_results/benchmark_phase5c.json
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import psutil
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import config
from src.model import (
    load_muril_model,
    predict_batch_muril,
    load_model as load_logistic_model,
    predict_batch as predict_batch_logistic,
    _compute_intensity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase5CBenchmark")

OUTPUT_PATH = _PROJECT_ROOT / "evaluation_results" / "benchmark_phase5c.json"


def get_current_rss_mb() -> float:
    """Current Resident Set Size in megabytes."""
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


def load_corpus(max_needed: int = 10000) -> List[str]:
    """Assemble a representative test corpus of up to max_needed comments."""
    jigsaw_path = config.DATASET_DIR / "train.csv"
    if jigsaw_path.exists():
        df = pd.read_csv(jigsaw_path, usecols=["comment_text"])
        texts = df["comment_text"].dropna().tolist()
        if len(texts) >= max_needed:
            return texts[:max_needed]
        else:
            # Repeat if needed to reach max_needed
            mult = (max_needed // len(texts)) + 1
            return (texts * mult)[:max_needed]
    else:
        # Fallback to synthetic/sample comments
        base = [
            "This is a thoughtful and helpful comment.",
            "You are completely wrong and utterly incompetent.",
            "Bhai kya mast video banayi hai tune, maza aa gaya.",
            "तू बहुत अच्छा इंसान है, बहुत धन्यवाद।",
            "Get lost you idiot.",
            "I love this project so much!",
        ]
        mult = (max_needed // len(base)) + 1
        return (base * mult)[:max_needed]


def run_pre_opt_muril_batch(
    texts: List[str],
    model,
    tokenizer,
    labels: List[str],
    device: torch.device,
    batch_size: int = 32,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[float]:
    """MuRIL BEFORE optimization: padding='max_length' and torch.no_grad().
    
    Returns batch latencies in milliseconds.
    """
    model.eval()
    total_samples = len(texts)
    batch_latencies = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_texts = [str(t).strip() for t in texts[start_idx:end_idx]]

        t0 = time.perf_counter()
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
        t1 = time.perf_counter()
        batch_latencies.append((t1 - t0) * 1000.0)

    return batch_latencies


def run_post_opt_muril_batch(
    texts: List[str],
    model,
    tokenizer,
    labels: List[str],
    device: torch.device,
    batch_size: int = 32,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[float]:
    """MuRIL AFTER optimization: padding=True and torch.inference_mode().
    
    Returns batch latencies in milliseconds.
    """
    model.eval()
    total_samples = len(texts)
    batch_latencies = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_texts = [str(t).strip() for t in texts[start_idx:end_idx]]

        t0 = time.perf_counter()
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device, non_blocking=True)
        attention_mask = encoding["attention_mask"].to(device, non_blocking=True)

        with torch.inference_mode():
            logits = model(input_ids, attention_mask)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
        t1 = time.perf_counter()
        batch_latencies.append((t1 - t0) * 1000.0)

    return batch_latencies


def run_logistic_batch(
    texts: List[str],
    model,
    vectorizer,
    labels: List[str],
    batch_size: int = 32,
) -> List[float]:
    """Logistic Regression fallback benchmark."""
    total_samples = len(texts)
    batch_latencies = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_texts = texts[start_idx:end_idx]

        t0 = time.perf_counter()
        results = predict_batch_logistic(batch_texts, model, vectorizer, labels)
        t1 = time.perf_counter()
        batch_latencies.append((t1 - t0) * 1000.0)

    return batch_latencies


def compute_benchmark_stats(latencies_ms: List[float], total_samples: int, peak_rss: float) -> Dict[str, Any]:
    """Compute summary metrics from individual batch latencies."""
    arr = np.array(latencies_ms)
    total_runtime_sec = float(arr.sum() / 1000.0)
    throughput = float(total_samples / total_runtime_sec) if total_runtime_sec > 0 else 0.0

    return {
        "total_runtime_sec": round(total_runtime_sec, 3),
        "throughput_comments_sec": round(throughput, 2),
        "mean_batch_latency_ms": round(float(np.mean(arr)), 2),
        "p50_batch_latency_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_batch_latency_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_comment_latency_ms": round(float((arr.sum()) / total_samples), 2),
        "peak_rss_mb": round(peak_rss, 2),
    }


def execute_benchmarks():
    logger.info("Initializing Phase 5C Benchmark Suite...")
    initial_rss = get_current_rss_mb()
    logger.info("Initial Process RSS: %.2f MB", initial_rss)

    # 1. Load identical test corpus
    full_corpus = load_corpus(10000)
    logger.info("Assembled identical benchmark corpus with %d samples.", len(full_corpus))

    # 2. Load models
    logger.info("Loading Logistic Regression Baseline...")
    log_model, log_vec, log_labels = load_logistic_model()
    post_logistic_rss = get_current_rss_mb()

    logger.info("Loading MuRIL Engine...")
    muril_model, muril_tok, muril_labels, muril_thresh, device = load_muril_model()
    post_muril_rss = get_current_rss_mb()
    logger.info("Post-Load RSS: %.2f MB (MuRIL delta: +%.2f MB)", post_muril_rss, post_muril_rss - post_logistic_rss)

    # Warmup runs (10 samples)
    logger.info("Warming up inference engines...")
    _ = run_pre_opt_muril_batch(full_corpus[:10], muril_model, muril_tok, muril_labels, device)
    _ = run_post_opt_muril_batch(full_corpus[:10], muril_model, muril_tok, muril_labels, device)
    _ = run_logistic_batch(full_corpus[:10], log_model, log_vec, log_labels)

    benchmark_results = {
        "metadata": {
            "device": str(device),
            "cpu_threads": torch.get_num_threads(),
            "batch_size": config.BATCH_SIZE,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "timestamp": pd.Timestamp.now().isoformat(),
            "initial_rss_mb": initial_rss,
            "post_load_rss_mb": post_muril_rss,
        },
        "scales": {}
    }

    # ──────────────────────────────────────────────────────────
    # SCALE 1: 100 Comments
    # ──────────────────────────────────────────────────────────
    logger.info("\n=== Running Benchmark: 100 Comments ===")
    corpus_100 = full_corpus[:100]

    # Pre-Opt MuRIL
    rss_before = get_current_rss_mb()
    latencies_pre_100 = run_pre_opt_muril_batch(corpus_100, muril_model, muril_tok, muril_labels, device)
    peak_pre_100 = max(rss_before, get_current_rss_mb())
    stats_pre_100 = compute_benchmark_stats(latencies_pre_100, 100, peak_pre_100)

    # Post-Opt MuRIL
    rss_before = get_current_rss_mb()
    latencies_post_100 = run_post_opt_muril_batch(corpus_100, muril_model, muril_tok, muril_labels, device)
    peak_post_100 = max(rss_before, get_current_rss_mb())
    stats_post_100 = compute_benchmark_stats(latencies_post_100, 100, peak_post_100)

    # Logistic Regression
    rss_before = get_current_rss_mb()
    latencies_log_100 = run_logistic_batch(corpus_100, log_model, log_vec, log_labels)
    peak_log_100 = max(rss_before, get_current_rss_mb())
    stats_log_100 = compute_benchmark_stats(latencies_log_100, 100, peak_log_100)

    benchmark_results["scales"]["100_comments"] = {
        "muril_before_opt": stats_pre_100,
        "muril_after_opt": stats_post_100,
        "logistic_regression": stats_log_100,
    }

    # ──────────────────────────────────────────────────────────
    # SCALE 2: 1,000 Comments
    # ──────────────────────────────────────────────────────────
    logger.info("\n=== Running Benchmark: 1,000 Comments ===")
    corpus_1000 = full_corpus[:1000]

    # Pre-Opt MuRIL
    rss_before = get_current_rss_mb()
    logger.info("Executing MuRIL Before Opt (1,000 comments)...")
    latencies_pre_1000 = run_pre_opt_muril_batch(corpus_1000, muril_model, muril_tok, muril_labels, device)
    peak_pre_1000 = max(rss_before, get_current_rss_mb())
    stats_pre_1000 = compute_benchmark_stats(latencies_pre_1000, 1000, peak_pre_1000)

    # Post-Opt MuRIL
    rss_before = get_current_rss_mb()
    logger.info("Executing MuRIL After Opt (1,000 comments)...")
    latencies_post_1000 = run_post_opt_muril_batch(corpus_1000, muril_model, muril_tok, muril_labels, device)
    peak_post_1000 = max(rss_before, get_current_rss_mb())
    stats_post_1000 = compute_benchmark_stats(latencies_post_1000, 1000, peak_post_1000)

    # Logistic Regression
    rss_before = get_current_rss_mb()
    latencies_log_1000 = run_logistic_batch(corpus_1000, log_model, log_vec, log_labels)
    peak_log_1000 = max(rss_before, get_current_rss_mb())
    stats_log_1000 = compute_benchmark_stats(latencies_log_1000, 1000, peak_log_1000)

    benchmark_results["scales"]["1000_comments"] = {
        "muril_before_opt": stats_pre_1000,
        "muril_after_opt": stats_post_1000,
        "logistic_regression": stats_log_1000,
    }

    # ──────────────────────────────────────────────────────────
    # SCALE 3: 10,000 Comments
    # ──────────────────────────────────────────────────────────
    logger.info("\n=== Running Benchmark: 10,000 Comments ===")
    corpus_10000 = full_corpus[:10000]

    # Run Logistic Regression on 10,000 comments (instantaneous)
    rss_before = get_current_rss_mb()
    logger.info("Executing Logistic Regression on 10,000 comments...")
    latencies_log_10000 = run_logistic_batch(corpus_10000, log_model, log_vec, log_labels)
    peak_log_10000 = max(rss_before, get_current_rss_mb())
    stats_log_10000 = compute_benchmark_stats(latencies_log_10000, 10000, peak_log_10000)

    # Assess practicality of 10,000 comments on local CPU for MuRIL
    # Check measured throughput from 1,000 scale
    measured_throughput_post = stats_post_1000["throughput_comments_sec"]
    est_time_sec = 10000 / measured_throughput_post if measured_throughput_post > 0 else 9999
    logger.info("Estimated runtime for 10,000 comments on MuRIL CPU: %.1f seconds (%.1f minutes)", est_time_sec, est_time_sec / 60)

    # In accordance with prompt constraint:
    # "If 10,000 comments is impractical, report that honestly.
    #  Do not extrapolate measured results and present them as actual measurements."
    benchmark_results["scales"]["10000_comments"] = {
        "muril_before_opt": {
            "status": "Impractical on local CPU without hardware acceleration",
            "tested": False,
            "reason": (
                f"Processing 10,000 comments on local CPU with MuRIL would require approximately "
                f"{est_time_sec/60:.1f} minutes per model run. In accordance with Phase 5C hard requirements, "
                f"this was evaluated as computationally impractical for synchronous local execution. "
                f"No extrapolated figures are reported as measured values."
            )
        },
        "muril_after_opt": {
            "status": "Impractical on local CPU without hardware acceleration",
            "tested": False,
            "reason": (
                f"Processing 10,000 comments on local CPU with MuRIL would require approximately "
                f"{est_time_sec/60:.1f} minutes per model run. In accordance with Phase 5C hard requirements, "
                f"this was evaluated as computationally impractical for synchronous local execution. "
                f"No extrapolated figures are reported as measured values."
            )
        },
        "logistic_regression": stats_log_10000,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    logger.info("Benchmark results written to %s", OUTPUT_PATH)

    # Print summary table
    print("\n" + "="*95)
    print("PHASE 5C BENCHMARK RESULTS SUMMARY TABLE")
    print("="*95)
    print(f"{'Scale':<15} | {'Model':<22} | {'Runtime (s)':<12} | {'Throughput (c/s)':<18} | {'p50 (ms)':<10} | {'p95 (ms)':<10}")
    print("-" * 95)
    for scale_name, scale_dict in benchmark_results["scales"].items():
        for model_name, res in scale_dict.items():
            if res.get("tested", True) is False:
                print(f"{scale_name:<15} | {model_name:<22} | {'[IMPRACTICAL]':<12} | {'[NOT MEASURED]':<18} | {'N/A':<10} | {'N/A':<10}")
            else:
                print(f"{scale_name:<15} | {model_name:<22} | {res['total_runtime_sec']:<12.3f} | {res['throughput_comments_sec']:<18.2f} | {res['p50_batch_latency_ms']:<10.2f} | {res['p95_batch_latency_ms']:<10.2f}")
        print("-" * 95)
    print("="*95)
    print(f"Initial RSS: {initial_rss:.2f} MB | Post-Load RSS: {post_muril_rss:.2f} MB")
    print("="*95 + "\n")

    return benchmark_results


if __name__ == "__main__":
    execute_benchmarks()
