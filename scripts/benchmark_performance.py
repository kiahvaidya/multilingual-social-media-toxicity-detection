"""
Performance Benchmarking Suite for Phase 5B.

Measures:
  - Cold-start model load time (seconds)
  - Process memory usage (RSS in MB)
  - Total processing time across batch sizes: [10, 100, 500, 1000]
  - Throughput (comments / second)

Compares:
  🧠 Fine-tuned MuRIL Transformer vs. ⚡ Logistic Regression Baseline
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import psutil
import torch

from src.config import config
from src.model import (
    load_model as load_logistic_model,
    load_muril_model,
    predict_batch as predict_batch_logistic,
    predict_batch_muril,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_current_rss_mb() -> float:
    """Return resident set size memory of current process in MB."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def load_benchmark_comments(target_count: int = 1000) -> List[str]:
    """Load representative real-world comments from dataset up to target_count."""
    dataset_path = config.DATASET_DIR / "train.csv"
    if not dataset_path.exists():
        dataset_path = config.DATASET_DIR / "real_world_domain_eval.csv"

    if dataset_path.exists():
        logger.info("Loading comments from %s...", dataset_path.name)
        df = pd.read_csv(dataset_path, nrows=target_count * 2)
        text_col = "comment_text" if "comment_text" in df.columns else df.columns[1]
        comments = df[text_col].dropna().astype(str).tolist()[:target_count]
    else:
        # Fallback synthetic pool if dataset missing
        logger.warning("Dataset CSV not found; generating synthetic sample pool.")
        base_samples = [
            "This is a wonderful video, loved every second of it!",
            "You are a completely worthless moron, shut up.",
            "Aapka video dekh kar bahut maza aaya bhai, keep it up.",
            "Saale kutte mai tujhe jaan se maar dunga agar dobara bola toh.",
            "Normal comment discussing technology and machine learning.",
        ]
        comments = (base_samples * (target_count // len(base_samples) + 1))[:target_count]

    logger.info("Loaded %d comments for benchmarking.", len(comments))
    return comments


def benchmark_engine(
    engine_name: str,
    load_fn,
    predict_batch_fn,
    comments_pool: List[str],
    batch_sizes: List[int],
) -> Dict[str, Any]:
    """Benchmark a single inference engine across cold start and batch sizes."""
    logger.info("==================================================")
    logger.info("  Benchmarking Engine: %s", engine_name)
    logger.info("==================================================")

    mem_before_load = get_current_rss_mb()

    # 1. Measure Cold Start / Load Time
    t_start = time.perf_counter()
    loaded_artifacts = load_fn()
    cold_start_time = time.perf_counter() - t_start
    mem_after_load = get_current_rss_mb()
    mem_delta_load = round(mem_after_load - mem_before_load, 2)

    logger.info(
        "[%s] Cold-start load time: %.3fs | Memory: %.1f MB (+%.1f MB)",
        engine_name, cold_start_time, mem_after_load, mem_delta_load
    )

    batch_metrics = {}

    # 2. Benchmark each batch size
    for size in batch_sizes:
        subset = comments_pool[:size]
        actual_size = len(subset)

        mem_before_infer = get_current_rss_mb()
        t0 = time.perf_counter()

        # Run batch prediction
        if "MuRIL" in engine_name:
            model, tokenizer, labels, thresholds, device = loaded_artifacts
            results = predict_batch_fn(
                subset,
                model,
                tokenizer,
                labels,
                device,
                batch_size=config.BATCH_SIZE,
                thresholds=thresholds,
            )
        else:
            model, vectorizer, labels = loaded_artifacts
            results = predict_batch_fn(subset, model, vectorizer, labels)

        infer_time = time.perf_counter() - t0
        mem_after_infer = get_current_rss_mb()
        throughput = round(actual_size / max(infer_time, 1e-6), 2)

        logger.info(
            "[%s] Batch %4d | Time: %6.3fs | Throughput: %7.2f comments/s | RSS: %6.1f MB",
            engine_name, actual_size, infer_time, throughput, mem_after_infer
        )

        batch_metrics[str(actual_size)] = {
            "batch_size": actual_size,
            "total_time_seconds": round(infer_time, 3),
            "throughput_comments_per_sec": throughput,
            "memory_rss_mb": mem_after_infer,
        }

    return {
        "engine_name": engine_name,
        "cold_start_seconds": round(cold_start_time, 3),
        "initial_memory_mb": mem_before_load,
        "post_load_memory_mb": mem_after_load,
        "load_memory_delta_mb": mem_delta_load,
        "batches": batch_metrics,
    }


def main():
    logger.info("Initializing Phase 5B Performance Benchmarks...")
    comments_pool = load_benchmark_comments(1000)
    batch_sizes = [10, 100, 500, 1000]

    # Benchmark MuRIL
    muril_results = benchmark_engine(
        engine_name="MuRIL Transformer",
        load_fn=load_muril_model,
        predict_batch_fn=predict_batch_muril,
        comments_pool=comments_pool,
        batch_sizes=batch_sizes,
    )

    # Benchmark Logistic Regression
    logistic_results = benchmark_engine(
        engine_name="Logistic Regression Baseline",
        load_fn=load_logistic_model,
        predict_batch_fn=predict_batch_logistic,
        comments_pool=comments_pool,
        batch_sizes=batch_sizes,
    )

    output_dir = _PROJECT_ROOT / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "performance_benchmark.json"

    benchmark_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_info": {
            "device": "CUDA (GPU)" if torch.cuda.is_available() else "CPU",
            "cpu_threads": torch.get_num_threads(),
            "python_version": sys.version.split()[0],
        },
        "muril": muril_results,
        "logistic_regression": logistic_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    logger.info("\nBenchmark results successfully saved to: %s", out_file)


if __name__ == "__main__":
    main()
