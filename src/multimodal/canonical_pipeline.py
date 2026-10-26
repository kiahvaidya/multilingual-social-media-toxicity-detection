"""
Canonical Multimodal Training & Evaluation Pipeline Entry Point.

Provides a reproducible, safeguarded training pipeline that:
1. Executes mandatory preflight validation checks before any model instantiation.
2. Rejects zero-vector visual placeholders, missing images, and mismatched splits.
3. Isolates research experiment artifacts completely from production serving paths.
4. Prevents accidental overwriting of the official production baseline.
"""

import sys
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np

from .data_contract import (
    MultimodalSample,
    FeatureCacheMetadata,
    DataContractViolation,
    MissingImageError,
    ZeroVectorPlaceholderError,
)
from .preflight import (
    PreflightReport,
    PreflightCheckFailure,
    run_preflight_checks,
)

logger = logging.getLogger("toxiscope.multimodal.pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PRODUCTION_PATHS = {
    _PROJECT_ROOT / "models" / "muril_model_b.pt",
    _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal" / "best_model.pt",
    _PROJECT_ROOT / "api.py",
}


@dataclass
class PipelineConfig:
    """Explicit configuration contract for a multimodal training run."""
    experiment_name: str
    output_dir: str
    dataset_name: str
    text_encoder_name: str = "google/muril-base-cased"
    image_encoder_name: str = "openai/clip-vit-base-patch32"
    proj_dim: int = 256
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 5
    modality_dropout: float = 0.15
    seed: int = 42
    strict_preflight: bool = True

    def validate_safety(self) -> None:
        """Ensures that training outputs can never overwrite production paths."""
        out_p = Path(self.output_dir).resolve()
        for prod_p in PROTECTED_PRODUCTION_PATHS:
            if out_p == prod_p.resolve() or out_p == prod_p.parent.resolve():
                raise DataContractViolation(
                    f"CRITICAL SAFETY VIOLATION: Pipeline output directory {out_p} matches "
                    f"protected production path {prod_p}. Research experiments must output to an "
                    "isolated directory under scratch/meme_experiments/<run_name>/."
                )


def run_canonical_training(
    config: PipelineConfig,
    train_samples: List[MultimodalSample],
    val_samples: List[MultimodalSample],
    expected_subgroups: List[str],
    test_samples: Optional[List[MultimodalSample]] = None,
    train_tensors: Optional[Any] = None,
    check_disk_images: bool = True,
) -> PreflightReport:
    """Executes the canonical multimodal pipeline with strict fail-fast validation.

    Step 1: Verify output directory safety.
    Step 2: Run all 10 mandatory preflight checks.
    Step 3: Block execution if preflight fails (e.g. missing Indic images or zero vectors).
    """
    logger.info("Initializing Canonical Multimodal Pipeline for experiment: %s", config.experiment_name)
    config.validate_safety()

    out_path = Path(config.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save explicit configuration
    with open(out_path / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info("Executing mandatory Preflight Validation Checks...")
    try:
        report = run_preflight_checks(
            train_samples=train_samples,
            val_samples=val_samples,
            expected_subgroups=expected_subgroups,
            test_samples=test_samples,
            train_tensors=train_tensors,
            check_disk_images=check_disk_images,
        )
    except PreflightCheckFailure as e:
        logger.error("PREFLIGHT VALIDATION FAILED: %s", e)
        # Record failure diagnostics
        fail_record = {
            "passed": False,
            "error": str(e),
            "experiment": config.experiment_name,
        }
        with open(out_path / "preflight_failure.json", "w", encoding="utf-8") as f:
            json.dump(fail_record, f, indent=2)
        raise

    logger.info("Preflight Validation PASSED (%d/%d checks).", report.checks_passed, report.checks_run)
    with open(out_path / "preflight_report.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    return report
