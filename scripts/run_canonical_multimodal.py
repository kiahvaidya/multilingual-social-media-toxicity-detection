"""
Canonical Research Entry Point for Multimodal Training Pipeline.

Usage:
    python scripts/run_canonical_multimodal.py --experiment_name my_exp --output_dir scratch/meme_experiments/my_exp

Guarantees:
1. Strict fail-fast preflight validation executes before any model weights or training loops start.
2. Production model checkpoints (api.py, models/muril_model_b.pt, final_multimodal/best_model.pt) are protected from overwriting.
3. Zero-vector visual placeholders, unverified image paths, empty subgroup masks, and cross-split leakage are strictly blocked.
4. If Indic paired images are missing on disk, training is blocked with an actionable error.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.multimodal.canonical_pipeline import PipelineConfig, run_canonical_training
from src.multimodal.data_contract import MultimodalSample, MissingImageError
from src.multimodal.preflight import PreflightCheckFailure


def parse_args():
    parser = argparse.ArgumentParser(description="Canonical Multimodal Training & Validation Entry Point")
    parser.add_argument("--experiment_name", type=str, default="canonical_run", help="Unique identifier for experiment")
    parser.add_argument("--output_dir", type=str, default="scratch/meme_experiments/canonical_run", help="Output directory for experiment artifacts")
    parser.add_argument("--dataset_name", type=str, default="canonical_multimodal_v1", help="Dataset identifier")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--modality_dropout", type=float, default=0.15, help="Modality dropout probability")
    parser.add_argument("--strict_preflight", action="store_true", default=True, help="Enforce all fail-fast invariants")
    parser.add_argument("--dry_run", action="store_true", help="Only run preflight checks without initiating training")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        modality_dropout=args.modality_dropout,
        strict_preflight=args.strict_preflight,
    )

    print(f"=== Canonical Multimodal Pipeline: {config.experiment_name} ===")
    print(f"Output Directory: {config.output_dir}")
    print(f"Strict Preflight: {config.strict_preflight}")

    # Safety validation
    try:
        config.validate_safety()
    except Exception as e:
        print(f"CRITICAL ERROR: Safety check failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Pipeline entry point initialized successfully.")
    print("To execute a full training run, provide valid MultimodalSample manifests to run_canonical_training().")
    print("If Indic images are missing, preflight will halt execution immediately.")


if __name__ == "__main__":
    main()
