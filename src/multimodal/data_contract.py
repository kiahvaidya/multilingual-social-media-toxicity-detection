"""
Data Contracts and Verification Engine for Multimodal Toxicity Detection.

Enforces strict invariants on multimodal datasets, samples, feature caches,
and metadata. Eliminates silent failure modes such as zero-vector image placeholders,
missing image files, and mismatched subgroup masks.
"""

import os
import re
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import torch
from PIL import Image

VALID_LANGUAGES: Set[str] = {"english", "hindi", "hinglish", "bengali", "tamil", "telugu"}
VALID_SOURCES: Set[str] = {"pridemm", "hasoc_2025", "trac_hinglish", "multi3hate", "indichm"}
VALID_MODALITIES: Set[str] = {"multimodal_paired", "text_only", "image_only"}


class DataContractViolation(Exception):
    """Raised when an explicit data contract invariant is violated."""
    pass


class MissingImageError(DataContractViolation):
    """Raised when an image required for multimodal supervision does not exist or cannot be read."""
    pass


class ZeroVectorPlaceholderError(DataContractViolation):
    """Raised when visual embeddings contain all-zero vectors used as silent substitutes for missing images."""
    pass


class IncompatibleCacheError(DataContractViolation):
    """Raised when cached tensors do not match schema, dimensions, or metadata contracts."""
    pass


class SubgroupMaskMismatchError(DataContractViolation):
    """Raised when a subgroup evaluation mask produces empty slices due to naming mismatches."""
    pass


@dataclass
class MultimodalSample:
    """Explicit contract for a single paired multimodal sample."""
    sample_id: str
    text: str
    label: float
    source: str
    language: str
    image_path: Optional[str] = None
    is_paired: bool = True

    def validate(self, verify_image_on_disk: bool = True) -> None:
        """Validates all sample invariants. Fails fast with clear actionable diagnostics."""
        if not self.sample_id or not isinstance(self.sample_id, str):
            raise DataContractViolation(f"Invalid sample_id: {self.sample_id!r}")

        if not isinstance(self.text, str) or len(self.text.strip()) == 0:
            raise DataContractViolation(f"Sample {self.sample_id}: text must be a non-empty string.")

        if self.label not in (0.0, 1.0, 0, 1):
            raise DataContractViolation(f"Sample {self.sample_id}: label must be binary 0.0 or 1.0, got {self.label!r}.")

        lang_norm = self.language.strip().lower()
        if lang_norm not in VALID_LANGUAGES:
            raise DataContractViolation(
                f"Sample {self.sample_id}: unknown language {self.language!r}. "
                f"Must be one of {sorted(VALID_LANGUAGES)}."
            )

        src_norm = self.source.strip().lower()
        if src_norm not in VALID_SOURCES:
            raise DataContractViolation(
                f"Sample {self.sample_id}: unknown source {self.source!r}. "
                f"Must be one of {sorted(VALID_SOURCES)}."
            )

        if self.is_paired:
            if not self.image_path:
                raise MissingImageError(
                    f"Sample {self.sample_id}: is_paired=True but image_path is None or empty. "
                    "Multimodal samples must have a valid physical image."
                )
            if verify_image_on_disk:
                p = Path(self.image_path)
                if not p.is_file():
                    raise MissingImageError(
                        f"Sample {self.sample_id}: image file not found at {p.resolve()}. "
                        "Cannot silently substitute missing images."
                    )
                try:
                    with Image.open(p) as img:
                        img.verify()
                except Exception as e:
                    raise MissingImageError(
                        f"Sample {self.sample_id}: unreadable or corrupted image file {p}: {e}"
                    )
        else:
            if self.image_path is not None:
                raise DataContractViolation(
                    f"Sample {self.sample_id}: is_paired=False but image_path is provided. "
                    "Unpaired samples must be declared explicitly as text_only."
                )


@dataclass
class FeatureCacheMetadata:
    """Explicit metadata header stored alongside any cached embedding tensor."""
    dataset_name: str
    split: str
    num_samples: int
    text_encoder: str
    image_encoder: str
    text_dim: int
    image_dim: int
    modality: str
    sample_ids: List[str]
    has_zero_vector_placeholders: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate_tensors(self, text_emb: torch.Tensor, image_emb: torch.Tensor, labels: torch.Tensor) -> None:
        """Validates that loaded tensors match the metadata specification and contain no zero-vector hacks."""
        if text_emb.shape[0] != self.num_samples:
            raise IncompatibleCacheError(
                f"Cache sample count mismatch: metadata specifies {self.num_samples}, "
                f"text_emb has {text_emb.shape[0]} rows."
            )
        if text_emb.shape[1] != self.text_dim:
            raise IncompatibleCacheError(
                f"Text dimension mismatch: metadata specifies {self.text_dim}, "
                f"text_emb has dim {text_emb.shape[1]}."
            )
        if image_emb.shape[0] != self.num_samples:
            raise IncompatibleCacheError(
                f"Image sample count mismatch: metadata specifies {self.num_samples}, "
                f"image_emb has {image_emb.shape[0]} rows."
            )
        if image_emb.shape[1] != self.image_dim:
            raise IncompatibleCacheError(
                f"Image dimension mismatch: metadata specifies {self.image_dim}, "
                f"image_emb has dim {image_emb.shape[1]}."
            )
        if labels.shape[0] != self.num_samples:
            raise IncompatibleCacheError(
                f"Labels count mismatch: metadata specifies {self.num_samples}, "
                f"labels has {labels.shape[0]} rows."
            )

        # Check for NaNs or Infinities
        if torch.isnan(text_emb).any() or torch.isinf(text_emb).any():
            raise IncompatibleCacheError("text_emb contains NaN or Inf values.")
        if torch.isnan(image_emb).any() or torch.isinf(image_emb).any():
            raise IncompatibleCacheError("image_emb contains NaN or Inf values.")

        # STRICT ZERO-VECTOR CHECK:
        # In a claimed multimodal paired cache, zero vectors indicate missing-image placeholders.
        if self.modality == "multimodal_paired":
            norms = torch.norm(image_emb, p=2, dim=-1)
            zero_count = int((norms < 1e-5).sum().item())
            if zero_count > 0:
                raise ZeroVectorPlaceholderError(
                    f"CRITICAL DATA INTEGRITY VIOLATION: Found {zero_count} zero-norm visual vectors "
                    f"in claimed multimodal cache '{self.dataset_name}_{self.split}'. "
                    "Missing images must NEVER be substituted with zero-vector placeholders."
                )


def verify_subgroup_mask(sources: np.ndarray, expected_key: str, min_count: int = 1) -> np.ndarray:
    """Verifies that a subgroup filter mask matches the intended records and is non-empty."""
    mask = (sources == expected_key)
    count = int(np.sum(mask))
    if count < min_count:
        unique_sources = list(np.unique(sources))
        raise SubgroupMaskMismatchError(
            f"Subgroup mask for {expected_key!r} produced {count} samples (expected >= {min_count}). "
            f"Available sources in dataset: {unique_sources}. "
            "Ensure source identifiers match exact registered keys (e.g. 'trac_hinglish' vs 'trac')."
        )
    return mask
