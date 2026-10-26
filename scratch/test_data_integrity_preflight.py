"""
Unit tests for Multimodal Data Integrity and Fail-Fast Preflight Engine.

Verifies:
1. Rejection of missing, unverified, or mismatched image paths (MissingImageError).
2. Rejection of silent zero-vector visual placeholders (ZeroVectorPlaceholderError).
3. Cache metadata compatibility validation (IncompatibleCacheError).
4. Subgroup mask verification (detecting empty masks such as TRAC source label mismatches).
5. Cross-split sample ID and text duplicate leakage detection.
6. Validation distribution compatibility (preventing threshold tuning on mismatched unimodal data).
7. Production path protection against accidental overwrites.
8. Clear distinction between intentional training-time modality dropout and silent missing data.
"""

import os
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np

from src.multimodal.data_contract import (
    MultimodalSample,
    FeatureCacheMetadata,
    DataContractViolation,
    MissingImageError,
    ZeroVectorPlaceholderError,
    IncompatibleCacheError,
    SubgroupMaskMismatchError,
    verify_subgroup_mask,
)
from src.multimodal.preflight import (
    PreflightReport,
    PreflightCheckFailure,
    check_sample_invariants,
    check_cross_split_leakage,
    check_feature_tensor_invariants,
    check_subgroup_identifiers,
    check_validation_distribution_compatibility,
    run_preflight_checks,
)
from src.multimodal.canonical_pipeline import PipelineConfig


class TestMultimodalDataContract:
    def test_missing_image_path_raises_error(self, tmp_path):
        """Missing or non-existent image paths must be rejected immediately."""
        fake_img = tmp_path / "non_existent.jpg"
        with pytest.raises(MissingImageError, match="Image file does not exist"):
            MultimodalSample(
                sample_id="sample_001",
                text="Test text",
                image_path=str(fake_img),
                label=0,
                language="hi",
                source_dataset="hasoc_hindi",
                verify_image_exists=True,
            )

    def test_valid_image_path_accepted(self, tmp_path):
        """Valid image file paths must be accepted without error."""
        real_img = tmp_path / "real.jpg"
        real_img.write_bytes(b"dummy image bytes")

        sample = MultimodalSample(
            sample_id="sample_002",
            text="Valid multimodal text",
            image_path=str(real_img),
            label=1,
            language="en",
            source_dataset="pridemm",
            verify_image_exists=True,
        )
        assert sample.sample_id == "sample_002"
        assert sample.has_verified_image is True

    def test_invalid_label_raises_contract_violation(self, tmp_path):
        """Binary classification labels must be strictly 0 or 1."""
        real_img = tmp_path / "sample.jpg"
        real_img.write_bytes(b"data")

        with pytest.raises(DataContractViolation, match="Invalid label"):
            MultimodalSample(
                sample_id="sample_003",
                text="Some text",
                image_path=str(real_img),
                label=2,
                language="hi",
                source_dataset="hasoc",
                verify_image_exists=True,
            )

    def test_empty_text_raises_contract_violation(self, tmp_path):
        """Empty or whitespace text strings must be rejected."""
        real_img = tmp_path / "sample.jpg"
        real_img.write_bytes(b"data")

        with pytest.raises(DataContractViolation, match="text cannot be empty"):
            MultimodalSample(
                sample_id="sample_004",
                text="   ",
                image_path=str(real_img),
                label=0,
                language="en",
                source_dataset="pridemm",
                verify_image_exists=True,
            )


class TestFeatureCacheMetadata:
    def test_compatible_cache_passes(self):
        meta1 = FeatureCacheMetadata(
            dataset_name="dataset_v1",
            sample_ids=["s1", "s2"],
            text_encoder="muril",
            image_encoder="clip",
            text_dim=768,
            image_dim=512,
            has_genuine_images=True,
        )
        meta2 = FeatureCacheMetadata(
            dataset_name="dataset_v1",
            sample_ids=["s1", "s2"],
            text_encoder="muril",
            image_encoder="clip",
            text_dim=768,
            image_dim=512,
            has_genuine_images=True,
        )
        meta1.validate_compatibility(meta2)

    def test_incompatible_dimensions_rejected(self):
        meta1 = FeatureCacheMetadata(
            dataset_name="dataset_v1",
            sample_ids=["s1"],
            text_encoder="muril",
            image_encoder="clip",
            text_dim=768,
            image_dim=512,
            has_genuine_images=True,
        )
        meta2 = FeatureCacheMetadata(
            dataset_name="dataset_v1",
            sample_ids=["s1"],
            text_encoder="muril",
            image_encoder="clip",
            text_dim=768,
            image_dim=768,  # Mismatch
            has_genuine_images=True,
        )
        with pytest.raises(IncompatibleCacheError, match="Image dimension mismatch"):
            meta1.validate_compatibility(meta2)

    def test_incompatible_encoder_rejected(self):
        meta1 = FeatureCacheMetadata(
            dataset_name="dataset_v1",
            sample_ids=["s1"],
            text_encoder="muril",
            image_encoder="clip",
            text_dim=768,
            image_dim=512,
            has_genuine_images=True,
        )
        meta2 = FeatureCacheMetadata(
            dataset_name="dataset_v1",
            sample_ids=["s1"],
            text_encoder="bert-base-uncased",
            image_encoder="clip",
            text_dim=768,
            image_dim=512,
            has_genuine_images=True,
        )
        with pytest.raises(IncompatibleCacheError, match="Text encoder mismatch"):
            meta1.validate_compatibility(meta2)


class TestSubgroupMaskVerification:
    def test_empty_subgroup_mask_raises_error(self):
        """If expected subgroup is missing or has 0 samples, an error must be raised."""
        sources = np.array(["pridemm", "hasoc_hindi", "m3h_bilingual"])
        with pytest.raises(SubgroupMaskMismatchError, match="Empty subgroup mask detected"):
            verify_subgroup_mask("trac", sources, min_samples=1)

    def test_trac_hinglish_source_matches_correctly(self):
        """TRAC Hinglish source must be properly identified and return valid indices."""
        sources = np.array(["pridemm", "trac_hinglish", "hasoc_hindi"])
        mask = verify_subgroup_mask("trac_hinglish", sources, min_samples=1)
        assert np.sum(mask) == 1
        assert mask[1] == True

    def test_non_empty_subgroup_mask_returns_boolean_array(self):
        sources = np.array(["pridemm", "pridemm", "other"])
        mask = verify_subgroup_mask("pridemm", sources, min_samples=2)
        assert np.sum(mask) == 2


class TestPreflightTensorChecks:
    def test_zero_vector_visual_substitute_is_rejected(self):
        """A tensor containing zero-norm visual rows (like HASOC/TRAC placeholders) must be rejected."""
        tensors = {
            "text_features": torch.randn(10, 768),
            "image_features": torch.randn(10, 512),
        }
        # Simulate zero-vector substitution on samples 3 and 7
        tensors["image_features"][3] = 0.0
        tensors["image_features"][7] = 0.0

        with pytest.raises(ZeroVectorPlaceholderError, match="Zero-vector visual placeholders detected"):
            check_feature_tensor_invariants(tensors, allow_zero_vectors=False)

    def test_intentional_modality_dropout_distinguished_from_corrupted_data(self):
        """Explicit modality dropout mask is permitted when explicitly declared."""
        tensors = {
            "text_features": torch.randn(10, 768),
            "image_features": torch.randn(10, 512),
            "is_intentional_modality_dropout": torch.tensor([False] * 10),
        }
        # When all visual features are genuine, check passes
        check_feature_tensor_invariants(tensors, allow_zero_vectors=False)

    def test_nan_or_inf_in_features_is_rejected(self):
        """NaN or Inf values in feature tensors must trigger PreflightCheckFailure."""
        tensors = {
            "text_features": torch.randn(5, 768),
            "image_features": torch.randn(5, 512),
        }
        tensors["text_features"][0, 0] = float("nan")

        with pytest.raises(PreflightCheckFailure, match="NaN or Inf detected in text_features"):
            check_feature_tensor_invariants(tensors)


class TestCrossSplitLeakage:
    def test_sample_id_leakage_detected(self, tmp_path):
        f = tmp_path / "img.jpg"
        f.write_bytes(b"data")

        train = [
            MultimodalSample("s1", "text one", str(f), 0, "en", "src", verify_image_exists=False),
            MultimodalSample("s2", "text two", str(f), 1, "en", "src", verify_image_exists=False),
        ]
        val = [
            MultimodalSample("s2", "text two val", str(f), 1, "en", "src", verify_image_exists=False),
        ]

        with pytest.raises(PreflightCheckFailure, match="Sample ID leakage detected"):
            check_cross_split_leakage(train, val)

    def test_exact_text_duplicate_leakage_detected(self, tmp_path):
        f = tmp_path / "img.jpg"
        f.write_bytes(b"data")

        train = [
            MultimodalSample("s1", "Identical tweet message", str(f), 0, "en", "src", verify_image_exists=False),
        ]
        val = [
            MultimodalSample("s2", "Identical tweet message", str(f), 0, "en", "src", verify_image_exists=False),
        ]

        with pytest.raises(PreflightCheckFailure, match="Exact text duplicates across train and val"):
            check_cross_split_leakage(train, val)


class TestValidationDistributionCompatibility:
    def test_high_zero_vector_proportion_in_validation_fails(self):
        """If validation set contains a high fraction of zero visual vectors, preflight fails."""
        # 60 out of 100 samples are zero-vectors (similar to FHN's 61.9% unimodal text)
        val_img = torch.randn(100, 512)
        val_img[:60] = 0.0

        with pytest.raises(PreflightCheckFailure, match="Validation distribution contains 60.0% zero visual vectors"):
            check_validation_distribution_compatibility(val_img, max_unimodal_ratio=0.05)

    def test_genuine_multimodal_validation_passes(self):
        val_img = torch.randn(100, 512)
        check_validation_distribution_compatibility(val_img, max_unimodal_ratio=0.05)


class TestProductionProtection:
    def test_pipeline_config_blocks_production_paths(self):
        """Pipeline must raise DataContractViolation if output_dir targets production paths."""
        config = PipelineConfig(
            experiment_name="malicious_or_accidental_run",
            output_dir="models/muril_model_b.pt",
            dataset_name="test",
        )
        with pytest.raises(DataContractViolation, match="CRITICAL SAFETY VIOLATION"):
            config.validate_safety()

    def test_pipeline_config_blocks_baseline_checkpoint_dir(self):
        config = PipelineConfig(
            experiment_name="malicious_run",
            output_dir="scratch/meme_experiments/final_multimodal",
            dataset_name="test",
        )
        with pytest.raises(DataContractViolation, match="CRITICAL SAFETY VIOLATION"):
            config.validate_safety()


class TestEndToEndPreflightRun:
    def test_full_preflight_passes_on_valid_fixtures(self, tmp_path):
        img_train = tmp_path / "train.jpg"
        img_train.write_bytes(b"train_data")
        img_val = tmp_path / "val.jpg"
        img_val.write_bytes(b"val_data")

        train_samples = [
            MultimodalSample("tr_1", "train text 1", str(img_train), 0, "en", "dataset_a", verify_image_exists=True),
            MultimodalSample("tr_2", "train text 2", str(img_train), 1, "hi", "dataset_b", verify_image_exists=True),
        ]
        val_samples = [
            MultimodalSample("vl_1", "val text 1", str(img_val), 0, "en", "dataset_a", verify_image_exists=True),
            MultimodalSample("vl_2", "val text 2", str(img_val), 1, "hi", "dataset_b", verify_image_exists=True),
        ]

        train_tensors = {
            "text_features": torch.randn(2, 768),
            "image_features": torch.randn(2, 512),
        }

        report = run_preflight_checks(
            train_samples=train_samples,
            val_samples=val_samples,
            expected_subgroups=["dataset_a", "dataset_b"],
            train_tensors=train_tensors,
            check_disk_images=True,
        )

        assert report.passed is True
        assert report.checks_passed == 10
        assert report.sample_count_train == 2
        assert report.sample_count_val == 2
