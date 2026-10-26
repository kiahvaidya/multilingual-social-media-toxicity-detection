"""
Multimodal Package for Multilingual Social Media Toxicity Detection.
"""

from .data_contract import (
    MultimodalSample,
    FeatureCacheMetadata,
    DataContractViolation,
    MissingImageError,
    ZeroVectorPlaceholderError,
    IncompatibleCacheError,
    SubgroupMaskMismatchError,
    verify_subgroup_mask,
)

from .preflight import (
    PreflightReport,
    PreflightCheckFailure,
    check_sample_invariants,
    check_cross_split_leakage,
    check_feature_tensor_invariants,
    check_subgroup_identifiers,
    check_validation_distribution_compatibility,
    run_preflight_checks,
)

__all__ = [
    "MultimodalSample",
    "FeatureCacheMetadata",
    "DataContractViolation",
    "MissingImageError",
    "ZeroVectorPlaceholderError",
    "IncompatibleCacheError",
    "SubgroupMaskMismatchError",
    "verify_subgroup_mask",
    "PreflightReport",
    "PreflightCheckFailure",
    "check_sample_invariants",
    "check_cross_split_leakage",
    "check_feature_tensor_invariants",
    "check_subgroup_identifiers",
    "check_validation_distribution_compatibility",
    "run_preflight_checks",
]
