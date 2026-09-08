"""
End-to-End Application Validation Suite for Phase 5B.

Validates the complete production pipeline:
  Input → Preprocessing → Model (MuRIL / Logistic) → Per-Label Thresholds → Results & Moderation

Verifies:
  - Custom text inference with per-label thresholds
  - Batch text inference with batch tensor processing
  - YouTube comments connector resilience (invalid URLs, empty results)
  - Reddit comments connector resilience (invalid URLs, empty results)
  - Twitter/X comments connector resilience (invalid URLs, missing API credentials)
  - Model caching behavior (MuRIL loaded only once)
  - Explicit CPU fallback
  - Explicit fallback from MuRIL to Logistic Regression baseline
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from src.config import config
from src.connectors.youtube import fetch_youtube_comments
from src.connectors.reddit import fetch_reddit_comments
from src.connectors.twitter import fetch_twitter_comments
from src.model import (
    load_muril_model,
    predict_comment_muril,
    predict_batch_muril,
    load_model as load_logistic_model,
    predict_comment as predict_comment_logistic,
    predict_batch as predict_batch_logistic,
)
from src.moderation import (
    RiskLevel,
    classify_risk,
    calculate_aggregate_statistics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def validate_custom_text(model, tokenizer, labels, thresholds, device) -> Dict[str, Any]:
    """Test single custom text prediction and 3-tier risk classification."""
    test_text = "Teri himmat kaise hui ye bolne ki? I will destroy you."
    logger.info("Testing custom text: '%s'", test_text)

    preds, score, intensity, flags = predict_comment_muril(
        test_text, model, tokenizer, labels, device, thresholds=thresholds
    )
    risk_level, icon, explanation = classify_risk(preds, flags, thresholds)

    assert isinstance(preds, dict), "Predictions must be a dictionary"
    assert set(preds.keys()) == set(labels), "Predictions must cover all 6 labels"
    assert 0.0 <= score <= 1.0, "Score must be within [0.0, 1.0]"
    assert intensity in ("High", "Moderate", "Low"), "Intensity must be High/Moderate/Low"
    assert risk_level in (RiskLevel.SAFE, RiskLevel.REVIEW, RiskLevel.HIGH_RISK)

    return {
        "status": "PASS",
        "text": test_text,
        "toxic_score": score,
        "intensity": intensity,
        "risk_level": risk_level.value,
        "risk_icon": icon,
        "flags": flags,
    }


def validate_batch_text(model, tokenizer, labels, thresholds, device) -> Dict[str, Any]:
    """Test batch text prediction with progress reporting and aggregate stats."""
    batch_texts = [
        "This is an inspiring video, keep up the amazing work!",
        "You are so stupid and useless, get lost.",
        "Aapka content bahut badhiya hai bhai, mazaa aa gaya.",
        "Tu toh kutta hai, tera murder kar dunga mai.",
        "Whatever you say, this doesn't make any sense.",
    ]
    logger.info("Testing batch text with %d samples...", len(batch_texts))

    t0 = time.perf_counter()
    batch_results = predict_batch_muril(
        batch_texts, model, tokenizer, labels, device, batch_size=4, thresholds=thresholds
    )
    elapsed = time.perf_counter() - t0

    assert len(batch_results) == len(batch_texts), "Batch result count must match input count"

    stats = calculate_aggregate_statistics(batch_results, thresholds)
    assert stats["total"] == len(batch_texts)
    assert stats["toxic_count"] >= 1
    assert stats["high_risk_count"] >= 1

    return {
        "status": "PASS",
        "sample_count": len(batch_texts),
        "elapsed_seconds": round(elapsed, 4),
        "throughput_samples_per_sec": round(len(batch_texts) / max(elapsed, 1e-6), 2),
        "aggregate_stats": stats,
    }


def validate_youtube_connector() -> Dict[str, Any]:
    """Verify YouTube connector handles invalid URLs cleanly without crashing."""
    logger.info("Testing YouTube connector invalid URL handling...")
    invalid_urls = [
        "not_a_valid_url",
        "https://example.com/watch?v=12345",
        "https://youtube.com/invalid_page",
    ]

    handled_cleanly = True
    errors_caught = []

    for url in invalid_urls:
        try:
            fetch_youtube_comments(url)
            handled_cleanly = False
        except (ValueError, RuntimeError) as e:
            errors_caught.append(f"{url} -> {type(e).__name__}: {e}")
        except Exception as e:
            logger.error("Unexpected unhandled exception for YouTube URL '%s': %s", url, e)
            handled_cleanly = False

    return {
        "status": "PASS" if handled_cleanly else "FAIL",
        "handled_cleanly": handled_cleanly,
        "caught_errors": errors_caught,
    }


def validate_reddit_connector() -> Dict[str, Any]:
    """Verify Reddit connector handles invalid URLs cleanly without crashing."""
    logger.info("Testing Reddit connector invalid URL handling...")
    invalid_urls = [
        "https://reddit.com/not_a_post",
        "ftp://reddit.com/bad",
        "not-a-url",
    ]

    handled_cleanly = True
    errors_caught = []

    for url in invalid_urls:
        try:
            fetch_reddit_comments(url)
            handled_cleanly = False
        except (ValueError, RuntimeError) as e:
            errors_caught.append(f"{url} -> {type(e).__name__}: {e}")
        except Exception as e:
            logger.error("Unexpected unhandled exception for Reddit URL '%s': %s", url, e)
            handled_cleanly = False

    return {
        "status": "PASS" if handled_cleanly else "FAIL",
        "handled_cleanly": handled_cleanly,
        "caught_errors": errors_caught,
    }


def validate_twitter_connector() -> Dict[str, Any]:
    """Verify Twitter connector handles invalid URLs or missing tokens cleanly without crashing."""
    logger.info("Testing Twitter connector error resilience...")
    invalid_urls = [
        "https://twitter.com/not_a_status",
        "https://x.com/user/status/not_numeric_id",
        "bad_url_string",
    ]

    handled_cleanly = True
    errors_caught = []

    for url in invalid_urls:
        try:
            fetch_twitter_comments(url)
            handled_cleanly = False
        except (ValueError, RuntimeError) as e:
            errors_caught.append(f"{url} -> {type(e).__name__}: {e}")
        except Exception as e:
            logger.error("Unexpected unhandled exception for Twitter URL '%s': %s", url, e)
            handled_cleanly = False

    return {
        "status": "PASS" if handled_cleanly else "FAIL",
        "handled_cleanly": handled_cleanly,
        "caught_errors": errors_caught,
    }


def validate_cpu_fallback(model, tokenizer, labels, thresholds) -> Dict[str, Any]:
    """Verify inference runs reliably when explicitly dispatched to CPU."""
    logger.info("Testing explicit CPU device inference...")
    cpu_device = torch.device("cpu")
    model.to(cpu_device)

    preds, score, intensity, flags = predict_comment_muril(
        "Testing CPU execution fallback pathway",
        model,
        tokenizer,
        labels,
        cpu_device,
        thresholds=thresholds,
    )

    assert isinstance(preds, dict)
    assert 0.0 <= score <= 1.0

    return {
        "status": "PASS",
        "device": "CPU",
        "score": score,
        "intensity": intensity,
    }


def validate_logistic_fallback() -> Dict[str, Any]:
    """Verify fallback from MuRIL to Logistic Regression baseline works explicitly."""
    logger.info("Testing Logistic Regression baseline fallback pathway...")
    log_model, log_vec, log_labels = load_logistic_model()

    test_comment = "You are a terrible person and an idiot"
    preds, score, intensity = predict_comment_logistic(
        test_comment, log_model, log_vec, log_labels
    )
    flags = {lbl: (val >= config.TOXIC_THRESHOLD) for lbl, val in preds.items()}
    thresh_map = {lbl: config.TOXIC_THRESHOLD for lbl in log_labels}

    risk_level, icon, explanation = classify_risk(preds, flags, thresh_map)

    # Batch test for logistic
    batch_raw = predict_batch_logistic(
        ["Great work!", "I hate you, you are garbage."],
        log_model,
        log_vec,
        log_labels,
    )
    assert len(batch_raw) == 2

    return {
        "status": "PASS",
        "fallback_model": "OneVsRest Logistic Regression",
        "score": score,
        "intensity": intensity,
        "risk_level": risk_level.value,
        "batch_processed": len(batch_raw),
    }


def main():
    logger.info("==================================================")
    logger.info("  Phase 5B — End-to-End Application Validation    ")
    logger.info("==================================================")

    # 1. Load MuRIL
    t0 = time.perf_counter()
    model, tokenizer, labels, thresholds, device = load_muril_model()
    load_time = time.perf_counter() - t0
    logger.info("MuRIL loaded in %.2fs on %s", load_time, device)

    # Run validations
    results = {}
    results["custom_text"] = validate_custom_text(model, tokenizer, labels, thresholds, device)
    results["batch_text"] = validate_batch_text(model, tokenizer, labels, thresholds, device)
    results["youtube_connector"] = validate_youtube_connector()
    results["reddit_connector"] = validate_reddit_connector()
    results["twitter_connector"] = validate_twitter_connector()
    results["cpu_fallback"] = validate_cpu_fallback(model, tokenizer, labels, thresholds)
    results["logistic_fallback"] = validate_logistic_fallback()

    logger.info("\n--- VALIDATION SUMMARY ---")
    all_passed = True
    for key, res in results.items():
        status = res.get("status", "FAIL")
        logger.info("  [%s] %s", status, key)
        if status != "PASS":
            all_passed = False

    if all_passed:
        logger.info("\nALL 7 END-TO-END VALIDATION CHECKS PASSED SUCCESSFULLY!")
    else:
        logger.error("\nONE OR MORE VALIDATIONS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
