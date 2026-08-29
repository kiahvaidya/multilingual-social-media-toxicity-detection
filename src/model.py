"""
Model loading, caching, and prediction.

Provides a thin wrapper around the serialized sklearn model and vectorizer
with proper error handling and structured output.
"""

import json
import logging
from typing import Dict, List, Tuple

import joblib
import numpy as np

from src.config import config
from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# ── Intensity thresholds ─────────────────────────────────────
_INTENSITY_HIGH = 0.7
_INTENSITY_MODERATE = 0.4


def load_model():
    """Load the serialized model, vectorizer, and label columns.

    Returns
    -------
    tuple
        (model, vectorizer, labels)

    Raises
    ------
    FileNotFoundError
        If any of the required .pkl files are missing.
    """
    for path, name in [
        (config.MODEL_PATH, "model"),
        (config.VECTORIZER_PATH, "vectorizer"),
        (config.COLUMNS_PATH, "columns"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file '{path.name}' not found at {path}. "
                f"Please place the serialized {name} file in the project root."
            )

    model = joblib.load(config.MODEL_PATH)
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    labels = joblib.load(config.COLUMNS_PATH)

    logger.info(
        "Model loaded: %s labels, vectorizer vocab size %d",
        len(labels),
        len(vectorizer.vocabulary_),
    )
    return model, vectorizer, labels


def load_thresholds(labels: list) -> Dict[str, float]:
    """Load per-label decision thresholds from toxic_thresholds.json or fallback to 0.5."""
    if config.THRESHOLDS_PATH.exists():
        try:
            with open(config.THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            logger.info("Loaded custom thresholds from %s", config.THRESHOLDS_PATH)
            return thresholds
        except Exception as e:
            logger.warning("Error loading thresholds file %s: %s", config.THRESHOLDS_PATH, e)
    
    return {label: 0.5 for label in labels}


def predict_comment(
    text: str,
    model,
    vectorizer,
    labels: list,
    thresholds: Dict[str, float] = None,
) -> Tuple[Dict[str, float], float, str]:
    """Predict toxicity labels for a single comment.

    Parameters
    ----------
    text : str
        Raw input text (will be preprocessed).
    model : sklearn estimator
        The loaded OneVsRestClassifier.
    vectorizer : TfidfVectorizer
        The loaded vectorizer.
    labels : list[str]
        Ordered label names.

    Returns
    -------
    predictions : dict[str, float]
        Per-label probability scores.
    toxic_score : float
        Maximum probability across all labels.
    intensity : str
        "High" / "Moderate" / "Low" based on toxic_score.
    """
    clean = preprocess_text(text, social_media=True)
    vec = vectorizer.transform([clean])

    # OneVsRestClassifier.predict_proba returns shape (1, n_labels)
    # where each value is P(label=1). Fall back to binary predict
    # only if predict_proba is genuinely unavailable.
    try:
        probs = model.predict_proba(vec)[0]
    except AttributeError:
        logger.warning(
            "Model does not support predict_proba; "
            "falling back to binary predict (scores will be 0 or 1)."
        )
        probs = model.predict(vec)[0].astype(float)

    predictions: Dict[str, float] = {}
    for i, label in enumerate(labels):
        if i < len(probs):
            predictions[label] = float(round(probs[i], 3))
        else:
            predictions[label] = 0.0

    toxic_score = max(predictions.values()) if predictions else 0.0

    if toxic_score > _INTENSITY_HIGH:
        intensity = "High"
    elif toxic_score > _INTENSITY_MODERATE:
        intensity = "Moderate"
    else:
        intensity = "Low"

    return predictions, toxic_score, intensity


def predict_batch(
    texts: List[str],
    model,
    vectorizer,
    labels: list,
) -> List[Tuple[Dict[str, float], float, str]]:
    """Predict toxicity for a batch of comments.

    Vectorizes all texts at once for efficiency, then produces
    per-comment results.
    """
    if not texts:
        return []

    cleaned = [preprocess_text(t, social_media=True) for t in texts]
    vecs = vectorizer.transform(cleaned)

    try:
        all_probs = model.predict_proba(vecs)
    except AttributeError:
        logger.warning(
            "Model does not support predict_proba; "
            "falling back to binary predict."
        )
        all_probs = model.predict(vecs).astype(float)

    results = []
    for probs in all_probs:
        predictions: Dict[str, float] = {}
        for i, label in enumerate(labels):
            if i < len(probs):
                predictions[label] = float(round(probs[i], 3))
            else:
                predictions[label] = 0.0

        toxic_score = max(predictions.values()) if predictions else 0.0

        if toxic_score > _INTENSITY_HIGH:
            intensity = "High"
        elif toxic_score > _INTENSITY_MODERATE:
            intensity = "Moderate"
        else:
            intensity = "Low"

        results.append((predictions, toxic_score, intensity))

    return results
