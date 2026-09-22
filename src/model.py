"""
Model loading, caching, and prediction.

Supports both:
  1. 🧠 Primary Multilingual Model: fine-tuned MuRIL (google/muril-base-cased)
     with frozen Phase 4C validation-optimized per-label thresholds.
  2. ⚡ Lightweight Fast Model: scikit-learn OneVsRest Logistic Regression baseline.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.config import config
from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# Standard label columns
DEFAULT_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ── Intensity thresholds ─────────────────────────────────────
_INTENSITY_HIGH = 0.7
_INTENSITY_MODERATE = 0.4


# =============================================================
# 🧠 MuRIL Multilingual Transformer Model Architecture
# =============================================================

class MuRILForMultiLabelToxicity(nn.Module):
    """Fine-tuned MuRIL transformer for multi-label toxicity detection."""

    def __init__(self, model_name: str = config.MURIL_MODEL_NAME, num_labels: int = 6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Bottom 11 layers frozen during training for efficient inference
        for param in self.encoder.encoder.layer[:11].parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits


def get_device() -> torch.device:
    """Detect and return available compute device (CUDA GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU with %d threads", torch.get_num_threads())
    return device


def load_muril_thresholds(
    labels: Optional[List[str]] = None,
    path: Optional[Path] = None,
) -> Dict[str, float]:
    """Load Phase 4C validation-optimized per-label thresholds.

    Falls back to frozen constants if file is missing.
    """
    lbls = labels or DEFAULT_LABELS
    target_path = path or config.MURIL_THRESHOLDS_PATH

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            logger.info("Loaded MuRIL thresholds from %s", target_path)
            return {lbl: float(thresholds.get(lbl, config.FROZEN_MURIL_THRESHOLDS.get(lbl, 0.5))) for lbl in lbls}
        except Exception as e:
            logger.warning("Failed to load thresholds from %s: %s. Using frozen defaults.", target_path, e)

    return {lbl: float(config.FROZEN_MURIL_THRESHOLDS.get(lbl, 0.5)) for lbl in lbls}


def load_muril_model(
    checkpoint_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[MuRILForMultiLabelToxicity, AutoTokenizer, List[str], Dict[str, float], torch.device]:
    """Load fine-tuned MuRIL model, tokenizer, labels, and optimized thresholds.

    Parameters
    ----------
    checkpoint_path : Path, optional
        Path to muril_toxicity_model.pt
    model_name : str, optional
        Base huggingface model name (default: google/muril-base-cased)
    device : torch.device, optional
        Target compute device.

    Returns
    -------
    tuple
        (model, tokenizer, labels, thresholds, device)

    Raises
    ------
    FileNotFoundError
        If the checkpoint file is not found.
    """
    ckpt_path = checkpoint_path or config.MURIL_CHECKPOINT_PATH
    m_name = model_name or config.MURIL_MODEL_NAME
    dev = device or get_device()

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"MuRIL model checkpoint '{ckpt_path.name}' not found at {ckpt_path}. "
            "Please ensure muril_toxicity_model.pt exists in the project root."
        )

    logger.info("Loading MuRIL tokenizer from %s...", m_name)
    tokenizer = AutoTokenizer.from_pretrained(m_name)

    logger.info("Initializing MuRIL architecture and loading weights from %s...", ckpt_path)
    model = MuRILForMultiLabelToxicity(model_name=m_name, num_labels=len(DEFAULT_LABELS))
    state_dict = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    labels = list(DEFAULT_LABELS)
    thresholds = load_muril_thresholds(labels)

    logger.info("MuRIL engine loaded successfully on %s with thresholds: %s", dev, thresholds)
    return model, tokenizer, labels, thresholds, dev


def _compute_intensity(score: float) -> str:
    if score > _INTENSITY_HIGH:
        return "High"
    elif score > _INTENSITY_MODERATE:
        return "Moderate"
    return "Low"


def predict_comment_muril(
    text: str,
    model: MuRILForMultiLabelToxicity,
    tokenizer: AutoTokenizer,
    labels: List[str],
    device: torch.device,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float, str, Dict[str, bool]]:
    """Predict toxicity labels for a single comment using MuRIL.

    Parameters
    ----------
    text : str
        Raw input string (multilingual text preserved natively).
    model : MuRILForMultiLabelToxicity
        Loaded PyTorch model.
    tokenizer : AutoTokenizer
        Loaded HuggingFace tokenizer.
    labels : list[str]
        Ordered label names.
    device : torch.device
        Compute device.
    thresholds : dict[str, float], optional
        Per-label classification thresholds.

    Returns
    -------
    predictions : dict[str, float]
        Per-label probability scores (0.0 to 1.0).
    toxic_score : float
        Primary toxicity probability (toxic label or maximum probability).
    intensity : str
        "High" / "Moderate" / "Low" based on toxic_score.
    is_toxic_flags : dict[str, bool]
        Per-label binary classification evaluated against optimal thresholds.
    """
    model.eval()
    thresh_dict = thresholds or config.FROZEN_MURIL_THRESHOLDS
    raw_text = str(text).strip() if text is not None else ""

    encoding = tokenizer(
        raw_text,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device, non_blocking=True)
    attention_mask = encoding["attention_mask"].to(device, non_blocking=True)

    with torch.inference_mode():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    predictions: Dict[str, float] = {}
    is_toxic_flags: Dict[str, bool] = {}

    for i, label in enumerate(labels):
        val = float(probs[i]) if i < len(probs) else 0.0
        val_rounded = float(round(val, 4))
        predictions[label] = val_rounded
        th = float(thresh_dict.get(label, 0.5))
        is_toxic_flags[label] = bool(val >= th)

    toxic_score = max(predictions.values()) if predictions else 0.0
    intensity = _compute_intensity(toxic_score)

    return predictions, toxic_score, intensity, is_toxic_flags


def predict_batch_muril(
    texts: List[str],
    model: MuRILForMultiLabelToxicity,
    tokenizer: AutoTokenizer,
    labels: List[str],
    device: torch.device,
    batch_size: int = config.BATCH_SIZE,
    thresholds: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[Tuple[Dict[str, float], float, str, Dict[str, bool]]]:
    """Predict toxicity for a batch of comments using MuRIL with batched tensors.

    Optimized in Phase 5C with torch.inference_mode() and dynamic batch padding (padding=True).

    Parameters
    ----------
    texts : list[str]
        List of raw comment strings.
    model : MuRILForMultiLabelToxicity
        Loaded PyTorch model.
    tokenizer : AutoTokenizer
        Loaded HuggingFace tokenizer.
    labels : list[str]
        Ordered label names.
    device : torch.device
        Compute device.
    batch_size : int, default 32
        Batch chunk size for GPU/CPU evaluation.
    thresholds : dict[str, float], optional
        Per-label classification thresholds.
    progress_callback : callable, optional
        Callback accepting progress fraction (0.0 to 1.0).

    Returns
    -------
    list of (predictions, toxic_score, intensity, is_toxic_flags)
    """
    if not texts:
        return []

    model.eval()
    thresh_dict = thresholds or config.FROZEN_MURIL_THRESHOLDS
    total_samples = len(texts)
    results = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_texts = [str(t).strip() for t in texts[start_idx:end_idx]]

        # Phase 5C: Dynamic padding (padding=True) avoids redundant padding compute for shorter batches
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

        for probs in batch_probs:
            predictions: Dict[str, float] = {}
            is_toxic_flags: Dict[str, bool] = {}

            for i, label in enumerate(labels):
                val = float(probs[i]) if i < len(probs) else 0.0
                val_rounded = float(round(val, 4))
                predictions[label] = val_rounded
                th = float(thresh_dict.get(label, 0.5))
                is_toxic_flags[label] = bool(val >= th)

            toxic_score = max(predictions.values()) if predictions else 0.0
            intensity = _compute_intensity(toxic_score)
            results.append((predictions, toxic_score, intensity, is_toxic_flags))

        if progress_callback is not None:
            progress_callback(end_idx / total_samples)

    return results


# =============================================================
# ⚡ Logistic Regression Fast Baseline Model
# =============================================================

def load_model():
    """Load the serialized Logistic Regression baseline model, vectorizer, and label columns.

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
        "Logistic model loaded: %s labels, vectorizer vocab size %d",
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
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float, str]:
    """Predict toxicity labels for a single comment using the Logistic Regression baseline."""
    clean = preprocess_text(text, social_media=True)
    vec = vectorizer.transform([clean])

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
    intensity = _compute_intensity(toxic_score)

    return predictions, toxic_score, intensity


def predict_batch(
    texts: List[str],
    model,
    vectorizer,
    labels: list,
) -> List[Tuple[Dict[str, float], float, str]]:
    """Predict toxicity for a batch of comments using the Logistic Regression baseline."""
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
        intensity = _compute_intensity(toxic_score)
        results.append((predictions, toxic_score, intensity))

    return results
