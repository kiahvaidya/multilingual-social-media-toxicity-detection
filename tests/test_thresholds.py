"""
Unit tests for per-label threshold loading and prediction.
"""

import json
import pytest
import numpy as np

from src.config import config
from src.model import load_thresholds, predict_comment


def test_load_thresholds_fallback(tmp_path, monkeypatch):
    """Test load_thresholds returns default 0.5 when thresholds file is missing."""
    non_existent = tmp_path / "non_existent.json"
    monkeypatch.setattr(config, "THRESHOLDS_PATH", non_existent)

    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    thresh = load_thresholds(labels)

    assert isinstance(thresh, dict)
    assert len(thresh) == 6
    for label in labels:
        assert thresh[label] == 0.5


def test_load_thresholds_valid_json(tmp_path, monkeypatch):
    """Test load_thresholds correctly reads thresholds from JSON."""
    json_file = tmp_path / "toxic_thresholds.json"
    expected = {
        "toxic": 0.45,
        "severe_toxic": 0.25,
        "obscene": 0.40,
        "threat": 0.15,
        "insult": 0.35,
        "identity_hate": 0.20,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(expected, f)

    monkeypatch.setattr(config, "THRESHOLDS_PATH", json_file)

    labels = list(expected.keys())
    thresh = load_thresholds(labels)

    assert thresh == expected


def test_predict_comment_with_thresholds(monkeypatch):
    """Test predict_comment respects provided per-label thresholds for binary classifications."""
    # Mock model, vectorizer, and labels
    class DummyModel:
        def predict_proba(self, vec):
            # [toxic, severe_toxic, obscene, threat, insult, identity_hate]
            return np.array([[0.40, 0.20, 0.35, 0.18, 0.30, 0.15]])

    class DummyVectorizer:
        def transform(self, texts):
            return [1]

    dummy_model = DummyModel()
    dummy_vec = DummyVectorizer()
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    thresholds = {
        "toxic": 0.45,        # 0.40 < 0.45 -> False
        "severe_toxic": 0.25, # 0.20 < 0.25 -> False
        "obscene": 0.30,      # 0.35 > 0.30 -> True
        "threat": 0.10,       # 0.18 > 0.10 -> True
        "insult": 0.50,       # 0.30 < 0.50 -> False
        "identity_hate": 0.20 # 0.15 < 0.20 -> False
    }

    preds, toxic_score, intensity = predict_comment(
        "test comment", dummy_model, dummy_vec, labels, thresholds=thresholds
    )

    assert preds["toxic"] == 0.40
    assert preds["obscene"] == 0.35
    assert toxic_score == 0.40
    assert intensity == "Low"
