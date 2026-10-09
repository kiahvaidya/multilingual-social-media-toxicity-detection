"""
Centralized configuration loaded from environment variables.

Usage:
    from src.config import config
    api_key = config.YOUTUBE_API_KEY
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


class _Config:
    """Application configuration sourced from environment variables."""

    # ── Paths ──────────────────────────────────────────────
    PROJECT_ROOT: Path = _PROJECT_ROOT
    MODEL_PATH: Path = _PROJECT_ROOT / "toxic_model.pkl"
    VECTORIZER_PATH: Path = _PROJECT_ROOT / "toxic_vectorizer.pkl"
    COLUMNS_PATH: Path = _PROJECT_ROOT / "toxic_columns.pkl"
    THRESHOLDS_PATH: Path = _PROJECT_ROOT / "toxic_thresholds.json"
    DATASET_DIR: Path = _PROJECT_ROOT / "dataset"

    # ── MuRIL Multilingual Model Paths & Settings ───────────
    MURIL_MODEL_NAME: str = "google/muril-base-cased"
    MURIL_CHECKPOINT_PATH: Path = _PROJECT_ROOT / "muril_toxicity_model.pt"
    MURIL_THRESHOLDS_PATH: Path = _PROJECT_ROOT / "evaluation_results" / "muril_optimized_thresholds.json"
    
    # Frozen Phase 4C validation-optimized per-label thresholds
    FROZEN_MURIL_THRESHOLDS: dict = {
        "toxic": 0.45,
        "severe_toxic": 0.62,
        "obscene": 0.65,
        "threat": 0.59,
        "insult": 0.61,
        "identity_hate": 0.60,
    }
    DEFAULT_MODEL_TYPE: str = "muril"
    MAX_SEQ_LENGTH: int = 64
    BATCH_SIZE: int = 32

    # ── API Keys (loaded from .env) ────────────────────────
    @property
    def YOUTUBE_API_KEY(self) -> str:
        val = os.getenv("YOUTUBE_API_KEY", "")
        if not val:
            logger.warning("YOUTUBE_API_KEY not set in environment")
        return val

    @property
    def TWITTER_BEARER_TOKEN(self) -> str:
        val = os.getenv("TWITTER_BEARER_TOKEN", "")
        if not val:
            logger.warning("TWITTER_BEARER_TOKEN not set in environment")
        return val

    @property
    def REDDIT_CLIENT_ID(self) -> str:
        val = os.getenv("REDDIT_CLIENT_ID", "")
        if not val:
            logger.warning("REDDIT_CLIENT_ID not set in environment")
        return val

    @property
    def REDDIT_CLIENT_SECRET(self) -> str:
        val = os.getenv("REDDIT_CLIENT_SECRET", "")
        if not val:
            logger.warning("REDDIT_CLIENT_SECRET not set in environment")
        return val

    @property
    def REDDIT_USER_AGENT(self) -> str:
        return os.getenv("REDDIT_USER_AGENT", "ToxicityDetectorBot/1.0")

    # ── JEV Contextual Moderation Settings (Phase 6A) ──────
    @property
    def JEV_ENABLED(self) -> bool:
        return os.getenv("JEV_ENABLED", "false").lower() in ("true", "1", "yes")

    @property
    def GEMINI_API_KEY(self) -> str:
        return os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def JEV_GEMINI_MODEL(self) -> str:
        return os.getenv("JEV_GEMINI_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash"

    @property
    def JEV_ENDPOINT_URL(self) -> str:
        return os.getenv("JEV_ENDPOINT_URL", "").strip()

    @property
    def JEV_API_KEY(self) -> str:
        return os.getenv("JEV_API_KEY", "").strip()

    @property
    def JEV_TIMEOUT_SECONDS(self) -> float:
        try:
            return float(os.getenv("JEV_TIMEOUT_SECONDS", "3.0"))
        except ValueError:
            return 3.0

    # ── App Settings ───────────────────────────────────────
    TOXIC_THRESHOLD: float = 0.5
    MAX_COMMENTS: int = 10


config = _Config()
