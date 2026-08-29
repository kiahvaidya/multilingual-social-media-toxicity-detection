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

    # ── App Settings ───────────────────────────────────────
    TOXIC_THRESHOLD: float = 0.5
    MAX_COMMENTS: int = 10


config = _Config()
