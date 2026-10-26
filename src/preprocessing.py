"""
Unified text preprocessing pipeline.

This module provides a single preprocessing path used by BOTH
the serving app and the evaluation script, eliminating train/serve skew.

The pipeline is structured in two stages:
  1. Social-media normalization (URLs, @mentions, Hinglish, char dedup)
  2. Training-compatible cleaning (matches what the TF-IDF vectorizer saw)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Hinglish / Code-Mixed & Devanagari Slang Map ───────────
# Maps romanised Hindi slurs, Devanagari abuse, and severe hate
# phrases to English equivalents so the TF-IDF model can recognise them.
HINGLISH_MAP = {
    "chutiya": "idiot",
    "chutya": "idiot",
    "chodu": "idiot",
    "mc": "motherfucker",
    "madarchod": "motherfucker",
    "bc": "fuck",
    "behenchod": "fuck",
    "bhosdi": "prostitute",
    "bhosdiwala": "prostitute",
    "bhosdika": "prostitute",
    "bhosdike": "prostitute",
    "bhadwa": "pimp",
    "bkl": "bastard",
    "gandu": "asshole",
    "gand": "asshole",
    "harami": "bastard",
    "haramzada": "bastard",
    "lauder": "dickhead",
    "kamine": "bastard",
    "kutta": "dog",
    "nalla": "loser",
    "bakwas": "nonsense",
    "tatti": "shit",
    "lodu": "idiot",
    "lawde": "dick",
    "lund": "dick",
    "saale": "bastard",
    "deshdrohi": "traitor bastard",
    "देशद्रोही": "traitor idiot hate",
    "साले": "bastard idiot",
    "निकल यहाँ से": "get lost idiot",
    "शर्म नहीं आती": "shameful idiot",
    "गद्दार": "traitor bastard",
    "कमीने": "bastard idiot",
    "हरामी": "bastard",
    "चूतिया": "idiot",
    "बकवास": "stupid nonsense",
    "despise your kind": "hate you racist idiot",
    "eradicated from existence": "kill destroy die threat",
}


def _clean_social_media(text: str) -> str:
    """Stage 1: Social-media-specific normalization.

    Applied before the training-compatible cleaning so that URLs,
    mentions, and code-mixed slang are handled for social content.
    These transforms are additive — they don't conflict with what
    the TF-IDF vectorizer expects.
    """
    text = re.sub(r"http\S+", " ", text)       # Remove URLs
    text = re.sub(r"@\w+", " ", text)           # Remove @mentions
    for slang, english in HINGLISH_MAP.items():  # Hinglish & Devanagari normalisation
        text = text.replace(slang, english)
    text = re.sub(r"(.)\1{2,}", r"\1", text)   # Deduplicate chars: "heeello" → "helo"
    return text


def _clean_for_model(text: str) -> str:
    """Stage 2: Training-compatible text cleaning.

    Matches the preprocessing applied in smtd3.ipynb before TF-IDF
    vectorisation:
      - punctuation removal  (regex: [^\\w\\s] → "")
      - number removal       (regex: \\d+ → "")
      - newline / CR removal
    """
    text = re.sub(r"[^\w\s]", "", text)    # Remove punctuation
    text = re.sub(r"\d+", "", text)         # Remove numbers
    text = text.replace("\n", " ")          # Remove newlines
    text = re.sub(r"\r", " ", text)         # Remove carriage returns
    return text.strip()


def preprocess_text(text: str, *, social_media: bool = True) -> str:
    """Full preprocessing pipeline.

    Parameters
    ----------
    text : str
        Raw input text.
    social_media : bool, default True
        If True, applies social-media normalization (URLs, mentions,
        Hinglish map, char dedup) before the model-compatible cleaning.
        Set to False for evaluation on the original training data.

    Returns
    -------
    str
        Cleaned text ready for TF-IDF vectorization.
    """
    text = str(text).lower()

    if social_media:
        text = _clean_social_media(text)

    text = _clean_for_model(text)
    return text
