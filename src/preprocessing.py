"""
Unified text preprocessing pipeline.

This module provides a single preprocessing path used by BOTH
the serving app and the evaluation script, eliminating train/serve skew.

The pipeline is structured in two stages:
  1. Social-media normalization (URLs, @mentions, Hinglish, char dedup)
  2. Training-compatible cleaning (matches what the TF-IDF vectorizer saw)
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# ── Unicode UTS #39 Confusables & Homoglyphs Map ───────────────
# Maps visually identical Cyrillic, Greek, and compatibility characters to Latin
HOMOGLYPH_MAP = {
    # Cyrillic lowercase to Latin
    '\u0430': 'a', '\u0441': 'c', '\u0435': 'e', '\u043E': 'o', '\u0440': 'p',
    '\u0455': 's', '\u0445': 'x', '\u0443': 'y', '\u0456': 'i', '\u0458': 'j',
    '\u0501': 'd', '\u051B': 'q',
    # Cyrillic uppercase to Latin
    '\u0410': 'A', '\u0412': 'B', '\u0421': 'C', '\u0415': 'E', '\u041D': 'H',
    '\u0406': 'I', '\u0408': 'J', '\u041A': 'K', '\u041C': 'M', '\u041E': 'O',
    '\u0420': 'P', '\u0422': 'T', '\u0425': 'X', '\u04AE': 'Y',
    # Greek lowercase to Latin
    '\u03B1': 'a', '\u03B2': 'b', '\u03B5': 'e', '\u03B9': 'i', '\u03BA': 'k',
    '\u03BF': 'o', '\u03C1': 'p', '\u03C4': 't', '\u03C5': 'u', '\u03BD': 'v',
    # Greek uppercase to Latin
    '\u0391': 'A', '\u0392': 'B', '\u0395': 'E', '\u0396': 'Z', '\u0397': 'H',
    '\u0399': 'I', '\u039A': 'K', '\u039C': 'M', '\u039D': 'N', '\u039F': 'O',
    '\u03A1': 'P', '\u03A4': 'T', '\u03A5': 'Y', '\u03A7': 'X',
}
_HOMOGLYPH_TRANSLATION_TABLE = str.maketrans(HOMOGLYPH_MAP)

# ── Spaced / Separator Evasion Patterns ────────────────────────
# Collapse letters separated by punctuation (e.g., k . i . l . l, f.u.c.k, s-h-i-t, b_h_o_s_d_i_k_e)
_PUNCT_SPACED_PATTERN = re.compile(r"\b[a-zA-Z](?:[\s\.\-_]*[\.\-_][\s\.\-_]*[a-zA-Z]){2,}\b")

# Targeted space-separated evasions to avoid false collation on natural short English words
_KNOWN_SPACED_EVASIONS = [
    (re.compile(r"\bk\s+i\s+l\s+l\b", re.IGNORECASE), "kill"),
    (re.compile(r"\bs\s+h\s+i\s+t\b", re.IGNORECASE), "shit"),
    (re.compile(r"\bf\s+u\s+c\s+k\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bb\s+i\s+t\s+c\s+h\b", re.IGNORECASE), "bitch"),
    (re.compile(r"\bd\s+i\s+e\b", re.IGNORECASE), "die"),
    (re.compile(r"\br\s+a\s+p\s+e\b", re.IGNORECASE), "rape"),
    (re.compile(r"\bh\s+a\s+t\s+e\b", re.IGNORECASE), "hate"),
    (re.compile(r"\bs\s+l\s+u\s+t\b", re.IGNORECASE), "slut"),
    (re.compile(r"\bc\s+u\s+n\s+t\b", re.IGNORECASE), "cunt"),
    (re.compile(r"\bl\s+o\s+s\s+e\s+r\b", re.IGNORECASE), "loser"),
    (re.compile(r"\bi\s+d\s+i\s+o\s+t\b", re.IGNORECASE), "idiot"),
    (re.compile(r"\bc\s+h\s+u\s+t\s+i\s+y\s+a\b", re.IGNORECASE), "chutiya"),
    (re.compile(r"\bb\s+h\s+o\s+s\s+d\s+i\s+k\s+e\b", re.IGNORECASE), "bhosdike"),
    (re.compile(r"\bm\s+a\s+d\s+a\s+r\s+c\s+h\s+o\s+d\b", re.IGNORECASE), "madarchod"),
    (re.compile(r"\bb\s+e\s+h\s+e\s+n\s+c\s+h\s+o\s+d\b", re.IGNORECASE), "behenchod"),
    (re.compile(r"\bg\s+a\s+n\s+d\s+u\b", re.IGNORECASE), "gandu"),
    (re.compile(r"\bh\s+a\s+r\s+a\s+m\s+i\b", re.IGNORECASE), "harami"),
]

# ── Masked Profanity Reconstruction Patterns ───────────────────
_MASKED_PROFANITY_PATTERNS = [
    (re.compile(r"\bf[\*#\$!@]+k\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bf[\*#\$!@]+ck\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bf[\*#\$!@]+c?k(?:ing|ed|er|s)?\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bb[!1\*#\$@]+tch(?:es)?\b", re.IGNORECASE), "bitch"),
    (re.compile(r"\bsh[!1\*#\$@]+t\b", re.IGNORECASE), "shit"),
    (re.compile(r"\b(?:a[\*\$#@]{2,}hole|[@a]\$\$hole)\b", re.IGNORECASE), "asshole"),
    (re.compile(r"\bk[!1\*#\$@]+ll\b", re.IGNORECASE), "kill"),
    (re.compile(r"\bd[!1\*#\$@]+ck\b", re.IGNORECASE), "dick"),
    (re.compile(r"\bc[!1\*#\$@]+nt\b", re.IGNORECASE), "cunt"),
    (re.compile(r"\bch[\*#\$!@]+tiy[a-z]*\b", re.IGNORECASE), "chutiya"),
    (re.compile(r"\bm[\*#\$!@]+d?arch[a-z]*\b", re.IGNORECASE), "madarchod"),
    (re.compile(r"\bb[\*#\$!@]+h?ench[a-z]*\b", re.IGNORECASE), "behenchod"),
    (re.compile(r"\bbh[\*#\$!@]+(?:s?dik|osdi)[a-z]*\b", re.IGNORECASE), "bhosdike"),
    (re.compile(r"\bg[\*#\$!@]+nd[a-z]*\b", re.IGNORECASE), "gandu"),
    (re.compile(r"\bh[\*#\$!@]+ram[a-z]*\b", re.IGNORECASE), "harami"),
    (re.compile(r"\bk[\*#\$!@]+min[a-z]*\b", re.IGNORECASE), "kamine"),
    (re.compile(r"\bl[\*#\$!@]+nd\b", re.IGNORECASE), "lund"),
    (re.compile(r"\bl[\*#\$!@]+wd[a-z]*\b", re.IGNORECASE), "lawde"),
]

# Leetspeak digit to Latin mapping for mixed alphanumeric tokens
_LEET_DIGIT_MAP = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '5': 's', '7': 't', '8': 'b'})


def deobfuscate_text(text: str) -> str:
    """Multi-pass deobfuscation for adversarial leetspeak, homoglyphs, and masked slurs.

    Follows industry best practices (Google Jigsaw Perspective / Unicode UTS #39 Confusables):
      Pass 1: Strip zero-width evasion characters while safeguarding Devanagari ligatures.
      Pass 2: Map UTS #39 Cyrillic & Greek homoglyphs to canonical Latin characters.
      Pass 3: Collapse punctuation-separated and spaced adversarial evasions.
      Pass 4: Expand masked profanity (e.g., f**k -> fuck, ch**tiya -> chutiya).
      Pass 5: Deobfuscate embedded leetspeak symbols within word contexts (! -> i, $ -> s, @ -> a, 0 -> o, etc.).
    """
    if not text:
        return ""

    t = str(text)

    # Pass 1: Zero-width character stripping (safeguards Devanagari conjuncts)
    t = re.sub(r"[\u200B\uFEFF\u200E\u200F]", "", t)
    t = re.sub(r"(?<=[a-zA-Z0-9])[\u200C\u200D]|[\u200C\u200D](?=[a-zA-Z0-9])", "", t)

    # Pass 2: Unicode UTS #39 Confusables / Homoglyphs
    t = t.translate(_HOMOGLYPH_TRANSLATION_TABLE)

    # Pass 3: Intra-word punctuation and spaced evasion collapse
    t = _PUNCT_SPACED_PATTERN.sub(lambda m: re.sub(r"[\s\.\-_]+", "", m.group(0)), t)
    for p, repl in _KNOWN_SPACED_EVASIONS:
        t = p.sub(repl, t)

    # Pass 4: Masked profanity expansion
    for p, repl in _MASKED_PROFANITY_PATTERNS:
        t = p.sub(repl, t)

    # Pass 5: Leetspeak symbol canonicalization within words
    # '!' followed by a letter (e.g. k!ll, sh!rt, b!tch, th!s) -> 'i'
    t = re.sub(r"!(?=[a-zA-Z])", "i", t)
    # '$' preceding or following letters (e.g. y0ur$elf, a$$hole) -> 's'
    t = re.sub(r"\$(?=[a-zA-Z])|(?<=[a-zA-Z])\$", "s", t)
    # '@' inside words (e.g. m@d) or explicit @ss / @sshole slur
    t = re.sub(r"(?<=[a-zA-Z])@(?=[a-zA-Z])", "a", t)
    t = re.sub(r"(?:^|(?<=\s))@ss(?:hole)?\b", "asshole", t, flags=re.IGNORECASE)

    # Replace digits inside mixed alphanumeric tokens (e.g. 1diot, l0ser, y0urself, c00l)
    def _replace_num_in_word(match):
        w = match.group(0)
        return w.translate(_LEET_DIGIT_MAP)

    t = re.sub(r"\b(?=[a-zA-Z0-9]*[a-zA-Z])(?=[a-zA-Z0-9]*[013578])[a-zA-Z0-9]+\b", _replace_num_in_word, t)

    return t

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


# Sort HINGLISH_MAP by key length descending to prevent sub-string prefix collisions
_SORTED_HINGLISH_ITEMS = sorted(HINGLISH_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)

# Precompile word-bounded regex patterns for robust Hinglish slang replacement
_HINGLISH_REGEX_PATTERNS = [
    (re.compile(r"\b" + re.escape(slang) + r"\b", re.IGNORECASE), english)
    for slang, english in _SORTED_HINGLISH_ITEMS
]


def _clean_social_media(text: str) -> str:
    """Stage 1: Social-media-specific normalization.

    Applied before the training-compatible cleaning so that URLs,
    mentions, adversarial leetspeak, and code-mixed slang are handled for social content.
    These transforms use word boundaries to avoid corrupting legitimate words.
    """
    text = re.sub(r"http\S+", " ", text)       # Remove URLs
    text = re.sub(r"@\w+", " ", text)           # Remove @mentions
    text = deobfuscate_text(text)
    for pattern, english in _HINGLISH_REGEX_PATTERNS:  # Hinglish normalisation with word boundaries
        text = pattern.sub(english, text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)   # Deduplicate excess chars: "heeello" → "heello", "sooo" → "soo"
    return text


def clean_text_for_transformer(text: str) -> str:
    """Non-destructive normalization for transformer models (MuRIL).

    Preserves punctuation, casing nuances, numbers, and multilingual scripts
    while canonicalizing adversarial homoglyphs/leetspeak, stripping distracting
    social noise (URLs, mentions), and collapsing excess whitespace.
    """
    if not text:
        return ""
    t = str(text).strip()
    t = re.sub(r"http\S+", "", t)        # Strip URLs
    t = re.sub(r"@\w+", "", t)           # Strip @mentions
    t = deobfuscate_text(t)
    t = re.sub(r"(.)\1{3,}", r"\1\1", t) # Collapse extreme character flooding (>3 to 2)
    t = re.sub(r"\s+", " ", t)           # Normalize whitespace
    return t.strip()


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
