"""
Multilingual Social Media Toxicity Detector — Streamlit Application.

Analyze text or fetch comments from YouTube, Reddit, or Twitter (X)
to detect toxic content across 6 categories:
  toxic, severe_toxic, obscene, threat, insult, identity_hate
"""

import logging
import streamlit as st

from src.model import load_model, predict_comment, predict_batch
from src.connectors.youtube import fetch_youtube_comments
from src.connectors.reddit import fetch_reddit_comments
from src.connectors.twitter import fetch_twitter_comments
from src.config import config

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Cached Model Loading ──────────────────────────────────
@st.cache_resource(show_spinner="Loading toxicity model...")
def _load_model():
    """Load and cache the model, vectorizer, and labels."""
    return load_model()


# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🧠",
    layout="wide",
)

st.title("💬 Advanced Hate Speech & Toxicity Detector")
st.markdown(
    "Analyze text or fetch comments from **YouTube**, **Reddit**, or **Twitter (X)**"
)

# Load model once (cached across reruns)
try:
    model, vectorizer, labels = _load_model()
except FileNotFoundError as e:
    st.error(f"⚠️ Model files missing: {e}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────
choice = st.sidebar.radio(
    "Choose Input Type",
    ["📝 Custom Text", "📺 YouTube", "👽 Reddit", "🐦 Twitter (X)"],
)


# ── Helper: display results for one comment ──────────────
def _display_result(text_preview: str, preds, score, intensity, *, use_expander=False):
    """Render toxicity results for a single comment."""
    container = st.expander(text_preview, expanded=False) if use_expander else st

    with container:
        if score > config.TOXIC_THRESHOLD:
            st.error("⚠️ Toxic Content Detected")
        else:
            st.success("✅ Safe Comment")

        st.write("### Toxicity Breakdown")
        for label, value in preds.items():
            st.write(f"**{label}**: {value}")

        st.progress(min(score, 1.0))
        st.write(f"**Intensity**: {intensity}")


# ── Custom Text Tab ──────────────────────────────────────
if choice == "📝 Custom Text":
    text = st.text_area("Type or paste text", height=150)

    if st.button("Analyze"):
        if text.strip():
            preds, score, intensity = predict_comment(text, model, vectorizer, labels)
            _display_result(text[:100], preds, score, intensity)
        else:
            st.warning("Please enter text")

# ── Social Media Tabs ────────────────────────────────────
else:
    url = st.text_input("Enter URL")

    if st.button("Fetch & Analyze"):
        if not url.strip():
            st.warning("Please enter a URL")
        else:
            # Fetch comments with error handling
            comments = []
            try:
                with st.spinner("Fetching comments..."):
                    if "YouTube" in choice:
                        comments = fetch_youtube_comments(url)
                    elif "Reddit" in choice:
                        comments = fetch_reddit_comments(url)
                    else:
                        comments = fetch_twitter_comments(url)
            except (ValueError, RuntimeError) as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")
                logger.exception("Unexpected error fetching comments")

            if not comments:
                if not st.session_state.get("_error_shown"):
                    st.info("No comments found.")
            else:
                st.success(f"✅ {len(comments)} comments fetched")

                # Batch prediction for efficiency
                results = predict_batch(comments, model, vectorizer, labels)

                for comment, (preds, score, intensity) in zip(comments, results):
                    # Safe truncation that doesn't break mid-character
                    preview = comment[:80]
                    if len(comment) > 80:
                        preview += "…"
                    _display_result(preview, preds, score, intensity, use_expander=True)

st.markdown("---")