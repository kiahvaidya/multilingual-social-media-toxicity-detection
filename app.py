"""
Multilingual Social Media Toxicity Detector — Streamlit Application.

Analyze text or fetch comments from YouTube, Reddit, or Twitter (X)
to detect toxic content across 6 categories:
  toxic, severe_toxic, obscene, threat, insult, identity_hate

Primary Inference Engine: 🧠 Fine-tuned MuRIL (google/muril-base-cased)
Fallback/Lightweight Engine: ⚡ OneVsRest Logistic Regression Baseline
"""

import logging
import streamlit as st

from src.config import config
from src.connectors.reddit import fetch_reddit_comments
from src.connectors.twitter import fetch_twitter_comments
from src.connectors.youtube import fetch_youtube_comments
from src.model import (
    get_device,
    load_model as load_logistic_model,
    load_muril_model,
    predict_batch as predict_batch_logistic,
    predict_batch_muril,
    predict_comment as predict_comment_logistic,
    predict_comment_muril,
)

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual Toxicity Detector",
    page_icon="🧠",
    layout="wide",
)


# ── Cached Model Loading ──────────────────────────────────
@st.cache_resource(show_spinner="Loading primary multilingual toxicity model (MuRIL)...")
def _get_muril_engine():
    """Load and cache the MuRIL model, tokenizer, labels, thresholds, and device."""
    return load_muril_model()


@st.cache_resource(show_spinner="Loading fast baseline model (Logistic Regression)...")
def _get_logistic_engine():
    """Load and cache the Logistic Regression baseline model, vectorizer, and labels."""
    return load_logistic_model()


# ── Initialize Available Engines ──────────────────────────
muril_available = False
muril_data = None
muril_error_msg = None

try:
    muril_data = _get_muril_engine()
    muril_available = True
except Exception as e:
    muril_error_msg = str(e)
    logger.warning("MuRIL engine not available: %s", e)

logistic_available = False
logistic_data = None
logistic_error_msg = None

try:
    logistic_data = _get_logistic_engine()
    logistic_available = True
except Exception as e:
    logistic_error_msg = str(e)
    logger.warning("Logistic engine not available: %s", e)

if not muril_available and not logistic_available:
    st.error(f"⚠️ Neither MuRIL nor Logistic Regression models could be loaded.\nMuRIL: {muril_error_msg}\nLogistic: {logistic_error_msg}")
    st.stop()


# ── Sidebar: Model Selection & Settings ───────────────────
st.sidebar.title("⚙️ Model & Input")

engine_options = []
if muril_available:
    engine_options.append("🧠 Multilingual Model (MuRIL)")
if logistic_available:
    engine_options.append("⚡ Fast Model (Logistic Regression)")

# MuRIL is the default engine
selected_engine_name = st.sidebar.radio(
    "Active Inference Engine",
    engine_options,
    index=0,
    help="MuRIL supports English, Hindi (Devanagari), and Hinglish with validation-optimized per-label thresholds.",
)

# Fallback warning if MuRIL is unavailable
if not muril_available:
    st.sidebar.warning(f"⚠️ MuRIL unavailable ({muril_error_msg}). Operating in Fast Baseline mode.")

# Active Model Status Badge
if "MuRIL" in selected_engine_name and muril_available:
    muril_model, muril_tokenizer, muril_labels, muril_thresholds, muril_device = muril_data
    dev_str = "CUDA (GPU)" if muril_device.type == "cuda" else "CPU"
    st.sidebar.info(
        f"**Engine**: `MuRIL Transformer`\n\n"
        f"**Backbone**: `{config.MURIL_MODEL_NAME}`\n\n"
        f"**Device**: `{dev_str}`\n\n"
        f"**Thresholds**: `Phase 4C Optimized`"
    )
else:
    log_model, log_vectorizer, log_labels = logistic_data
    st.sidebar.info(
        "**Engine**: `Logistic Regression Baseline`\n\n"
        "**Features**: `TF-IDF Word + Char N-Grams`\n\n"
        "**Threshold**: `0.50 Default`"
    )

st.sidebar.markdown("---")

choice = st.sidebar.radio(
    "Choose Input Type",
    ["📝 Custom Text", "📺 YouTube", "👽 Reddit", "🐦 Twitter (X)"],
)


# ── Main Header ───────────────────────────────────────────
st.title("💬 Advanced Multilingual Toxicity & Hate Speech Detector")
st.markdown(
    "Analyze comments in **English**, **Hindi (हिंदी)**, and **Hinglish** across 6 toxicity categories:\n"
    "`toxic` • `severe_toxic` • `obscene` • `threat` • `insult` • `identity_hate`"
)


# ── Helper: Unified Prediction Dispatcher ─────────────────
def _predict_single(text: str):
    if "MuRIL" in selected_engine_name and muril_available:
        preds, score, intensity, is_toxic_flags = predict_comment_muril(
            text, muril_model, muril_tokenizer, muril_labels, muril_device, muril_thresholds
        )
        return preds, score, intensity, is_toxic_flags, muril_thresholds
    else:
        preds, score, intensity = predict_comment_logistic(
            text, log_model, log_vectorizer, log_labels
        )
        is_toxic_flags = {lbl: (val >= config.TOXIC_THRESHOLD) for lbl, val in preds.items()}
        thresh_map = {lbl: config.TOXIC_THRESHOLD for lbl in log_labels}
        return preds, score, intensity, is_toxic_flags, thresh_map


def _predict_multiple(texts: list, progress_bar=None):
    if "MuRIL" in selected_engine_name and muril_available:
        def p_cb(frac):
            if progress_bar is not None:
                progress_bar.progress(frac, text=f"Analyzing with MuRIL... {int(frac * 100)}%")

        return predict_batch_muril(
            texts, muril_model, muril_tokenizer, muril_labels, muril_device,
            batch_size=config.BATCH_SIZE, thresholds=muril_thresholds, progress_callback=p_cb
        ), muril_thresholds
    else:
        raw_results = predict_batch_logistic(texts, log_model, log_vectorizer, log_labels)
        adapted = []
        for preds, score, intensity in raw_results:
            flags = {lbl: (val >= config.TOXIC_THRESHOLD) for lbl, val in preds.items()}
            adapted.append((preds, score, intensity, flags))
        if progress_bar is not None:
            progress_bar.progress(1.0, text="Analysis complete!")
        thresh_map = {lbl: config.TOXIC_THRESHOLD for lbl in log_labels}
        return adapted, thresh_map


# ── Helper: Display Result Card ───────────────────────────
def _display_result(text_preview: str, preds, score, intensity, is_toxic_flags, thresholds, *, use_expander=False):
    """Render structured toxicity breakdown for a single comment."""
    container = st.expander(text_preview, expanded=False) if use_expander else st

    with container:
        is_any_toxic = any(is_toxic_flags.values())
        detected_categories = [lbl for lbl, flag in is_toxic_flags.items() if flag]

        if is_any_toxic:
            st.error(f"⚠️ **Toxic Content Detected** (Categories: {', '.join(detected_categories)})")
        else:
            st.success("✅ **Safe / Non-Toxic Comment**")

        st.markdown("##### Category Breakdown & Threshold Comparison")
        cols = st.columns(3)
        for i, (label, prob) in enumerate(preds.items()):
            col = cols[i % 3]
            th = thresholds.get(label, 0.5)
            flagged = is_toxic_flags.get(label, False)
            status_icon = "🚨" if flagged else "🟢"
            with col:
                st.metric(
                    label=f"{status_icon} {label.replace('_', ' ').title()}",
                    value=f"{prob:.3f}",
                    delta=f"{'+' if prob >= th else ''}{prob - th:.2f} (Threshold: {th:.2f})",
                    delta_color="inverse" if flagged else "normal",
                )

        st.progress(min(score, 1.0), text=f"Overall Toxicity Intensity: {intensity} ({score:.3f})")


# ── Custom Text Tab ──────────────────────────────────────
if choice == "📝 Custom Text":
    text = st.text_area(
        "Type or paste text (English, Hindi, or Hinglish)",
        placeholder="e.g., You are doing an awesome job! OR तुम बहुत अच्छे हो OR Bhai bohot mast video tha...",
        height=140,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze Text", type="primary")

    if analyze_btn:
        if text.strip():
            with st.spinner("Analyzing text..."):
                preds, score, intensity, is_toxic_flags, thresholds = _predict_single(text)
            _display_result(text[:100], preds, score, intensity, is_toxic_flags, thresholds)
        else:
            st.warning("Please enter some text to analyze.")


# ── Social Media Tabs ────────────────────────────────────
else:
    platform_name = choice.split()[-1]
    url = st.text_input(f"Enter {platform_name} URL to fetch and analyze comments")

    if st.button("🚀 Fetch & Analyze Comments", type="primary"):
        if not url.strip():
            st.warning(f"Please enter a valid {platform_name} URL.")
        else:
            comments = []
            try:
                with st.spinner(f"Fetching latest comments from {platform_name}..."):
                    if "YouTube" in choice:
                        comments = fetch_youtube_comments(url)
                    elif "Reddit" in choice:
                        comments = fetch_reddit_comments(url)
                    else:
                        comments = fetch_twitter_comments(url)
            except (ValueError, RuntimeError) as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"⚠️ Unexpected error fetching comments: {e}")
                logger.exception("Unexpected error fetching comments")

            if not comments:
                if not st.session_state.get("_error_shown"):
                    st.info("No comments found or comments could not be fetched.")
            else:
                st.success(f"✅ Fetched **{len(comments)}** comments. Processing with {selected_engine_name}...")

                p_bar = st.progress(0.0, text="Starting batch inference...")
                results, thresholds = _predict_multiple(comments, progress_bar=p_bar)

                # Summary metrics
                toxic_count = sum(1 for _, _, _, flags in results if any(flags.values()))
                safe_count = len(results) - toxic_count

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Comments", len(results))
                col_b.metric("Toxic Flagged", toxic_count, delta=f"{(toxic_count / len(results) * 100):.1f}%", delta_color="inverse")
                col_c.metric("Safe Comments", safe_count, delta=f"{(safe_count / len(results) * 100):.1f}%")

                st.markdown("---")
                st.markdown("#### Individual Comment Breakdown")

                for comment, (preds, score, intensity, is_toxic_flags) in zip(comments, results):
                    preview = comment[:85]
                    if len(comment) > 85:
                        preview += "…"
                    _display_result(preview, preds, score, intensity, is_toxic_flags, thresholds, use_expander=True)

st.markdown("---")
st.caption("Multilingual Toxicity Detection System • Powered by fine-tuned MuRIL & Streamlit")