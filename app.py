"""
Multilingual Social Media Toxicity Detector — Streamlit Application.

Analyze text or fetch comments from YouTube, Reddit, or Twitter (X)
to detect toxic content across 6 categories:
  toxic, severe_toxic, obscene, threat, insult, identity_hate

Primary Inference Engine: 🧠 Fine-tuned MuRIL (google/muril-base-cased)
Fallback/Lightweight Engine: ⚡ OneVsRest Logistic Regression Baseline
Moderation Workflow: Assistive 3-tier risk triage & human review
"""

import hashlib
import io
import logging
import pandas as pd
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
from src.moderation import (
    RiskLevel,
    ReviewStatus,
    classify_risk,
    calculate_aggregate_statistics,
    get_review_status,
    set_review_status,
)

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual Toxicity Detector & Moderation Suite",
    page_icon="🧠",
    layout="wide",
)

# ── Session State Initialization ──────────────────────────
if "moderation_reviews" not in st.session_state:
    st.session_state["moderation_reviews"] = {}

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None


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
    st.error(
        f"⚠️ Neither MuRIL nor Logistic Regression models could be loaded.\n"
        f"MuRIL: {muril_error_msg}\nLogistic: {logistic_error_msg}"
    )
    st.stop()


# ── Sidebar: Model Selection & Settings ───────────────────
st.sidebar.title("⚙️ Model & Controls")

engine_options = []
if muril_available:
    engine_options.append("🧠 Multilingual Model (MuRIL)")
if logistic_available:
    engine_options.append("⚡ Fast Model (Logistic Regression)")

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
    active_model_label = "MuRIL Transformer (google/muril-base-cased)"
    st.sidebar.info(
        f"**Engine**: `MuRIL Transformer`\n\n"
        f"**Backbone**: `{config.MURIL_MODEL_NAME}`\n\n"
        f"**Device**: `{dev_str}`\n\n"
        f"**Thresholds**: `Phase 4C Optimized`"
    )
else:
    log_model, log_vectorizer, log_labels = logistic_data
    active_model_label = "Logistic Regression Baseline (TF-IDF)"
    st.sidebar.info(
        "**Engine**: `Logistic Regression Baseline`\n\n"
        "**Features**: `TF-IDF Word + Char N-Grams`\n\n"
        "**Threshold**: `0.50 Default`"
    )

st.sidebar.markdown("---")

# Session Review Stats
reviews = st.session_state["moderation_reviews"]
if reviews:
    st.sidebar.subheader("📋 Session Moderation Summary")
    rev_counts = {
        ReviewStatus.CONFIRMED_TOXIC.value: sum(1 for v in reviews.values() if v == ReviewStatus.CONFIRMED_TOXIC.value),
        ReviewStatus.FALSE_POSITIVE.value: sum(1 for v in reviews.values() if v == ReviewStatus.FALSE_POSITIVE.value),
        ReviewStatus.REVIEW.value: sum(1 for v in reviews.values() if v == ReviewStatus.REVIEW.value),
    }
    st.sidebar.caption(f"Confirmed Toxic: **{rev_counts[ReviewStatus.CONFIRMED_TOXIC.value]}**")
    st.sidebar.caption(f"False Positives: **{rev_counts[ReviewStatus.FALSE_POSITIVE.value]}**")
    st.sidebar.caption(f"Pending Review: **{rev_counts[ReviewStatus.REVIEW.value]}**")
    if st.sidebar.button("🧹 Clear Review History", help="Reset session-only review tags"):
        st.session_state["moderation_reviews"] = {}
        st.rerun()

st.sidebar.markdown("---")
choice = st.sidebar.radio(
    "Choose Input Source",
    ["📝 Custom Text", "📁 Batch Text / CSV", "📺 YouTube", "👽 Reddit", "🐦 Twitter (X)"],
)


# ── Main Header ───────────────────────────────────────────
st.title("🛡️ Multilingual Toxicity Detection & Moderation Studio")
st.markdown(
    "Assistive AI toxicity classification across **English**, **Hindi (हिंदी)**, and **Hinglish**.\n\n"
    "> ℹ️ **Assistive Workflow**: *Detection → Risk Assessment → Human Review → Action*. "
    "Model scores are algorithmic probabilistic indicators requiring human moderation judgment."
)


# ── Helpers: Unified Prediction Dispatchers ───────────────
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

        results = predict_batch_muril(
            texts,
            muril_model,
            muril_tokenizer,
            muril_labels,
            muril_device,
            batch_size=config.BATCH_SIZE,
            thresholds=muril_thresholds,
            progress_callback=p_cb,
        )
        return results, muril_thresholds
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


# ── Helper: Display Moderation Result Card ────────────────
def _display_moderation_card(
    comment_text: str,
    preds: dict,
    score: float,
    intensity: str,
    is_toxic_flags: dict,
    thresholds: dict,
    *,
    use_expander: bool = False,
    card_index: int = 0,
):
    """Render structured 3-tier moderation result card with human review actions."""
    risk_level, badge_icon, explanation = classify_risk(preds, is_toxic_flags, thresholds)
    detected_categories = [lbl for lbl, flag in is_toxic_flags.items() if flag]

    # Generate persistent unique hash for session tracking
    c_hash = hashlib.sha256(f"{card_index}_{comment_text}".encode("utf-8")).hexdigest()[:12]
    current_review = get_review_status(st.session_state["moderation_reviews"], c_hash)

    # Title header for card
    preview = comment_text[:90] + ("…" if len(comment_text) > 90 else "")
    header_title = f"{badge_icon} [{risk_level.value.upper()}] {preview}"
    if current_review != ReviewStatus.PENDING.value:
        header_title += f" • 🏷️ {current_review}"

    container = st.expander(header_title, expanded=not use_expander) if use_expander else st.container()

    with container:
        # 1. Status Banner
        if risk_level == RiskLevel.HIGH_RISK:
            st.error(f"🔴 **High Risk Detected** — {explanation}")
        elif risk_level == RiskLevel.REVIEW:
            st.warning(f"🟡 **Moderator Review Recommended** — {explanation}")
        else:
            st.success(f"🟢 **Content Appears Safe** — {explanation}")

        # Metadata Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.caption(f"**Active Engine**: {active_model_label}")
        mcol2.caption(f"**Peak Score**: `{score:.3f}` ({intensity})")
        mcol3.caption(f"**Flagged Categories**: `{', '.join(detected_categories) if detected_categories else 'None'}`")
        review_badge = f"**Review Status**: `{current_review}`"
        mcol4.caption(review_badge)

        # Full Comment Text Display
        if use_expander:
            st.text_area("Full Comment Text", value=comment_text, height=75, disabled=True, key=f"txt_{c_hash}")

        # 2. Detailed Category Breakdown with Thresholds
        st.markdown("###### Multi-Label Breakdown & Threshold Comparison")
        cols = st.columns(6)
        for i, (label, prob) in enumerate(preds.items()):
            col = cols[i % 6]
            th = thresholds.get(label, 0.5)
            flagged = is_toxic_flags.get(label, False)
            icon = "🚨" if flagged else "🟢"
            with col:
                st.metric(
                    label=f"{icon} {label.replace('_', ' ').title()}",
                    value=f"{prob:.3f}",
                    delta=f"{'+' if prob >= th else ''}{prob - th:.2f} (th: {th:.2f})",
                    delta_color="inverse" if flagged else "normal",
                )

        # 3. Human Review Action Controls (Assistive Workflow)
        st.markdown("###### 👤 Human Moderation Action")
        rcol1, rcol2, rcol3, rcol4 = st.columns([2, 2, 2, 4])

        with rcol1:
            if st.button("🟡 Flag for Review", key=f"btn_rev_{c_hash}"):
                set_review_status(st.session_state["moderation_reviews"], c_hash, ReviewStatus.REVIEW.value)
                st.rerun()
        with rcol2:
            if st.button("🔴 Confirm Toxic", key=f"btn_tox_{c_hash}"):
                set_review_status(st.session_state["moderation_reviews"], c_hash, ReviewStatus.CONFIRMED_TOXIC.value)
                st.rerun()
        with rcol3:
            if st.button("🟢 Mark False Positive", key=f"btn_fp_{c_hash}"):
                set_review_status(st.session_state["moderation_reviews"], c_hash, ReviewStatus.FALSE_POSITIVE.value)
                st.rerun()
        with rcol4:
            if current_review != ReviewStatus.PENDING.value:
                st.info(f"Marked as: **{current_review}** (Session-only tag)")


# ── Helper: Display Aggregate Dashboard ───────────────────
def _render_aggregate_dashboard(comments: list, results: list, thresholds: dict):
    """Render aggregate statistics and filtered comment stream."""
    stats = calculate_aggregate_statistics(results, thresholds)

    st.markdown("### 📊 Batch Toxicity & Risk Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Analyzed", stats["total"])
    c2.metric("Toxic Flagged", stats["toxic_count"], delta=f"{stats['toxic_percentage']}%", delta_color="inverse")
    c3.metric("🔴 High Risk", stats["high_risk_count"])
    c4.metric("🟡 Under Review", stats["review_count"])
    c5.metric("🟢 Safe Content", stats["safe_risk_count"])

    # Category counts
    st.markdown("##### Category Distribution")
    cat_cols = st.columns(6)
    for i, (cat, count) in enumerate(stats["category_counts"].items()):
        pct = stats["category_percentages"].get(cat, 0.0)
        with cat_cols[i % 6]:
            st.metric(cat.replace("_", " ").title(), count, delta=f"{pct}%")

    st.markdown("---")
    st.markdown("### 🔎 Individual Comments Moderation Feed")

    # Filter selector
    filter_choice = st.selectbox(
        "Filter comment feed by risk/status:",
        ["All Comments", "🔴 High Risk Only", "🟡 Review Only", "🚨 Toxic Flagged Only", "🟢 Safe Only"],
    )

    # Filter items
    displayed_items = []
    for idx, (comment, res) in enumerate(zip(comments, results)):
        preds, score, intensity, is_toxic_flags = res
        r_level, _, _ = classify_risk(preds, is_toxic_flags, thresholds)
        is_toxic = any(is_toxic_flags.values())

        if filter_choice == "🔴 High Risk Only" and r_level != RiskLevel.HIGH_RISK:
            continue
        if filter_choice == "🟡 Review Only" and r_level != RiskLevel.REVIEW:
            continue
        if filter_choice == "🚨 Toxic Flagged Only" and not is_toxic:
            continue
        if filter_choice == "🟢 Safe Only" and r_level != RiskLevel.SAFE:
            continue

        displayed_items.append((idx, comment, preds, score, intensity, is_toxic_flags))

    st.caption(f"Showing **{len(displayed_items)}** of **{len(comments)}** comments.")

    for idx, comment, preds, score, intensity, is_toxic_flags in displayed_items:
        _display_moderation_card(
            comment, preds, score, intensity, is_toxic_flags, thresholds, use_expander=True, card_index=idx
        )


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
            with st.spinner("Analyzing text with active model..."):
                preds, score, intensity, is_toxic_flags, thresholds = _predict_single(text)
            _display_moderation_card(
                text, preds, score, intensity, is_toxic_flags, thresholds, use_expander=False, card_index=0
            )
        else:
            st.warning("Please enter some text to analyze.")


# ── Batch Text / CSV Tab ──────────────────────────────────
elif choice == "📁 Batch Text / CSV":
    st.subheader("Batch Text & CSV File Analysis")
    input_mode = st.radio("Batch Input Format", ["Paste Multi-line Text", "Upload CSV File"], horizontal=True)

    batch_texts = []
    if input_mode == "Paste Multi-line Text":
        pasted = st.text_area("Paste comments (one comment per line):", height=150, placeholder="Line 1: Great video!\nLine 2: You are an idiot\nLine 3: Wah kya baat hai...")
        if st.button("🔍 Analyze Multi-line Batch", type="primary"):
            batch_texts = [line.strip() for line in pasted.splitlines() if line.strip()]
    else:
        uploaded_file = st.file_uploader("Upload a CSV file containing comments", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded CSV:", df.head(3))
                text_col = st.selectbox("Select text column to analyze:", df.columns.tolist())
                if st.button("🔍 Analyze CSV Batch", type="primary"):
                    batch_texts = df[text_col].dropna().astype(str).tolist()
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

    if batch_texts:
        st.info(f"Loaded **{len(batch_texts)}** comments. Executing batched inference...")
        p_bar = st.progress(0.0, text="Starting batch inference...")
        results, thresholds = _predict_multiple(batch_texts, progress_bar=p_bar)
        _render_aggregate_dashboard(batch_texts, results, thresholds)


# ── Social Media Tabs ────────────────────────────────────
else:
    platform_name = choice.split()[-1]
    st.subheader(f"Fetch & Moderate Comments from {platform_name}")
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
                st.info(f"No comments found or comments could not be fetched from {platform_name}.")
            else:
                st.success(f"✅ Fetched **{len(comments)}** comments. Running batched inference with {selected_engine_name}...")
                p_bar = st.progress(0.0, text="Starting batch inference...")
                results, thresholds = _predict_multiple(comments, progress_bar=p_bar)
                _render_aggregate_dashboard(comments, results, thresholds)

st.markdown("---")
st.caption("Multilingual Toxicity Moderation Studio • Phase 5B Production Evaluation • Powered by fine-tuned MuRIL")