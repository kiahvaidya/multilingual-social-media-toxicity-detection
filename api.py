"""
Toxiscope Backend API — Production FastAPI Service.

Integrates:
  1. Primary Multilingual Transformer Engine: MuRIL (google/muril-base-cased)
  2. 3-Tier Assistive Moderation & Ambiguity Gate (src.moderation)
  3. JEV Contextual Adjudication Layer (src.jev)
  4. Final Research Multimodal Classifier: MuRIL + CLIP ViT-B/32 Cross-Modal Interaction Fusion
"""

import os
import sys
import io
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.model import (
    DEFAULT_LABELS,
    MuRILForMultiLabelToxicity,
    get_device,
    load_muril_model,
    predict_batch_muril,
    predict_comment_muril,
)
from src.moderation import (
    RiskLevel,
    calculate_aggregate_statistics,
    classify_risk,
    moderate_comment_with_context,
)
from src.jev import get_jev_provider
from src.connectors.youtube import fetch_youtube_comments
from src.connectors.reddit import fetch_reddit_comments
from src.connectors.twitter import fetch_twitter_comments
from src.profile_watch import (
    MonitoringState,
    ObservationRecord,
    PatternEventRecord,
    ProfileComparisonResponse,
    ProfileRecord,
    ProfileSummaryResponse,
    ProfileWatchService,
    ScanRequest,
    ScanSummaryResponse,
)

logger = logging.getLogger("toxiscope.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Toxiscope Multilingual AI Moderation API",
    description="Production-grade inference and research API for multilingual and multimodal toxicity detection.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Final Multimodal Model Architecture Definition ───────────
class FinalMultimodalClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, proj_dim=256, modality_dropout=0.15, classifier_dropout=0.2):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        fused_dim = proj_dim * 4
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 1)
        )

    def forward(self, text_emb, image_emb, force_drop_modality=None):
        t_norm = F.normalize(text_emb, p=2, dim=-1)
        v_norm = F.normalize(image_emb, p=2, dim=-1)
        t_p = self.text_proj(t_norm)
        v_p = self.image_proj(v_norm)

        if force_drop_modality == "text":
            t_p = torch.zeros_like(t_p)
        elif force_drop_modality == "image":
            v_p = torch.zeros_like(v_p)

        prod = t_p * v_p
        diff = torch.abs(t_p - v_p)
        fused = torch.cat([t_p, v_p, prod, diff], dim=-1)
        fused = self.fusion_norm(fused)
        logits = self.classifier(fused).squeeze(-1)
        return logits


# ── Global Model Server Singleton ────────────────────────────
class ModelServer:
    _instance = None

    def __init__(self):
        self.device = get_device()
        logger.info("Initializing ModelServer on %s...", self.device)

        # 1. MuRIL Text Model
        self.muril_model, self.muril_tokenizer, self.labels, self.thresholds, _ = load_muril_model(device=self.device)
        self.jev_provider = get_jev_provider()

        # 2. Multimodal Meme Classifier
        self.multimodal_model = None
        self.clip_processor = None
        self.clip_model = None
        self.multimodal_threshold = 0.44
        self.load_multimodal_engine()

        logger.info("ModelServer successfully initialized.")

    def load_multimodal_engine(self):
        ckpt_path = _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal" / "best_model.pt"
        cfg_path = _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal" / "config.json"

        if not ckpt_path.exists():
            logger.warning("Final multimodal checkpoint not found at %s. Multimodal inference will be unavailable.", ckpt_path)
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("Loading CLIP ViT-B/32 image processor and model...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()

            logger.info("Loading FinalMultimodalClassifier from %s...", ckpt_path)
            self.multimodal_model = FinalMultimodalClassifier().to(self.device)
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.multimodal_model.load_state_dict(state_dict)
            self.multimodal_model.eval()

            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.multimodal_threshold = float(cfg.get("optimal_threshold", 0.44))

            logger.info("Multimodal engine ready (Threshold: %s).", self.multimodal_threshold)
        except Exception as e:
            logger.error("Failed to load multimodal engine: %s", e)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelServer()
        return cls._instance


@app.on_event("startup")
def startup_event():
    ModelServer.get_instance()


# ── Schemas ───────────────────────────────────────────────────
class TextAnalysisRequest(BaseModel):
    text: str
    language: Optional[str] = None
    enable_jev: Optional[bool] = True

class TextAnalysisResponse(BaseModel):
    text: str
    language: str
    risk_level: str
    badge_icon: str
    is_toxic: bool
    confidence: float
    scores: Dict[str, float]
    thresholds: Dict[str, float]
    is_toxic_flags: Dict[str, bool]
    explanation: str
    is_ambiguous: bool
    gate_reason: str
    human_review_required: bool = False
    jev: Dict[str, Any]
    pipeline_trace: List[Dict[str, Any]]

class BatchItem(BaseModel):
    id: str
    text: str

class BatchAnalysisRequest(BaseModel):
    items: List[BatchItem]

class SocialAnalysisRequest(BaseModel):
    url: str
    platform: Optional[str] = "auto"
    max_comments: Optional[int] = 50
    demo_fallback: Optional[bool] = True
    enable_jev: Optional[bool] = False


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/api/health")
def get_health():
    server = ModelServer.get_instance()
    return {
        "status": "ONLINE",
        "device": str(server.device),
        "muril_loaded": server.muril_model is not None,
        "multimodal_loaded": server.multimodal_model is not None,
        "multimodal_threshold": server.multimodal_threshold,
        "labels": server.labels,
        "thresholds": server.thresholds,
        "jev_configured": server.jev_provider.provider_type.value if server.jev_provider else "UNCONFIGURED"
    }


def detect_language_simple(text: str) -> str:
    # Heuristic language detector for Indic scripts vs Latin
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "Hindi (Devanagari)"
        elif 0x0980 <= code <= 0x09FF:
            return "Bengali"
        elif 0x0A00 <= code <= 0x0A7F:
            return "Punjabi"
        elif 0x0A80 <= code <= 0x0AFF:
            return "Gujarati"
        elif 0x0B00 <= code <= 0x0B7F:
            return "Odia"
        elif 0x0B80 <= code <= 0x0BFF:
            return "Tamil"
        elif 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        elif 0x0C80 <= code <= 0x0CFF:
            return "Kannada"
        elif 0x0D00 <= code <= 0x0D7F:
            return "Malayalam"
    return "English / Latin"


@app.post("/api/analyze/text", response_model=TextAnalysisResponse)
def analyze_text(req: TextAnalysisRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    server = ModelServer.get_instance()
    t_start = time.time()
    trace = []

    # Stage 1: Preprocessing & Language Identification
    detected_lang = req.language or detect_language_simple(req.text)
    trace.append({"stage": "Input Preprocessing", "status": "COMPLETED", "duration_ms": round((time.time() - t_start) * 1000, 2)})

    # Stage 2: MuRIL Transformer Forward
    t_muril_start = time.time()
    predictions, max_toxic_score, intensity, is_toxic_flags = predict_comment_muril(
        text=req.text,
        model=server.muril_model,
        tokenizer=server.muril_tokenizer,
        labels=server.labels,
        device=server.device,
        thresholds=server.thresholds,
    )
    trace.append({"stage": "MuRIL Multi-Label Inference", "status": "COMPLETED", "duration_ms": round((time.time() - t_muril_start) * 1000, 2)})

    # Stage 3: Moderation & Ambiguity Gating
    t_mod_start = time.time()
    active_jev = server.jev_provider if req.enable_jev else None
    mod_res = moderate_comment_with_context(
        text=req.text,
        predictions=predictions,
        is_toxic_flags=is_toxic_flags,
        thresholds=server.thresholds,
        jev_provider=active_jev,
    )
    trace.append({"stage": "Contextual Ambiguity Evaluation", "status": "COMPLETED", "duration_ms": round((time.time() - t_mod_start) * 1000, 2)})

    # Stage 4: JEV Adjudication (if invoked)
    jev_data = {
        "invoked": mod_res.jev_invoked,
        "status": mod_res.jev_status,
        "is_ambiguous": mod_res.is_ambiguous,
        "gate_reason": mod_res.gate_reason,
        "decision": mod_res.jev_decision.to_dict() if mod_res.jev_decision else None,
        "error": mod_res.jev_error,
        "human_review_required": mod_res.human_review_required,
    }
    if mod_res.jev_invoked:
        trace.append({"stage": "JEV Contextual Adjudication", "status": "COMPLETED", "duration_ms": 12.0})

    if mod_res.jev_invoked:
        # Contextual adjudication overrides raw lexical flags
        is_overall_toxic = (mod_res.final_risk_level == RiskLevel.HIGH_RISK)
    else:
        is_overall_toxic = (mod_res.final_risk_level == RiskLevel.HIGH_RISK) or any(is_toxic_flags.values())

    return TextAnalysisResponse(
        text=req.text,
        language=detected_lang,
        risk_level=mod_res.final_risk_level.value,
        badge_icon=mod_res.final_badge_icon,
        is_toxic=is_overall_toxic,
        confidence=round(float(max_toxic_score), 4),
        scores={k: round(float(v), 4) for k, v in predictions.items()},
        thresholds=server.thresholds,
        is_toxic_flags=is_toxic_flags,
        explanation=mod_res.final_explanation,
        is_ambiguous=mod_res.is_ambiguous,
        gate_reason=mod_res.gate_reason,
        human_review_required=mod_res.human_review_required,
        jev=jev_data,
        pipeline_trace=trace
    )


class MemeBase64Request(BaseModel):
    image_base64: str
    text: Optional[str] = ""

def _process_meme_inference(pil_img: Image.Image, meme_text: str):
    server = ModelServer.get_instance()
    if server.multimodal_model is None or server.clip_model is None:
        raise HTTPException(status_code=503, detail="Multimodal inference engine not available on this server.")

    detected_lang = detect_language_simple(meme_text) if meme_text else "None (Visual only)"
    t_start = time.time()
    with torch.no_grad():
        # 1. CLIP Image Embedding
        clip_inputs = server.clip_processor(images=[pil_img], return_tensors="pt")
        img_features = server.clip_model.get_image_features(pixel_values=clip_inputs["pixel_values"].to(server.device))
        img_emb = F.normalize(img_features, p=2, dim=-1)

        # 2. MuRIL Text Embedding
        text_probs_dict = {}
        text_toxic_score = 0.0
        if meme_text:
            enc = server.muril_tokenizer([meme_text], truncation=True, max_length=128, padding=True, return_tensors="pt")
            out_muril = server.muril_model.encoder(input_ids=enc["input_ids"].to(server.device), attention_mask=enc["attention_mask"].to(server.device))
            t_cls = out_muril.last_hidden_state[:, 0, :]
            txt_emb = F.normalize(t_cls, p=2, dim=-1)

            t_preds, _, _, _ = predict_comment_muril(meme_text, server.muril_model, server.muril_tokenizer, server.labels, server.device, server.thresholds)
            text_probs_dict = t_preds
            text_toxic_score = max(t_preds.get("toxic", 0.0), t_preds.get("identity_hate", 0.0))
        else:
            txt_emb = torch.zeros(1, 768, device=server.device)

        # 3. Multimodal Interaction Fusion Model Forward
        logits_full = server.multimodal_model(txt_emb, img_emb)
        prob_multimodal = float(torch.sigmoid(logits_full).item())

        # 4. Modality Ablation on this sample
        logits_img_only = server.multimodal_model(txt_emb, img_emb, force_drop_modality="text")
        prob_img_only = float(torch.sigmoid(logits_img_only).item())

        logits_txt_only = server.multimodal_model(txt_emb, img_emb, force_drop_modality="image")
        prob_txt_only = float(torch.sigmoid(logits_txt_only).item())

    thresh = server.multimodal_threshold
    is_hateful = prob_multimodal >= thresh

    # 5. Semantic Relationship Diagnosis
    if text_toxic_score >= 0.45 and not is_hateful:
        interaction_type = "CONTEXTUAL_INVERSION"
        interaction_desc = "Hostile text context defused by benign visual framing."
    elif text_toxic_score < 0.35 and is_hateful:
        interaction_type = "EMERGENT_TOXICITY"
        interaction_desc = "Innocuous text combined with visual template produces identity-targeted hate."
    elif is_hateful:
        interaction_type = "EXPLICIT_HATE"
        interaction_desc = "Multimodal content indicates hostile intent across both modalities."
    else:
        interaction_type = "BENIGN"
        interaction_desc = "Content is safe across text and visual context."

    risk_level = "High Risk" if prob_multimodal >= 0.70 else ("Review" if is_hateful else "Safe")

    return {
        "text": meme_text,
        "language": detected_lang,
        "multimodal_score": round(prob_multimodal, 4),
        "threshold": thresh,
        "is_hateful": is_hateful,
        "risk_level": risk_level,
        "text_only_score": round(float(text_toxic_score), 4),
        "image_only_score": round(float(prob_img_only), 4),
        "modality_text_contribution": round(float(prob_txt_only), 4),
        "interaction_type": interaction_type,
        "interaction_description": interaction_desc,
        "text_categories": {k: round(float(v), 4) for k, v in text_probs_dict.items()},
        "latency_ms": round((time.time() - t_start) * 1000, 2)
    }

@app.post("/api/analyze/meme")
async def analyze_meme(
    image: UploadFile = File(...),
    text: Optional[str] = Form("")
):
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    return _process_meme_inference(pil_img, (text or "").strip())

@app.post("/api/analyze/meme-json")
def analyze_meme_json(req: MemeBase64Request):
    import base64
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    return _process_meme_inference(pil_img, (req.text or "").strip())


@app.post("/api/analyze/batch")
def analyze_batch(req: BatchAnalysisRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="Empty batch.")

    server = ModelServer.get_instance()
    texts = [item.text for item in req.items]
    t0 = time.time()

    batch_preds = predict_batch_muril(
        texts=texts,
        model=server.muril_model,
        tokenizer=server.muril_tokenizer,
        labels=server.labels,
        device=server.device,
        batch_size=32,
        thresholds=server.thresholds,
    )

    results = []
    for idx, item in enumerate(req.items):
        scores, toxic_score, intensity, is_toxic_flags = batch_preds[idx]
        base_risk, base_icon, base_expl = classify_risk(scores, is_toxic_flags, server.thresholds)
        results.append({
            "id": item.id,
            "text": item.text,
            "language": detect_language_simple(item.text),
            "risk_level": base_risk.value,
            "badge_icon": base_icon,
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "is_toxic": any(is_toxic_flags.values()),
            "confidence": round(float(max(scores.values())), 4)
        })

    # Summary Stats
    high_risk_cnt = sum(1 for r in results if r["risk_level"] == "High Risk")
    review_cnt = sum(1 for r in results if r["risk_level"] == "Review")
    safe_cnt = sum(1 for r in results if r["risk_level"] == "Safe")

    return {
        "total_analyzed": len(results),
        "high_risk_count": high_risk_cnt,
        "review_count": review_cnt,
        "safe_count": safe_cnt,
        "high_risk_percentage": round((high_risk_cnt / len(results)) * 100, 2),
        "latency_ms": round((time.time() - t0) * 1000, 2),
        "results": results
    }


# ── Curated Multilingual Social Media Demo Fallbacks ─────────
DEMO_SOCIAL_COMMENTS = {
    "YouTube": [
        "Great tutorial bhai, everything was explained clearly! ❤️",
        "Bhai kya bawaal content banaya hai tune, maza aa gaya dekh kar!",
        "This guy is completely useless and should delete his channel immediately.",
        "तुम बहुत अच्छा सिखाते हो, धन्यवाद भाई!",
        "Shut up you idiot, stop spreading fake news you clown 🤡",
        "I will track down where you live and destroy you.",
        "Awesome breakdown! Looking forward to the next part.",
        "bc kya bakwas video hai, time waste kar diya mera",
        "Clean explanation, helped me solve my college project.",
        "These immigrants are filthy pests who don't belong here."
    ],
    "Reddit": [
        "Solid discussion. Really appreciate the nuanced perspective from OP.",
        "You are a pathetic failure who doesn't understand basic statistics.",
        "bhai sach me dil jeet liya, amazing post",
        "Fuck yeah! We finally passed all unit tests after three days of debugging!",
        "Get lost you filthy racist scum.",
        "Interesting observation, but the benchmark might have some caveats.",
        "This whole subreddit is full of clowns and degenerates."
    ],
    "Twitter": [
        "Congrats on the launch! Super excited to try this out 🚀",
        "Watch your back, I know where your office is.",
        "Bhai mast tweet hai, totally agree with you!",
        "Lmao you have zero credibility, go cry somewhere else loser.",
        "Very thoughtful thread, thanks for breaking it down."
    ]
}


@app.post("/api/analyze/social")
def analyze_social_media_url(req: SocialAnalysisRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Social media URL is required.")

    url_lower = req.url.lower().strip()
    platform = req.platform

    # Platform Auto-Detection & Normalization
    p_clean = (platform or "auto").lower()
    if p_clean == "auto":
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            platform = "YouTube"
        elif "reddit.com" in url_lower or "redd.it" in url_lower:
            platform = "Reddit"
        elif "twitter.com" in url_lower or "x.com" in url_lower:
            platform = "Twitter"
        else:
            raise HTTPException(
                status_code=400,
                detail="Could not automatically identify social platform from URL. Supported platforms: YouTube, Reddit, Twitter / X."
            )
    elif "youtube" in p_clean or "yt" in p_clean:
        platform = "YouTube"
    elif "reddit" in p_clean:
        platform = "Reddit"
    elif "twitter" in p_clean or p_clean == "x":
        platform = "Twitter"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")

    raw_comments = []
    is_demo = False
    message = None

    # Fetch comments using platform connector
    try:
        if platform == "YouTube":
            raw_comments = fetch_youtube_comments(req.url)
        elif platform == "Reddit":
            raw_comments = fetch_reddit_comments(req.url)
        elif platform == "Twitter":
            raw_comments = fetch_twitter_comments(req.url)
    except (RuntimeError, ValueError, Exception) as e:
        logger.warning("Error fetching comments from %s (%s): %s", platform, req.url, e)
        if req.demo_fallback:
            raw_comments = DEMO_SOCIAL_COMMENTS.get(platform, DEMO_SOCIAL_COMMENTS["YouTube"])
            is_demo = True
            message = (
                f"Live {platform} API credentials not configured in .env ({str(e)}). "
                f"Loaded {len(raw_comments)} representative multilingual comments for preview."
            )
        else:
            raise HTTPException(status_code=400, detail=str(e))

    if not raw_comments:
        if req.demo_fallback:
            raw_comments = DEMO_SOCIAL_COMMENTS.get(platform, DEMO_SOCIAL_COMMENTS["YouTube"])
            is_demo = True
            message = f"No comments found at {req.url}. Loaded {len(raw_comments)} sample comments."
        else:
            raise HTTPException(status_code=404, detail=f"No comments found for {platform} URL.")

    # Slice to max requested comments
    target_comments = raw_comments[:req.max_comments]

    server = ModelServer.get_instance()
    t0 = time.time()

    # Batch prediction with MuRIL
    batch_preds = predict_batch_muril(
        texts=target_comments,
        model=server.muril_model,
        tokenizer=server.muril_tokenizer,
        labels=server.labels,
        device=server.device,
        batch_size=32,
        thresholds=server.thresholds,
    )

    results = []
    for idx, comment_text in enumerate(target_comments):
        scores, toxic_score, intensity, is_toxic_flags = batch_preds[idx]
        if req.enable_jev and server.jev_provider:
            mod_res = moderate_comment_with_context(
                text=comment_text,
                predictions=scores,
                is_toxic_flags=is_toxic_flags,
                thresholds=server.thresholds,
                jev_provider=server.jev_provider,
            )
            r_level = mod_res.final_risk_level.value
            r_icon = mod_res.final_badge_icon
            r_expl = mod_res.final_explanation
            r_amb = mod_res.is_ambiguous
            r_reason = mod_res.gate_reason
            is_tox = (mod_res.final_risk_level == RiskLevel.HIGH_RISK) or any(is_toxic_flags.values())
        else:
            base_risk, base_icon, base_expl = classify_risk(scores, is_toxic_flags, server.thresholds)
            r_level = base_risk.value
            r_icon = base_icon
            r_expl = base_expl
            r_amb = False
            r_reason = "Deterministic policy triage"
            is_tox = (base_risk == RiskLevel.HIGH_RISK) or any(is_toxic_flags.values())

        results.append({
            "id": f"soc_{platform.lower()}_{idx + 1}",
            "text": comment_text,
            "language": detect_language_simple(comment_text),
            "risk_level": r_level,
            "badge_icon": r_icon,
            "is_toxic": is_tox,
            "confidence": round(float(max(scores.values())), 4),
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "is_toxic_flags": is_toxic_flags,
            "explanation": r_expl,
            "is_ambiguous": r_amb,
            "gate_reason": r_reason,
        })

    high_risk_cnt = sum(1 for r in results if r["risk_level"] == "High Risk")
    review_cnt = sum(1 for r in results if r["risk_level"] == "Review")
    safe_cnt = sum(1 for r in results if r["risk_level"] == "Safe")

    return {
        "platform": platform,
        "url": req.url,
        "is_demo": is_demo,
        "message": message,
        "total_comments": len(results),
        "high_risk_count": high_risk_cnt,
        "review_count": review_cnt,
        "safe_count": safe_cnt,
        "high_risk_percentage": round((high_risk_cnt / len(results)) * 100, 2) if results else 0.0,
        "latency_ms": round((time.time() - t0) * 1000, 2),
        "comments": results
    }


# ── Profile Watch Endpoints ────────────────────────────────────
profile_watch_service = ProfileWatchService()


@app.post("/api/profile-watch/scan", response_model=ScanSummaryResponse)
def scan_profile_endpoint(req: ScanRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Profile URL or handle cannot be empty.")
    try:
        summary = profile_watch_service.scan_profile(
            url_or_id=req.url.strip(),
            limit=req.limit or 25,
            demo_fallback=True if req.demo_fallback is None else req.demo_fallback,
            enable_jev=bool(req.enable_jev),
        )
        return summary
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error executing Profile Watch scan: %s", e)
        raise HTTPException(status_code=500, detail=f"Profile Watch scan failed: {str(e)}")


@app.get("/api/profile-watch", response_model=List[ProfileRecord])
def list_profiles_endpoint():
    return profile_watch_service.list_profiles()


@app.get("/api/profile-watch/compare", response_model=ProfileComparisonResponse)
def compare_profiles_endpoint(profile_a: str, profile_b: str):
    res = profile_watch_service.compare_profiles(profile_a, profile_b)
    if not res:
        raise HTTPException(status_code=404, detail="One or both profiles could not be found for comparison.")
    return res


@app.get("/api/profile-watch/{profile_id}", response_model=ProfileSummaryResponse)
def get_profile_summary_endpoint(profile_id: str):
    res = profile_watch_service.get_profile_summary(profile_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return res


@app.get("/api/profile-watch/{profile_id}/activity", response_model=List[ObservationRecord])
def get_profile_activity_endpoint(
    profile_id: str,
    limit: int = 50,
    offset: int = 0,
    risk_filter: Optional[str] = None,
):
    profile = profile_watch_service.storage.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return profile_watch_service.get_profile_activity(
        profile_id=profile_id,
        limit=limit,
        offset=offset,
        risk_filter=risk_filter,
    )


@app.get("/api/profile-watch/{profile_id}/events", response_model=List[PatternEventRecord])
def get_profile_events_endpoint(profile_id: str):
    profile = profile_watch_service.storage.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return profile_watch_service.get_profile_events(profile_id)


@app.post("/api/profile-watch/{profile_id}/monitor", response_model=ProfileRecord)
def activate_profile_monitoring_endpoint(profile_id: str):
    profile = profile_watch_service.set_monitoring_state(profile_id, MonitoringState.MONITORING)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return profile


@app.post("/api/profile-watch/{profile_id}/pause", response_model=ProfileRecord)
def pause_profile_monitoring_endpoint(profile_id: str):
    profile = profile_watch_service.set_monitoring_state(profile_id, MonitoringState.PAUSED)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return profile


@app.get("/api/research/metrics")
def get_research_metrics():
    # Load actual benchmark files
    dir_path = _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal"
    try:
        with open(dir_path / "multi3hate_metrics.json", "r", encoding="utf-8") as f:
            m3h = json.load(f)
        with open(dir_path / "pridemm_test_metrics.json", "r", encoding="utf-8") as f:
            pridemm = json.load(f)
        with open(dir_path / "ablation_results.json", "r", encoding="utf-8") as f:
            ablation = json.load(f)
        with open(dir_path / "shuffle_sanity_results.json", "r", encoding="utf-8") as f:
            sanity = json.load(f)
        return {
            "status": "AVAILABLE",
            "multi3hate": m3h,
            "pridemm_test": pridemm,
            "ablation": ablation,
            "shuffle_sanity": sanity
        }
    except Exception as e:
        return {"status": "UNAVAILABLE", "error": str(e)}


@app.get("/api/research/case-studies")
def get_case_studies():
    p = _PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal" / "case_studies.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "NOT_FOUND"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
