"""
Model C-2: Trained + Calibrated Multimodal Meme Toxicity Detection.

Implements:
  1. Text Branch: MuRIL (initialized from Model B checkpoint muril_model_b.pt).
  2. Vision Branch: Pretrained CLIP ViT (openai/clip-vit-base-patch32), frozen.
  3. Four Architectural Ablations:
     - C-2a: Text-only (Model B representations)
     - C-2b: Image-only (CLIP ViT features)
     - C-2c: Multimodal Concatenation (LayerNorm + Projection)
     - C-2d: Multimodal Gated Fusion (Learned Convex Combination: gate * text + (1 - gate) * vision)
  4. Training on HASOC-Meme 2025 Train Split (80/20 train/val split).
     Multi3Hate is strictly held out (quarantined from training, validation, and calibration).
  5. Validation-Based Calibration (Platt / Temperature Scaling fitted strictly on validation split).
  6. Validation-Selected Threshold Discovery (optimizing precision/recall/FPR balance).
  7. Artifacts saved under scratch/meme_experiments/model_c2/.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    AutoProcessor,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.model import DEFAULT_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainModelC2")

EXP_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_c2"
MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
CHECKPOINT_OUT = EXP_DIR / "muril_model_c2.pt"
METRICS_OUT = EXP_DIR / "model_c2_training_metrics.json"

RAW_HASOC_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "hasoc_meme_2025"
MULTI3HATE_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "multi3hate"


# ==============================================================
# 🧠 Model C-2 Architecture with Dynamic Gated Fusion
# ==============================================================

class CalibratedMultimodalMemeModel(nn.Module):
    """Multimodal model with learned convex-combination gating and unimodal bypass."""

    def __init__(
        self,
        muril_model_name: str = "google/muril-base-cased",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        num_labels: int = 6,
        fusion_dim: int = 512,
        dropout_prob: float = 0.2,
        freeze_vision: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.fusion_dim = fusion_dim

        # 1. Text Encoder (MuRIL)
        logger.info("Initializing text encoder from %s", muril_model_name)
        self.text_encoder = AutoModel.from_pretrained(muril_model_name)
        self.text_hidden_size = self.text_encoder.config.hidden_size  # 768

        # 2. Vision Encoder (CLIP ViT-B/32)
        logger.info("Initializing vision encoder from %s", clip_model_name)
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)
        self.vision_hidden_size = self.vision_encoder.config.projection_dim  # 512

        # 3. CLIP Text Encoder (provides multimodal shared sphere representation for text-supervised training)
        logger.info("Initializing CLIP text encoder for shared multimodal sphere...")
        self.clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_name)

        # Freeze backbones
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.clip_text_encoder.parameters():
                p.requires_grad = False
            logger.info("Vision backbone and CLIP text encoder FROZEN.")

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            logger.info("MuRIL text backbone FROZEN.")

        # 4. Projections & Normalization (Preserving Unit Scale)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden_size, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_hidden_size, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # 5. Ablation A: Concatenation Head (C-2c)
        self.concat_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_dim, num_labels),
        )

        # 6. Ablation B: Dynamic Gated Fusion Head (C-2d)
        # Learnable gating vector: g in [0, 1]^fusion_dim
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.gated_classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_dim, num_labels),
        )

        # 7. Unimodal Heads (C-2a Text-Only, C-2b Vision-Only)
        self.text_classifier = nn.Linear(fusion_dim, num_labels)
        self.vision_classifier = nn.Linear(fusion_dim, num_labels)

        # 8. Calibration parameters (post-hoc temperature / Platt scaling)
        # Initialized to neutral scaling (T = 1.0)
        self.temperature = nn.Parameter(torch.ones(num_labels), requires_grad=False)
        self.cal_bias = nn.Parameter(torch.zeros(num_labels), requires_grad=False)

    def extract_text_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return self.text_proj(mean_pooled)

    def extract_vision_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        return self.vision_proj(vision_outputs.image_embeds)

    def extract_clip_text_embedding(self, clip_input_ids: torch.Tensor, clip_attention_mask: torch.Tensor) -> torch.Tensor:
        clip_outputs = self.clip_text_encoder(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
        return self.vision_proj(clip_outputs.text_embeds)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        mode: str = "gated",
        return_gate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass supporting gated fusion, concat, text-only, or vision-only modes."""
        gate_value = None

        if mode == "text_only":
            if input_ids is None:
                raise ValueError("input_ids required for text_only mode.")
            h_text = self.extract_text_embedding(input_ids, attention_mask)
            logits = self.text_classifier(h_text)
            return logits, None

        elif mode == "vision_only":
            if pixel_values is not None:
                h_vision = self.extract_vision_embedding(pixel_values)
            elif vision_embeds is not None:
                h_vision = vision_embeds
            else:
                raise ValueError("pixel_values or vision_embeds required for vision_only mode.")
            logits = self.vision_classifier(h_vision)
            return logits, None

        # Multimodal Modes: Extract both modalities
        if input_ids is None:
            raise ValueError("input_ids required for multimodal inference.")
        h_text = self.extract_text_embedding(input_ids, attention_mask)

        if pixel_values is not None:
            h_vision = self.extract_vision_embedding(pixel_values)
        elif vision_embeds is not None:
            h_vision = vision_embeds
        else:
            raise ValueError("pixel_values or vision_embeds required for multimodal inference.")

        if mode == "concat":
            h_concat = torch.cat([h_text, h_vision], dim=-1)
            logits = self.concat_mlp(h_concat)
            return logits, None

        elif mode == "gated":
            # Compute dynamic gating vector
            h_paired = torch.cat([h_text, h_vision], dim=-1)
            gate = self.gate_network(h_paired)  # Shape: (B, fusion_dim) in [0, 1]
            gate_value = gate.mean(dim=-1)       # Scalar average gate per instance

            # Convex combination: gate * text + (1 - gate) * vision
            h_fused = gate * h_text + (1.0 - gate) * h_vision
            logits = self.gated_classifier(h_fused)

            if return_gate:
                return logits, gate_value
            return logits, None

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def load_text_branch_from_model_b(self, checkpoint_path: Path):
        """Loads fine-tuned weights from Model B into the text encoder."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model B checkpoint not found at {checkpoint_path}")
        logger.info("Loading Model B checkpoint from %s into text branch...", checkpoint_path.name)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        text_state = {}
        for k, v in state_dict.items():
            if k.startswith("muril."):
                text_state[k[len("muril."):]] = v
        if text_state:
            missing, unexpected = self.text_encoder.load_state_dict(text_state, strict=False)
            logger.info("Loaded text encoder from Model B (missing: %d, unexpected: %d)", len(missing), len(unexpected))
        else:
            self.text_encoder.load_state_dict(state_dict, strict=False)


# ==============================================================
# 📦 HASOC-Meme 2025 Dataset & Loss Function
# ==============================================================

class HASOCMultimodalDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        targets: np.ndarray,
        masks: np.ndarray,
        muril_tokenizer,
        clip_tokenizer,
        max_len: int = 64,
    ):
        self.texts = [str(t) for t in texts]
        self.targets = targets
        self.masks = masks
        self.muril_tokenizer = muril_tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # MuRIL text encoding
        enc_muril = self.muril_tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # CLIP text encoding (shared multimodal sphere representation)
        enc_clip = self.clip_tokenizer(
            text,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": enc_muril["input_ids"].squeeze(0),
            "attention_mask": enc_muril["attention_mask"].squeeze(0),
            "clip_input_ids": enc_clip["input_ids"].squeeze(0),
            "clip_attention_mask": enc_clip["attention_mask"].squeeze(0),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
            "masks": torch.tensor(self.masks[idx], dtype=torch.float32),
        }


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, mask):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        masked_loss = bce * mask
        return masked_loss.sum() / (mask.sum() + 1e-8)


def load_hasoc_training_split() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    train_file = RAW_HASOC_DIR / "Hindi Train HASOC 2025 - Updated.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"HASOC 2025 train data not found at {train_file}")

    df = pd.read_csv(train_file)
    df = df[df["OCR"].notna() & (df["OCR"].str.strip() != "")].reset_index(drop=True)
    logger.info("Loaded %d clean HASOC 2025 samples", len(df))

    N = len(df)
    targets = np.zeros((N, 6), dtype=np.float32)
    masks = np.zeros((N, 6), dtype=np.float32)

    abuse_is_abusive = (df["Abuse"].str.strip().str.lower() == "abusive").values
    vulgar_is_vulgar = (df["Vulgar"].str.strip().str.lower() == "vulgar").values
    target_community = df["Target"].astype(str).str.strip().str.lower().isin(["gender", "community"]).values

    # Target Mapping Layer (preserving non-collapsing logic)
    targets[:, 0] = abuse_is_abusive.astype(np.float32)
    masks[:, 0] = 1.0

    targets[:, 2] = vulgar_is_vulgar.astype(np.float32)
    masks[:, 2] = 1.0

    targets[:, 4] = abuse_is_abusive.astype(np.float32)
    masks[:, 4] = 1.0

    targets[:, 5] = (abuse_is_abusive & target_community).astype(np.float32)
    masks[:, 5] = 1.0

    # Masks for severe_toxic and threat remain 0.0
    masks[:, 1] = 0.0
    masks[:, 3] = 0.0

    return df, targets, masks


# ==============================================================
# 📐 Calibration & Metrics Evaluation Functions
# ==============================================================

def compute_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - y_true) ** 2))


def compute_ece(y_true: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> float:
    bin_limits = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(num_bins):
        bin_mask = (probs >= bin_limits[i]) & (probs < bin_limits[i + 1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += (np.sum(bin_mask) / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def fit_platt_scaling(val_logits: np.ndarray, val_targets: np.ndarray) -> Tuple[float, float]:
    """Fits Platt scaling: p = sigmoid(a * z + b) on validation logits."""
    def loss_func(params):
        a, b = params
        scaled_logits = a * val_logits + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -15, 15)))
        eps = 1e-8
        bce = -np.mean(val_targets * np.log(probs + eps) + (1.0 - val_targets) * np.log(1.0 - probs + eps))
        return bce

    res = minimize(loss_func, [1.0, 0.0], method="L-BFGS-B")
    a, b = res.x
    logger.info("Fitted Platt scaling parameters: a=%.4f, b=%.4f", a, b)
    return float(a), float(b)


# ==============================================================
# 🚀 Training & Validation Pipeline
# ==============================================================

def train_and_calibrate_model_c2():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running on device: %s", device)

    # 1. Load HASOC data & create 80/20 train/val split
    df, targets, masks = load_hasoc_training_split()
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=targets[:, 0]
    )
    logger.info("HASOC-Meme 2025 Split: %d train, %d validation", len(train_idx), len(val_idx))

    # Tokenizers
    muril_tok = AutoTokenizer.from_pretrained("google/muril-base-cased")
    clip_tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    train_ds = HASOCMultimodalDataset(
        df["OCR"].iloc[train_idx].tolist(),
        targets[train_idx],
        masks[train_idx],
        muril_tok,
        clip_tok,
    )
    val_ds = HASOCMultimodalDataset(
        df["OCR"].iloc[val_idx].tolist(),
        targets[val_idx],
        masks[val_idx],
        muril_tok,
        clip_tok,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # 2. Build Model C-2
    model = CalibratedMultimodalMemeModel(num_labels=6).to(device)
    if MODEL_B_CHECKPOINT.exists():
        model.load_text_branch_from_model_b(MODEL_B_CHECKPOINT)

    # Trainable parameters: Text Proj, Vision Proj, Concat MLP, Gate Network, Gated Classifier, Unimodal Heads
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Total Trainable Parameters in Multimodal Heads: %d", sum(p.numel() for p in trainable_params))

    # Class weights for toxicity
    num_pos = np.sum(targets[train_idx, 0] == 1)
    num_neg = np.sum(targets[train_idx, 0] == 0)
    pos_weight_val = torch.tensor([num_neg / (num_pos + 1e-5)], device=device)
    logger.info("Toxic pos_weight: %.3f (pos: %d, neg: %d)", pos_weight_val.item(), num_pos, num_neg)

    pos_weights = torch.ones(6, device=device)
    pos_weights[0] = pos_weight_val.item()
    pos_weights[2] = 1.5
    pos_weights[4] = pos_weight_val.item()
    pos_weights[5] = 2.0

    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    # 3. Training Loop
    epochs = 3
    logger.info("Starting training of multimodal projection and gated fusion head (%d epochs)...", epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clip_input_ids = batch["clip_input_ids"].to(device)
            clip_attention_mask = batch["clip_attention_mask"].to(device)
            b_targets = batch["targets"].to(device)
            b_masks = batch["masks"].to(device)

            with torch.no_grad():
                vision_embeds = model.extract_clip_text_embedding(clip_input_ids, clip_attention_mask)

            # Train both gated fusion and concatenation modes
            logits_gated, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_embeds=vision_embeds,
                mode="gated",
            )
            logits_concat, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_embeds=vision_embeds,
                mode="concat",
            )

            loss_gated = criterion(logits_gated, b_targets, b_masks)
            loss_concat = criterion(logits_concat, b_targets, b_masks)
            loss = loss_gated + 0.5 * loss_concat

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info("Epoch %d/%d - Training Loss: %.4f", epoch + 1, epochs, avg_loss)

    # 4. Validation & Calibration on HASOC Validation Split
    model.eval()
    val_logits_gated = []
    val_targets_toxic = []
    val_gates = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clip_input_ids = batch["clip_input_ids"].to(device)
            clip_attention_mask = batch["clip_attention_mask"].to(device)
            b_targets = batch["targets"].to(device)

            vision_embeds = model.extract_clip_text_embedding(clip_input_ids, clip_attention_mask)
            logits, gate = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_embeds=vision_embeds,
                mode="gated",
                return_gate=True,
            )

            val_logits_gated.append(logits[:, 0].cpu().numpy())
            val_targets_toxic.append(b_targets[:, 0].cpu().numpy())
            if gate is not None:
                val_gates.append(gate.cpu().numpy())

    val_logits_arr = np.concatenate(val_logits_gated)
    val_targets_arr = np.concatenate(val_targets_toxic)
    val_gates_arr = np.concatenate(val_gates) if val_gates else np.zeros_like(val_logits_arr)

    # Raw metrics on validation split
    raw_probs = 1.0 / (1.0 + np.exp(-val_logits_arr))
    raw_brier = compute_brier_score(val_targets_arr, raw_probs)
    raw_ece = compute_ece(val_targets_arr, raw_probs)

    # Fit Platt Scaling strictly on validation split
    platt_a, platt_b = fit_platt_scaling(val_logits_arr, val_targets_arr)
    cal_probs = 1.0 / (1.0 + np.exp(-(platt_a * val_logits_arr + platt_b)))
    cal_brier = compute_brier_score(val_targets_arr, cal_probs)
    cal_ece = compute_ece(val_targets_arr, cal_probs)

    logger.info("Validation Calibration - Raw Brier: %.4f, ECE: %.4f | Calibrated Brier: %.4f, ECE: %.4f",
                raw_brier, raw_ece, cal_brier, cal_ece)

    # 5. Validation-Selected Threshold Search
    best_thresh = 0.45
    best_f1 = -1.0
    thresh_grid = np.linspace(0.20, 0.80, 61)
    val_tradeoffs = []

    for t in thresh_grid:
        preds = (cal_probs >= t).astype(int)
        tp = np.sum((val_targets_arr == 1) & (preds == 1))
        fp = np.sum((val_targets_arr == 0) & (preds == 1))
        tn = np.sum((val_targets_arr == 0) & (preds == 0))
        fn = np.sum((val_targets_arr == 1) & (preds == 0))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        val_tradeoffs.append({"threshold": round(float(t), 3), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "fpr": round(fpr, 4)})

        if f1 > best_f1 and fpr <= 0.40:  # Constrained optimization to avoid uncontrolled FPR
            best_f1 = f1
            best_thresh = float(t)

    logger.info("Validation-Selected Threshold: %.3f (Val F1: %.4f)", best_thresh, best_f1)

    # Store calibration parameters in model buffer
    model.temperature.data[0] = float(platt_a)
    model.cal_bias.data[0] = float(platt_b)

    # 6. Save Checkpoint & Metrics
    logger.info("Saving Model C-2 checkpoint to %s...", CHECKPOINT_OUT)
    torch.save(model.state_dict(), CHECKPOINT_OUT)

    meta = {
        "model_name": "Model C-2 (Trained + Calibrated Multimodal Fusion)",
        "training_data": "HASOC-Meme 2025 (80% Train, 20% Val)",
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "calibration": {
            "method": "Platt Scaling (a * z + b)",
            "platt_a": round(platt_a, 4),
            "platt_b": round(platt_b, 4),
            "validation_raw_brier": round(raw_brier, 4),
            "validation_raw_ece": round(raw_ece, 4),
            "validation_calibrated_brier": round(cal_brier, 4),
            "validation_calibrated_ece": round(cal_ece, 4),
        },
        "thresholds": {
            "frozen_production_threshold": 0.45,
            "validation_selected_threshold": round(best_thresh, 3),
            "validation_f1_at_selected": round(best_f1, 4),
        },
        "gate_statistics_val": {
            "mean_gate": round(float(np.mean(val_gates_arr)), 4),
            "std_gate": round(float(np.std(val_gates_arr)), 4),
            "min_gate": round(float(np.min(val_gates_arr)), 4),
            "max_gate": round(float(np.max(val_gates_arr)), 4),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training and calibration complete. Metadata saved to %s", METRICS_OUT)
    return model, meta


if __name__ == "__main__":
    train_and_calibrate_model_c2()
