"""
Rigorous Inference Latency & Throughput Benchmark
Measures:
1. Cached-Feature Inference (MLP vs Top-4 Transformer Head)
2. Full End-to-End Inference (Raw Text + Raw Image -> Tokenizer/Processor -> Backbones -> Fusion Head)
Using identical hardware, identical warm-up, and precise microsecond timers.
"""

import sys
import os
import time
import json
import psutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPVisionModelWithProjection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_MM_CKPT = PROJECT_ROOT / "scratch" / "meme_experiments" / "final_multimodal" / "best_model.pt"
FHN_CKPT = PROJECT_ROOT / "scratch" / "meme_experiments" / "final_hard_negative_pass" / "best_model.pt"

# Sample test meme image and text from Multi3Hate
SAMPLE_IMG_PATH = PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "multi3hate" / "Multi3Hate_repo" / "data" / "memes" / "en" / "skeptical-black-kid" / "0.jpg"
SAMPLE_TEXT_EN = "So You're telling me You eat too much and you want to lose weight?"
SAMPLE_TEXT_HI = "तो आप मुझे बता रहे हैं आप बहुत अधिक खाते हैं और वजन कम करना चाहते हैं?"

class FinalMultimodalClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, proj_dim=256, classifier_dropout=0.2):
        super().__init__()
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

    def forward(self, text_emb, image_emb):
        t_norm = F.normalize(text_emb, p=2, dim=-1)
        v_norm = F.normalize(image_emb, p=2, dim=-1)
        t_p = self.text_proj(t_norm)
        v_p = self.image_proj(v_norm)
        prod = t_p * v_p
        diff = torch.abs(t_p - v_p)
        fused = torch.cat([t_p, v_p, prod, diff], dim=-1)
        fused = self.fusion_norm(fused)
        return self.classifier(fused).squeeze(-1)

class Top4MuRILMultimodalClassifier(nn.Module):
    def __init__(self, muril_layers_8_to_11, proj_dim=256, classifier_dropout=0.25):
        super().__init__()
        self.muril_top4 = muril_layers_8_to_11
        self.proj_dim = proj_dim
        self.text_proj = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.fusion_norm = nn.LayerNorm(proj_dim * 4)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 1)
        )

    def forward_from_h7(self, h7, ext_mask, image_emb):
        h = h7
        for layer in self.muril_top4:
            h = layer(h, attention_mask=ext_mask)[0]
        cls_emb = h[:, 0, :]
        t_norm = F.normalize(cls_emb, p=2, dim=-1)
        v_has_signal = (image_emb.abs().sum(dim=-1, keepdim=True) > 1e-5).float()
        v_norm = F.normalize(image_emb + 1e-8, p=2, dim=-1) * v_has_signal
        t_p = self.text_proj(t_norm)
        v_p = self.image_proj(v_norm) * v_has_signal
        prod = t_p * v_p
        diff = torch.abs(t_p - v_p)
        fused = torch.cat([t_p, v_p, prod, diff], dim=-1)
        fused = self.fusion_norm(fused)
        return self.classifier(fused).squeeze(-1)

def run_benchmarks():
    print("=" * 65)
    print("  RIGOROUS INFERENCE PERFORMANCE BENCHMARK (CPU)")
    print("=" * 65)

    device = torch.device("cpu")
    torch.set_num_threads(4)

    # 1. Instantiate Models
    print("Loading models and tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    muril_full = AutoModel.from_pretrained("google/muril-base-cased")
    muril_full.eval()

    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_vision = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    clip_vision.eval()

    model_base = FinalMultimodalClassifier()
    model_base.load_state_dict(torch.load(FINAL_MM_CKPT, map_location=device, weights_only=False))
    model_base.eval()

    top4_layers = muril_full.encoder.layer[8:]
    model_fhn = Top4MuRILMultimodalClassifier(top4_layers)
    model_fhn.load_state_dict(torch.load(FHN_CKPT, map_location=device, weights_only=False))
    model_fhn.eval()

    # Pre-extract dummy features for cached-feature test
    sample_img = Image.open(SAMPLE_IMG_PATH).convert("RGB")
    clip_inputs = clip_proc(images=sample_img, return_tensors="pt")
    with torch.no_grad():
        img_feature_512 = clip_vision(**clip_inputs).image_embeds # (1, 512)

    text_inputs = tokenizer(SAMPLE_TEXT_EN, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        out_muril = muril_full(**text_inputs)
        text_feature_768 = out_muril.last_hidden_state[:, 0, :] # (1, 768)

        # Precompute h7 for FHN cached test
        embs = muril_full.embeddings(text_inputs["input_ids"], token_type_ids=text_inputs["token_type_ids"])
        ext_mask = muril_full.get_extended_attention_mask(text_inputs["attention_mask"], text_inputs["input_ids"].shape)
        h = embs
        for layer in muril_full.encoder.layer[:8]:
            h = layer(h, attention_mask=ext_mask)[0]
        h7_feature = h # (1, 64, 768)

    # -------------------------------------------------------------
    # PART A: Cached-Feature Inference Benchmark (Batch size = 1)
    # -------------------------------------------------------------
    print("\n--- Part A: Cached-Feature Inference (Latency / Throughput, Batch Size = 1) ---")
    N_ITER = 200
    WARMUP = 20

    # Warmup Baseline
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model_base(text_feature_768, img_feature_512)

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        with torch.no_grad():
            _ = model_base(text_feature_768, img_feature_512)
    dur_base = time.perf_counter() - t0
    lat_base_cached_ms = (dur_base / N_ITER) * 1000.0
    tput_base_cached = N_ITER / dur_base

    # Warmup FHN
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model_fhn.forward_from_h7(h7_feature, ext_mask, img_feature_512)

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        with torch.no_grad():
            _ = model_fhn.forward_from_h7(h7_feature, ext_mask, img_feature_512)
    dur_fhn = time.perf_counter() - t0
    lat_fhn_cached_ms = (dur_fhn / N_ITER) * 1000.0
    tput_fhn_cached = N_ITER / dur_fhn

    print(f"Baseline Head (Cached): Latency = {lat_base_cached_ms:.3f} ms | Throughput = {tput_base_cached:.1f} samples/sec")
    print(f"FHN Head (Top-4 Cached): Latency = {lat_fhn_cached_ms:.3f} ms | Throughput = {tput_fhn_cached:.1f} samples/sec")

    # -------------------------------------------------------------
    # PART B: Full End-to-End Inference (Raw Input -> Tokenizer/Vision -> Prediction)
    # -------------------------------------------------------------
    print("\n--- Part B: Full End-to-End Inference (Raw Text + Raw Image -> Tokenizer -> Backbones -> Head) ---")
    N_E2E = 20
    WARMUP_E2E = 3

    # Pipeline 1: Full Baseline End-to-End
    def pipeline_baseline_e2e(text, img):
        t_in = tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        i_in = clip_proc(images=img, return_tensors="pt")
        with torch.no_grad():
            t_out = muril_full(**t_in)
            t_cls = t_out.last_hidden_state[:, 0, :]
            v_emb = clip_vision(**i_in).image_embeds
            logit = model_base(t_cls, v_emb)
            prob = torch.sigmoid(logit).item()
        return prob >= 0.44

    # Pipeline 2: Full FHN End-to-End
    def pipeline_fhn_e2e(text, img):
        t_in = tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        i_in = clip_proc(images=img, return_tensors="pt")
        with torch.no_grad():
            embs = muril_full.embeddings(t_in["input_ids"], token_type_ids=t_in["token_type_ids"])
            ext_mask = muril_full.get_extended_attention_mask(t_in["attention_mask"], t_in["input_ids"].shape)
            h = embs
            for layer in muril_full.encoder.layer[:8]:
                h = layer(h, attention_mask=ext_mask)[0]
            v_emb = clip_vision(**i_in).image_embeds
            logit = model_fhn.forward_from_h7(h, ext_mask, v_emb)
            prob = torch.sigmoid(logit).item()
        return prob >= 0.40

    # Warmup E2E
    for _ in range(WARMUP_E2E):
        pipeline_baseline_e2e(SAMPLE_TEXT_EN, sample_img)
        pipeline_fhn_e2e(SAMPLE_TEXT_EN, sample_img)

    t0 = time.perf_counter()
    for _ in range(N_E2E):
        pipeline_baseline_e2e(SAMPLE_TEXT_EN, sample_img)
    dur_e2e_base = time.perf_counter() - t0
    lat_e2e_base_ms = (dur_e2e_base / N_E2E) * 1000.0
    tput_e2e_base = N_E2E / dur_e2e_base

    t0 = time.perf_counter()
    for _ in range(N_E2E):
        pipeline_fhn_e2e(SAMPLE_TEXT_EN, sample_img)
    dur_e2e_fhn = time.perf_counter() - t0
    lat_e2e_fhn_ms = (dur_e2e_fhn / N_E2E) * 1000.0
    tput_e2e_fhn = N_E2E / dur_e2e_fhn

    print(f"Baseline End-to-End: Latency = {lat_e2e_base_ms:.2f} ms | Throughput = {tput_e2e_base:.2f} samples/sec")
    print(f"FHN End-to-End:      Latency = {lat_e2e_fhn_ms:.2f} ms | Throughput = {tput_e2e_fhn:.2f} samples/sec")

    bench_results = {
        "cached_feature_inference": {
            "baseline": {
                "latency_ms": round(lat_base_cached_ms, 4),
                "throughput_sps": round(tput_base_cached, 1)
            },
            "fhn_candidate": {
                "latency_ms": round(lat_fhn_cached_ms, 4),
                "throughput_sps": round(tput_fhn_cached, 1)
            }
        },
        "full_end_to_end_inference": {
            "baseline": {
                "latency_ms": round(lat_e2e_base_ms, 2),
                "throughput_sps": round(tput_e2e_base, 2)
            },
            "fhn_candidate": {
                "latency_ms": round(lat_e2e_fhn_ms, 2),
                "throughput_sps": round(tput_e2e_fhn, 2)
            }
        }
    }

    out_file = PROJECT_ROOT / "scratch" / "inference_benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bench_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    run_benchmarks()
