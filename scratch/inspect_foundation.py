"""
Detailed foundation inspection script for Step 1, Step 2, and Step 8.
Audits Multi3Hate, HASOC-Meme 2025, and Jigsaw datasets on disk.
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
M3H_DIR = _ROOT / "scratch" / "meme_data" / "raw" / "multi3hate"
HASOC_DIR = _ROOT / "scratch" / "meme_data" / "raw" / "hasoc_meme_2025"
JIGSAW_PATH = _ROOT / "dataset" / "train.csv"

out = {}

# ==========================================
# 1. MULTI3HATE INSPECTION
# ==========================================
out["multi3hate"] = {}

for fname in ["final_annotations.csv", "raw_annotations.csv", "en_captions.csv", "hi_captions.csv"]:
    fpath = M3H_DIR / fname
    if fpath.exists():
        df = pd.read_csv(fpath)
        col_info = {}
        for c in df.columns:
            val_counts = df[c].value_counts(dropna=False).to_dict()
            # If large, summarize
            if len(val_counts) > 20:
                summary_vc = {str(k): int(v) for k, v in list(val_counts.items())[:10]}
                summary_vc["_truncated_num_unique"] = len(val_counts)
            else:
                summary_vc = {str(k): int(v) for k, v in val_counts.items()}

            col_info[c] = {
                "dtype": str(df[c].dtype),
                "num_null": int(df[c].isna().sum()),
                "num_unique": int(df[c].nunique(dropna=False)),
                "value_distribution": summary_vc,
            }

        out["multi3hate"][fname] = {
            "num_rows": len(df),
            "columns": list(df.columns),
            "column_info": col_info,
        }

# Check Multi3Hate images
m3h_images_base = M3H_DIR / "Multi3Hate_repo" / "data" / "memes"
m3h_hi_imgs = list((m3h_images_base / "hi").rglob("*.jpg")) if (m3h_images_base / "hi").exists() else []
m3h_en_imgs = list((m3h_images_base / "en").rglob("*.jpg")) if (m3h_images_base / "en").exists() else []
out["multi3hate"]["image_files"] = {
    "hi_images_count": len(m3h_hi_imgs),
    "en_images_count": len(m3h_en_imgs),
}

# Verify opening sample images
can_open_hi = 0
for p in m3h_hi_imgs[:10]:
    try:
        with Image.open(p) as im:
            im.verify()
        can_open_hi += 1
    except Exception:
        pass
out["multi3hate"]["image_files"]["sample_hi_verified_openable"] = can_open_hi

# Check Multi3Hate paired instances count: image + text + label
df_annot = pd.read_csv(M3H_DIR / "final_annotations.csv")
df_hi = pd.read_csv(M3H_DIR / "hi_captions.csv")
df_en = pd.read_csv(M3H_DIR / "en_captions.csv")

hi_map = {r["Meme ID"]: r for _, r in df_hi.iterrows()}
en_map = {r["Meme ID"]: r for _, r in df_en.iterrows()}

m3h_paired_hi = 0
m3h_paired_en = 0
m3h_paired_total = 0
missing_img = 0
missing_txt = 0
missing_lbl = 0

for _, row in df_annot.iterrows():
    mid = row["Meme ID"]
    lbl_in = row.get("IN")
    lbl_us = row.get("US")

    # HI
    h_row = hi_map.get(mid)
    txt_h = h_row.get("Translation") if h_row is not None else None
    matches_h = list((m3h_images_base / "hi").rglob(f"{mid}.jpg"))
    img_h = matches_h[0] if matches_h else None

    if img_h and img_h.exists() and txt_h and pd.notna(lbl_in):
        m3h_paired_hi += 1
    else:
        if not img_h: missing_img += 1
        if not txt_h: missing_txt += 1
        if pd.isna(lbl_in): missing_lbl += 1

    # EN
    e_row = en_map.get(mid)
    txt_e = e_row.get("Original (English)") if e_row is not None else None
    matches_e = list((m3h_images_base / "en").rglob(f"{mid}.jpg"))
    img_e = matches_e[0] if matches_e else None

    if img_e and img_e.exists() and txt_e and pd.notna(lbl_in):
        m3h_paired_en += 1
    else:
        if not img_e: missing_img += 1
        if not txt_e: missing_txt += 1
        if pd.isna(lbl_in): missing_lbl += 1

out["multi3hate"]["paired_stats"] = {
    "paired_hi": m3h_paired_hi,
    "paired_en": m3h_paired_en,
    "paired_total": m3h_paired_hi + m3h_paired_en,
    "missing_img": missing_img,
    "missing_txt": missing_txt,
    "missing_lbl": missing_lbl,
}

# ==========================================
# 2. HASOC-MEME 2025 INSPECTION
# ==========================================
out["hasoc_meme_2025"] = {}

for fname in ["Hindi Train HASOC 2025 - Updated.csv", "Hindi Test HASOC 2025 - Updated.csv"]:
    fpath = HASOC_DIR / fname
    if fpath.exists():
        df = pd.read_csv(fpath)
        col_info = {}
        for c in df.columns:
            val_counts = df[c].value_counts(dropna=False).to_dict()
            if len(val_counts) > 20:
                summary_vc = {str(k): int(v) for k, v in list(val_counts.items())[:10]}
                summary_vc["_truncated_num_unique"] = len(val_counts)
            else:
                summary_vc = {str(k): int(v) for k, v in val_counts.items()}

            col_info[c] = {
                "dtype": str(df[c].dtype),
                "num_null": int(df[c].isna().sum()),
                "num_unique": int(df[c].nunique(dropna=False)),
                "value_distribution": summary_vc,
            }

        out["hasoc_meme_2025"][fname] = {
            "num_rows": len(df),
            "columns": list(df.columns),
            "column_info": col_info,
        }

# Check for physical image files in HASOC
hasoc_jpgs = list(HASOC_DIR.rglob("*.jpg"))
hasoc_pngs = list(HASOC_DIR.rglob("*.png"))
hasoc_all_imgs = hasoc_jpgs + hasoc_pngs
out["hasoc_meme_2025"]["image_files"] = {
    "total_image_files_found": len(hasoc_all_imgs),
    "jpg_count": len(hasoc_jpgs),
    "png_count": len(hasoc_pngs),
}

# ==========================================
# 3. JIGSAW INSPECTION
# ==========================================
if JIGSAW_PATH.exists():
    df_jigsaw = pd.read_csv(JIGSAW_PATH, nrows=5000)
    out["jigsaw"] = {
        "exists": True,
        "columns": list(df_jigsaw.columns),
        "target_cols": [c for c in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] if c in df_jigsaw.columns],
    }

with open(_ROOT / "scratch" / "foundation_audit.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("Foundation audit saved to scratch/foundation_audit.json")
