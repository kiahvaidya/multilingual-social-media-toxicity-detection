import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import MuRILForMultiLabelToxicity

OUT_DIR = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_d_pridemm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_B_CHECKPOINT = _PROJECT_ROOT / "scratch" / "meme_experiments" / "model_b" / "muril_model_b.pt"
PRIDEMM_MANIFEST = _PROJECT_ROOT / "scratch" / "meme_data" / "processed" / "pridemm_manifest.json"
MULTI3HATE_DIR = _PROJECT_ROOT / "scratch" / "meme_data" / "raw" / "multi3hate"
MULTI3HATE_IMAGES_BASE = MULTI3HATE_DIR / "Multi3Hate_repo" / "data" / "memes"

def extract_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threads = os.cpu_count()
    torch.set_num_threads(threads)
    print(f"Extraction running on: {device} with {threads} threads")

    # 1. Load Models
    print("Loading Model B MuRIL encoder...")
    model_b = MuRILForMultiLabelToxicity(model_name="google/muril-base-cased", num_labels=6)
    model_b.load_state_dict(torch.load(MODEL_B_CHECKPOINT, map_location=device, weights_only=True))
    model_b.to(device).eval()
    muril_tok = AutoTokenizer.from_pretrained("google/muril-base-cased")

    print("Loading CLIP ViT-B/32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Helper function to extract embeddings for a list of items
    def process_items(items, name, save_path):
        if os.path.exists(save_path):
            print(f"Embeddings already exist for {name} at {save_path}, loading...")
            return torch.load(save_path, weights_only=False)

        print(f"Extracting embeddings for {name} ({len(items)} items)...")
        text_embs = []
        img_embs = []
        labels = []
        sample_ids = []
        batch_size = 32

        t0 = time.time()
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            texts = [it["text"] for it in batch]
            imgs = []
            valid_batch_indices = []

            for b_idx, it in enumerate(batch):
                try:
                    with Image.open(it["image_path"]) as img:
                        imgs.append(img.convert("RGB"))
                        valid_batch_indices.append(b_idx)
                except Exception as e:
                    print(f"Error opening image {it['image_path']}: {e}")
                    imgs.append(Image.new("RGB", (224, 224), color="black"))
                    valid_batch_indices.append(b_idx)

            with torch.no_grad():
                # MuRIL text embeddings (CLS token)
                enc = muril_tok(texts, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
                text_outputs = model_b.encoder(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                t_emb = text_outputs.last_hidden_state[:, 0, :] # (B, 768)
                t_emb_norm = F.normalize(t_emb, p=2, dim=-1).cpu()

                # CLIP image embeddings
                img_inputs = clip_proc(images=imgs, return_tensors="pt")
                v_emb = clip_model.get_image_features(pixel_values=img_inputs["pixel_values"].to(device)) # (B, 512)
                v_emb_norm = F.normalize(v_emb, p=2, dim=-1).cpu()

            text_embs.append(t_emb_norm)
            img_embs.append(v_emb_norm)
            for it in batch:
                labels.append(it["label"])
                sample_ids.append(it["sample_id"])

            if (i // batch_size) % 10 == 0 or (i + batch_size) >= len(items):
                elapsed = time.time() - t0
                done = min(i + batch_size, len(items))
                print(f"  [{name}] {done}/{len(items)} processed ({done/elapsed:.1f} items/s)")

        data = {
            "text_embeddings": torch.cat(text_embs, dim=0),
            "image_embeddings": torch.cat(img_embs, dim=0),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "sample_ids": sample_ids,
            "items": items
        }
        torch.save(data, save_path)
        print(f"Saved {name} embeddings to {save_path}.")
        return data

    # 2. Process PrideMM splits
    print("Loading PrideMM manifest...")
    with open(PRIDEMM_MANIFEST, "r", encoding="utf-8") as f:
        pridemm_records = json.load(f)

    train_items = []
    val_items = []
    test_items = []

    for r in pridemm_records:
        it = {
            "sample_id": r["sample_id"],
            "text": r["raw_text"],
            "image_path": r["image_path"],
            "label": 1.0 if r["labels"]["hate"] else 0.0,
            "target": r.get("target"),
            "stance": r.get("stance"),
            "humor": r.get("humor")
        }
        split = r["source_split"]
        if split == "train":
            train_items.append(it)
        elif split == "val":
            val_items.append(it)
        elif split == "test":
            test_items.append(it)

    print(f"PrideMM items: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    process_items(train_items, "pridemm_train", OUT_DIR / "pridemm_train_embeddings.pt")
    process_items(val_items, "pridemm_val", OUT_DIR / "pridemm_val_embeddings.pt")
    process_items(test_items, "pridemm_test", OUT_DIR / "pridemm_test_embeddings.pt")

    # 3. Process Multi3Hate held-out evaluation
    print("Loading Multi3Hate instances...")
    annot_csv = MULTI3HATE_DIR / "final_annotations.csv"
    hi_csv = MULTI3HATE_DIR / "hi_captions.csv"
    en_csv = MULTI3HATE_DIR / "en_captions.csv"

    df_annot = pd.read_csv(annot_csv)
    df_hi = pd.read_csv(hi_csv)
    df_en = pd.read_csv(en_csv)

    hi_map = {r["Meme ID"]: r for _, r in df_hi.iterrows()}
    en_map = {r["Meme ID"]: r for _, r in df_en.iterrows()}

    m3h_items = []
    for _, row in df_annot.iterrows():
        mid = int(row["Meme ID"])
        gt_in = int(row["IN"])
        gt_us = int(row["US"])
        gt_de = int(row["DE"])
        gt_mx = int(row["MX"])
        gt_cn = int(row["CN"])

        # English
        e_row = en_map[mid]
        e_txt = str(e_row["Original (English)"]).replace("<sep>", " ").strip()
        e_matches = list((MULTI3HATE_IMAGES_BASE / "en").rglob(f"{mid}.jpg")) + list((MULTI3HATE_IMAGES_BASE / "en").rglob(f"{mid}.png"))
        if not e_matches:
            raise FileNotFoundError(f"Missing English image for Meme ID {mid}")

        m3h_items.append({
            "sample_id": f"multi3hate_en_{mid}",
            "meme_id": mid,
            "language": "english",
            "text": e_txt,
            "image_path": str(e_matches[0]),
            "label": float(gt_in),
            "IN": gt_in,
            "US": gt_us,
            "DE": gt_de,
            "MX": gt_mx,
            "CN": gt_cn,
        })

        # Hindi
        h_row = hi_map[mid]
        h_txt = str(h_row["Translation"]).replace("<sep>", " ").strip()
        h_matches = list((MULTI3HATE_IMAGES_BASE / "hi").rglob(f"{mid}.jpg")) + list((MULTI3HATE_IMAGES_BASE / "hi").rglob(f"{mid}.png"))
        if not h_matches:
            raise FileNotFoundError(f"Missing Hindi image for Meme ID {mid}")

        m3h_items.append({
            "sample_id": f"multi3hate_hi_{mid}",
            "meme_id": mid,
            "language": "hindi",
            "text": h_txt,
            "image_path": str(h_matches[0]),
            "label": float(gt_in),
            "IN": gt_in,
            "US": gt_us,
            "DE": gt_de,
            "MX": gt_mx,
            "CN": gt_cn,
        })

    print(f"Multi3Hate items: {len(m3h_items)}")
    process_items(m3h_items, "multi3hate", OUT_DIR / "multi3hate_embeddings.pt")
    print("All feature extractions completed successfully!")

if __name__ == "__main__":
    extract_features()
