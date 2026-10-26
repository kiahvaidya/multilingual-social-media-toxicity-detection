import os
import glob
import hashlib
import json
import re
import pandas as pd
from difflib import SequenceMatcher

def compute_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_audit():
    print("=== MULTI3HATE AUDIT ===")
    multi3hate_en_imgs = glob.glob('scratch/meme_data/raw/multi3hate/Multi3Hate_repo/data/memes/en/**/*.jpg', recursive=True) + \
                         glob.glob('scratch/meme_data/raw/multi3hate/Multi3Hate_repo/data/memes/en/**/*.png', recursive=True)
    multi3hate_hi_imgs = glob.glob('scratch/meme_data/raw/multi3hate/Multi3Hate_repo/data/memes/hi/**/*.jpg', recursive=True) + \
                         glob.glob('scratch/meme_data/raw/multi3hate/Multi3Hate_repo/data/memes/hi/**/*.png', recursive=True)
    
    print(f"Multi3Hate EN images found: {len(multi3hate_en_imgs)}")
    print(f"Multi3Hate HI images found: {len(multi3hate_hi_imgs)}")
    
    multi3hate_hashes = {}
    for p in multi3hate_en_imgs + multi3hate_hi_imgs:
        h = compute_hash(p)
        multi3hate_hashes[h] = p
        
    print(f"Total unique image SHA256 hashes in Multi3Hate: {len(multi3hate_hashes)}")
    
    # Load Multi3Hate texts
    ann_path = 'scratch/meme_data/raw/multi3hate/final_annotations.csv'
    df_ann = pd.read_csv(ann_path)
    print(f"Multi3Hate annotations rows: {len(df_ann)}")
    
    en_captions_path = 'scratch/meme_data/raw/multi3hate/en_captions.csv'
    hi_captions_path = 'scratch/meme_data/raw/multi3hate/hi_captions.csv'
    df_en = pd.read_csv(en_captions_path)
    df_hi = pd.read_csv(hi_captions_path)
    
    m3h_texts = set()
    m3h_norm_texts = {}
    
    for _, r in df_en.iterrows():
        t = str(r['text']) if 'text' in r else str(r.iloc[1])
        m3h_texts.add(t)
        norm = normalize_text(t)
        if norm:
            m3h_norm_texts[norm] = ('en', t)
            
    for _, r in df_hi.iterrows():
        t = str(r['text']) if 'text' in r else str(r.iloc[1])
        m3h_texts.add(t)
        norm = normalize_text(t)
        if norm:
            m3h_norm_texts[norm] = ('hi', t)
            
    print(f"Total unique raw text captions in Multi3Hate: {len(m3h_texts)}")
    print(f"Total unique normalized captions in Multi3Hate: {len(m3h_norm_texts)}")
    
    # Now check HASOC 2025 Hindi Train & Test
    hasoc_train_path = 'scratch/meme_data/raw/hasoc_meme_2025/Hindi Train HASOC 2025 - Updated.csv'
    hasoc_test_path = 'scratch/meme_data/raw/hasoc_meme_2025/Hindi Test HASOC 2025 - Updated.csv'
    
    df_hasoc_train = pd.read_csv(hasoc_train_path)
    df_hasoc_test = pd.read_csv(hasoc_test_path)
    
    print(f"HASOC 2025 Train rows: {len(df_hasoc_train)}")
    print(f"HASOC 2025 Test rows: {len(df_hasoc_test)}")
    
    exact_text_overlap = []
    near_text_overlap = []
    
    all_hasoc = pd.concat([df_hasoc_train.assign(split='train'), df_hasoc_test.assign(split='test')], ignore_index=True)
    
    for idx, row in all_hasoc.iterrows():
        ocr = str(row['OCR']) if pd.notna(row['OCR']) else ""
        norm_ocr = normalize_text(ocr)
        if not norm_ocr:
            continue
            
        if norm_ocr in m3h_norm_texts:
            lang, orig_m3h = m3h_norm_texts[norm_ocr]
            exact_text_overlap.append({
                'hasoc_id': str(row['Ids']),
                'hasoc_split': row['split'],
                'hasoc_ocr': ocr,
                'm3h_lang': lang,
                'm3h_text': orig_m3h
            })
        else:
            # Check for near duplicate (>0.85 similarity)
            # Sample check or prefix match
            for m_norm, (lang, orig_m3h) in m3h_norm_texts.items():
                if abs(len(norm_ocr) - len(m_norm)) > 20:
                    continue
                ratio = SequenceMatcher(None, norm_ocr, m_norm).ratio()
                if ratio >= 0.85:
                    near_text_overlap.append({
                        'hasoc_id': str(row['Ids']),
                        'hasoc_split': row['split'],
                        'hasoc_ocr': ocr,
                        'm3h_lang': lang,
                        'm3h_text': orig_m3h,
                        'similarity': ratio
                    })
                    break

    print(f"\nExact text overlap with Multi3Hate: {len(exact_text_overlap)}")
    print(f"Near-duplicate text overlap with Multi3Hate: {len(near_text_overlap)}")
    
    # Check template overlap:
    # Multi3Hate templates are directory names in Multi3Hate_repo/data/memes/en/
    m3h_templates = set([os.path.basename(os.path.dirname(p)).lower().replace('-', ' ') for p in multi3hate_en_imgs])
    print(f"Multi3Hate template classes ({len(m3h_templates)}): {sorted(list(m3h_templates))[:10]}...")
    
    # Check if HASOC Ids or OCR explicitly reference these templates
    hasoc_template_matches = []
    for idx, row in all_hasoc.iterrows():
        id_str = str(row['Ids']).lower()
        ocr_str = str(row['OCR']).lower()
        matched = []
        for tmpl in m3h_templates:
            if tmpl in id_str or tmpl in ocr_str:
                matched.append(tmpl)
        if matched:
            hasoc_template_matches.append({
                'hasoc_id': str(row['Ids']),
                'matched_templates': matched
            })
            
    print(f"HASOC items with detectable Multi3Hate template name mentions: {len(hasoc_template_matches)}")
    
    results = {
        'multi3hate_image_count': len(multi3hate_hashes),
        'hasoc_exact_text_overlap_count': len(exact_text_overlap),
        'hasoc_near_text_overlap_count': len(near_text_overlap),
        'hasoc_template_matches_count': len(hasoc_template_matches),
        'exact_text_overlap_samples': exact_text_overlap[:5],
        'near_text_overlap_samples': near_text_overlap[:5]
    }
    
    with open('scratch/multi3hate_leakage_audit_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("Audit finished successfully. Results saved to scratch/multi3hate_leakage_audit_result.json")

if __name__ == '__main__':
    run_audit()
