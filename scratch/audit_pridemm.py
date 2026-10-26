import os
import glob
import json
import hashlib
import re
from collections import Counter
from difflib import SequenceMatcher
import pandas as pd
from PIL import Image

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(16384):
            hasher.update(chunk)
    return hasher.hexdigest()

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_pridemm_audit():
    base_dir = 'scratch/meme_data/raw/pridemm'
    print("=== STARTING PRIDEMM AUDIT ===")
    
    # 1. Locate metadata files and image folders
    csv_files = glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True) + \
                glob.glob(os.path.join(base_dir, '**', '*.json*'), recursive=True) + \
                glob.glob(os.path.join(base_dir, '**', '*.tsv'), recursive=True)
    print(f"Found metadata files: {csv_files}")
    
    # Find all image files
    img_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')
    all_images = [f for f in glob.glob(os.path.join(base_dir, '**', '*.*'), recursive=True) 
                  if f.lower().endswith(img_extensions)]
    print(f"Total physical image files in PrideMM dir: {len(all_images)}")
    
    # Find metadata dataframes
    dfs = []
    for f in csv_files:
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(f)
                print(f"Loaded {os.path.basename(f)}: shape {df.shape}, cols {df.columns.tolist()}")
                dfs.append((f, df))
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    # Combine or find the main info file
    main_df = None
    if len(dfs) == 1:
        main_df = dfs[0][1]
    elif len(dfs) > 1:
        # Check if train/val/test splits or one single master CSV
        split_dfs = []
        for f, df in dfs:
            base = os.path.basename(f).lower()
            if 'train' in base:
                split_dfs.append(df.assign(split_source='train'))
            elif 'val' in base or 'dev' in base:
                split_dfs.append(df.assign(split_source='val'))
            elif 'test' in base:
                split_dfs.append(df.assign(split_source='test'))
            else:
                main_df = df
        if split_dfs and main_df is None:
            main_df = pd.concat(split_dfs, ignore_index=True)
        elif main_df is None:
            main_df = dfs[0][1]

    if main_df is None:
        print("ERROR: Could not load PrideMM metadata DataFrame.")
        return

    print(f"Master metadata rows: {len(main_df)}")
    print(f"Columns: {main_df.columns.tolist()}")
    
    # Identify key columns
    col_map = {}
    for c in main_df.columns:
        cl = c.lower()
        if 'img' in cl or 'image' in cl or 'file' in cl:
            col_map['image'] = c
        elif 'text' in cl or 'sentence' in cl or 'caption' in cl:
            col_map['text'] = c
        elif cl == 'hate':
            col_map['hate'] = c
        elif 'target' in cl:
            col_map['target'] = c
        elif 'stance' in cl:
            col_map['stance'] = c
        elif 'humor' in cl or 'humour' in cl:
            col_map['humor'] = c
        elif 'split' in cl:
            col_map['split'] = c

    print("Detected Column Mapping:", col_map)
    
    # Check image files resolution on disk
    image_name_to_path = {}
    for p in all_images:
        image_name_to_path[os.path.basename(p)] = p
        
    img_col = col_map.get('image', main_df.columns[0])
    text_col = col_map.get('text', 'text' if 'text' in main_df.columns else None)
    hate_col = col_map.get('hate', 'hate' if 'hate' in main_df.columns else None)
    
    valid_images = 0
    missing_images = 0
    image_paths_matched = []
    
    for idx, r in main_df.iterrows():
        raw_name = str(r[img_col])
        base_name = os.path.basename(raw_name)
        if base_name in image_name_to_path:
            valid_images += 1
            image_paths_matched.append(image_name_to_path[base_name])
        else:
            missing_images += 1
            image_paths_matched.append(None)
            
    valid_text = 0
    text_lengths = []
    for idx, r in main_df.iterrows():
        t = str(r[text_col]) if text_col and pd.notna(r[text_col]) else ""
        if t.strip() and t.strip().lower() != 'nan':
            valid_text += 1
            text_lengths.append(len(t.split()))
            
    # Label counts
    hate_1_count = 0
    hate_0_count = 0
    if hate_col:
        hate_vals = main_df[hate_col].value_counts().to_dict()
        hate_1_count = int(hate_vals.get(1, hate_vals.get('1', 0)))
        hate_0_count = int(hate_vals.get(0, hate_vals.get('0', 0)))
        
    # Split counts
    split_counts = {}
    if 'split' in col_map:
        split_counts = main_df[col_map['split']].value_counts().to_dict()
    elif 'split_source' in main_df.columns:
        split_counts = main_df['split_source'].value_counts().to_dict()
        
    # Target, Stance, Humor counts
    target_counts = main_df[col_map['target']].value_counts().to_dict() if 'target' in col_map else {}
    stance_counts = main_df[col_map['stance']].value_counts().to_dict() if 'stance' in col_map else {}
    humor_counts = main_df[col_map['humor']].value_counts().to_dict() if 'humor' in col_map else {}

    # Target distribution among hateful samples only
    target_hateful_counts = {}
    if 'target' in col_map and hate_col:
        hateful_sub = main_df[main_df[hate_col].astype(str).isin(['1', '1.0', 'hate'])]
        target_hateful_counts = hateful_sub[col_map['target']].value_counts().to_dict()

    # Image hashes & duplicates
    print("Computing SHA256 hashes for all physical images...")
    img_hashes = {}
    dup_hash_count = 0
    for p in all_images:
        h = compute_sha256(p)
        if h in img_hashes:
            dup_hash_count += 1
        else:
            img_hashes[h] = p

    # Duplicate text
    norm_texts = {}
    dup_text_count = 0
    for idx, r in main_df.iterrows():
        t = str(r[text_col]) if text_col and pd.notna(r[text_col]) else ""
        nt = normalize_text(t)
        if nt:
            if nt in norm_texts:
                dup_text_count += 1
            else:
                norm_texts[nt] = idx
                
    # Image dimensions sample
    dims = []
    for p in all_images[:200]:
        try:
            with Image.open(p) as img:
                dims.append(img.size) # (w, h)
        except Exception:
            pass

    # === MULTI3HATE LEAKAGE CHECK ===
    print("Running Multi3Hate leakage check...")
    m3h_imgs = glob.glob('scratch/meme_data/raw/multi3hate/Multi3Hate_repo/data/memes/en/**/*.jpg', recursive=True) + \
               glob.glob('scratch/meme_data/raw/multi3hate/Multi3Hate_repo/data/memes/en/**/*.png', recursive=True) + \
               glob.glob('scratch/meme_data/raw/multi3hate/Multi3Hate_repo/data/memes/hi/**/*.jpg', recursive=True) + \
               glob.glob('scratch/meme_data/raw/multi3hate/Multi3Hate_repo/data/memes/hi/**/*.png', recursive=True)
               
    m3h_hashes = {}
    for p in m3h_imgs:
        m3h_hashes[compute_sha256(p)] = p
        
    exact_image_overlap = []
    for h, p in img_hashes.items():
        if h in m3h_hashes:
            exact_image_overlap.append((p, m3h_hashes[h]))
            
    # Text overlap with Multi3Hate
    df_en = pd.read_csv('scratch/meme_data/raw/multi3hate/en_captions.csv')
    df_hi = pd.read_csv('scratch/meme_data/raw/multi3hate/hi_captions.csv')
    
    m3h_norm_texts = {}
    for _, r in df_en.iterrows():
        t = str(r.iloc[1])
        nt = normalize_text(t)
        if nt: m3h_norm_texts[nt] = ('en', t)
    for _, r in df_hi.iterrows():
        t = str(r.iloc[1])
        nt = normalize_text(t)
        if nt: m3h_norm_texts[nt] = ('hi', t)
        
    exact_text_overlap = []
    near_text_overlap = []
    
    for idx, r in main_df.iterrows():
        t = str(r[text_col]) if text_col and pd.notna(r[text_col]) else ""
        nt = normalize_text(t)
        if not nt: continue
        if nt in m3h_norm_texts:
            exact_text_overlap.append({
                'pridemm_id': str(r[img_col]),
                'pridemm_text': t,
                'm3h_lang': m3h_norm_texts[nt][0],
                'm3h_text': m3h_norm_texts[nt][1]
            })
        else:
            for m_nt, (lang, orig) in m3h_norm_texts.items():
                if abs(len(nt) - len(m_nt)) > 15: continue
                ratio = SequenceMatcher(None, nt, m_nt).ratio()
                if ratio >= 0.85:
                    near_text_overlap.append({
                        'pridemm_id': str(r[img_col]),
                        'pridemm_text': t,
                        'm3h_lang': lang,
                        'm3h_text': orig,
                        'similarity': ratio
                    })
                    break

    # Template overlap
    m3h_templates = set([os.path.basename(os.path.dirname(p)).lower().replace('-', ' ') for p in m3h_imgs])
    template_overlap = []
    for idx, r in main_df.iterrows():
        t = str(r[text_col]).lower() if text_col and pd.notna(r[text_col]) else ""
        fn = str(r[img_col]).lower()
        matched = []
        for tmpl in m3h_templates:
            if tmpl in t or tmpl in fn:
                matched.append(tmpl)
        if matched:
            template_overlap.append({
                'pridemm_id': str(r[img_col]),
                'matched_templates': matched
            })

    results = {
        'total_images_found': len(all_images),
        'unique_image_hashes': len(img_hashes),
        'duplicate_image_hashes': dup_hash_count,
        'metadata_rows': len(main_df),
        'unique_image_ids': int(main_df[img_col].nunique()),
        'valid_matched_images': valid_images,
        'missing_images': missing_images,
        'valid_text_rows': valid_text,
        'duplicate_text_rows': dup_text_count,
        'hate_1_count': hate_1_count,
        'hate_0_count': hate_0_count,
        'split_distribution': split_counts,
        'target_distribution': target_counts,
        'target_hateful_distribution': target_hateful_counts,
        'stance_distribution': stance_counts,
        'humor_distribution': humor_counts,
        'avg_text_words': sum(text_lengths)/len(text_lengths) if text_lengths else 0,
        'sample_dimensions': dims[:5],
        'leakage': {
            'exact_image_overlap_count': len(exact_image_overlap),
            'exact_image_overlap_samples': exact_image_overlap[:3],
            'exact_text_overlap_count': len(exact_text_overlap),
            'exact_text_overlap_samples': exact_text_overlap[:3],
            'near_text_overlap_count': len(near_text_overlap),
            'near_text_overlap_samples': near_text_overlap[:3],
            'template_overlap_count': len(template_overlap),
            'template_overlap_samples': template_overlap[:3]
        }
    }
    
    with open('scratch/pridemm_audit_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("PrideMM audit completed. Report saved to scratch/pridemm_audit_report.json")

if __name__ == '__main__':
    run_pridemm_audit()
