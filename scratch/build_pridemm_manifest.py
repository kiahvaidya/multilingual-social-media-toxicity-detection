import os
import json
import pandas as pd

def build_pridemm_manifest():
    csv_path = 'scratch/meme_data/raw/pridemm/PrideMM/PrideMM.csv'
    img_dir = 'scratch/meme_data/raw/pridemm/PrideMM/Images'
    output_path = 'scratch/meme_data/processed/pridemm_manifest.json'
    
    df = pd.read_csv(csv_path)
    print(f"Loaded PrideMM CSV with {len(df)} rows.")
    
    target_map = {
        0: 'undirected',
        0.0: 'undirected',
        1: 'individual',
        1.0: 'individual',
        2: 'community',
        2.0: 'community',
        3: 'organization',
        3.0: 'organization'
    }
    
    stance_map = {
        0: 'neutral',
        1: 'support',
        2: 'oppose'
    }
    
    humor_map = {
        0: 'no_humor',
        1: 'humor'
    }
    
    records = []
    
    for idx, row in df.iterrows():
        img_filename = str(row['name'])
        full_img_path = os.path.join(img_dir, img_filename)
        
        is_hate = bool(row['hate'] == 1)
        raw_target = row['target']
        target_str = target_map.get(raw_target, None) if pd.notna(raw_target) else None
        
        raw_stance = row['stance']
        stance_str = stance_map.get(raw_stance, None) if pd.notna(raw_stance) else None
        
        raw_humor = row['humour']
        humor_str = humor_map.get(raw_humor, None) if pd.notna(raw_humor) else None
        
        split = str(row['split']).strip().lower()
        if split == 'train':
            role = 'AUXILIARY_DOMAIN_SPECIFIC_TRAINING'
        elif split == 'val':
            role = 'AUXILIARY_DOMAIN_SPECIFIC_VAL'
        elif split == 'test':
            role = 'AUXILIARY_DOMAIN_SPECIFIC_TEST'
        else:
            role = 'AUXILIARY_DOMAIN_SPECIFIC_OTHER'
            
        rhetorical = []
        if humor_str == 'humor':
            rhetorical.append('humor')
            
        record = {
            'sample_id': f"pridemm_{os.path.splitext(img_filename)[0]}",
            'dataset': 'pridemm',
            'dataset_role': role,
            'image_path': full_img_path.replace('\\', '/'),
            'image_reference': f"pridemm://Images/{img_filename}",
            'raw_text': str(row['text']),
            'ocr_text': str(row['text']),
            'language': 'english',
            'script': 'Latin',
            'labels': {
                'hate': is_hate,
                'target_raw': raw_target,
                'stance_raw': raw_stance,
                'humor_raw': raw_humor
            },
            'target': target_str,
            'stance': stance_str,
            'humor': humor_str,
            'source_split': split,
            'unified_ontology': {
                'hate': is_hate,
                'hate_types': ['lgbtq_hate'] if is_hate else [],
                'target_type': target_str if is_hate else None,
                'rhetorical_mechanisms': rhetorical,
                'domain': 'lgbtq_pride'
            }
        }
        records.append(record)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {output_path} with {len(records)} records.")
    
    # Split summary
    split_summary = {}
    for r in records:
        s = r['source_split']
        split_summary[s] = split_summary.get(s, 0) + 1
    print("Split summary:", split_summary)

if __name__ == '__main__':
    build_pridemm_manifest()
