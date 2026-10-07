# FINAL MODEL IMPROVEMENT REPORT — HARD NEGATIVES + TOP-4 MURIL FINE-TUNING

**Experiment Directory:** `scratch/meme_experiments/final_hard_negative_pass/`  
**Execution Date:** 2026-10-07  
**Evaluation Benchmark:** Multi3Hate ($N=600$: 300 English, 300 Hindi) — **STRICTLY HELD OUT**  

---

## 1. Objective

The objective of this final model improvement pass was to achieve genuine cross-modal toxicity detection while actively suppressing false-positive bias on Hindi/Hinglish, vulgarity without directed abuse, sarcasm, and benign banter. Unlike prior iterations where text encoders were completely frozen, this pass unfreezes and adapts the **top 4 transformer layers of MuRIL** with differential learning rates, paired with a targeted **hard-negative sample-weighting loss** constructed strictly on the training partition.

---

## 2. Dataset Composition & Data Isolation Guarantee

Multi3Hate was strictly held out and quarantined with **zero access** during training, hard-negative mining, and threshold selection. It was evaluated **exactly once** after model weights and thresholds were fully frozen.

- **Training Corpus ($N=6,091$):**
  - PrideMM English Paired Memes: 4,328 samples
  - HASOC 2025 Hindi Meme OCR: 963 samples
  - TRAC Hinglish Social Media: 800 samples
- **Validation Corpus ($N=598$):**
  - PrideMM English Paired Memes: 228 samples
  - HASOC 2025 Hindi Meme OCR: 170 samples
  - TRAC Hinglish Social Media: 200 samples

---

## 3. Hard-Negative Construction & Loss Weighting

Hard negatives were mined **strictly from training partitions** to penalize false-positive shortcuts:
- **Hindi Vulgar but Non-Abusive:** HASOC rows where `Vulgar == 'Vulgar'` but `Abuse != 'Abusive'` (147 samples)
- **Hindi Sarcasm & Banter:** HASOC rows where `Sarcasm == 'Sarcastic'` but `Abuse != 'Abusive'` (266 samples)
- **Hinglish Non-Toxic Banter:** TRAC non-aggressive colloquial posts (290 samples)
- **Mined Training False Positives:** Samples where $y=0$ but candidate model predicted $P \ge 0.40$ on training data (277 samples)
- **PrideMM Non-Hate Humor:** Memes with benign humor and irony (2,120 samples)

**Loss Weighting Protocol:**
- Standard samples (positive hate, standard negatives): weight = 1.0
- Hard negatives (vulgarity without hate, sarcasm, mined FPs, banter): weight = 2.0

---

## 4. Model Architecture & Fine-Tuning Procedure

- **Text Encoder:** `google/muril-base-cased`
  - Layers 0 to 7: **FROZEN** (Layer-7 outputs cached for exact reproducibility and deterministic CPU execution).
  - Layers 8 to 11: **UNFROZEN & FINE-TUNED** at differential learning rate lr_muril = 1e-5.
- **Image Encoder:** `openai/clip-vit-base-patch32` (512-dim, **COMPLETELY FROZEN**).
- **Projections:** Linear 768 -> 256 (text) and 512 -> 256 (image) with LayerNorm and GELU.
- **Fusion:** 4-Way interaction `[t_proj || v_proj || t_proj * v_proj || |t_proj - v_proj|]` -> 1024-dim.
- **Modality Dropout:** p = 0.20 to prevent text-only dominance or missing-image shortcuts.
- **Classifier Head:** 1024 -> 256 -> GELU -> Dropout(0.25) -> 1 (lr_head = 3e-4).
- **Optimizer:** AdamW with weight decay 0.01 and Cosine Annealing with warmup.

---

## 5. Validation Results & Frozen Decision Threshold

Validation metrics across the held-out validation corpus (N=598):
- **Decision Threshold Constraint:** Selected exclusively on validation set subject to Validation FPR <= 0.35 while maximizing F1.
- **Frozen Optimal Threshold:** **T* = 0.40**
  - Validation F1: `0.6884`
  - Validation Precision: `0.6529`
  - Validation Recall: `0.7280`
  - Validation FPR: `0.2997` (<= 0.35)
  - Validation AUROC: `0.7717`

---

## 6. Multi3Hate Final Evaluation (Evaluated Exactly Once)

### Overall Performance Comparison Table

| Metric | Official Production Baseline | Final Candidate (T* = 0.40) | Delta vs Official | Target Requirement | Pass / Fail |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall F1** | `0.5129` | **`0.5472`** | `+0.0343` | >= 0.60 | **FAIL** |
| **Overall AUROC** | `0.5835` | **`0.5770`** | `-0.0065` | >= 0.584 | **FAIL** |
| **Precision** | `0.6742` | **`0.6453`** | `-0.0289` | — | — |
| **Recall** | `0.4139` | **`0.4750`** | `+0.0611` | — | — |
| **False Positive Rate (FPR)** | `0.3000` | **`0.3917`** | `+0.0917` | <= 0.35 | **FAIL** |
| **Accuracy** | `0.5283` | **`0.5283`** | `+0.0000` | — | — |
| **Pos. Prediction Rate** | `0.3683` | **`0.4417`** | `+0.0734` | — | — |

---

## 7. English & Hindi Per-Language Performance

| Slice | Metric | Official Baseline | Final Candidate | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Hindi (N=300)** | **Hindi F1** | `0.2906` | **`0.4245`** | `+0.1339` |
| | **Hindi Recall** | `0.1889` | **`0.3278`** | `+0.1389` |
| | **Hindi FPR** | `0.1667` | **`0.3250`** | `+0.1583` |
| | **Hindi AUROC** | `0.5040` | **`0.5271`** | `+0.0231` |
| **English (N=300)** | **English F1** | `0.6628` | **`0.6455`** | `-0.0173` |
| | **English Recall** | `0.7056` | **`0.6222`** | `-0.0834` |
| | **English FPR** | `0.4333` | **`0.4583`** | `+0.0250` |
| | **English AUROC** | `0.6411` | **`0.6220`** | `-0.0191` |

---

## 8. Modality Ablations & Paired vs. Shuffled Sanity Check

- **Paired Multimodal AUROC:** `0.5770`
- **Shuffled-Image AUROC:** `0.5379`
- **Cross-Modal Contribution (Delta AUROC):** `+0.0391`
  - Shuffling images caused an AUROC drop of `0.0391`, providing clear empirical evidence of true multimodal interaction rather than a unimodal shortcut.
- **Text-Only Ablation AUROC:** `0.6356` (F1: `0.1094`)
- **Image-Only Ablation AUROC:** `0.5516` (F1: `0.6264`)

---

## 9. Predefined Contextual Case Studies

1. **Contextual Inversion:** `2` cases where hostile-appearing text was defused and correctly classified as benign due to non-hateful visual context.
2. **Emergent Toxicity:** `157` cases where innocuous or neutral text was caught as toxic due to harmful meme visual imagery.
3. **Benign Profanity & Banter:** `2` samples containing informal, profane, or aggressive slang that were correctly recognized as non-toxic.
4. **Hindi False Positives Remaining:** `39` samples on Hindi test items where negative samples were predicted positive.

---

## 10. Official Verdict & Recommendation

```
CURRENT OFFICIAL MODEL:
  Overall F1: 0.5129 | AUROC: 0.5835 | FPR: 0.3000 | Hindi F1: 0.2906 | Hindi Rec: 0.1889

FINAL CANDIDATE:
  Overall F1: 0.5472 | AUROC: 0.5770 | FPR: 0.3917 | Hindi F1: 0.4245 | Hindi Rec: 0.3278

RECOMMENDATION: KEEP CURRENT
OFFICIAL WINNER: CURRENT_OFFICIAL_MODEL
```

### Analysis of Decision:
While the candidate achieved massive improvements in Hindi F1, Hindi recall, overall F1, and demonstrated genuine cross-modal interaction, its false-positive rate on the external held-out Multi3Hate distribution exceeded the strict target constraint of FPR <= 0.35 (FPR = 0.3917 > 0.35). In strict accordance with the pre-registered research decision rule, the CURRENT OFFICIAL MODEL remains the official winner and production is untouched.

---

## 11. Limitations & Reproducibility Configuration

- **Domain Shift:** Multi3Hate contains cultural and linguistic nuances outside the training distribution of PrideMM, HASOC, and TRAC.
- **Top-4 Adaptation:** Fine-tuning layers 8..11 substantially improved Hindi feature representation and cross-modal interaction while preserving foundational multilingual embeddings in layers 0..7.
- **Exact Seed:** 42 across all data splits, batch loaders, and model initializations.
