# Research Experiment Report: Model D Multimodal Training on PrideMM

**Experiment Identifier:** `MODEL_D_PRIDEMM_STAGE1`  
**Dataset Role:** `AUXILIARY_DOMAIN_SPECIFIC_TRAINING` (PrideMM: 4,328 train, 228 val, 507 test)  
**External Benchmark:** `STRICTLY_HELD_OUT` (Multi3Hate: 600 memes, 300 EN / 300 HI)  
**Baseline:** Frozen Text-Only Model B (`muril_model_b.pt`)  
**Production Impact:** **ZERO** (Isolated research sandbox in `scratch/meme_experiments/model_d_pridemm/`)

---

## 1. Executive Summary & Objective

### Core Question
> **"Does actual paired image + text training improve hate detection compared with our existing text-only Model B?"**

### Bottom-Line Scientific Finding
1. **In-Domain (PrideMM Test, $N=507$):**
   - **YES, dramatically.** Multimodal Model D improves F1 from **0.3198** (Model B toxic) / **0.0080** (Model B identity_hate) to **0.7266**, and AUROC from **0.5607** to **0.7794**.
2. **External Held-Out Cross-Cultural Benchmark (Multi3Hate, $N=600$):**
   - **Nominally higher F1 (0.6720 vs 0.2633), BUT driven by severe over-prediction and high False Positive Rate (FPR: 0.5667).**
   - Multimodal AUROC on Multi3Hate is **0.6018** (compared to 0.6147 for Model B toxic head and 0.6248 for Model B identity_hate head).
   - On Hindi memes, AUROC collapses to **0.5289** (barely above random guessing: 0.50).
3. **Modality Ablation & Sanity Check Diagnostics:**
   - **Ablation:** When text embeddings are zeroed out, the visual branch alone achieves F1 **0.7083** on PrideMM and F1 **0.7125** on Multi3Hate. Conversely, zeroing out image embeddings causes recall to collapse to near zero (0.0243 on PrideMM, 0.0028 on Multi3Hate). The MLP fusion learned to rely almost exclusively on CLIP visual features.
   - **Sanity Checks:** Shuffling meme images on PrideMM collapses AUROC to **0.5039** (demonstrating true image sensitivity in-domain). However, on Multi3Hate, shuffling images causes F1 to drop by only **-0.0176** (from 0.6720 to 0.6544).

### Formal Verdict
> **"Domain-specific improvement without demonstrated cross-cultural generalization."**

---

## 2. Experimental Setup & Architecture

### Model D Architecture
- **Text Encoder:** Pretrained MuRIL (`google/muril-base-cased` backbone from Model B), 768-dim CLS embedding, independently $L_2$-normalized.
- **Image Encoder:** Pretrained CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), 512-dim visual embedding, independently $L_2$-normalized.
- **Fusion Layer:** Concatenation $[e_{text} \;\|\; e_{img}] \in \mathbb{R}^{1280}$.
- **Classifier:** 
  $$\text{Linear}(1280 \to 256) \to \text{GELU}() \to \text{Dropout}(0.2) \to \text{Linear}(256 \to 1) \to \text{Binary Hate Logit}$$
- **Training Constraints (Stage 1):**
  - MuRIL and CLIP encoders were **strictly frozen**.
  - Trained only the fusion MLP parameters (328,449 trainable params).
  - Optimizer: AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-2}$).
  - Loss: `BCEWithLogitsLoss`.
  - Early stopping: Best PrideMM validation F1 (patience = 15 epochs, random seed = 42).
  - Best checkpoint achieved at **Epoch 6** (Val F1: 0.7287, Val Loss: 0.5638, Val AUROC: 0.7644).

---

## 3. Comparative Results

### Primary Comparison Table: Model B (Text-Only) vs Model D (Multimodal)

| Evaluation Set | Model / Head | Precision | Recall | F1 Score | AUROC | FPR | Pred Pos Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PrideMM Test** ($N=507$) | **Model B** (toxic $\ge 0.45$) | 0.5670 | 0.2227 | 0.3198 | 0.5607 | 0.1615 | 19.1% |
| | **Model B** (identity_hate $\ge 0.60$) | 0.5000 | 0.0040 | 0.0080 | 0.5120 | 0.0038 | 0.4% |
| | **Model D (Multimodal)** | **0.7019** | **0.7530** | **0.7266** | **0.7794** | 0.3038 | 52.3% |
| **Multi3Hate Held-Out** ($N=600$) | **Model B** (toxic $\ge 0.45$) | **0.7808** | 0.1583 | 0.2633 | **0.6147** | **0.0667** | 12.2% |
| | **Model B** (identity_hate $\ge 0.60$) | 0.0000 | 0.0000 | 0.0000 | **0.6248** | **0.0000** | 0.0% |
| | **Model D (Multimodal)** | 0.6486 | **0.6972** | **0.6720** | 0.6018 | 0.5667 | 64.5% |

### Confusion Matrices

#### PrideMM Test ($N=507$):
- **Model D:**
  - $\text{TP} = 186, \quad \text{FP} = 79, \quad \text{TN} = 181, \quad \text{FN} = 61$
  - Accuracy: **72.39%**
- **Model B (toxic $\ge 0.45$):**
  - $\text{TP} = 55, \quad \text{FP} = 42, \quad \text{TN} = 218, \quad \text{FN} = 192$
  - Accuracy: **53.85%**

#### Multi3Hate Held-Out ($N=600$, IN Ground Truth):
- **Model D:**
  - $\text{TP} = 251, \quad \text{FP} = 136, \quad \text{TN} = 104, \quad \text{FN} = 109$
  - Accuracy: **59.17%**
- **Model B (toxic $\ge 0.45$):**
  - $\text{TP} = 57, \quad \text{FP} = 16, \quad \text{TN} = 224, \quad \text{FN} = 303$
  - Accuracy: **46.83%**

---

## 4. Multi3Hate Slices & Cultural Consensus Breakdown

### Language Slices (English vs Hindi)

| Multi3Hate Slice | Model D Precision | Model D Recall | Model D F1 | Model D AUROC | Model D FPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **English Slice** ($N=300$) | 0.6885 | 0.7000 | **0.6942** | **0.6631** | 0.4750 |
| **Hindi Slice** ($N=300$) | 0.6127 | 0.6944 | **0.6510** | **0.5289** | 0.6583 |

**Critical Observation:** In English, Model D achieves moderate ranking capability ($\text{AUROC} = 0.6631$). But in Hindi, AUROC collapses to **0.5289**, and FPR increases to **65.83%**. Because PrideMM contains 100% English memes, the text encoder failed to generalize cross-lingually in the multimodal fusion space.

### Cultural Consensus Targets

| Annotator Region | Model D Precision | Model D Recall | Model D F1 | Model D AUROC | Model D FPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **India (IN)** (Primary GT) | 0.6486 | 0.6972 | 0.6720 | 0.6018 | 0.5667 |
| **United States (US)** | 0.5762 | 0.7240 | 0.6417 | 0.6137 | 0.5616 |
| **Germany (DE)** | 0.6589 | 0.7123 | 0.6846 | 0.6237 | 0.5455 |
| **Mexico (MX)** | 0.6098 | 0.7066 | 0.6546 | 0.6023 | 0.5677 |
| **China (CN)** | 0.6718 | 0.6842 | 0.6780 | 0.5938 | 0.5773 |

The model exhibits a consistent over-flagging bias across all annotator groups, predicting positive on **64.5%** of all samples ($\text{pred\_pos} = 387$).

---

## 5. Modality Ablation Analysis

Evaluating the exact frozen Model D classifier under:
1. **Both Modalities:** $[e_{text} \;\|\; e_{img}]$
2. **Text Only:** $[e_{text} \;\|\; \mathbf{0}]$
3. **Image Only:** $[\mathbf{0} \;\|\; e_{img}]$

| Dataset | Modality Setting | Accuracy | Precision | Recall | F1 Score | AUROC | FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PrideMM Test** | **Both Modalities** | **0.7239** | 0.7019 | 0.7530 | **0.7266** | **0.7794** | 0.3038 |
| | Text Only (Zero Image) | 0.5148 | 0.5455 | 0.0243 | 0.0465 | 0.6299 | 0.0192 |
| | Image Only (Zero Text) | 0.6588 | 0.6069 | 0.8502 | 0.7083 | 0.7751 | 0.5231 |
| **Multi3Hate** | **Both Modalities** | **0.5917** | 0.6486 | 0.6972 | **0.6720** | **0.6018** | 0.5667 |
| | Text Only (Zero Image) | 0.4017 | 1.0000 | 0.0028 | 0.0055 | 0.5594 | 0.0000 |
| | Image Only (Zero Text) | 0.5750 | 0.5996 | 0.8778 | 0.7125 | 0.5982 | 0.8792 |

### Forensic Ablation Insights:
- **Visual Dominance:** Image embeddings account for virtually all activation energy in the MLP classifier. Image-only achieves an AUROC of **0.7751** on PrideMM (nearly matching the full multimodal model's 0.7794).
- **Text Suppression Role:** The text branch does not trigger hate detection on its own (recall = 0.0243); rather, it acts primarily as a regularizer/suppressor to slightly reduce false positives produced by the image branch (lowering FPR from 0.5231 to 0.3038).

---

## 6. Anti-Cheating & Modality Shuffling Sanity Checks

To verify whether Model D relies on genuine cross-modal semantic pairing or merely univariate distribution shifts, we evaluated Model D with randomized pairing.

| Test Split | Condition | Accuracy | Precision | Recall | F1 Score | AUROC | F1 $\Delta$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PrideMM Test** | Paired (Normal) | **0.7239** | 0.7019 | 0.7530 | **0.7266** | **0.7794** | Baseline |
| | Shuffled Images | 0.5148 | 0.5018 | 0.5587 | 0.5287 | **0.5039** | **-0.1979** |
| | Shuffled Text | 0.7061 | 0.6801 | 0.7490 | 0.7129 | 0.7687 | -0.0137 |
| **Multi3Hate** | Paired (Normal) | **0.5917** | 0.6486 | 0.6972 | **0.6720** | **0.6018** | Baseline |
| | Shuffled Images | 0.5617 | 0.6209 | 0.6917 | 0.6544 | 0.5292 | **-0.0176** |

### Sanity Check Diagnoses:
1. **PrideMM In-Domain Integrity:**
   - Shuffling images collapses in-domain AUROC from **0.7794** to **0.5039** (pure coin flip). This proves the model learned real visual patterns tied to PrideMM labels.
   - However, shuffling text produces almost no drop in AUROC (0.7794 $\to$ 0.7687), corroborating the ablation finding that text plays a minimal discriminative role.
2. **Multi3Hate Out-of-Domain Leakage Check:**
   - Shuffling images on Multi3Hate results in an F1 drop of only **-0.0176** (0.6720 $\to$ 0.6544).
   - This proves that **Model D's apparent high F1 (0.6720) on Multi3Hate is NOT driven by fine-grained image-text semantic alignment**, but rather by a broad visual prior from CLIP that happens to flag high base-rate visual features present in memes.

---

## 7. Semantic & Cross-Modal Case Studies on Multi3Hate

Representative cases from `multi3hate_predictions.csv` categorized by multimodal relationship:

### Case A: Contextual Inversion (Hostile Text, Benign Meme Context)
*Text contains aggressive or sensitive tokens, but meme context makes it innocuous.*

| Meme ID | Lang | Meme Text | GT (IN) | Model B Toxic | Model D Score | Prediction | Analysis |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **14** | EN | `WHEN PEOPLE LEAVE THE DOOR OPEN   OH NO YOU DIDNT !` | 0 | 0.5754 | 0.6268 | **1 (FP)** | **Failure:** Both Model B and Model D falsely flag everyday frustration as toxic hate. |
| **17** | EN | `YOU AIN'T GETTIN' NO FRIED CHICKEN   TILL' YOU FINISH ALL  YOUR COOL-AID` | 0 | 0.6451 | 0.3802 | **0 (TN)** | **Success:** Multimodal fusion successfully overrides Model B's high text toxicity score (0.6451 $\to$ 0.3802), correctly classifying it as benign. |

### Case B: Emergent Hate (Benign Text, Hateful Meme Context)
*Text appears harmless in isolation, but the paired visual template introduces hateful meaning.*

| Meme ID | Lang | Meme Text | GT (IN) | Model B Toxic | Model D Score | Prediction | Analysis |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **113** | EN | `You are good at the olympics   but terrible at money management.` | 1 | 0.2955 | 0.8737 | **1 (TP)** | **Multimodal Success:** Model B completely misses the racial stereotype (0.2955), while Model D's visual embedding detects the targeted template (0.8737). |
| **205** | EN | `Cops...   Only laugh at their own jokes` | 1 | 0.2962 | 0.8326 | **1 (TP)** | **Multimodal Success:** Model B treats the text as benign (0.2962), but Model D detects hateful visual context. |

### Case C: Explicit Hate (Both Modalities Clearly Hateful)

| Meme ID | Lang | Meme Text | GT (IN) | Model B Toxic | Model D Score | Prediction | Analysis |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **9** | EN | `Support Our Troops, Motherfucker!   Or we'll lock your ass up for tax evasion!!!` | 1 | 0.6501 | 0.8499 | **1 (TP)** | High confidence across both modalities. |
| **9** | HI | *(Hindi translation of above)* | 1 | 0.6696 | 0.5573 | **1 (TP)** | Model B text remains confident; Model D score drops closer to the decision boundary (0.5573). |

### Case D: Benign Memes (Both Modalities Benign)

| Meme ID | Lang | Meme Text | GT (IN) | Model B Toxic | Model D Score | Prediction | Analysis |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **7** | EN | `so you're telling me   you already adjusted our grades` | 0 | 0.2967 | 0.2665 | **0 (TN)** | Correctly rejected by both models. |
| **148** | HI | *(Hindi meme about religious identity inquiry)* | 0 | 0.2968 | 0.5213 | **1 (FP)** | **False Positive:** Model D flags benign Hindi meme due to high visual prior. |

---

## 8. Final Research Synthesis & Recommendations

### Key Scientific Conclusions
1. **In-Domain Multimodal Efficacy:**
   Paired image-text training on PrideMM is highly effective for PrideMM. The multimodal fusion model achieved **0.7266 F1** and **0.7794 AUROC**, thoroughly outperforming text-only Model B (0.3198 F1).
2. **Lack of Cross-Cultural Generalization:**
   PrideMM consists exclusively of Western/English LGBTQ+ memes from Reddit/Twitter. When evaluated on Multi3Hate (South Asian / Indian cultural contexts, Hindi language, global templates):
   - AUROC is marginal (**0.6018** overall, **0.5289** on Hindi).
   - False Positive Rate is unsustainably high (**56.67%**).
   - Shuffling images only drops F1 by **0.0176**, indicating that the model's out-of-domain predictions are largely ungrounded in cross-modal alignment.
3. **Architecture Insight:**
   A simple concatenation + MLP fusion head quickly overfits to the modality with the strongest separability (CLIP visual features) and ignores the text branch. More sophisticated cross-attention or modality gating (trained on diverse multilingual meme data) is required before deploying multimodal models to production.

---

*Artifacts generated in `scratch/meme_experiments/model_d_pridemm/`:*
- `model_d_fusion.pt`
- `metrics.json`
- `training_history.json`
- `pridemm_test_predictions.csv`
- `multi3hate_predictions.csv`
- `ablation_results.json`
- `shuffled_sanity_results.json`
- `case_studies.json`
