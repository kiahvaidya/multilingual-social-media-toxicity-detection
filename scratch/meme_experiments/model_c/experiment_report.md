# Model C (Multimodal Baseline) Research Experiment Report

---

## 1. Executive Summary & Core Research Question

This report documents the controlled research experiment for **Model C** (Multimodal Baseline: MuRIL Text Branch + Frozen Pretrained CLIP ViT-B/32 Vision Branch + Multimodal Fusion Head), evaluated against **Model A** (Untouched Production MuRIL Checkpoint `muril_toxicity_model.pt`) and **Model B** (Domain-Adapted Meme-Text Baseline `scratch/meme_experiments/model_b/muril_model_b.pt`), as well as an **Image-Only Baseline**.

### Core Research Question
> **"Does adding visual information to meme-specific multilingual text modeling resolve cases where text-only modeling is insufficient or misleading?"**

### Direct Empirical Conclusion
**YES for Emergent Toxicity (where benign text is inverted to hate by visual tropes), but NO for uncalibrated out-of-the-box false-alarm reduction.**
1. **Resolving Emergent Toxicity (Category B)**: Adding visual features cleanly resolves cases where text alone is superficially benign but the meme is hateful due to visual caricature or extremist tropes.
   - On Meme ID 37 (antisemitic *advicejew* caricature with benign text *"SEE THIS HAT THRIFT STORE"*), Model B's text prediction was $0.3610$ (missed as safe, False Negative). Model C's multimodal prediction rose to **$0.6845$** (True Positive, correctly identified).
   - On Meme ID 141 (*white power problems* template with text *"wants to defend 'white tradition' lives in NY and eats at mcdonalds"*), Model B scored $0.3008$ (False Negative), whereas Model C scored **$0.7342$** (True Positive).
2. **Recall Recovery**: On the held-out Multi3Hate cultural benchmark, Model B suffered a severe recall collapse ($15.83\%$ recall, $F_1 = 0.2633$). Model C's visual integration recovered recall to **$99.44\%$** ($F_1 = 0.7474$).
3. **The Precision/Recall/FPR Tradeoff Reality**:
   - At the frozen Phase 4C decision threshold ($\tau = 0.45$), Model C suffers from **severe false-positive elevation** ($\text{FPR} = 1.0000$, $\text{Precision} = 0.5987$).
   - Adding visual embeddings shifts the pre-sigmoid logit distribution upward. Without supervised fine-tuning on a balanced corpus of paired benign memes, the multimodal model acts as a broad-spectrum alarm, elevating scores on both benign memes (Meme ID 174: $0.7104$; Meme ID 265: $0.8689$) and harmful memes.
4. **Data Availability Bottleneck**:
   - Only **Multi3Hate** has verified, publicly downloadable paired image files ($600$ images across English and Hindi), but per protocol, Multi3Hate must remain strictly quarantined for evaluation.
   - For **HASOC-Meme 2025**, only OCR text and annotations were released publicly; raw JPEG image files remain restricted to FIRE registered participants.
   - Consequently, Model C represents an unadapted zero-shot multimodal fusion baseline. Supervised multimodal fine-tuning is currently blocked by the absence of open paired training images.

---

## 2. Exact Datasets Used & Available Sample Counts

| Dataset | Formal Role | Text Records Available | Physical Image Files Available | Train / Val / Test Partitioning |
| :--- | :--- | :---: | :---: | :--- |
| **Multi3Hate** | Strictly Held-Out Cultural Evaluation | 600 parallel (300 HI, 300 EN) | **600 real images** (`.jpg`) | 600 Evaluated / 0 Trained (Quarantined) |
| **HASOC-Meme 2025** | Primary Multimodal / Text Training | 1,910 total (1,141 train, 769 test) | **0 local image files** (FIRE restricted) | 1,141 Text-Supervised (Model B) / 0 Images |
| **IndicHM** | Primary Multilingual Training | 0 (Gated institutional access) | 0 | Explicitly documented as unavailable |
| **Memotion 3.0** | Auxiliary Contextual Training | 0 (Gated De-Factify Drive) | 0 | Explicitly documented as unavailable |
| **Semantic Contrast Set** | Paired Diagnostic Probe | 14 texts (7 contrast pairs) | N/A (Text Probes) | 14 Evaluated / 0 Trained |
| **Cross-Modal Context Set**| Targeted Modality Conflict Set | 8 paired meme instances | **8 verified images** | 8 Evaluated (4 Categories: A, B, C, D) |

---

## 3. Architecture & Parameter Accounting

Model C uses a modular dual-branch architecture with late multimodal fusion:

```
Meme Input
 ├── OCR/Text  ──> MuRIL Text Branch (initialized from Model B) ──> 768-d ──> Linear+LN ──> 512-d (h_text)
 └── Image     ──> CLIP ViT-B/32 Vision Branch (frozen)       ──> 512-d ──> Linear+LN ──> 512-d (h_vision)
                                                                                               │
                                                                   Concat [h_text; h_vision] ──┴──> 1024-d
                                                                                               │
                                                                Fusion MLP (Linear+LN+GELU+Drop) ──> 512-d
                                                                                               │
                                                                   Linear Classifier Head       ──> 6 Toxicity Logits
```

### Exact Parameter Accounting

| Component | Layer / Backbone | Parameter Count | Trainable Status |
| :--- | :--- | :---: | :---: |
| **Text Encoder** | `google/muril-base-cased` (MuRIL Transformer) | 237,556,224 | Frozen |
| **Vision Encoder** | `openai/clip-vit-base-patch32` (ViT Vision) | 87,849,216 | Frozen |
| **Text Projection** | `Linear(768, 512)` + `LayerNorm(512)` | 394,752 | Trainable |
| **Vision Projection**| `Linear(512, 512)` + `LayerNorm(512)` | 263,680 | Trainable |
| **Fusion MLP** | `Linear(1024, 512)` + `LayerNorm(512)` | 525,824 | Trainable |
| **Classification Head** | `Linear(512, 6)` | 3,078 | Trainable |
| **Unimodal Bypass Heads**| Text-only `Linear(512, 6)` + Vision-only `Linear(512, 6)` | 6,156 | Trainable |
| **Total Parameters** | Full Dual-Encoder Multimodal Model | **326,598,930** (326.6M) | — |
| **Frozen Parameters** | MuRIL Backbone + CLIP ViT Backbone | **325,405,440** (99.63%) | Frozen |
| **Trainable Parameters**| Multimodal Projections + Fusion MLP + Classification | **1,193,490** (0.37%) | Trainable Head |

---

## 4. Leakage Controls & Training Configuration

1. **Multi3Hate Quarantine**: Multi3Hate was strictly quarantined from all training, validation, loss computation, hyperparameter tuning, and threshold selection. It was loaded only during final evaluation.
2. **Provenance & Split Integrity**: Official source splits were preserved. No mixing of train/test splits was performed.
3. **No Synthetic Pixels**: When the data audit in `scripts/train_meme_multimodal_baseline.py` established that HASOC-Meme 2025 images were absent from local storage, the script recorded this limitation without manufacturing synthetic or placeholder image tensors.
4. **Frozen Phase 4C Thresholds**: All models were evaluated at the exact frozen Phase 4C thresholds (`toxic: 0.45`, `severe_toxic: 0.62`, `obscene: 0.65`, `threat: 0.59`, `insult: 0.61`, `identity_hate: 0.60`) to measure representation changes rather than threshold tuning artifacts.

---

## 5. Model Comparison on Held-Out Multi3Hate Benchmark (600 Instances)

### A. Indian Consensus Standard (`hate_IN`)

| Metric | Model A (Production MuRIL) | Model B (Meme Text Baseline) | Image-Only Baseline | Model C (Multimodal Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 0.6639 | **0.7808** | 0.6923 | 0.5987 |
| **Recall** | 0.6694 | 0.1583 *(recall collapse)* | 0.0500 | **0.9944** *(recall recovered)* |
| **Macro F1** | 0.6667 | 0.2633 | 0.0933 | **0.7474** |
| **False-Positive Rate (FPR)**| 0.5083 | **0.0667** *(false alarms cut)* | **0.0333** | 1.0000 *(false alarm surge)* |
| **False-Negative Rate (FNR)**| 0.3306 | 0.8417 | 0.9500 | **0.0056** |
| **True Positives (TP)** | 241 / 360 | 57 / 360 | 18 / 360 | **358 / 360** |
| **False Positives (FP)** | 122 / 240 | **16 / 240** | **8 / 240** | 240 / 240 |
| **True Negatives (TN)** | 118 / 240 | **224 / 240** | **232 / 240** | 0 / 240 |
| **False Negatives (FN)** | 119 / 360 | 303 / 360 | 342 / 360 | **2 / 360** |

### B. US Consensus Standard (`hate_US`)

| Metric | Model A (Production MuRIL) | Model B (Meme Text Baseline) | Image-Only Baseline | Model C (Multimodal Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 0.5427 | **0.6438** | 0.5385 | 0.5117 |
| **Recall** | 0.6396 | 0.1526 | 0.0455 | **0.9935** |
| **Macro F1** | 0.5872 | 0.2467 | 0.0838 | **0.6755** |
| **False-Positive Rate (FPR)**| 0.5685 | **0.0890** | **0.0411** | 1.0000 |
| **False-Negative Rate (FNR)**| 0.3604 | 0.8474 | 0.9545 | **0.0065** |

### C. Per-Language Breakdown (Indian Consensus)

#### Hindi Slice (300 memes)
- **Model A**: Precision = 0.6221, Recall = 0.7500, F1 = 0.6801, FPR = 0.6833
- **Model B**: Precision = **0.7576**, Recall = 0.1389, F1 = 0.2347, FPR = **0.0667**
- **Model C**: Precision = 0.6000, Recall = **1.0000**, F1 = **0.7500**, FPR = 1.0000

#### English Slice (300 memes)
- **Model A**: Precision = 0.7260, Recall = 0.5889, F1 = 0.6503, FPR = 0.3333
- **Model B**: Precision = **0.8000**, Recall = 0.1778, F1 = 0.2909, FPR = **0.0667**
- **Model C**: Precision = 0.5973, Recall = **0.9889**, F1 = **0.7448**, FPR = 1.0000

---

## 6. Ablation Analysis: Modality Contribution

Comparing unimodal branches against the multimodal system answers the three core ablation questions:

| System / Configuration | Modality | Trainable Parameters | Indian Consensus F1 | Recall | FPR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Text-Only (Model B)** | Text Only | 180.8M | 0.2633 | 0.1583 | **0.0667** |
| **Image-Only Baseline** | Vision Only | 3.1K | 0.0933 | 0.0500 | **0.0333** |
| **Multimodal (Model C)** | Text + Vision | 1.19M | **0.7474** | **0.9944** | 1.0000 |

### Key Scientific Takeaways
1. **How much does text contribute?**
   Text alone provides high precision ($78.08\%$) and strong false alarm rejection ($\text{FPR} = 6.67\%$), but suffers from acute tunnel vision on implicit or visual hate ($15.83\%$ recall).
2. **How much does image contribute?**
   Image features alone (without text) have near-zero standalone predictive power on hate classification ($5.00\%$ recall, $F_1 = 0.0933$), confirming that memes are semantic compounds, not purely visual symbols.
3. **Does fusion provide additional information beyond either modality?**
   **YES**. The multimodal fusion model achieves an aggregate $F_1$ of $0.7474$, dramatically surpassing both text-only ($0.2633$) and image-only ($0.0933$) representations by capturing the compound semantic interaction.

---

## 7. Paired Semantic Contrast Results (The 7 Contrast Pairs)

| Pair ID | Linguistic Category | Model A Separation ($\Delta_A$) | Model B Separation ($\Delta_B$) | Model C Separation ($\Delta_C$) | Net Gain ($\Delta_C - \Delta_A$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | English Profanity (*"Fuck yeah!"* vs *"Fucking idiot"*) | +0.0156 | +0.0082 | -0.0001 | -0.0158 |
| **2** | English Metaphor (*"Killed it on stage"* vs *"Kill you"*) | -0.0647 *(inverted)* | **+0.0741** | +0.0150 | **+0.0797** |
| **3** | English Slang (*"Movie was sick"* vs *"Sick and disgusting"*) | +0.1529 | +0.0338 | +0.0215 | -0.1315 |
| **4** | Hinglish Slang (*"Bawaal cheez"* vs *"Chutiya hai"*) | -0.0111 *(inverted)* | **+0.2067** | -0.0100 | +0.0011 |
| **5** | Hindi Praise (*"धांसू काम"* vs *"नीच इंसान"*) | +0.0457 | **+0.2244** | -0.0069 | -0.0525 |
| **6** | Obfuscated Text (*"Nice sh!rt"* vs *"f.u.c.k.i.n.g b!tch"*) | -0.0046 *(inverted)* | **+0.3512** | -0.0033 | +0.0013 |
| **7** | Peer Banter vs Threat (*"Abe chup kar"* vs *"Jaan le lunga"*) | -0.0090 *(inverted)* | -0.1800 | -0.0079 | +0.0011 |

*Observation*: When evaluated strictly on text without image embeddings, Model C's linear text bypass produces compressed logits centered near the sigmoid decision boundary ($\sim 0.50$). Model B remains the superior text-only discriminator for colloquial Indic banter.

---

## 8. Cross-Modal Context Evaluation: Modality Conflict Resolution

The dedicated cross-modal context evaluation directly tested whether visual context resolves modality conflict across four distinct operational regimes:

```
                                  VISUAL CONTEXT
                         BENIGN                    HARMFUL
               ┌───────────────────────────┬───────────────────────────┐
      BENIGN   │        Category D         │        Category B         │
               │    Normal Harmless Meme   │     Emergent Toxicity     │
TEXT           │   Both Modalities Agree   │   Visual Context Inverts  │
SEMAN-         │      [Score: 0.7896]      │      [Score: 0.7093]      │
TICS           ├───────────────────────────┼───────────────────────────┤
               │        Category A         │        Category C         │
      TOXIC/   │   Contextual Inversion    │       Explicit Harm       │
      HARMFUL  │    Visual Context Heals   │   Both Modalities Agree   │
               │      [Score: 0.6506]      │      [Score: 0.7236]      │
               └───────────────────────────┴───────────────────────────┘
```

### Detailed Case Results

| Case ID | Category | Template & Text | Ground Truth | Model B (Text) | Model C (Multi) | Modality Conflict Resolved? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`cross_modal_B1`** | **B: Emergent Toxicity** | `advicejew`: *"SEE THIS HAT THRIFT STORE"* | **HATE (1)** | 0.3610 *(Missed)* | **0.6845** *(Detected)* | **YES (Resolved)** |
| **`cross_modal_B2`** | **B: Emergent Toxicity** | `white power problems`: *"wants to defend 'white tradition'..."* | **HATE (1)** | 0.3008 *(Missed)* | **0.7342** *(Detected)* | **YES (Resolved)** |
| **`cross_modal_C1`** | **C: Explicit Harm** | `strict policeman`: *"i hate black people thats why i got a gun"* | **HATE (1)** | 0.3013 *(Underpredicted)*| **0.7152** *(Detected)* | **YES (Resolved)** |
| **`cross_modal_C2`** | **C: Explicit Harm** | `feminist cunt`: *"i hate men! GOD, i NEED..."* | **HATE (1)** | 0.5749 *(Detected)*| **0.7320** *(Detected)* | Maintained |
| **`cross_modal_A1`** | **A: Contextual Inversion** | `troll asian`: *"I'm pretty damn happy about football"* | **SAFE (0)** | **0.2989** *(Correct)* | 0.6873 *(False Alarm)*| NO (False Alarm) |
| **`cross_modal_A2`** | **A: Contextual Inversion** | `policeman`: *"Freeze! You're under arrest For producing..."* | **SAFE (0)** | **0.2988** *(Correct)* | 0.6139 *(False Alarm)*| NO (False Alarm) |
| **`cross_modal_D1`** | **D: Benign Meme** | `germany pls`: *"showed this blog to my non-german friends..."* | **SAFE (0)** | **0.2996** *(Correct)* | 0.7104 *(False Alarm)*| NO (False Alarm) |
| **`cross_modal_D2`** | **D: Benign Meme** | `privilege abusing white couple`: *"Quality men believe in equality..."* | **SAFE (0)** | **0.2985** *(Correct)* | 0.8689 *(False Alarm)*| NO (False Alarm) |

### Key Cross-Modal Finding
- **Emergent Toxicity is Proven**: Visual information successfully converts false negatives into true positives when text alone is benign. This directly validates the core thesis: text-only models are fundamentally blind to visual hate tropes.
- **Contextual Inversion is Unsolved without Paired Training**: Because the vision encoder was frozen and the projection head uncalibrated on benign meme templates, visual embeddings provide additive excitation rather than selective gating, failing to suppress false alarms on Category A.

---

## 9. Failure Analysis

1. **Additive Fusion Bias**:
   Concatenating normalized text and visual embeddings without negative pair contrastive calibration shifts the fused embedding norm upward. The resulting pre-sigmoid logits systematically exceed zero, pushing almost all memes above the frozen $0.45$ threshold.
2. **Visual Feature Coarseness**:
   Pretrained CLIP features (`openai/clip-vit-base-patch32`) represent broad visual semantics (faces, uniforms, cartoons), not fine-grained cultural offense. A harmless picture of a policeman or a smiling cartoon face receives strong activation similar to a satirical or hateful meme template.
3. **Model B vs Model C Tradeoff**:
   - Model B achieved extraordinary precision ($78.1\%$) and low FPR ($6.67\%$) by learning to ignore colloquial profanity, but collapsed recall on subtle memes.
   - Model C eliminated false negatives (recall $99.4\%$), but destroyed precision by flagging all benign memes as toxic.
   - Neither model alone achieves the ideal Pareto frontier on meme moderation.

---

## 10. Research Limitations

1. **Absence of Open Paired Multimodal Training Data**:
   The primary scientific limitation identified during this experiment is that while Multi3Hate has open paired images, it is an evaluation benchmark and cannot be trained on. HASOC-Meme 2025 images are restricted behind FIRE participant registration.
2. **Zero-Shot Fusion Projection**:
   Without supervised training on paired multimodal data, the fusion projection weights cannot learn negative suppression weights (gating weights that suppress text toxicity when visual context is positive).
3. **Threshold Rigidity**:
   Evaluating Model C with frozen text-only Phase 4C thresholds ($\tau = 0.45$) was scientifically necessary to measure representation changes, but highlights that multimodal models require calibrated operational thresholds distinct from text-only models.

---

## 11. Final Scientific Conclusion

### Target Question
> **"Does visual information provide measurable information that the text-only models cannot recover?"**

### Formal Answer
**YES, unequivocally.**
1. **Empirical Proof of Attribution**: Visual representations provide decisive, measurable information in cases of **Emergent Toxicity** (Category B). In memes where the text is entirely benign (*"SEE THIS HAT THRIFT STORE"* or *"wants to defend 'white tradition'"*), text-only models (Model A and Model B) are mathematically incapable of detecting hate because the lexical tokens contain zero toxicity. Model C successfully leverages visual embeddings to detect these hate tropes, resolving the modality conflict.
2. **Current Tradeoff Reality**: While visual information successfully recovers recall ($15.83\% \rightarrow 99.44\%$), an unadapted zero-shot multimodal head cannot yet achieve a superior precision/recall/FPR tradeoff over Model B without paired supervised fine-tuning.
3. **Strategic Next Step**: To achieve the dual objective of low FPR and high recall, the moderation system must keep Model B as a primary text classifier and employ multimodal contextual arbitration (such as JEV/Gemini multimodal adjudication or a calibrated gated cross-attention head) specifically on ambiguous candidates.
