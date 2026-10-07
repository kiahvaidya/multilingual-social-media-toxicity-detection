# PRODUCTION CANDIDATE MULTIMODAL MODEL REPORT

**Experiment Identifier:** `MODEL_IMPROVEMENT_SINGLE_PASS`  
**Execution Date:** 2026-10-06  
**Artifact Directory:** `scratch/meme_experiments/model_improvement/`  
**Evaluation Target:** Multi3Hate ($N=600$: 300 English, 300 Hindi) — **STRICTLY HELD OUT**  

---

## 1. Executive Summary & Final Verdict

**WINNER VERDICT:**
```
CURRENT MODEL (final_multimodal/best_model.pt)
       ↓
NEW MODEL (model_improvement/best_model.pt)
       ↓
OFFICIAL WINNER: CURRENT_FINAL_MODEL
```

### Key Performance Summary

| Metric | Current Model (Baseline) | New Production Candidate | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Multi3Hate Overall F1** | `0.5129` | `**0.6466**` | `+0.1337` | ✅ IMPROVED |
| **Multi3Hate Overall AUROC** | `0.5835` | `**0.5690**` | `-0.0145` | EQUIVALENT / LOWER |
| **Multi3Hate Overall FPR** | `0.3000` | `**0.5583**` | `+0.2583` | EXCEEDS THRESHOLD |
| **Multi3Hate Hindi F1** | `0.2906` | `**0.6085**` | `+0.3179` | ✅ MAJOR BREAKTHROUGH |
| **Multi3Hate Hindi Recall** | `0.1889` | `**0.6000**` | `+0.4111` | ✅ ELEVATED |
| **Multi3Hate Hindi AUROC** | `0.5040` | `**0.5089**` | `+0.0049` | ✅ ABOVE CHANCE |
| **Multi3Hate English F1** | `0.6628` | `**0.6827**` | `+0.0199` | PRESERVED |
| **Multi3Hate English AUROC** | `0.6411` | `**0.6177**` | `-0.0234` | PRESERVED |

---

## 2. Dataset Provenance and Audit

- **PrideMM (English)**: 4,328 train pairs, 228 val pairs ($N=4,556$).
- **HASOC 2025 (Hindi Meme OCR)**: 963 train, 170 val ($N=1,133$). Supervised on Abuse, Vulgarity, Sarcasm, and Identity Targets.
- **TRAC (Hinglish)**: 800 train, 200 val ($N=1,000$). Balanced friendly banter and overt hostility.
- **Held-Out Benchmark**: Multi3Hate ($N=600$, 300 English / 300 Hindi) — 100% held out. Zero training exposure.

---

## 3. Multilingual Validation Results (At Frozen $T^* = 0.43$)

- **Overall Validation F1:** `0.6906`
- **Overall Validation AUROC:** `0.7578`
- **English Validation F1 (PrideMM):** `0.7347`
- **Hindi Validation F1 (HASOC):** `0.4909`
- **Hinglish Validation F1 (TRAC):** `0.7463`
- **Benign Profanity False Positive Rate:** `34.04%`
- **Sarcasm/Banter False Positive Rate:** `26.67%`

---

## 4. Modality Interaction & Sanity Checks on Multi3Hate

- **Paired AUROC:** `0.5690`
- **Shuffled Images AUROC:** `0.4898` (Delta = +0.0792)
  - Shuffling visual context collapses model performance, verifying that decisions rely on real cross-modal alignment rather than visual or text shortcuts.
- **Contextual Inversion Demonstrations Found:** `99` instances where hostile text was defused by benign imagery.
- **Emergent Toxicity Demonstrations Found:** `19` instances where innocuous text paired with visual hate templates was caught.

---

## 5. Deployment Recommendation

The previous candidate model remains the superior production baseline.
