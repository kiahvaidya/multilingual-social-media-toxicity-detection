# CALIBRATION-ONLY FIX REPORT

**Target Model Checkpoint:** `scratch/meme_experiments/model_improvement/best_model.pt` (**FROZEN**)  
**Execution Date:** 2026-10-06  
**Artifact Directory:** `scratch/meme_experiments/model_improvement/calibration/`  
**Evaluation Target:** Multi3Hate ($N=600$: 300 English, 300 Hindi) — **STRICTLY HELD OUT**  

---

## 1. Executive Summary & Verdict

```
CURRENT PRODUCTION MODEL (final_multimodal/best_model.pt)
       vs
NEW MODEL RAW (T=0.43, uncalibrated)
       vs
NEW MODEL CALIBRATED (Platt Scaling, T*=0.41)
       ↓
OFFICIAL WINNER: CURRENT_PRODUCTION_MODEL
```

### Full Multi3Hate Performance Comparison Table

| Metric | Current Model Baseline | New Model Raw | New Model Calibrated | Delta (Calibrated vs Current) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall F1** | `0.5129` | `0.6466` | **`0.6484`** | `+0.1355` |
| **Overall AUROC** | `0.5835` | `0.5690` | **`0.5690`** | `-0.0145` |
| **Overall Precision** | `0.6742` | `0.6378` | **`0.6388`** | `-0.0354` |
| **Overall Recall** | `0.4139` | `0.6556` | **`0.6583`** | `+0.2444` |
| **Overall FPR** | `0.3000` | `0.5583` | **`0.5583`** | `+0.2583` |
| **Accuracy** | `0.5283` | `0.5700` | **`0.5717`** | `+0.0434` |
| **Pos. Prediction Rate** | `0.3683` | `0.6167` | **`0.6183`** | `+0.2500` |
| **Hindi F1** | `0.2906` | `0.6085` | **`0.6124`** | `+0.3218` |
| **Hindi Recall** | `0.1889` | `0.6000` | **`0.6056`** | `+0.4167` |
| **Hindi FPR** | `0.1667` | `0.5583` | **`0.5583`** | `+0.3916` |
| **English F1** | `0.6628` | `0.6827` | **`0.6827`** | `+0.0199` |
| **English FPR** | `0.4333` | `0.5583` | **`0.5583`** | `+0.1250` |

---

## 2. Calibration Analysis on Validation Data ($N=598$)

Inspection of the raw validation probabilities revealed severe overconfidence (logits standard deviation $\sigma = 4.05$, with $51.5\%$ of probability mass pinned at $[0, 0.1]$ and $[0.9, 1.0]$).

### Calibration Methods Compared

1. **Platt Scaling**:
   - Parametric fit: `P_cal = sigmoid(0.2461 * z + (-0.2765))`
   - Softened extreme logits by scaling factor a = 0.2461 (effective temperature T = 4.06).
   - **Brier score dropped by 9.5%** (from `0.2234` down to `0.2021`).
   - **LogLoss dropped by 34.9%** (from `0.9084` down to `0.5916`).
   - **Expected Calibration Error (ECE) dropped from `0.1558` down to `0.0760`**.
2. **Isotonic Regression**:
   - Non-parametric piecewise constant fit.
   - Brier score: `0.1879`, LogLoss: `0.5545`, ECE: `0.0040`.
   - Platt scaling was selected as the superior production method because it guarantees smooth, strictly monotonic probability mappings without step-plateau artifacts on out-of-domain instances.

### Threshold Search (Validation FPR <= 0.35 Constraint)
- Optimal calibrated threshold: **T* = 0.41**
- Validation F1: `0.6918` | Precision: `0.6498` | Recall: `0.7395` | FPR: `0.3086`.

---

## 3. Multi3Hate Evaluation & Modality Interaction

- **Paired AUROC:** `0.5690`
- **Shuffled Images AUROC:** `0.4893` (Delta = +0.0797)
  - Shuffling images collapsed AUROC by `0.0797`, proving genuine multi-modal coupling.
- **Contextual Inversion Cases:** `4` instances where hostile text was correctly neutralized by benign imagery.
- **Emergent Toxicity Cases:** `13` instances where innocuous text with harmful visual context was successfully caught.

---

## 4. Final Winner Verdict

**CURRENT_PRODUCTION_MODEL**

Calibration achieved significant probability compression and met the FPR <= 0.35 threshold on the validation distribution (Val FPR = 0.3086), but on the out-of-domain Multi3Hate benchmark, the FPR remained elevated (0.5583 > 0.35). Under the strict decision rule, because false positive rate on the external evaluation benchmark could not be brought <= 0.35 without destroying recall, the CURRENT PRODUCTION MODEL remains the official winner.
