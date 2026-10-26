# Research Audit & Qualification Note — Version 1.0 (2026-10-09)

**Target:** Multilingual Multimodal Toxicity Detection Research Artifacts  
**Scope:** `scratch/meme_experiments/` (`model_d_pridemm`, `final_multimodal`, `model_improvement`, `final_hard_negative_pass`)

---

## 1. Executive Summary

A forensic ML pipeline audit conducted on 2026-10-09 identified critical methodological and data-level defects in previous experimental runs. In accordance with scientific integrity guidelines, this note formally qualifies historical results, documents confirmed code/data defects, and prevents future recurrence. Historical result JSONs and model checkpoints are preserved intact for provenance, but their scientific interpretations are qualified as follows.

---

## 2. Confirmed Defects & Corrections

### Defect A: Missing-Image Zero-Vector Substitution (Critical)
* **Historical Finding:** In `model_improvement/build_dataset_and_features.py` and `final_hard_negative_pass/prepare_corpus_and_mined_hard_negatives.py`, HASOC 2025 Hindi Meme OCR ($N=1,113$) and TRAC Hinglish ($N=1,000$) samples were assigned all-zero visual embeddings (`torch.zeros(len(texts), 512)`) because raw images were unavailable.
* **Impact:** 28.9% of training data and 61.9% of validation data were unimodal text. The cross-modal interaction layers received zero visual gradients on all Indic samples.
* **Correction/Qualification:** Claims that candidate models achieved "cross-cultural multimodal alignment" are **withdrawn**. Candidate models achieved multilingual text adaptation on noisy OCR, but zero paired visual-textual supervision on Indic memes.

### Defect B: Validation-Test Modality Distribution Skew (High)
* **Historical Finding:** Threshold calibration ($T^* = 0.40$ in FHN, $T^* = 0.43$ in Model Improvement) was optimized on a validation partition containing 61.9% zero-vector visual inputs.
* **Impact:** On unimodal text, visual false alarms cannot occur ($v_p = \mathbf{0}$). Validation FPR ($0.2997$) severely underestimated real-world test FPR when evaluated on Multi3Hate ($100\%$ true multimodal memes), causing test FPR to escalate to $0.3917 - 0.5583$.
* **Correction/Qualification:** Validation metrics from `final_hard_negative_pass/validation_metrics.json` cannot be cited as representative of multimodal performance.

### Defect C: TRAC Subgroup Mask String Mismatch (Bug)
* **Historical Finding:** Line 378 of `train_and_evaluate_pass.py` filtered validation sources using `val_sources == "trac"`. The dataset records stored the identifier as `"trac_hinglish"`, resulting in an empty mask and zeroed-out subgroup metrics in `validation_metrics.json`.
* **Impact:** Hinglish subgroup performance was unrecorded.
* **Correction:** Line 378 has been corrected to `(val_sources == "trac_hinglish") | (val_sources == "trac")`. Historical JSON is preserved; future evaluations will record accurate TRAC slices.

---

## 3. Preserved Status of Official Baseline & Production

1. **Official Baseline Integrity:**
   * Model: `scratch/meme_experiments/final_multimodal/best_model.pt`
   * Training Data: PrideMM English paired memes only ($N=4,328$).
   * Status: **UNCOMPROMISED**. The baseline was trained strictly on genuine paired memes. Its reported Multi3Hate metrics (F1: 0.5129, AUROC: 0.5835, FPR: 0.3000, Hindi Recall: 0.1889) are technically valid.
2. **Production Protection:**
   * Production `api.py` loads the official baseline exclusively.
   * None of the candidate experimental checkpoints have been staged or promoted to production.
