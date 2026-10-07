# Zero-Shot CLIP ViT-B/32 & Model B Late Fusion Evaluation

**Artifact Directory**: `scratch/meme_experiments/clip_zero_shot/`  
**Dataset**: Held-Out Multi3Hate ($N = 600$ paired instances: 300 English, 300 Hindi)  
**Ground Truth**: Indian Consensus (`IN == 1` positive: 360, `IN == 0` negative: 240)  

### Key Findings
1. **Model B Baseline**: Precision = 0.7808, Recall = 0.1583, F1 = 0.2633, FPR = 0.0667
2. **Zero-Shot CLIP Mean Ensemble (at 0.50)**: AUROC = 0.5905, AP = 0.6883, Precision = 0.6307, Recall = 0.8444, FPR = 0.7417
3. **OR Late Fusion (`p_text >= 0.45 OR clip >= 0.50`)**:
   - Precision: 0.6336 (Δ -0.1472)
   - Recall: 0.8694 (Δ 0.7111)
   - F1: 0.733 (Δ 0.4697)
   - FPR: 0.7542 (Δ 0.6875)

All artifacts generated automatically without model retraining or threshold tuning.
