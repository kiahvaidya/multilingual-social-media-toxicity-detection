/**
 * RESEARCH METRICS SOURCE OF TRUTH
 * Source of truth: scratch/meme_experiments/final_hard_negative_pass/
 * Generated from official experiment run artifacts:
 * - multi3hate_metrics.json
 * - per_language_metrics.json
 * - ablation_results.json
 * - comparison_against_official.json
 * - shuffle_sanity_results.json
 */

export interface ModelMetricSet {
  f1: number;
  auroc: number;
  fpr: number;
  precision?: number;
  recall?: number;
  accuracy?: number;
  positive_prediction_rate?: number;
  hindi_f1?: number;
  hindi_recall?: number;
  hindi_fpr?: number;
  hindi_auroc?: number;
  english_f1?: number;
  english_recall?: number;
  english_fpr?: number;
  english_auroc?: number;
}

export interface LanguageSliceMetric {
  language: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auroc: number;
  fpr: number;
  fnr: number;
  positive_prediction_rate: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface AblationStudyResult {
  condition: string;
  description: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auroc: number;
  fpr: number;
}

export const OFFICIAL_PRODUCTION_BASELINE: ModelMetricSet = {
  f1: 0.5129,
  auroc: 0.5835,
  fpr: 0.3000,
  hindi_f1: 0.2906,
  hindi_recall: 0.1889,
};

export const FINAL_CANDIDATE_RESULTS: ModelMetricSet = {
  f1: 0.5472,
  auroc: 0.5770,
  fpr: 0.3917,
  precision: 0.6453,
  recall: 0.4750,
  accuracy: 0.5283,
  positive_prediction_rate: 0.4417,
  hindi_f1: 0.4245,
  hindi_recall: 0.3278,
  hindi_fpr: 0.3250,
  hindi_auroc: 0.5271,
  english_f1: 0.6455,
  english_recall: 0.6222,
  english_fpr: 0.4583,
  english_auroc: 0.6220,
};

export const EXPERIMENT_STATUS = {
  officialWinner: 'CURRENT_OFFICIAL_MODEL',
  officialRecommendation: 'KEEP CURRENT',
  statusExplanation:
    'The final candidate improved recall and Hindi performance but did not outperform the official model on the held-out benchmark under the predefined decision criteria.',
};

// Supported language slices in Multi3Hate (N=600: 300 English, 300 Hindi)
export const EVALUATED_LANGUAGE_SLICES: LanguageSliceMetric[] = [
  {
    language: 'English',
    accuracy: 0.5900,
    precision: 0.6707,
    recall: 0.6222,
    f1: 0.6455,
    auroc: 0.6220,
    fpr: 0.4583,
    fnr: 0.3778,
    positive_prediction_rate: 0.5567,
    tp: 112,
    fp: 55,
    tn: 65,
    fn: 68,
  },
  {
    language: 'Hindi',
    accuracy: 0.4667,
    precision: 0.6020,
    recall: 0.3278,
    f1: 0.4245,
    auroc: 0.5271,
    fpr: 0.3250,
    fnr: 0.6722,
    positive_prediction_rate: 0.3267,
    tp: 59,
    fp: 39,
    tn: 81,
    fn: 121,
  },
];

// Shuffled image control & ablation results
export const SHUFFLED_IMAGE_RESULTS = {
  paired_auroc: 0.5770,
  shuffled_auroc: 0.5379,
  delta_auroc: 0.0391,
  description:
    'Shuffling image-text pairings reduced AUROC by 0.0391, indicating sensitivity to cross-modal correspondence.',
};

export const ABLATION_STUDY_METRICS: AblationStudyResult[] = [
  {
    condition: 'Paired Full Multimodal',
    description: 'MuRIL top-4 FT + CLIP ViT-B/32 + 4-way fusion',
    accuracy: 0.5283,
    precision: 0.6453,
    recall: 0.4750,
    f1: 0.5472,
    auroc: 0.5770,
    fpr: 0.3917,
  },
  {
    condition: 'Shuffled Image Control',
    description: 'Images permuted within batch to break cross-modal semantics',
    accuracy: 0.5200,
    precision: 0.6268,
    recall: 0.4944,
    f1: 0.5528,
    auroc: 0.5379,
    fpr: 0.4417,
  },
  {
    condition: 'Text-Only Ablation',
    description: 'Visual vector zeroed out at fusion bottleneck',
    accuracy: 0.4300,
    precision: 0.8750,
    recall: 0.0583,
    f1: 0.1094,
    auroc: 0.6356,
    fpr: 0.0125,
  },
  {
    condition: 'Image-Only Ablation',
    description: 'Text vector zeroed out at fusion bottleneck',
    accuracy: 0.5467,
    precision: 0.6196,
    recall: 0.6333,
    f1: 0.6264,
    auroc: 0.5516,
    fpr: 0.5833,
  },
];
