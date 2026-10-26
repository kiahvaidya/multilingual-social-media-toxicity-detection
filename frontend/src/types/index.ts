export type RiskLevel = 'Safe' | 'Review' | 'High Risk';

export interface ToxicityScores {
  toxic: number;
  severe_toxic: number;
  obscene: number;
  threat: number;
  insult: number;
  identity_hate: number;
}

export interface PipelineStage {
  stage: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'BYPASSED';
  duration_ms: number;
}

export interface JEVDecisionInfo {
  intent: string;
  contextual_toxicity: boolean;
  moderation: RiskLevel;
  human_review_required: boolean;
  human_review: boolean;
  confidence: number;
  reason: string;
  provider_type: string;
}

export interface JEVData {
  invoked: boolean;
  status: string;
  is_ambiguous: boolean;
  gate_reason: string;
  decision: JEVDecisionInfo | null;
  error?: string | null;
}

export interface TextAnalysisResult {
  text: string;
  language: string;
  risk_level: RiskLevel;
  badge_icon: string;
  is_toxic: boolean;
  confidence: number;
  scores: ToxicityScores;
  thresholds: ToxicityScores;
  is_toxic_flags: Record<keyof ToxicityScores, boolean>;
  explanation: string;
  is_ambiguous: boolean;
  gate_reason: string;
  jev: JEVData;
  pipeline_trace: PipelineStage[];
  timestamp?: string;
}

export interface MemeAnalysisResult {
  text: string;
  language: string;
  multimodal_score: number;
  threshold: number;
  is_hateful: boolean;
  risk_level: RiskLevel;
  text_only_score: number;
  image_only_score: number;
  modality_text_contribution: number;
  interaction_type: 'CONTEXTUAL_INVERSION' | 'EMERGENT_TOXICITY' | 'EXPLICIT_HATE' | 'BENIGN';
  interaction_description: string;
  text_categories: Partial<ToxicityScores>;
  latency_ms: number;
  imageUrl?: string;
  timestamp?: string;
}

export interface BatchItemResult {
  id: string;
  text: string;
  language: string;
  risk_level: RiskLevel;
  badge_icon: string;
  scores: ToxicityScores;
  is_toxic: boolean;
  confidence: number;
}

export interface BatchAnalysisResponse {
  total_analyzed: number;
  high_risk_count: number;
  review_count: number;
  safe_count: number;
  high_risk_percentage: number;
  latency_ms: number;
  results: BatchItemResult[];
}

export interface SystemHealth {
  status: string;
  device: string;
  muril_loaded: boolean;
  multimodal_loaded: boolean;
  multimodal_threshold: number;
  labels: string[];
  thresholds: Record<string, number>;
  jev_configured: string;
}

export interface ModerationQueueItem {
  id: string;
  content: string;
  imageUrl?: string;
  type: 'TEXT' | 'MEME' | 'SOCIAL';
  language: string;
  risk_level: RiskLevel;
  score: number;
  jev_status: string;
  status: 'PENDING' | 'APPROVED' | 'REMOVED' | 'ESCALATED' | 'FALSE_POSITIVE';
  timestamp: string;
}

export interface SocialCommentResult {
  id: string;
  text: string;
  language: string;
  risk_level: RiskLevel;
  badge_icon: string;
  is_toxic: boolean;
  confidence: number;
  scores: ToxicityScores;
  is_toxic_flags: Record<keyof ToxicityScores, boolean>;
  explanation: string;
  is_ambiguous: boolean;
  gate_reason: string;
}

export interface SocialAnalysisResponse {
  platform: 'YouTube' | 'Reddit' | 'Twitter' | string;
  url: string;
  is_demo: boolean;
  message?: string | null;
  total_comments: number;
  high_risk_count: number;
  review_count: number;
  safe_count: number;
  high_risk_percentage: number;
  latency_ms: number;
  comments: SocialCommentResult[];
}

export type MonitoringState = 'NOT_MONITORED' | 'MONITORING' | 'PAUSED' | 'ERROR';

export type ProfileActivityRisk = 'LOW' | 'MODERATE' | 'ELEVATED' | 'HIGH';

export type PatternType =
  | 'HIGH_RISK_FREQUENCY'
  | 'REPEATED_CATEGORY'
  | 'REPLY_CONCENTRATION'
  | 'TEMPORAL_CONCENTRATION'
  | 'LANGUAGE_PATTERN'
  | 'TOXICITY_SPIKE'
  | 'REPEATED_INSULT'
  | 'REPEATED_IDENTITY_HATE'
  | 'REPEATED_THREAT'
  | 'ESCALATION'
  | 'ACTIVITY_BURST';

export type EventSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface ProfileRecord {
  id: string;
  platform: string;
  handle: string;
  profile_url: string;
  monitoring_state: MonitoringState;
  created_at: string;
  last_scanned_at: string | null;
  error_message: string | null;
}

export interface ObservationRecord {
  id: string;
  profile_id: string;
  platform_item_id: string;
  permalink?: string | null;
  timestamp: string;
  text: string;
  media_url?: string | null;
  activity_type?: string;
  language: string;
  scores: Record<string, number>;
  base_risk: string;
  jev_invoked: boolean;
  jev_status: string;
  final_risk_level: string;
  human_review_required: boolean;
  final_explanation: string;
  analyzed_at: string;
}

export interface PatternEventRecord {
  id: string;
  profile_id: string;
  pattern_type: PatternType;
  title: string;
  description: string;
  evidence_count: number;
  total_count: number;
  severity: EventSeverity;
  detected_at: string;
}

export interface TemporalPeriodStat {
  period: string;
  volume: number;
  safe_count: number;
  review_count: number;
  high_risk_count: number;
  toxicity_rate: number;
  high_risk_rate: number;
  dominant_category: string;
}

export interface RetrievalAudit {
  requested_limit: number;
  retrieved_count: number;
  earliest_timestamp?: string | null;
  latest_timestamp?: string | null;
  is_demo: boolean;
  retrieval_limitations: string;
  rate_limit_encountered: boolean;
}

export interface AggregateMetrics {
  total_observations: number;
  safe_count: number;
  review_count: number;
  high_risk_count: number;
  toxicity_rate: number;
  high_risk_rate: number;
  review_rate?: number;
  safe_rate?: number;
  average_toxicity_score?: number;
  profile_activity_risk?: ProfileActivityRisk;
  risk_definition?: string;
  average_scores: Record<string, number>;
  category_flag_counts: Record<string, number>;
  language_distribution?: Record<string, number>;
  language_percentages?: Record<string, number>;
  activity_type_breakdown?: Record<string, { total: number; safe: number; review: number; high_risk: number; toxicity_rate: number }>;
  toxicity_concentration?: {
    dominant_category: string;
    dominant_category_flags: number;
    reply_toxicity_rate: number;
    post_toxicity_rate: number;
    reply_vs_post_difference: number;
  };
  historical_timeline?: TemporalPeriodStat[];
  activity_by_date?: Array<{ date: string; count: number; high_risk_count?: number; high_risk?: number }>;
}

export interface ProfileDossierReport {
  profile: ProfileRecord;
  retrieval_audit: RetrievalAudit;
  activity_risk: ProfileActivityRisk;
  risk_explanation: string;
  aggregates: AggregateMetrics;
  events: PatternEventRecord[];
  recent_observations: ObservationRecord[];
  representative_evidence: ObservationRecord[];
}

export interface ProfileSummaryResponse {
  profile: ProfileRecord;
  aggregates: AggregateMetrics;
  recent_observations: ObservationRecord[];
  events: PatternEventRecord[];
  retrieval_audit?: RetrievalAudit;
  activity_risk?: ProfileActivityRisk;
  representative_evidence?: ObservationRecord[];
}

export interface ScanSummaryResponse {
  profile: ProfileRecord;
  discovered: number;
  new: number;
  skipped: number;
  analyzed: number;
  safe: number;
  review: number;
  high_risk: number;
  aggregates: AggregateMetrics;
  events_detected: PatternEventRecord[];
  is_demo: boolean;
  message?: string | null;
  retrieval_audit?: RetrievalAudit;
  dossier?: ProfileDossierReport;
}

export interface ProfileComparisonResponse {
  profile_a: ProfileSummaryResponse;
  profile_b: ProfileSummaryResponse;
  delta_metrics: Record<string, any>;
  comparative_notes: string[];
}



