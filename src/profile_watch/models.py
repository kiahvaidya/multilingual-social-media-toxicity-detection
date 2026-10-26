"""
Data models and schemas for Historical Profile Intelligence Scanner.
Strictly evidence-first analysis of public social interactions without psychoanalysis or surveillance.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MonitoringState(str, Enum):
    """Retained for backwards compatibility with existing records."""
    NOT_MONITORED = "NOT_MONITORED"
    MONITORING = "MONITORING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class ProfileActivityRisk(str, Enum):
    """
    Overall classification of observed public activity volume and severity.
    STRICTLY DEFINED: Risk level of observed public activity only.
    DOES NOT infer identity, personality, beliefs, mental state, or future behavior.
    """
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"


class PatternType(str, Enum):
    """Deterministic, evidence-grounded observable activity patterns."""
    HIGH_RISK_FREQUENCY = "HIGH_RISK_FREQUENCY"
    REPEATED_CATEGORY = "REPEATED_CATEGORY"
    REPLY_CONCENTRATION = "REPLY_CONCENTRATION"
    TEMPORAL_CONCENTRATION = "TEMPORAL_CONCENTRATION"
    LANGUAGE_PATTERN = "LANGUAGE_PATTERN"
    ACTIVITY_BURST = "ACTIVITY_BURST"
    TOXICITY_SPIKE = "TOXICITY_SPIKE"
    ESCALATION = "ESCALATION"
    REPEATED_INSULT = "REPEATED_INSULT"
    REPEATED_IDENTITY_HATE = "REPEATED_IDENTITY_HATE"
    REPEATED_THREAT = "REPEATED_THREAT"


class EventSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RawProfileItem(BaseModel):
    """Raw public activity item ingested from platform connector."""
    item_id: str
    text: str
    timestamp: str  # ISO8601 or parseable date string
    permalink: str
    media_url: Optional[str] = None
    activity_type: str = "comment"  # "post", "comment", "reply"
    is_demo: bool = False


class ResolvedProfile(BaseModel):
    """Normalized profile identity."""
    platform: str  # Reddit, YouTube, Twitter
    handle: str
    profile_url: str
    is_demo: bool = False
    message: Optional[str] = None


class ObservationRecord(BaseModel):
    """Analyzed and persisted single observation."""
    id: str
    profile_id: str
    platform_item_id: str
    permalink: str
    timestamp: str
    text: str
    media_url: Optional[str] = None
    activity_type: str = "comment"  # "post", "comment", "reply"
    language: str = "English"
    scores: Dict[str, float]  # toxic, severe_toxic, obscene, threat, insult, identity_hate
    base_risk: str
    jev_invoked: bool = False
    jev_status: str = "BYPASSED_CLEAR_CASE"
    final_risk_level: str
    human_review_required: bool = False
    final_explanation: str
    analyzed_at: str


class PatternEventRecord(BaseModel):
    """Detected behavioral pattern with evidence counts."""
    id: str
    profile_id: str
    pattern_type: PatternType
    title: str
    description: str  # Evidence-first: e.g. "14 of 31 analyzed interactions contained elevated insult scores."
    evidence_count: int
    total_count: int
    severity: EventSeverity
    detected_at: str


class TemporalPeriodStat(BaseModel):
    """Historical timeline statistical bucket (e.g. monthly)."""
    period: str  # e.g. "2026-08" or "2026-W34"
    volume: int
    safe_count: int
    review_count: int
    high_risk_count: int
    toxicity_rate: float
    high_risk_rate: float
    dominant_category: str


class RetrievalAudit(BaseModel):
    """Transparent audit of public API ingestion limits and data origin."""
    requested_limit: int = 25
    retrieved_count: int = 0
    earliest_timestamp: Optional[str] = None
    latest_timestamp: Optional[str] = None
    is_demo: bool = False
    retrieval_limitations: str = ""
    rate_limit_encountered: bool = False


class AggregateMetrics(BaseModel):
    """Deterministic summary statistics calculated across profile observations."""
    total_observations: int = 0
    safe_count: int = 0
    review_count: int = 0
    high_risk_count: int = 0
    toxicity_rate: float = 0.0  # percentage (0.0 to 100.0) of flagged or high-risk items
    high_risk_rate: float = 0.0  # percentage of high_risk items
    review_rate: float = 0.0
    safe_rate: float = 0.0
    average_toxicity_score: float = 0.0
    profile_activity_risk: ProfileActivityRisk = ProfileActivityRisk.LOW
    risk_definition: str = "Risk level of observed public activity only. Does not characterize the person."
    average_scores: Dict[str, float] = Field(default_factory=dict)
    category_flag_counts: Dict[str, int] = Field(default_factory=dict)
    language_distribution: Dict[str, int] = Field(default_factory=dict)
    language_percentages: Dict[str, float] = Field(default_factory=dict)
    activity_type_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    toxicity_concentration: Dict[str, Any] = Field(default_factory=dict)
    historical_timeline: List[TemporalPeriodStat] = Field(default_factory=list)
    activity_by_date: List[Dict[str, Any]] = Field(default_factory=list)  # Backwards compatibility


class ProfileRecord(BaseModel):
    """Profile record persisted in SQLite."""
    id: str
    platform: str
    handle: str
    profile_url: str
    monitoring_state: MonitoringState = MonitoringState.NOT_MONITORED
    created_at: str
    last_scanned_at: Optional[str] = None
    error_message: Optional[str] = None


class ProfileDossierReport(BaseModel):
    """Comprehensive historical profile intelligence dossier."""
    profile: ProfileRecord
    retrieval_audit: RetrievalAudit
    activity_risk: ProfileActivityRisk
    risk_explanation: str
    aggregates: AggregateMetrics
    events: List[PatternEventRecord]
    recent_observations: List[ObservationRecord]
    representative_evidence: List[ObservationRecord]


class ProfileSummaryResponse(BaseModel):
    """Consolidated profile view."""
    profile: ProfileRecord
    aggregates: AggregateMetrics
    recent_observations: List[ObservationRecord]
    events: List[PatternEventRecord]
    retrieval_audit: Optional[RetrievalAudit] = None
    activity_risk: Optional[ProfileActivityRisk] = None
    representative_evidence: List[ObservationRecord] = Field(default_factory=list)


class ScanRequest(BaseModel):
    url: str
    limit: Optional[int] = 25
    demo_fallback: Optional[bool] = True
    enable_jev: Optional[bool] = False


class ScanSummaryResponse(BaseModel):
    profile: ProfileRecord
    discovered: int
    new: int
    skipped: int
    analyzed: int
    safe: int
    review: int
    high_risk: int
    aggregates: AggregateMetrics
    events_detected: List[PatternEventRecord]
    is_demo: bool = False
    message: Optional[str] = None
    retrieval_audit: Optional[RetrievalAudit] = None
    dossier: Optional[ProfileDossierReport] = None


class ProfileComparisonResponse(BaseModel):
    """Comparative evaluation of two observed public activity profiles."""
    profile_a: ProfileSummaryResponse
    profile_b: ProfileSummaryResponse
    delta_metrics: Dict[str, Any]
    comparative_notes: List[str]

