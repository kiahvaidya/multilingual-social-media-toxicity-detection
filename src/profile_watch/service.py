"""
Core service orchestrating Profile Watch:
Resolution -> Ingestion -> Deduplication -> Moderation Reuse -> Persistence -> Aggregation -> Pattern Detection.
"""

from collections import defaultdict
from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional
import uuid

from src.profile_watch.connectors import fetch_profile_activity, resolve_profile_url
from src.profile_watch.models import (
    AggregateMetrics,
    MonitoringState,
    ObservationRecord,
    PatternEventRecord,
    ProfileActivityRisk,
    ProfileComparisonResponse,
    ProfileDossierReport,
    ProfileRecord,
    ProfileSummaryResponse,
    RetrievalAudit,
    ScanSummaryResponse,
    TemporalPeriodStat,
)
from src.profile_watch.pattern_engine import detect_profile_patterns
from src.profile_watch.storage import ProfileWatchStorage

logger = logging.getLogger(__name__)

HINGLISH_KEYWORDS = {
    "bhai", "yaar", "kya", "nahi", "nahin", "tera", "teri", "meri", "mera", "tere",
    "karo", "karna", "hoga", "hogi", "raha", "rahi", "rahe", "hai", "hain", "bawaal",
    "mast", "faltu", "bakwaas", "chup", "sharm", "thodi", "niklo", "yahan", "wahan",
    "dekho", "samjho", "achha", "accha", "bahut", "bohot", "bilkul", "sach", "jhuth",
    "apna", "apni", "desh", "deshi", "log", "paisa", "ladki", "ladka", "chutiya",
    "madarchod", "bhenchod", "saale", "kamina", "dost", "dosti", "suno", "bolo",
    "kaise", "kisko", "hum", "tum", "aap", "unka", "inhe", "mat", "kyu", "kyun",
}


def detect_language(text: str) -> str:
    """Heuristic script and lexical detection for Indic, Hinglish, and English."""
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "Hindi"
        elif 0x0980 <= code <= 0x09FF:
            return "Bengali"
        elif 0x0A00 <= code <= 0x0A7F:
            return "Punjabi"
        elif 0x0A80 <= code <= 0x0AFF:
            return "Gujarati"
        elif 0x0B00 <= code <= 0x0B7F:
            return "Odia"
        elif 0x0B80 <= code <= 0x0BFF:
            return "Tamil"
        elif 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        elif 0x0C80 <= code <= 0x0CFF:
            return "Kannada"
        elif 0x0D00 <= code <= 0x0D7F:
            return "Malayalam"

    words = re.findall(r"[a-zA-Z]+", text.lower())
    hinglish_matches = [w for w in words if w in HINGLISH_KEYWORDS]
    if len(hinglish_matches) >= 2 or (len(words) <= 5 and len(hinglish_matches) >= 1):
        return "Hinglish"

    return "English"


class ProfileWatchService:
    """End-to-end service for managing public profile activity monitoring."""

    def __init__(self, storage: Optional[ProfileWatchStorage] = None, model_server: Optional[Any] = None):
        self.storage = storage or ProfileWatchStorage()
        self._model_server = model_server

    @property
    def model_server(self):
        """Lazily access ModelServer singleton to avoid circular imports during startup."""
        if self._model_server is None:
            from api import ModelServer
            self._model_server = ModelServer.get_instance()
        return self._model_server

    # ── Scan & Ingestion Flow ─────────────────────────────────────

    def scan_profile(
        self,
        url_or_id: str,
        limit: int = 25,
        demo_fallback: bool = True,
        enable_jev: bool = False,
    ) -> ScanSummaryResponse:
        """Execute a complete scan cycle for a public profile.

        Steps:
          1. Resolve profile identity and URL.
          2. Fetch public activity from platform connector.
          3. Deduplicate against already persisted observation items.
          4. Execute existing moderation pipeline on new items.
          5. Persist new observations with atomic SQLite transactions.
          6. Recalculate cumulative aggregate metrics.
          7. Run evidence-first pattern detection across timeline.
          8. Persist new monitoring events and update profile timestamp.
          9. Return concise scan telemetry.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Resolve Profile
        existing_profile = self.storage.get_profile(url_or_id)
        if existing_profile:
            resolved = resolve_profile_url(existing_profile.profile_url)
            profile_url = existing_profile.profile_url
            profile_id = existing_profile.id
            profile = existing_profile
        else:
            resolved = resolve_profile_url(url_or_id)
            profile_url = resolved.profile_url
            clean_handle = resolved.handle.replace("@", "").replace("/", "_").lower()
            profile_id = f"prof_{resolved.platform.lower()}_{clean_handle}"
            profile = self.storage.get_or_create_profile(
                profile_id=profile_id,
                platform=resolved.platform,
                handle=resolved.handle,
                profile_url=profile_url,
                created_at=now_iso,
            )

        # Step 2: Fetch Public Activity
        try:
            fetch_res = fetch_profile_activity(
                platform=resolved.platform,
                handle=resolved.handle,
                profile_url=profile_url,
                limit=limit,
                demo_fallback=demo_fallback,
            )
            if len(fetch_res) == 4:
                raw_items, is_demo, fetch_msg, audit = fetch_res
            else:
                raw_items, is_demo, fetch_msg = fetch_res
                audit = None
        except Exception as e:
            self.storage.update_profile_state(profile_id, MonitoringState.ERROR, error_message=str(e))
            raise

        discovered_count = len(raw_items)

        # Fallback audit construction if connector did not provide one
        if audit is None:
            ts_list = [it.timestamp for it in raw_items if it.timestamp]
            audit = RetrievalAudit(
                requested_limit=limit,
                retrieved_count=discovered_count,
                earliest_timestamp=min(ts_list) if ts_list else None,
                latest_timestamp=max(ts_list) if ts_list else None,
                is_demo=is_demo,
                retrieval_limitations=fetch_msg or "Standard historical public activity retrieval.",
                rate_limit_encountered=False,
            )

        # Step 3: Identify New Items (Deduplication)
        discovered_ids = [item.item_id for item in raw_items]
        existing_ids = self.storage.get_existing_item_ids(profile_id, discovered_ids)
        new_items = [item for item in raw_items if item.item_id not in existing_ids]

        skipped_count = discovered_count - len(new_items)
        analyzed_count = len(new_items)

        new_observations: List[ObservationRecord] = []
        batch_safe = 0
        batch_review = 0
        batch_high_risk = 0

        # Step 4: Execute Existing Moderation Pipeline on New Items
        if new_items:
            server = self.model_server
            from src.model import predict_batch_muril
            from src.moderation import RiskLevel, classify_risk, moderate_comment_with_context

            texts = [item.text for item in new_items]
            if getattr(server, "muril_model", None) is not None or not hasattr(server, "predict_batch"):
                predictions_list = predict_batch_muril(
                    texts=texts,
                    model=server.muril_model,
                    tokenizer=server.muril_tokenizer,
                    labels=server.labels,
                    device=server.device,
                    batch_size=32,
                    thresholds=server.thresholds,
                )
            else:
                predictions_list = server.predict_batch(
                    texts=texts,
                    batch_size=32,
                )

            for idx, item in enumerate(new_items):
                scores, max_toxic_score, intensity, is_toxic_flags = predictions_list[idx]

                if enable_jev and server.jev_provider:
                    mod_res = moderate_comment_with_context(
                        text=item.text,
                        predictions=scores,
                        is_toxic_flags=is_toxic_flags,
                        thresholds=server.thresholds,
                        jev_provider=server.jev_provider,
                    )
                    final_risk = mod_res.final_risk_level.value
                    explanation = mod_res.final_explanation
                    jev_invoked = mod_res.jev_invoked
                    jev_status = mod_res.jev_status
                    human_review = mod_res.human_review_required
                    base_risk_str = mod_res.base_risk_level.value
                else:
                    base_risk, base_icon, base_expl = classify_risk(
                        predictions=scores,
                        is_toxic_flags=is_toxic_flags,
                        thresholds=server.thresholds,
                    )
                    final_risk = base_risk.value
                    explanation = base_expl
                    jev_invoked = False
                    jev_status = "BYPASSED_CLEAR_CASE"
                    human_review = (base_risk == RiskLevel.REVIEW)
                    base_risk_str = base_risk.value

                if final_risk == "High Risk":
                    batch_high_risk += 1
                elif final_risk == "Review":
                    batch_review += 1
                else:
                    batch_safe += 1

                new_observations.append(
                    ObservationRecord(
                        id=f"obs_{profile_id}_{uuid.uuid4().hex[:10]}",
                        profile_id=profile_id,
                        platform_item_id=item.item_id,
                        permalink=item.permalink,
                        timestamp=item.timestamp,
                        text=item.text,
                        media_url=item.media_url,
                        activity_type=item.activity_type,
                        language=detect_language(item.text),
                        scores={k: round(float(v), 4) for k, v in scores.items()},
                        base_risk=base_risk_str,
                        jev_invoked=jev_invoked,
                        jev_status=jev_status,
                        final_risk_level=final_risk,
                        human_review_required=human_review,
                        final_explanation=explanation,
                        analyzed_at=now_iso,
                    )
                )

            # Step 5: Persist New Observations
            self.storage.save_observations(new_observations)

        # Step 6: Recalculate Cumulative Aggregates
        all_observations = self.storage.get_all_observations_for_profile(profile_id)
        aggregates = self.calculate_aggregate_metrics(all_observations)

        # Step 7 & 8: Pattern Detection & Event Persistence
        server = self.model_server
        detected_patterns = detect_profile_patterns(
            profile_id=profile_id,
            observations=all_observations,
            thresholds=server.thresholds if server else None,
        )
        self.storage.save_events(detected_patterns)

        # Step 9: Update Profile Last Scanned Timestamp
        self.storage.update_profile_last_scanned(profile_id, now_iso)
        updated_profile = self.storage.get_profile(profile_id) or profile

        # Representative evidence: prioritized by high-risk, review, and toxicity magnitude
        sorted_evidence = sorted(
            all_observations,
            key=lambda o: (
                o.final_risk_level == "High Risk",
                o.final_risk_level == "Review",
                max(o.scores.values()) if o.scores else 0.0,
            ),
            reverse=True,
        )[:10]

        dossier = ProfileDossierReport(
            profile=updated_profile,
            retrieval_audit=audit,
            activity_risk=aggregates.profile_activity_risk,
            risk_explanation=(
                f"Observed public activity presents {aggregates.toxicity_rate:.1f}% toxicity rate and "
                f"{aggregates.high_risk_rate:.1f}% high-risk classification across {aggregates.total_observations} historical items. "
                "Classifications strictly evaluate observed public messages and do not represent psychological traits or future actions."
            ),
            aggregates=aggregates,
            events=detected_patterns,
            recent_observations=all_observations[:25],
            representative_evidence=sorted_evidence,
        )

        return ScanSummaryResponse(
            profile=updated_profile,
            discovered=discovered_count,
            new=analyzed_count,
            skipped=skipped_count,
            analyzed=analyzed_count,
            safe=batch_safe,
            review=batch_review,
            high_risk=batch_high_risk,
            aggregates=aggregates,
            events_detected=detected_patterns,
            is_demo=is_demo,
            message=fetch_msg,
            retrieval_audit=audit,
            dossier=dossier,
        )

    # ── Analytics & Aggregate Calculations ───────────────────────

    @staticmethod
    def calculate_aggregate_metrics(observations: List[ObservationRecord]) -> AggregateMetrics:
        """Deterministic calculations for activity volume, risk distribution, category breakdown, language, and timeline."""
        if not observations:
            return AggregateMetrics()

        total = len(observations)
        safe_cnt = sum(1 for o in observations if o.final_risk_level == "Safe")
        review_cnt = sum(1 for o in observations if o.final_risk_level == "Review")
        high_risk_cnt = sum(1 for o in observations if o.final_risk_level == "High Risk")

        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        sum_scores: Dict[str, float] = defaultdict(float)
        flag_counts: Dict[str, int] = defaultdict(int)

        frozen_th = {
            "toxic": 0.45,
            "severe_toxic": 0.62,
            "obscene": 0.65,
            "threat": 0.59,
            "insult": 0.61,
            "identity_hate": 0.60,
        }

        # Language distribution
        lang_dist: Dict[str, int] = defaultdict(int)

        # Date and period buckets
        date_counts: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "safe": 0,
            "review": 0,
            "high_risk": 0,
            "scores_sum": defaultdict(float),
        })
        daily_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "high_risk": 0})

        act_type_counts: Dict[str, Dict[str, Any]] = {
            "post": {"total": 0, "safe": 0, "review": 0, "high_risk": 0, "toxicity_rate": 0.0},
            "comment": {"total": 0, "safe": 0, "review": 0, "high_risk": 0, "toxicity_rate": 0.0},
            "reply": {"total": 0, "safe": 0, "review": 0, "high_risk": 0, "toxicity_rate": 0.0},
        }

        total_toxic_score_sum = 0.0

        for obs in observations:
            # Language
            lang = obs.language or "English"
            lang_dist[lang] += 1

            # Activity type
            a_type = obs.activity_type if obs.activity_type in act_type_counts else "comment"
            act_type_counts[a_type]["total"] += 1
            if obs.final_risk_level == "Safe":
                act_type_counts[a_type]["safe"] += 1
            elif obs.final_risk_level == "Review":
                act_type_counts[a_type]["review"] += 1
            elif obs.final_risk_level == "High Risk":
                act_type_counts[a_type]["high_risk"] += 1

            # Scores
            t_score = float(obs.scores.get("toxic", 0.0))
            total_toxic_score_sum += t_score

            for l in labels:
                sc = float(obs.scores.get(l, 0.0))
                sum_scores[l] += sc
                if sc >= frozen_th.get(l, 0.5):
                    flag_counts[l] += 1

            # Daily bucket for legacy activity_by_date
            day_str = obs.timestamp[:10] if len(obs.timestamp) >= 10 else "Unknown"
            daily_counts[day_str]["count"] += 1
            if obs.final_risk_level == "High Risk":
                daily_counts[day_str]["high_risk"] += 1

            # Monthly bucket for timeline (e.g. "2026-08")
            period_key = obs.timestamp[:7] if len(obs.timestamp) >= 7 else "Unknown"
            date_counts[period_key]["count"] += 1
            if obs.final_risk_level == "High Risk":
                date_counts[period_key]["high_risk"] += 1
            elif obs.final_risk_level == "Review":
                date_counts[period_key]["review"] += 1
            else:
                date_counts[period_key]["safe"] += 1

            for l in labels:
                date_counts[period_key]["scores_sum"][l] += float(obs.scores.get(l, 0.0))

        # Calculate toxicity rate per activity type
        for k, v in act_type_counts.items():
            if v["total"] > 0:
                flagged_t = v["review"] + v["high_risk"]
                v["toxicity_rate"] = round((flagged_t / v["total"]) * 100.0, 2)

        avg_scores = {l: round(sum_scores[l] / total, 4) for l in labels}
        toxic_or_high_risk = sum(1 for o in observations if o.final_risk_level in ("High Risk", "Review"))
        toxicity_rate = round((toxic_or_high_risk / total) * 100.0, 2)
        high_risk_rate = round((high_risk_cnt / total) * 100.0, 2)
        review_rate = round((review_cnt / total) * 100.0, 2)
        safe_rate = round((safe_cnt / total) * 100.0, 2)
        avg_toxic_score = round(total_toxic_score_sum / total, 4)

        # Profile Activity Risk classification
        if high_risk_rate >= 30.0 or toxicity_rate >= 55.0:
            profile_risk = ProfileActivityRisk.HIGH
        elif high_risk_rate >= 15.0 or toxicity_rate >= 35.0:
            profile_risk = ProfileActivityRisk.ELEVATED
        elif high_risk_rate >= 5.0 or toxicity_rate >= 15.0:
            profile_risk = ProfileActivityRisk.MODERATE
        else:
            profile_risk = ProfileActivityRisk.LOW

        # Language percentages
        lang_percentages = {lang: round((cnt / total) * 100.0, 2) for lang, cnt in lang_dist.items()}

        # Dominant category & concentration
        dom_cat = max(flag_counts.items(), key=lambda x: x[1])[0] if flag_counts and any(flag_counts.values()) else (
            max(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else "toxic"
        )
        rep_rate = act_type_counts["reply"]["toxicity_rate"]
        post_rate = act_type_counts["post"]["toxicity_rate"]
        toxicity_concentration = {
            "dominant_category": dom_cat,
            "dominant_category_flags": flag_counts.get(dom_cat, 0),
            "reply_toxicity_rate": rep_rate,
            "post_toxicity_rate": post_rate,
            "reply_vs_post_difference": round(rep_rate - post_rate, 2),
        }

        # Historical Timeline
        historical_timeline: List[TemporalPeriodStat] = []
        for period, pdata in sorted(date_counts.items()):
            p_vol = pdata["count"]
            p_hr = pdata["high_risk"]
            p_rev = pdata["review"]
            p_safe = pdata["safe"]
            p_flagged = p_hr + p_rev
            p_dom = max(pdata["scores_sum"].items(), key=lambda x: x[1])[0] if pdata["scores_sum"] else "toxic"
            historical_timeline.append(
                TemporalPeriodStat(
                    period=period,
                    volume=p_vol,
                    safe_count=p_safe,
                    review_count=p_rev,
                    high_risk_count=p_hr,
                    toxicity_rate=round((p_flagged / p_vol) * 100.0, 2) if p_vol > 0 else 0.0,
                    high_risk_rate=round((p_hr / p_vol) * 100.0, 2) if p_vol > 0 else 0.0,
                    dominant_category=p_dom,
                )
            )

        legacy_activity_by_date = [
            {"date": d, "count": data["count"], "high_risk_count": data["high_risk"]}
            for d, data in sorted(daily_counts.items())
        ]

        return AggregateMetrics(
            total_observations=total,
            safe_count=safe_cnt,
            review_count=review_cnt,
            high_risk_count=high_risk_cnt,
            toxicity_rate=toxicity_rate,
            high_risk_rate=high_risk_rate,
            review_rate=review_rate,
            safe_rate=safe_rate,
            average_toxicity_score=avg_toxic_score,
            profile_activity_risk=profile_risk,
            risk_definition="Risk level of observed public activity only. Does not characterize person, intent, or future behavior.",
            average_scores=avg_scores,
            category_flag_counts=dict(flag_counts),
            language_distribution=dict(lang_dist),
            language_percentages=lang_percentages,
            activity_type_breakdown=act_type_counts,
            toxicity_concentration=toxicity_concentration,
            historical_timeline=historical_timeline,
            activity_by_date=legacy_activity_by_date,
        )

    # ── Queries & State Management ───────────────────────────────

    def get_profile_summary(self, profile_id: str) -> Optional[ProfileSummaryResponse]:
        """Fetch consolidated profile view including metrics, recent items, and patterns."""
        profile = self.storage.get_profile(profile_id)
        if not profile:
            return None

        all_obs = self.storage.get_all_observations_for_profile(profile_id)
        aggregates = self.calculate_aggregate_metrics(all_obs)
        recent_obs = self.storage.get_observations(profile_id, limit=25)
        events = self.storage.get_events(profile_id, limit=20)

        sorted_evidence = sorted(
            all_obs,
            key=lambda o: (
                o.final_risk_level == "High Risk",
                o.final_risk_level == "Review",
                max(o.scores.values()) if o.scores else 0.0,
            ),
            reverse=True,
        )[:10]

        ts_list = [o.timestamp for o in all_obs if o.timestamp]
        audit = RetrievalAudit(
            requested_limit=len(all_obs),
            retrieved_count=len(all_obs),
            earliest_timestamp=min(ts_list) if ts_list else None,
            latest_timestamp=max(ts_list) if ts_list else None,
            is_demo=any("demo" in o.platform_item_id.lower() for o in all_obs),
            retrieval_limitations="Persisted historical observation window.",
            rate_limit_encountered=False,
        )

        return ProfileSummaryResponse(
            profile=profile,
            aggregates=aggregates,
            recent_observations=recent_obs,
            events=events,
            retrieval_audit=audit,
            activity_risk=aggregates.profile_activity_risk,
            representative_evidence=sorted_evidence,
        )

    def get_profile_activity(
        self,
        profile_id: str,
        limit: int = 50,
        offset: int = 0,
        risk_filter: Optional[str] = None,
    ) -> List[ObservationRecord]:
        """Fetch paginated observations for audit and drilldown."""
        return self.storage.get_observations(
            profile_id=profile_id,
            limit=limit,
            offset=offset,
            risk_filter=risk_filter,
        )

    def get_profile_events(self, profile_id: str) -> List[PatternEventRecord]:
        """Fetch active detected pattern events for a profile."""
        return self.storage.get_events(profile_id)

    def set_monitoring_state(self, profile_id: str, state: MonitoringState) -> Optional[ProfileRecord]:
        """Update monitoring state (e.g. MONITORING, PAUSED)."""
        ok = self.storage.update_profile_state(profile_id, state)
        if not ok:
            return None
        return self.storage.get_profile(profile_id)

    def list_profiles(self) -> List[ProfileRecord]:
        """List all tracked profiles."""
        return self.storage.list_profiles()

    def compare_profiles(self, profile_id_a: str, profile_id_b: str) -> Optional[ProfileComparisonResponse]:
        """Compare observed activity metrics across two scanned profiles without personal characterization."""
        summary_a = self.get_profile_summary(profile_id_a)
        summary_b = self.get_profile_summary(profile_id_b)
        if not summary_a or not summary_b:
            return None

        agg_a = summary_a.aggregates
        agg_b = summary_b.aggregates

        delta_vol = agg_a.total_observations - agg_b.total_observations
        delta_tox = round(agg_a.toxicity_rate - agg_b.toxicity_rate, 2)
        delta_hr = round(agg_a.high_risk_rate - agg_b.high_risk_rate, 2)

        dom_a = agg_a.toxicity_concentration.get("dominant_category", "none")
        dom_b = agg_b.toxicity_concentration.get("dominant_category", "none")

        notes = [
            f"Observed activity volume: {summary_a.profile.handle} ({agg_a.total_observations} items) vs {summary_b.profile.handle} ({agg_b.total_observations} items). Delta: {delta_vol:+d}.",
            f"Toxicity rate: {agg_a.toxicity_rate:.1f}% vs {agg_b.toxicity_rate:.1f}%. Delta: {delta_tox:+.1f}%.",
            f"High-risk rate: {agg_a.high_risk_rate:.1f}% vs {agg_b.high_risk_rate:.1f}%. Delta: {delta_hr:+.1f}%.",
            f"Dominant toxicity category: '{dom_a}' ({summary_a.profile.handle}) vs '{dom_b}' ({summary_b.profile.handle}).",
        ]

        delta_metrics = {
            "volume_delta": delta_vol,
            "toxicity_rate_delta": delta_tox,
            "high_risk_rate_delta": delta_hr,
            "profile_a_dominant_category": dom_a,
            "profile_b_dominant_category": dom_b,
            "profile_a_languages": agg_a.language_percentages,
            "profile_b_languages": agg_b.language_percentages,
        }

        return ProfileComparisonResponse(
            profile_a=summary_a,
            profile_b=summary_b,
            delta_metrics=delta_metrics,
            comparative_notes=notes,
        )

