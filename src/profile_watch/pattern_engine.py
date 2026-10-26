"""
Deterministic, evidence-first pattern detection engine for Profile Watch.
Reports observable behavioral activity patterns with strict mathematical proof.
NEVER infers personality, morality, or psychological characteristics.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from src.profile_watch.models import (
    EventSeverity,
    ObservationRecord,
    PatternEventRecord,
    PatternType,
)


def detect_profile_patterns(
    profile_id: str,
    observations: List[ObservationRecord],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[PatternEventRecord]:
    """Analyze a chronological list of observations and detect active behavioral patterns.

    Parameters
    ----------
    profile_id : str
        The unique profile identifier.
    observations : List[ObservationRecord]
        All historical observations for this profile, sorted chronologically.
    thresholds : dict[str, float], optional
        Per-label classification thresholds (defaults to frozen production thresholds).

    Returns
    -------
    List[PatternEventRecord]
        Detected patterns supported by measurable counts.
    """
    if not observations:
        return []

    th = thresholds or {
        "toxic": 0.45,
        "severe_toxic": 0.62,
        "obscene": 0.65,
        "threat": 0.59,
        "insult": 0.61,
        "identity_hate": 0.60,
    }

    # Ensure chronological order (oldest to newest)
    sorted_obs = sorted(observations, key=lambda o: o.timestamp)
    total = len(sorted_obs)
    now_iso = datetime.now(timezone.utc).isoformat()
    detected_events: List[PatternEventRecord] = []

    # ── Pattern 0A: HIGH_RISK_FREQUENCY ──────────────────────────
    hr_items = [o for o in sorted_obs if o.final_risk_level == "High Risk"]
    hr_cnt = len(hr_items)
    if hr_cnt >= 2 or (total >= 4 and hr_cnt >= 1 and (hr_cnt / total) >= 0.20):
        sev = EventSeverity.CRITICAL if (hr_cnt / total) >= 0.40 else EventSeverity.HIGH
        detected_events.append(
            PatternEventRecord(
                id=f"pat_hrfreq_{profile_id}_{uuid.uuid4().hex[:8]}",
                profile_id=profile_id,
                pattern_type=PatternType.HIGH_RISK_FREQUENCY,
                title="High-Risk Interaction Frequency",
                description=f"{hr_cnt} of {total} analyzed items were classified HIGH RISK ({hr_cnt / total * 100:.1f}%).",
                evidence_count=hr_cnt,
                total_count=total,
                severity=sev,
                detected_at=now_iso,
            )
        )

    # ── Pattern 0B: REPEATED_CATEGORY ─────────────────────────────
    for cat_name, label_title in [
        ("identity_hate", "Identity-Targeted Toxicity"),
        ("threat", "Explicit Threat Signals"),
        ("insult", "Insulting Language"),
        ("obscene", "Obscene Content"),
        ("toxic", "General Toxicity"),
    ]:
        c_th = th.get(cat_name, 0.60)
        c_items = [o for o in sorted_obs if o.scores.get(cat_name, 0.0) >= c_th]
        c_cnt = len(c_items)
        if c_cnt >= 2:
            detected_events.append(
                PatternEventRecord(
                    id=f"pat_repcat_{cat_name}_{profile_id}_{uuid.uuid4().hex[:8]}",
                    profile_id=profile_id,
                    pattern_type=PatternType.REPEATED_CATEGORY,
                    title=f"Repeated Category: {label_title}",
                    description=f"{cat_name.replace('_', ' ').title()} predictions appeared repeatedly across the observed activity ({c_cnt} of {total} interactions).",
                    evidence_count=c_cnt,
                    total_count=total,
                    severity=EventSeverity.HIGH if c_cnt >= 3 else EventSeverity.MEDIUM,
                    detected_at=now_iso,
                )
            )
            break

    # ── Pattern 0C: REPLY_CONCENTRATION ───────────────────────────
    replies = [o for o in sorted_obs if o.activity_type == "reply"]
    non_replies = [o for o in sorted_obs if o.activity_type in ("post", "comment")]
    if len(replies) >= 2 and len(non_replies) >= 1:
        rep_flagged = sum(1 for o in replies if o.final_risk_level in ("High Risk", "Review"))
        non_flagged = sum(1 for o in non_replies if o.final_risk_level in ("High Risk", "Review"))
        rep_rate = rep_flagged / len(replies)
        non_rate = non_flagged / len(non_replies)
        if (rep_rate - non_rate) >= 0.15 and rep_flagged >= 2:
            detected_events.append(
                PatternEventRecord(
                    id=f"pat_reply_conc_{profile_id}_{uuid.uuid4().hex[:8]}",
                    profile_id=profile_id,
                    pattern_type=PatternType.REPLY_CONCENTRATION,
                    title="Reply-Concentrated Toxicity",
                    description=(
                        f"Toxic predictions were substantially more frequent in replies ({rep_rate * 100:.1f}%, "
                        f"{rep_flagged} of {len(replies)}) than original posts/comments ({non_rate * 100:.1f}%, "
                        f"{non_flagged} of {len(non_replies)})."
                    ),
                    evidence_count=rep_flagged,
                    total_count=len(replies),
                    severity=EventSeverity.HIGH if rep_rate >= 0.5 else EventSeverity.MEDIUM,
                    detected_at=now_iso,
                )
            )

    # ── Pattern 0D: TEMPORAL_CONCENTRATION ────────────────────────
    if total >= 6:
        split_point = total // 2
        earlier = sorted_obs[:split_point]
        later = sorted_obs[split_point:]
        later_hr = sum(1 for o in later if o.final_risk_level == "High Risk")
        earlier_hr = sum(1 for o in earlier if o.final_risk_level == "High Risk")
        total_hr = later_hr + earlier_hr
        if total_hr >= 2 and later_hr >= 2 and (later_hr / total_hr) >= 0.65:
            detected_events.append(
                PatternEventRecord(
                    id=f"pat_temp_conc_{profile_id}_{uuid.uuid4().hex[:8]}",
                    profile_id=profile_id,
                    pattern_type=PatternType.TEMPORAL_CONCENTRATION,
                    title="Temporal Risk Concentration",
                    description=(
                        f"High-risk activity was concentrated in the most recent observed period: "
                        f"{later_hr} of {total_hr} high-risk items ({later_hr / total_hr * 100:.1f}%) "
                        f"occurred in the latest observation period."
                    ),
                    evidence_count=later_hr,
                    total_count=len(later),
                    severity=EventSeverity.HIGH,
                    detected_at=now_iso,
                )
            )

    # ── Pattern 0E: LANGUAGE_PATTERN ──────────────────────────────
    lang_counts: Dict[str, int] = {}
    for o in sorted_obs:
        lang_counts[o.language] = lang_counts.get(o.language, 0) + 1

    for lang, cnt in lang_counts.items():
        if "hinglish" in lang.lower() or "hindi" in lang.lower():
            pct = (cnt / total) * 100.0
            if pct >= 20.0 and cnt >= 2:
                detected_events.append(
                    PatternEventRecord(
                        id=f"pat_lang_{profile_id}_{uuid.uuid4().hex[:8]}",
                        profile_id=profile_id,
                        pattern_type=PatternType.LANGUAGE_PATTERN,
                        title=f"Language Activity Pattern: {lang}",
                        description=f"{lang} represented {pct:.1f}% of analyzed activity.",
                        evidence_count=cnt,
                        total_count=total,
                        severity=EventSeverity.LOW,
                        detected_at=now_iso,
                    )
                )
                break

    # ── Pattern 1: REPEATED_INSULT ────────────────────────────────
    insult_th = th.get("insult", 0.61)
    insult_items = [o for o in sorted_obs if o.scores.get("insult", 0.0) >= insult_th]
    insult_cnt = len(insult_items)
    if insult_cnt >= 2 and (insult_cnt >= 3 or total <= 8):
        sev = EventSeverity.HIGH if (insult_cnt / total) >= 0.4 else EventSeverity.MEDIUM
        detected_events.append(
            PatternEventRecord(
                id=f"pat_insult_{profile_id}_{uuid.uuid4().hex[:8]}",
                profile_id=profile_id,
                pattern_type=PatternType.REPEATED_INSULT,
                title="Repeated Insulting Language",
                description=(
                    f"{insult_cnt} of {total} analyzed public interactions contained elevated insult "
                    f"scores exceeding the {insult_th:.2f} decision threshold."
                ),
                evidence_count=insult_cnt,
                total_count=total,
                severity=sev,
                detected_at=now_iso,
            )
        )

    # ── Pattern 2: REPEATED_IDENTITY_HATE ────────────────────────
    hate_th = th.get("identity_hate", 0.60)
    hate_items = [o for o in sorted_obs if o.scores.get("identity_hate", 0.0) >= hate_th]
    hate_cnt = len(hate_items)
    if hate_cnt >= 2:
        sev = EventSeverity.CRITICAL if hate_cnt >= 3 else EventSeverity.HIGH
        detected_events.append(
            PatternEventRecord(
                id=f"pat_hate_{profile_id}_{uuid.uuid4().hex[:8]}",
                profile_id=profile_id,
                pattern_type=PatternType.REPEATED_IDENTITY_HATE,
                title="Recurring Identity-Targeted Toxicity",
                description=(
                    f"{hate_cnt} of {total} analyzed public interactions met or exceeded the "
                    f"identity_hate classification threshold ({hate_th:.2f})."
                ),
                evidence_count=hate_cnt,
                total_count=total,
                severity=sev,
                detected_at=now_iso,
            )
        )

    # ── Pattern 3: REPEATED_THREAT ────────────────────────────────
    threat_th = th.get("threat", 0.59)
    threat_items = [o for o in sorted_obs if o.scores.get("threat", 0.0) >= threat_th]
    threat_cnt = len(threat_items)
    if threat_cnt >= 2:
        detected_events.append(
            PatternEventRecord(
                id=f"pat_threat_{profile_id}_{uuid.uuid4().hex[:8]}",
                profile_id=profile_id,
                pattern_type=PatternType.REPEATED_THREAT,
                title="Recurring Explicit Threat Indicators",
                description=(
                    f"{threat_cnt} of {total} analyzed public interactions contained severe threat "
                    f"signals exceeding the {threat_th:.2f} threshold."
                ),
                evidence_count=threat_cnt,
                total_count=total,
                severity=EventSeverity.CRITICAL,
                detected_at=now_iso,
            )
        )

    # ── Pattern 4: TOXICITY_SPIKE ────────────────────────────────
    # Compare recent window against historical baseline
    if total >= 6:
        split_idx = max(total - max(4, total // 3), 2)
        baseline_obs = sorted_obs[:split_idx]
        recent_obs = sorted_obs[split_idx:]

        base_hr = sum(1 for o in baseline_obs if o.final_risk_level == "High Risk")
        recent_hr = sum(1 for o in recent_obs if o.final_risk_level == "High Risk")

        base_rate = (base_hr / len(baseline_obs)) if baseline_obs else 0.0
        recent_rate = (recent_hr / len(recent_obs)) if recent_obs else 0.0

        # Trigger if recent rate is materially higher (>= 2x baseline and >= 35%, or baseline=0 and recent >= 40%)
        is_spike = False
        if recent_hr >= 2:
            if base_rate > 0.0 and recent_rate >= (1.8 * base_rate) and recent_rate >= 0.35:
                is_spike = True
            elif base_rate == 0.0 and recent_rate >= 0.40:
                is_spike = True

        if is_spike:
            detected_events.append(
                PatternEventRecord(
                    id=f"pat_spike_{profile_id}_{uuid.uuid4().hex[:8]}",
                    profile_id=profile_id,
                    pattern_type=PatternType.TOXICITY_SPIKE,
                    title="Toxicity Rate Spike Above Baseline",
                    description=(
                        f"Recent activity displayed a {recent_rate * 100:.1f}% high-risk rate "
                        f"({recent_hr} of {len(recent_obs)} interactions), markedly exceeding the "
                        f"historical baseline of {base_rate * 100:.1f}% ({base_hr} of {len(baseline_obs)} interactions)."
                    ),
                    evidence_count=recent_hr,
                    total_count=len(recent_obs),
                    severity=EventSeverity.HIGH,
                    detected_at=now_iso,
                )
            )

    # ── Pattern 5: ESCALATION ────────────────────────────────────
    # Test if high-risk ratio is monotonically or strongly increasing over chronological segments
    if total >= 8:
        mid = total // 2
        first_half = sorted_obs[:mid]
        second_half = sorted_obs[mid:]

        h1_hr = sum(1 for o in first_half if o.final_risk_level == "High Risk")
        h2_hr = sum(1 for o in second_half if o.final_risk_level == "High Risk")

        r1 = h1_hr / len(first_half)
        r2 = h2_hr / len(second_half)

        if (r2 - r1) >= 0.25 and h2_hr >= 3:
            detected_events.append(
                PatternEventRecord(
                    id=f"pat_esc_{profile_id}_{uuid.uuid4().hex[:8]}",
                    profile_id=profile_id,
                    pattern_type=PatternType.ESCALATION,
                    title="Sequential Toxicity Escalation",
                    description=(
                        f"High-risk interaction frequency escalated from {r1 * 100:.1f}% in earlier observations "
                        f"({h1_hr} of {len(first_half)}) to {r2 * 100:.1f}% in subsequent observations "
                        f"({h2_hr} of {len(second_half)})."
                    ),
                    evidence_count=h2_hr,
                    total_count=len(second_half),
                    severity=EventSeverity.HIGH,
                    detected_at=now_iso,
                )
            )

    # ── Pattern 6: ACTIVITY_BURST ────────────────────────────────
    # Check for concentrated bursts (e.g. >= 4 items within a 2-hour window)
    parsed_timestamps = []
    for o in sorted_obs:
        try:
            # Handle ISO string with or without Z
            clean_ts = o.timestamp.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_ts)
            parsed_timestamps.append(dt)
        except Exception:
            continue

    if len(parsed_timestamps) >= 4:
        for i in range(len(parsed_timestamps) - 3):
            window_items = parsed_timestamps[i : i + 4]
            diff_hours = (window_items[-1] - window_items[0]).total_seconds() / 3600.0
            if 0 <= diff_hours <= 2.5:
                # Count total items within this burst window
                burst_end = window_items[-1]
                burst_start = window_items[0]
                burst_count = sum(1 for t in parsed_timestamps if burst_start <= t <= burst_end)
                date_str = burst_start.strftime("%Y-%m-%d")
                detected_events.append(
                    PatternEventRecord(
                        id=f"pat_burst_{profile_id}_{uuid.uuid4().hex[:8]}",
                        profile_id=profile_id,
                        pattern_type=PatternType.ACTIVITY_BURST,
                        title="High-Density Activity Burst",
                        description=(
                            f"Activity burst observed: {burst_count} public interactions were posted "
                            f"within {diff_hours:.1f} hours on {date_str}."
                        ),
                        evidence_count=burst_count,
                        total_count=total,
                        severity=EventSeverity.MEDIUM,
                        detected_at=now_iso,
                    )
                )
                break  # Record one consolidated burst event per scan cycle

    return detected_events
