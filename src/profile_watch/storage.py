"""
SQLite persistence layer for Profile Watch.
Uses Python's built-in sqlite3.
"""

from contextlib import contextmanager
import json
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.profile_watch.models import (
    MonitoringState,
    ObservationRecord,
    PatternEventRecord,
    PatternType,
    EventSeverity,
    ProfileRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "profile_watch.db"


class ProfileWatchStorage:
    """Thread-safe SQLite storage for profiles, observations, and monitoring events."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize tables and indexes with exact constraints."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Profiles Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    handle TEXT NOT NULL,
                    profile_url TEXT NOT NULL UNIQUE,
                    monitoring_state TEXT NOT NULL DEFAULT 'NOT_MONITORED',
                    created_at TEXT NOT NULL,
                    last_scanned_at TEXT,
                    error_message TEXT
                );
            """)

            # 2. Observations Table (with UNIQUE constraint on profile_id + platform_item_id)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    platform_item_id TEXT NOT NULL,
                    permalink TEXT,
                    timestamp TEXT NOT NULL,
                    text TEXT NOT NULL,
                    media_url TEXT,
                    activity_type TEXT DEFAULT 'comment',
                    language TEXT DEFAULT 'English',
                    scores TEXT NOT NULL,
                    base_risk TEXT NOT NULL,
                    jev_invoked INTEGER NOT NULL DEFAULT 0,
                    jev_status TEXT NOT NULL DEFAULT '',
                    final_risk_level TEXT NOT NULL,
                    human_review_required INTEGER NOT NULL DEFAULT 0,
                    final_explanation TEXT NOT NULL,
                    analyzed_at TEXT NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE,
                    UNIQUE(profile_id, platform_item_id)
                );
            """)

            # Safe schema evolution for existing databases
            try:
                cursor.execute("ALTER TABLE observations ADD COLUMN activity_type TEXT DEFAULT 'comment';")
            except Exception:
                pass

            # 3. Monitoring Events Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_events (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
                );
            """)

            # Indexes for high performance querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_profile_ts ON observations(profile_id, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_profile_analyzed ON observations(profile_id, analyzed_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_profile_ts ON monitoring_events(profile_id, detected_at);")
            conn.commit()

    # ── Profile Operations ────────────────────────────────────────

    def get_or_create_profile(
        self,
        profile_id: str,
        platform: str,
        handle: str,
        profile_url: str,
        created_at: str,
    ) -> ProfileRecord:
        """Retrieve existing profile or create a new one idempotently."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM profiles WHERE profile_url = ?;", (profile_url,))
            row = cursor.fetchone()
            if row:
                return ProfileRecord(**dict(row))

            cursor.execute(
                """
                INSERT INTO profiles (id, platform, handle, profile_url, monitoring_state, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (profile_id, platform, handle, profile_url, MonitoringState.NOT_MONITORED.value, created_at),
            )
            conn.commit()
            return ProfileRecord(
                id=profile_id,
                platform=platform,
                handle=handle,
                profile_url=profile_url,
                monitoring_state=MonitoringState.NOT_MONITORED,
                created_at=created_at,
            )

    def get_profile(self, profile_id: str) -> Optional[ProfileRecord]:
        """Fetch profile by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM profiles WHERE id = ?;", (profile_id,))
            row = cursor.fetchone()
            if row:
                return ProfileRecord(**dict(row))
        return None

    def get_profile_by_url(self, profile_url: str) -> Optional[ProfileRecord]:
        """Fetch profile by URL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM profiles WHERE profile_url = ?;", (profile_url,))
            row = cursor.fetchone()
            if row:
                return ProfileRecord(**dict(row))
        return None

    def list_profiles(self) -> List[ProfileRecord]:
        """List all tracked profiles."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM profiles ORDER BY created_at DESC;")
            return [ProfileRecord(**dict(row)) for row in cursor.fetchall()]

    def update_profile_state(
        self,
        profile_id: str,
        state: MonitoringState,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update monitoring state of a profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE profiles
                SET monitoring_state = ?, error_message = ?
                WHERE id = ?;
                """,
                (state.value, error_message, profile_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_profile_last_scanned(
        self,
        profile_id: str,
        scanned_at: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """Record the timestamp of a successful or completed scan."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE profiles
                SET last_scanned_at = ?, error_message = ?
                WHERE id = ?;
                """,
                (scanned_at, error_message, profile_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    # ── Observation Operations & Deduplication ───────────────────

    def get_existing_item_ids(self, profile_id: str, item_ids: List[str]) -> Set[str]:
        """Return the subset of item_ids that already exist in observations for this profile."""
        if not item_ids:
            return set()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in item_ids)
            query = f"""
                SELECT platform_item_id FROM observations
                WHERE profile_id = ? AND platform_item_id IN ({placeholders});
            """
            cursor.execute(query, [profile_id] + item_ids)
            return {row["platform_item_id"] for row in cursor.fetchall()}

    def save_observations(self, observations: List[ObservationRecord]) -> int:
        """Insert new observations using INSERT OR IGNORE for strict idempotency."""
        if not observations:
            return 0
        inserted_count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for obs in observations:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO observations (
                        id, profile_id, platform_item_id, permalink, timestamp,
                        text, media_url, activity_type, language, scores, base_risk,
                        jev_invoked, jev_status, final_risk_level,
                        human_review_required, final_explanation, analyzed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        obs.id,
                        obs.profile_id,
                        obs.platform_item_id,
                        obs.permalink,
                        obs.timestamp,
                        obs.text,
                        obs.media_url,
                        getattr(obs, "activity_type", "comment") or "comment",
                        obs.language,
                        json.dumps(obs.scores),
                        obs.base_risk,
                        1 if obs.jev_invoked else 0,
                        obs.jev_status,
                        obs.final_risk_level,
                        1 if obs.human_review_required else 0,
                        obs.final_explanation,
                        obs.analyzed_at,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted_count += 1
            conn.commit()
        return inserted_count

    def get_observations(
        self,
        profile_id: str,
        limit: int = 100,
        offset: int = 0,
        risk_filter: Optional[str] = None,
    ) -> List[ObservationRecord]:
        """Retrieve paginated observations for a profile ordered chronologically descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM observations WHERE profile_id = ?"
            params: List[Any] = [profile_id]
            if risk_filter and risk_filter.upper() != "ALL":
                query += " AND final_risk_level = ?"
                params.append(risk_filter)
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
            params.extend([limit, offset])

            cursor.execute(query, params)
            records = []
            for row in cursor.fetchall():
                d = dict(row)
                d["scores"] = json.loads(d["scores"])
                d["jev_invoked"] = bool(d["jev_invoked"])
                d["human_review_required"] = bool(d["human_review_required"])
                d["activity_type"] = d.get("activity_type") or "comment"
                records.append(ObservationRecord(**d))
            return records

    def get_all_observations_for_profile(self, profile_id: str) -> List[ObservationRecord]:
        """Fetch all observations for metric calculations and pattern detection."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM observations WHERE profile_id = ? ORDER BY timestamp ASC;",
                (profile_id,),
            )
            records = []
            for row in cursor.fetchall():
                d = dict(row)
                d["scores"] = json.loads(d["scores"])
                d["jev_invoked"] = bool(d["jev_invoked"])
                d["human_review_required"] = bool(d["human_review_required"])
                d["activity_type"] = d.get("activity_type") or "comment"
                records.append(ObservationRecord(**d))
            return records

    # ── Monitoring Events Operations ─────────────────────────────

    def save_events(self, events: List[PatternEventRecord]) -> int:
        """Save newly detected behavioral pattern events."""
        if not events:
            return 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for ev in events:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO monitoring_events (
                        id, profile_id, pattern_type, title, description,
                        evidence_count, total_count, severity, detected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        ev.id,
                        ev.profile_id,
                        ev.pattern_type.value,
                        ev.title,
                        ev.description,
                        ev.evidence_count,
                        ev.total_count,
                        ev.severity.value,
                        ev.detected_at,
                    ),
                )
            conn.commit()
            return len(events)

    def get_events(self, profile_id: str, limit: int = 50) -> List[PatternEventRecord]:
        """Retrieve detected pattern events for a profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM monitoring_events
                WHERE profile_id = ?
                ORDER BY detected_at DESC
                LIMIT ?;
                """,
                (profile_id, limit),
            )
            return [
                PatternEventRecord(
                    id=row["id"],
                    profile_id=row["profile_id"],
                    pattern_type=PatternType(row["pattern_type"]),
                    title=row["title"],
                    description=row["description"],
                    evidence_count=row["evidence_count"],
                    total_count=row["total_count"],
                    severity=EventSeverity(row["severity"]),
                    detected_at=row["detected_at"],
                )
                for row in cursor.fetchall()
            ]
