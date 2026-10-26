"""
Comprehensive unit and integration test suite for Profile Watch (Public Social Activity Monitoring).
Tests resolution, persistence, deduplication, idempotency, pattern detection, aggregates, and API contracts.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import app
from src.profile_watch.connectors import resolve_profile_url
from src.profile_watch.models import (
    EventSeverity,
    MonitoringState,
    ObservationRecord,
    PatternType,
    RawProfileItem,
)
from src.profile_watch.pattern_engine import detect_profile_patterns
from src.profile_watch.service import ProfileWatchService
from src.profile_watch.storage import ProfileWatchStorage


class TestProfileResolution(unittest.TestCase):
    """Test public profile resolution across Reddit, YouTube, and Twitter/X."""

    def test_reddit_url_formats(self):
        urls = [
            ("https://www.reddit.com/user/test_analyst", "Reddit", "test_analyst"),
            ("https://reddit.com/u/data_researcher", "Reddit", "data_researcher"),
            ("u/indic_coder", "Reddit", "indic_coder"),
        ]
        for url, exp_platform, exp_handle in urls:
            res = resolve_profile_url(url)
            self.assertEqual(res.platform, exp_platform)
            self.assertEqual(res.handle, exp_handle)
            self.assertTrue(res.profile_url.startswith("https://www.reddit.com/user/"))

    def test_youtube_url_formats(self):
        urls = [
            ("https://www.youtube.com/@ToxiscopeLab", "YouTube", "@ToxiscopeLab"),
            ("https://youtube.com/channel/UC1234567890123456789012", "YouTube", "UC1234567890123456789012"),
            ("https://youtube.com/c/AIEthicsChannel", "YouTube", "AIEthicsChannel"),
            ("@MultilingualCreator", "YouTube", "@MultilingualCreator"),
        ]
        for url, exp_platform, exp_handle in urls:
            res = resolve_profile_url(url)
            self.assertEqual(res.platform, exp_platform)
            self.assertEqual(res.handle, exp_handle)

    def test_twitter_url_formats(self):
        urls = [
            ("https://twitter.com/researcher_ai", "Twitter", "@researcher_ai"),
            ("https://x.com/nlp_security", "Twitter", "@nlp_security"),
        ]
        for url, exp_platform, exp_handle in urls:
            res = resolve_profile_url(url)
            self.assertEqual(res.platform, exp_platform)
            self.assertEqual(res.handle, exp_handle)

    def test_invalid_url_raises_value_error(self):
        invalid_inputs = [
            "",
            "   ",
            "https://facebook.com/someone",
            "https://linkedin.com/in/someone",
            "not a url at all",
        ]
        for bad_url in invalid_inputs:
            with self.assertRaises(ValueError):
                resolve_profile_url(bad_url)


class TestProfileWatchStorage(unittest.TestCase):
    """Test SQLite persistence layer, idempotency, and constraints."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test_pw.db"
        self.storage = ProfileWatchStorage(db_path=self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_profile_creation_and_idempotency(self):
        p1 = self.storage.get_or_create_profile(
            profile_id="prof_reddit_user1",
            platform="Reddit",
            handle="user1",
            profile_url="https://www.reddit.com/user/user1",
            created_at="2026-10-14T00:00:00Z",
        )
        self.assertEqual(p1.id, "prof_reddit_user1")
        self.assertEqual(p1.monitoring_state, MonitoringState.NOT_MONITORED)

        # Re-creating same profile URL returns existing without error
        p2 = self.storage.get_or_create_profile(
            profile_id="prof_reddit_user1_dup",
            platform="Reddit",
            handle="user1",
            profile_url="https://www.reddit.com/user/user1",
            created_at="2026-10-14T01:00:00Z",
        )
        self.assertEqual(p2.id, "prof_reddit_user1")

    def test_observation_unique_constraint_and_deduplication(self):
        self.storage.get_or_create_profile(
            profile_id="prof_1",
            platform="Reddit",
            handle="user1",
            profile_url="https://www.reddit.com/user/user1",
            created_at="2026-10-14T00:00:00Z",
        )

        obs1 = ObservationRecord(
            id="obs_1",
            profile_id="prof_1",
            platform_item_id="item_100",
            permalink="https://reddit.com/1",
            timestamp="2026-10-14T00:00:00Z",
            text="First post text",
            scores={"toxic": 0.1, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.05, "insult": 0.05, "identity_hate": 0.05},
            base_risk="Safe",
            final_risk_level="Safe",
            final_explanation="All safe",
            analyzed_at="2026-10-14T00:05:00Z",
        )

        # First insert succeeds
        ins1 = self.storage.save_observations([obs1])
        self.assertEqual(ins1, 1)

        # Check existing item IDs
        existing = self.storage.get_existing_item_ids("prof_1", ["item_100", "item_200"])
        self.assertEqual(existing, {"item_100"})

        # Inserting duplicate item_100 is ignored and does NOT duplicate
        obs_dup = ObservationRecord(
            id="obs_2_dup",
            profile_id="prof_1",
            platform_item_id="item_100",
            permalink="https://reddit.com/1",
            timestamp="2026-10-14T00:00:00Z",
            text="Duplicate content attempt",
            scores={"toxic": 0.9, "severe_toxic": 0.1, "obscene": 0.1, "threat": 0.1, "insult": 0.1, "identity_hate": 0.1},
            base_risk="High Risk",
            final_risk_level="High Risk",
            final_explanation="Duplicate",
            analyzed_at="2026-10-14T01:00:00Z",
        )
        ins2 = self.storage.save_observations([obs_dup])
        self.assertEqual(ins2, 0)

        # Total stored observations should still be 1
        stored = self.storage.get_all_observations_for_profile("prof_1")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].id, "obs_1")

    def test_state_transitions(self):
        self.storage.get_or_create_profile(
            profile_id="prof_state_test",
            platform="YouTube",
            handle="@test",
            profile_url="https://youtube.com/@test",
            created_at="2026-10-14T00:00:00Z",
        )
        self.storage.update_profile_state("prof_state_test", MonitoringState.MONITORING)
        p = self.storage.get_profile("prof_state_test")
        self.assertEqual(p.monitoring_state, MonitoringState.MONITORING)

        self.storage.update_profile_state("prof_state_test", MonitoringState.PAUSED)
        p = self.storage.get_profile("prof_state_test")
        self.assertEqual(p.monitoring_state, MonitoringState.PAUSED)


class TestPatternEngine(unittest.TestCase):
    """Test deterministic evidence-first pattern detection rules."""

    def _make_obs(self, idx: int, timestamp: str, risk: str, insult: float = 0.1, hate: float = 0.1, threat: float = 0.1):
        return ObservationRecord(
            id=f"o_{idx}",
            profile_id="prof_pat",
            platform_item_id=f"item_{idx}",
            permalink=f"https://reddit.com/{idx}",
            timestamp=timestamp,
            text=f"Sample text {idx}",
            scores={"toxic": 0.2, "severe_toxic": 0.05, "obscene": 0.05, "threat": threat, "insult": insult, "identity_hate": hate},
            base_risk=risk,
            final_risk_level=risk,
            final_explanation="Verdict",
            analyzed_at=timestamp,
        )

    def test_repeated_insult_pattern(self):
        obs = [
            self._make_obs(1, "2026-10-01T10:00:00Z", "High Risk", insult=0.75),
            self._make_obs(2, "2026-10-02T10:00:00Z", "High Risk", insult=0.68),
            self._make_obs(3, "2026-10-03T10:00:00Z", "High Risk", insult=0.72),
            self._make_obs(4, "2026-10-04T10:00:00Z", "Safe", insult=0.20),
            self._make_obs(5, "2026-10-05T10:00:00Z", "Safe", insult=0.15),
        ]
        patterns = detect_profile_patterns("prof_pat", obs)
        insult_pats = [p for p in patterns if p.pattern_type == PatternType.REPEATED_INSULT]
        self.assertEqual(len(insult_pats), 1)
        pat = insult_pats[0]
        self.assertEqual(pat.evidence_count, 3)
        self.assertEqual(pat.total_count, 5)
        self.assertIn("3 of 5 analyzed public interactions contained elevated insult scores", pat.description)
        # Ensure evidence-first language without personality claims
        self.assertNotIn("aggressive person", pat.description.lower())
        self.assertNotIn("personality", pat.description.lower())

    def test_repeated_identity_hate_and_threat(self):
        obs = [
            self._make_obs(1, "2026-10-01T10:00:00Z", "High Risk", hate=0.85, threat=0.70),
            self._make_obs(2, "2026-10-02T10:00:00Z", "High Risk", hate=0.80, threat=0.75),
            self._make_obs(3, "2026-10-03T10:00:00Z", "Safe", hate=0.10, threat=0.10),
        ]
        patterns = detect_profile_patterns("prof_pat", obs)
        types = {p.pattern_type for p in patterns}
        self.assertIn(PatternType.REPEATED_IDENTITY_HATE, types)
        self.assertIn(PatternType.REPEATED_THREAT, types)

    def test_toxicity_spike_pattern(self):
        # 6 baseline safe items followed by 3 recent high-risk items
        obs = [
            self._make_obs(1, "2026-10-01T10:00:00Z", "Safe"),
            self._make_obs(2, "2026-10-02T10:00:00Z", "Safe"),
            self._make_obs(3, "2026-10-03T10:00:00Z", "Safe"),
            self._make_obs(4, "2026-10-04T10:00:00Z", "Safe"),
            self._make_obs(5, "2026-10-05T10:00:00Z", "Safe"),
            self._make_obs(6, "2026-10-06T10:00:00Z", "Safe"),
            self._make_obs(7, "2026-10-07T10:00:00Z", "High Risk", insult=0.8),
            self._make_obs(8, "2026-10-08T10:00:00Z", "High Risk", threat=0.8),
        ]
        patterns = detect_profile_patterns("prof_pat", obs)
        types = {p.pattern_type for p in patterns}
        self.assertIn(PatternType.TOXICITY_SPIKE, types)
        spike = next(p for p in patterns if p.pattern_type == PatternType.TOXICITY_SPIKE)
        self.assertIn("Recent activity displayed a", spike.description)
        self.assertIn("historical baseline", spike.description)

    def test_activity_burst_pattern(self):
        # 4 items within 1 hour
        obs = [
            self._make_obs(1, "2026-10-10T14:00:00Z", "Safe"),
            self._make_obs(2, "2026-10-10T14:15:00Z", "Safe"),
            self._make_obs(3, "2026-10-10T14:30:00Z", "Safe"),
            self._make_obs(4, "2026-10-10T14:45:00Z", "Safe"),
        ]
        patterns = detect_profile_patterns("prof_pat", obs)
        burst_pats = [p for p in patterns if p.pattern_type == PatternType.ACTIVITY_BURST]
        self.assertEqual(len(burst_pats), 1)
        self.assertIn("Activity burst observed: 4 public interactions were posted within", burst_pats[0].description)


class TestAggregateMetricsCalculation(unittest.TestCase):
    """Test deterministic aggregate behavioral metrics."""

    def test_aggregates_with_empty_list(self):
        agg = ProfileWatchService.calculate_aggregate_metrics([])
        self.assertEqual(agg.total_observations, 0)
        self.assertEqual(agg.toxicity_rate, 0.0)

    def test_aggregates_with_varied_observations(self):
        obs = [
            ObservationRecord(
                id="1", profile_id="p", platform_item_id="i1", permalink="",
                timestamp="2026-10-10T10:00:00Z", text="Safe text",
                scores={"toxic": 0.1, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.05, "insult": 0.05, "identity_hate": 0.05},
                base_risk="Safe", final_risk_level="Safe", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="2", profile_id="p", platform_item_id="i2", permalink="",
                timestamp="2026-10-10T11:00:00Z", text="Review text",
                scores={"toxic": 0.42, "severe_toxic": 0.1, "obscene": 0.1, "threat": 0.1, "insult": 0.1, "identity_hate": 0.1},
                base_risk="Review", final_risk_level="Review", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="3", profile_id="p", platform_item_id="i3", permalink="",
                timestamp="2026-10-11T12:00:00Z", text="Toxic text",
                scores={"toxic": 0.85, "severe_toxic": 0.7, "obscene": 0.6, "threat": 0.65, "insult": 0.8, "identity_hate": 0.75},
                base_risk="High Risk", final_risk_level="High Risk", final_explanation="", analyzed_at=""
            ),
        ]
        agg = ProfileWatchService.calculate_aggregate_metrics(obs)
        self.assertEqual(agg.total_observations, 3)
        self.assertEqual(agg.safe_count, 1)
        self.assertEqual(agg.review_count, 1)
        self.assertEqual(agg.high_risk_count, 1)
        self.assertAlmostEqual(agg.high_risk_rate, 33.33, places=1)
        self.assertAlmostEqual(agg.toxicity_rate, 66.67, places=1)
        self.assertEqual(len(agg.activity_by_date), 2)
        self.assertAlmostEqual(agg.average_scores["toxic"], (0.1 + 0.42 + 0.85) / 3, places=3)


class TestScanExecutionAndDeduplication(unittest.TestCase):
    """Test full scan pipeline, duplicate skipping, and moderation reuse."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test_service.db"
        self.storage = ProfileWatchStorage(db_path=self.db_path)

        # Mock ModelServer so tests do not require 950MB weights in memory
        self.mock_model_server = MagicMock()
        self.mock_model_server.muril_model = MagicMock()
        self.mock_model_server.muril_tokenizer = MagicMock()
        self.mock_model_server.labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.mock_model_server.thresholds = {"toxic": 0.45, "severe_toxic": 0.62, "obscene": 0.65, "threat": 0.59, "insult": 0.61, "identity_hate": 0.60}
        self.mock_model_server.device = "cpu"
        self.mock_model_server.jev_provider = None

        self.service = ProfileWatchService(storage=self.storage, model_server=self.mock_model_server)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    @patch("src.profile_watch.service.fetch_profile_activity")
    @patch("src.model.predict_batch_muril")
    def test_first_scan_and_subsequent_duplicate_scan(self, mock_predict, mock_fetch):
        # 1. First Scan: 3 items discovered
        mock_fetch.return_value = (
            [
                RawProfileItem(item_id="item_1", text="Hello world", timestamp="2026-10-14T01:00:00Z", permalink="https://reddit.com/1"),
                RawProfileItem(item_id="item_2", text="You are an idiot", timestamp="2026-10-14T02:00:00Z", permalink="https://reddit.com/2"),
                RawProfileItem(item_id="item_3", text="Great project", timestamp="2026-10-14T03:00:00Z", permalink="https://reddit.com/3"),
            ],
            False,
            None,
        )

        mock_predict.return_value = [
            ({"toxic": 0.1, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.05, "insult": 0.05, "identity_hate": 0.05}, 0.1, "low", {}),
            ({"toxic": 0.85, "severe_toxic": 0.2, "obscene": 0.2, "threat": 0.1, "insult": 0.8, "identity_hate": 0.1}, 0.85, "high", {"toxic": True, "insult": True}),
            ({"toxic": 0.15, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.05, "insult": 0.05, "identity_hate": 0.05}, 0.15, "low", {}),
        ]

        res1 = self.service.scan_profile("https://www.reddit.com/user/scanner_test", limit=10)
        self.assertEqual(res1.discovered, 3)
        self.assertEqual(res1.new, 3)
        self.assertEqual(res1.skipped, 0)
        self.assertEqual(res1.analyzed, 3)
        self.assertEqual(res1.safe, 2)
        self.assertEqual(res1.high_risk, 1)

        # 2. Second Scan with same 3 items: must detect all 3 as skipped (idempotent!)
        res2 = self.service.scan_profile("https://www.reddit.com/user/scanner_test", limit=10)
        self.assertEqual(res2.discovered, 3)
        self.assertEqual(res2.new, 0)
        self.assertEqual(res2.skipped, 3)
        self.assertEqual(res2.analyzed, 0)
        # Total cumulative observations remains 3
        self.assertEqual(res2.aggregates.total_observations, 3)


class TestProfileWatchAPIContracts(unittest.TestCase):
    """Test FastAPI endpoints for Profile Watch."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("api.profile_watch_service.scan_profile")
    def test_post_scan_endpoint(self, mock_scan):
        from src.profile_watch.models import AggregateMetrics, ProfileRecord, ScanSummaryResponse

        mock_profile = ProfileRecord(
            id="prof_reddit_apitester",
            platform="Reddit",
            handle="apitester",
            profile_url="https://www.reddit.com/user/apitester",
            monitoring_state=MonitoringState.NOT_MONITORED,
            created_at="2026-10-14T00:00:00Z",
        )
        mock_scan.return_value = ScanSummaryResponse(
            profile=mock_profile,
            discovered=5,
            new=5,
            skipped=0,
            analyzed=5,
            safe=3,
            review=1,
            high_risk=1,
            aggregates=AggregateMetrics(total_observations=5, safe_count=3, review_count=1, high_risk_count=1),
            events_detected=[],
        )

        response = self.client.post("/api/profile-watch/scan", json={"url": "https://www.reddit.com/user/apitester"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["profile"]["id"], "prof_reddit_apitester")
        self.assertEqual(data["discovered"], 5)
        self.assertEqual(data["new"], 5)
        self.assertEqual(data["safe"], 3)

    def test_empty_url_returns_400(self):
        response = self.client.post("/api/profile-watch/scan", json={"url": ""})
        self.assertEqual(response.status_code, 400)

    @patch("api.profile_watch_service.storage.get_profile")
    def test_get_nonexistent_profile_returns_404(self, mock_get):
        mock_get.return_value = None
        response = self.client.get("/api/profile-watch/nonexistent_prof")
        self.assertEqual(response.status_code, 404)

    @patch("api.profile_watch_service.set_monitoring_state")
    def test_monitor_and_pause_endpoints(self, mock_set_state):
        from src.profile_watch.models import ProfileRecord

        mock_set_state.return_value = ProfileRecord(
            id="prof_active",
            platform="Reddit",
            handle="active_user",
            profile_url="https://www.reddit.com/user/active_user",
            monitoring_state=MonitoringState.MONITORING,
            created_at="2026-10-14T00:00:00Z",
        )
        r_mon = self.client.post("/api/profile-watch/prof_active/monitor")
        self.assertEqual(r_mon.status_code, 200)
        self.assertEqual(r_mon.json()["monitoring_state"], "MONITORING")

        mock_set_state.return_value.monitoring_state = MonitoringState.PAUSED
        r_pause = self.client.post("/api/profile-watch/prof_active/pause")
        self.assertEqual(r_pause.status_code, 200)
        self.assertEqual(r_pause.json()["monitoring_state"], "PAUSED")

    @patch("api.profile_watch_service.compare_profiles")
    def test_compare_profiles_endpoint(self, mock_compare):
        from src.profile_watch.models import (
            AggregateMetrics,
            ProfileActivityRisk,
            ProfileComparisonResponse,
            ProfileRecord,
            ProfileSummaryResponse,
        )

        p_a = ProfileRecord(id="p1", platform="Reddit", handle="user_a", profile_url="https://reddit.com/user/user_a", created_at="2026-10-14T00:00:00Z")
        p_b = ProfileRecord(id="p2", platform="Twitter", handle="user_b", profile_url="https://twitter.com/user_b", created_at="2026-10-14T00:00:00Z")
        sum_a = ProfileSummaryResponse(profile=p_a, aggregates=AggregateMetrics(total_observations=10, toxicity_rate=20.0), recent_observations=[], events=[])
        sum_b = ProfileSummaryResponse(profile=p_b, aggregates=AggregateMetrics(total_observations=15, toxicity_rate=40.0), recent_observations=[], events=[])

        mock_compare.return_value = ProfileComparisonResponse(
            profile_a=sum_a,
            profile_b=sum_b,
            delta_metrics={"volume_delta": -5, "toxicity_rate_delta": -20.0},
            comparative_notes=["Observed activity volume delta: -5."],
        )

        resp = self.client.get("/api/profile-watch/compare?profile_a=p1&profile_b=p2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["delta_metrics"]["volume_delta"], -5)
        self.assertEqual(len(data["comparative_notes"]), 1)


class TestHistoricalIntelligenceScannerExtended(unittest.TestCase):
    """Test language detection, new pattern types, profile activity risk, and retrieval audit."""

    def test_language_detection(self):
        from src.profile_watch.service import detect_language

        # Pure English
        self.assertEqual(detect_language("This is a purely English sentence analyzing neural benchmarks."), "English")

        # Devanagari Hindi
        self.assertEqual(detect_language("यह बहुत गलत और अस्वीकार्य बात है।"), "Hindi")

        # Hinglish Code-Mixed
        self.assertEqual(detect_language("bhai sach me dil jeet liya yaar, mast post banaya hai"), "Hinglish")
        self.assertEqual(detect_language("faltu me bakwaas mat karo chup chap raho"), "Hinglish")

    def test_new_behavioral_patterns(self):
        from src.profile_watch.models import ObservationRecord, PatternType, ProfileActivityRisk
        from src.profile_watch.pattern_engine import detect_profile_patterns

        obs = [
            ObservationRecord(
                id="o1", profile_id="prof_ext", platform_item_id="i1", permalink="",
                timestamp="2026-08-01T10:00:00Z", text="Standard post",
                activity_type="post", language="English",
                scores={"toxic": 0.1, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.05, "insult": 0.05, "identity_hate": 0.05},
                base_risk="Safe", final_risk_level="Safe", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="o2", profile_id="prof_ext", platform_item_id="i2", permalink="",
                timestamp="2026-08-02T10:00:00Z", text="Safe reply",
                activity_type="reply", language="English",
                scores={"toxic": 0.1, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.05, "insult": 0.05, "identity_hate": 0.05},
                base_risk="Safe", final_risk_level="Safe", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="o3", profile_id="prof_ext", platform_item_id="i3", permalink="",
                timestamp="2026-09-01T10:00:00Z", text="Toxic reply one",
                activity_type="reply", language="Hinglish",
                scores={"toxic": 0.85, "severe_toxic": 0.2, "obscene": 0.2, "threat": 0.1, "insult": 0.85, "identity_hate": 0.70},
                base_risk="High Risk", final_risk_level="High Risk", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="o4", profile_id="prof_ext", platform_item_id="i4", permalink="",
                timestamp="2026-09-02T10:00:00Z", text="Toxic reply two",
                activity_type="reply", language="Hinglish",
                scores={"toxic": 0.90, "severe_toxic": 0.3, "obscene": 0.2, "threat": 0.1, "insult": 0.80, "identity_hate": 0.65},
                base_risk="High Risk", final_risk_level="High Risk", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="o5", profile_id="prof_ext", platform_item_id="i5", permalink="",
                timestamp="2026-09-03T10:00:00Z", text="Third high risk reply",
                activity_type="reply", language="Hinglish",
                scores={"toxic": 0.88, "severe_toxic": 0.3, "obscene": 0.2, "threat": 0.1, "insult": 0.75, "identity_hate": 0.62},
                base_risk="High Risk", final_risk_level="High Risk", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="o6", profile_id="prof_ext", platform_item_id="i6", permalink="",
                timestamp="2026-09-04T10:00:00Z", text="Safe comment",
                activity_type="comment", language="English",
                scores={"toxic": 0.12, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.05, "insult": 0.05, "identity_hate": 0.05},
                base_risk="Safe", final_risk_level="Safe", final_explanation="", analyzed_at=""
            ),
        ]

        patterns = detect_profile_patterns("prof_ext", obs)
        types = {p.pattern_type for p in patterns}

        # Check that observable evidence patterns were detected
        self.assertIn(PatternType.HIGH_RISK_FREQUENCY, types)
        self.assertIn(PatternType.REPEATED_CATEGORY, types)
        self.assertIn(PatternType.REPLY_CONCENTRATION, types)
        self.assertIn(PatternType.TEMPORAL_CONCENTRATION, types)
        self.assertIn(PatternType.LANGUAGE_PATTERN, types)

        hr_pat = next(p for p in patterns if p.pattern_type == PatternType.HIGH_RISK_FREQUENCY)
        self.assertIn("3 of 6 analyzed items were classified HIGH RISK", hr_pat.description)

        lang_pat = next(p for p in patterns if p.pattern_type == PatternType.LANGUAGE_PATTERN)
        self.assertIn("Hinglish represented 50.0% of analyzed activity", lang_pat.description)

        reply_pat = next(p for p in patterns if p.pattern_type == PatternType.REPLY_CONCENTRATION)
        self.assertIn("frequent in replies", reply_pat.description)

        # Ensure no personal claims in descriptions
        for p in patterns:
            self.assertNotIn("is hateful", p.description.lower())
            self.assertNotIn("is toxic", p.description.lower())
            self.assertNotIn("aggressive personality", p.description.lower())

    def test_profile_activity_risk_and_aggregation(self):
        from src.profile_watch.models import ObservationRecord, ProfileActivityRisk
        from src.profile_watch.service import ProfileWatchService

        obs = [
            ObservationRecord(
                id="o1", profile_id="p", platform_item_id="1", permalink="",
                timestamp="2026-08-10T00:00:00Z", text="Safe 1", activity_type="post", language="English",
                scores={"toxic": 0.05, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.0},
                base_risk="Safe", final_risk_level="Safe", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="o2", profile_id="p", platform_item_id="2", permalink="",
                timestamp="2026-08-15T00:00:00Z", text="Safe 2", activity_type="comment", language="English",
                scores={"toxic": 0.08, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.0},
                base_risk="Safe", final_risk_level="Safe", final_explanation="", analyzed_at=""
            ),
        ]
        agg_safe = ProfileWatchService.calculate_aggregate_metrics(obs)
        self.assertEqual(agg_safe.profile_activity_risk, ProfileActivityRisk.LOW)
        self.assertIn("observed public activity only", agg_safe.risk_definition.lower())

        # Add high-risk items
        obs_high = obs + [
            ObservationRecord(
                id="o3", profile_id="p", platform_item_id="3", permalink="",
                timestamp="2026-09-10T00:00:00Z", text="High Risk", activity_type="reply", language="Hindi",
                scores={"toxic": 0.95, "severe_toxic": 0.8, "obscene": 0.5, "threat": 0.4, "insult": 0.8, "identity_hate": 0.7},
                base_risk="High Risk", final_risk_level="High Risk", final_explanation="", analyzed_at=""
            ),
            ObservationRecord(
                id="o4", profile_id="p", platform_item_id="4", permalink="",
                timestamp="2026-09-12T00:00:00Z", text="High Risk 2", activity_type="reply", language="Hinglish",
                scores={"toxic": 0.92, "severe_toxic": 0.7, "obscene": 0.5, "threat": 0.4, "insult": 0.85, "identity_hate": 0.75},
                base_risk="High Risk", final_risk_level="High Risk", final_explanation="", analyzed_at=""
            ),
        ]
        agg_high = ProfileWatchService.calculate_aggregate_metrics(obs_high)
        self.assertEqual(agg_high.profile_activity_risk, ProfileActivityRisk.HIGH)
        self.assertEqual(agg_high.high_risk_rate, 50.0)
        self.assertEqual(agg_high.activity_type_breakdown["reply"]["high_risk"], 2)
        self.assertEqual(len(agg_high.historical_timeline), 2)  # 2026-08 and 2026-09
        self.assertIn("English", agg_high.language_distribution)
        self.assertIn("Hindi", agg_high.language_distribution)
        self.assertIn("Hinglish", agg_high.language_distribution)

    def test_retrieval_audit_and_demo_tagging(self):
        from src.profile_watch.connectors import fetch_profile_activity

        # Demo fallback for Reddit when credentials absent / invalid
        items, is_demo, msg, audit = fetch_profile_activity(
            platform="Reddit",
            handle="nonexistent_test_123456",
            profile_url="https://reddit.com/user/nonexistent_test_123456",
            limit=5,
            demo_fallback=True,
        )
        self.assertTrue(is_demo)
        self.assertTrue(audit.is_demo)
        self.assertEqual(audit.requested_limit, 5)
        self.assertEqual(audit.retrieved_count, 5)
        self.assertIn("DEMO DATA", msg)
        self.assertTrue(all(item.is_demo for item in items))
        self.assertTrue(all(item.activity_type in ("post", "comment", "reply") for item in items))


if __name__ == "__main__":
    unittest.main()

