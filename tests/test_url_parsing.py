"""
Tests for YouTube URL parsing (src.connectors.youtube.extract_video_id).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.connectors.youtube import extract_video_id


class TestStandardURLs:
    """Standard youtube.com/watch?v=ID format."""

    def test_basic_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_watch_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_watch_url_v_not_first_param(self):
        url = "https://www.youtube.com/watch?feature=related&v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_http_without_www(self):
        url = "http://youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestShortenedURLs:
    """youtu.be/ID short format."""

    def test_youtu_be(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_youtu_be_with_params(self):
        url = "https://youtu.be/dQw4w9WgXcQ?t=42"
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestShortsURLs:
    """youtube.com/shorts/ID format."""

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_shorts_url_with_params(self):
        url = "https://youtube.com/shorts/dQw4w9WgXcQ?feature=share"
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestLiveURLs:
    """youtube.com/live/ID format."""

    def test_live_url(self):
        url = "https://www.youtube.com/live/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestEmbedURLs:
    """youtube.com/embed/ID format."""

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestInvalidURLs:
    """URLs that should NOT match."""

    def test_empty_string(self):
        assert extract_video_id("") is None

    def test_non_youtube_url(self):
        assert extract_video_id("https://google.com") is None

    def test_youtube_homepage(self):
        assert extract_video_id("https://www.youtube.com") is None

    def test_youtube_channel(self):
        assert extract_video_id("https://www.youtube.com/channel/UCxxxxxxx") is None

    def test_random_text(self):
        assert extract_video_id("hello world") is None

    def test_partial_video_id(self):
        # Video IDs are exactly 11 chars; a shorter string shouldn't match
        url = "https://www.youtube.com/watch?v=short"
        assert extract_video_id(url) is None


class TestEdgeCases:
    """Edge cases for video ID extraction."""

    def test_id_with_hyphens(self):
        url = "https://www.youtube.com/watch?v=abc-def_123"
        assert extract_video_id(url) == "abc-def_123"

    def test_id_with_underscores(self):
        url = "https://youtu.be/abc_def_123"
        assert extract_video_id(url) == "abc_def_123"

    def test_mobile_url(self):
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
