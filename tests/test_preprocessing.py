"""
Tests for src.preprocessing — unified text preprocessing pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.preprocessing import preprocess_text, _clean_social_media, _clean_for_model


# ── Basic lowercasing ────────────────────────────────────

class TestLowercasing:
    def test_uppercase_is_lowered(self):
        assert preprocess_text("HELLO WORLD") == "hello world"

    def test_mixed_case(self):
        result = preprocess_text("HeLLo WoRLd")
        assert result == "hello world"


# ── Punctuation removal ─────────────────────────────────

class TestPunctuationRemoval:
    def test_basic_punctuation(self):
        result = preprocess_text("hello, world! how's it?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_special_chars(self):
        result = preprocess_text("f*** you @#$%")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result


# ── Number removal ───────────────────────────────────────

class TestNumberRemoval:
    def test_numbers_removed(self):
        result = preprocess_text("hello 123 world 456")
        assert "123" not in result
        assert "456" not in result

    def test_mixed_alphanum(self):
        # Numbers in words should be removed
        result = preprocess_text("h8ers gonna h8")
        assert "8" not in result


# ── URL removal (social_media=True only) ─────────────────

class TestURLRemoval:
    def test_http_url_removed(self):
        result = preprocess_text("check http://example.com out")
        assert "http" not in result
        assert "example" not in result

    def test_https_url_removed(self):
        result = preprocess_text("visit https://t.co/abc123 now")
        assert "https" not in result

    def test_url_not_removed_without_social(self):
        # In training mode, URLs are NOT specifically removed
        # (they get partially cleaned by punctuation/number removal)
        result = preprocess_text("http://example.com", social_media=False)
        # The :// and . get removed by punctuation removal,
        # but "http" and "examplecom" may remain
        assert "http" in result


# ── @mention removal (social_media=True only) ────────────

class TestMentionRemoval:
    def test_mention_removed(self):
        result = preprocess_text("hello @username how are you")
        assert "username" not in result

    def test_mention_kept_without_social(self):
        result = preprocess_text("hello @username", social_media=False)
        # @ is removed as punctuation but "username" stays
        assert "username" in result


# ── Hinglish normalization ───────────────────────────────

class TestHinglishMap:
    def test_chutiya_mapped(self):
        result = preprocess_text("tu chutiya hai")
        assert "idiot" in result
        assert "chutiya" not in result

    def test_bc_mapped(self):
        result = preprocess_text("saale bc")
        assert "fuck" in result

    def test_multiple_slang(self):
        result = preprocess_text("gandu harami")
        assert "asshole" in result
        assert "bastard" in result

    def test_hinglish_not_applied_without_social(self):
        result = preprocess_text("chutiya", social_media=False)
        assert "chutiya" in result
        assert "idiot" not in result


# ── Character deduplication (social_media=True only) ─────

class TestCharDeduplication:
    def test_repeated_chars(self):
        result = preprocess_text("heeello")
        assert "eee" not in result

    def test_normal_text_unchanged(self):
        result = preprocess_text("hello")
        assert "helo" in result or "hello" in result


# ── Newline removal ──────────────────────────────────────

class TestNewlineRemoval:
    def test_newline_replaced(self):
        result = preprocess_text("line1\nline2")
        assert "\n" not in result

    def test_carriage_return_replaced(self):
        result = preprocess_text("line1\rline2")
        assert "\r" not in result


# ── Edge cases ───────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self):
        result = preprocess_text("")
        assert result == ""

    def test_none_input(self):
        # str(None) -> "none"
        result = preprocess_text(None)
        assert result == "none"

    def test_numeric_input(self):
        result = preprocess_text(12345)
        assert result == ""  # All digits removed

    def test_only_punctuation(self):
        result = preprocess_text("!@#$%^&*()")
        assert result.strip() == ""

    def test_whitespace_preserved_between_words(self):
        result = preprocess_text("hello world")
        assert "hello" in result
        assert "world" in result


# ── social_media flag ────────────────────────────────────

class TestSocialMediaFlag:
    def test_social_media_true_applies_url_removal(self):
        text = "check https://example.com"
        result_social = preprocess_text(text, social_media=True)
        result_train = preprocess_text(text, social_media=False)
        # Social should have removed the URL more thoroughly
        assert len(result_social) <= len(result_train)

    def test_both_modes_lowercase(self):
        assert preprocess_text("HELLO", social_media=True) == preprocess_text("HELLO", social_media=False)
