"""
YouTube comment fetcher with robust URL parsing and error handling.
"""

import re
import logging
from typing import List, Optional

from src.config import config

logger = logging.getLogger(__name__)

# ── Patterns for extracting video IDs ────────────────────────
_VIDEO_ID_PATTERNS = [
    # Standard: https://www.youtube.com/watch?v=VIDEO_ID
    re.compile(r"[?&]v=([a-zA-Z0-9_-]{11})"),
    # Shortened: https://youtu.be/VIDEO_ID
    re.compile(r"youtu\.be/([a-zA-Z0-9_-]{11})"),
    # Shorts: https://www.youtube.com/shorts/VIDEO_ID
    re.compile(r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
    # Live: https://www.youtube.com/live/VIDEO_ID
    re.compile(r"youtube\.com/live/([a-zA-Z0-9_-]{11})"),
    # Embed: https://www.youtube.com/embed/VIDEO_ID
    re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
]


def extract_video_id(url: str) -> Optional[str]:
    """Extract a YouTube video ID from various URL formats.

    Supports standard watch URLs, youtu.be short links, /shorts/,
    /live/, and /embed/ paths.

    Returns
    -------
    str or None
        The 11-character video ID, or None if no match.
    """
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_comments(url: str) -> List[str]:
    """Fetch top-level comments from a YouTube video.

    Parameters
    ----------
    url : str
        Any YouTube video URL.

    Returns
    -------
    list[str]
        Up to ``config.MAX_COMMENTS`` comment texts.

    Raises
    ------
    ValueError
        If the video ID cannot be extracted from the URL.
    RuntimeError
        If the YouTube API call fails.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(
            f"Could not extract a video ID from URL: {url!r}. "
            "Supported formats: youtube.com/watch?v=, youtu.be/, "
            "youtube.com/shorts/, youtube.com/live/"
        )

    api_key = config.YOUTUBE_API_KEY
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY is not configured. "
            "Please set it in your .env file."
        )

    try:
        import requests
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/commentThreads",
            params={
                "part": "snippet",
                "videoId": video_id,
                "maxResults": config.MAX_COMMENTS,
                "textFormat": "plainText",
                "key": api_key,
            },
            timeout=12,
        )
        if resp.status_code != 200:
            err_data = resp.json().get("error", {})
            err_msg = err_data.get("message", resp.text[:200])
            raise RuntimeError(f"YouTube API error (status {resp.status_code}): {err_msg}")
        response = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch YouTube comments: {e}") from e

    comments: List[str] = []
    for item in response.get("items", []):
        try:
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
        except (KeyError, TypeError):
            logger.warning("Skipping malformed comment item: %s", item)

    logger.info("Fetched %d YouTube comments for video %s", len(comments), video_id)
    return comments
