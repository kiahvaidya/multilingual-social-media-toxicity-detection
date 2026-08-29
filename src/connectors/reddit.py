"""
Reddit comment fetcher with error handling.
"""

import logging
from typing import List

from src.config import config

logger = logging.getLogger(__name__)


def fetch_reddit_comments(url: str) -> List[str]:
    """Fetch top-level comments from a Reddit submission.

    Parameters
    ----------
    url : str
        A Reddit post URL.

    Returns
    -------
    list[str]
        Up to ``config.MAX_COMMENTS`` comment bodies.

    Raises
    ------
    RuntimeError
        If Reddit credentials are missing or the API call fails.
    """
    client_id = config.REDDIT_CLIENT_ID
    client_secret = config.REDDIT_CLIENT_SECRET

    if not client_id or not client_secret:
        raise RuntimeError(
            "Reddit credentials are not configured. "
            "Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env."
        )

    try:
        import praw

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=config.REDDIT_USER_AGENT,
        )
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)

        comments = [c.body for c in submission.comments[: config.MAX_COMMENTS]]
        logger.info("Fetched %d Reddit comments", len(comments))
        return comments

    except Exception as e:
        raise RuntimeError(f"Failed to fetch Reddit comments: {e}") from e
