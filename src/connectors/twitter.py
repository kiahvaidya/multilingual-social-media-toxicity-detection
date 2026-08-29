"""
Twitter / X comment fetcher with error handling.

Note: The free tier of Twitter API v2 does NOT support the
tweets/search/recent endpoint. This connector will return an
informative error if the API rejects the request.
"""

import re
import logging
from typing import List

import requests

from src.config import config

logger = logging.getLogger(__name__)


def fetch_twitter_comments(url: str) -> List[str]:
    """Fetch replies to a tweet using Twitter API v2.

    Parameters
    ----------
    url : str
        A Twitter/X status URL containing a tweet ID.

    Returns
    -------
    list[str]
        Up to ``config.MAX_COMMENTS`` reply texts.

    Raises
    ------
    ValueError
        If the tweet ID cannot be extracted from the URL.
    RuntimeError
        If the bearer token is missing or the API call fails.
    """
    match = re.search(r"status/(\d+)", url)
    if not match:
        raise ValueError(
            f"Could not extract a tweet ID from URL: {url!r}. "
            "Expected a URL containing /status/<tweet_id>."
        )

    tweet_id = match.group(1)
    bearer = config.TWITTER_BEARER_TOKEN

    if not bearer:
        raise RuntimeError(
            "TWITTER_BEARER_TOKEN is not configured. "
            "Please set it in your .env file."
        )

    headers = {"Authorization": f"Bearer {bearer}"}
    params = {
        "query": f"conversation_id:{tweet_id}",
        "max_results": config.MAX_COMMENTS,
        "tweet.fields": "text",
    }

    try:
        response = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers=headers,
            params=params,
            timeout=15,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Network error contacting Twitter API: {e}") from e

    if response.status_code == 403:
        raise RuntimeError(
            "Twitter API returned 403 Forbidden. "
            "The free tier does not support tweets/search/recent. "
            "Upgrade your API access or use a different endpoint."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Twitter API returned status {response.status_code}: "
            f"{response.text[:200]}"
        )

    data = response.json()
    tweets = [tweet["text"] for tweet in data.get("data", [])]
    logger.info("Fetched %d Twitter replies for tweet %s", len(tweets), tweet_id)
    return tweets
