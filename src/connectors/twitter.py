"""
Twitter / X comment fetcher with error handling.

Note: The free tier of Twitter API v2 does NOT support the
tweets/search/recent endpoint. This connector will return an
informative error if the API rejects the request.
"""

import re
import logging
import urllib.parse
from typing import List

import requests

from src.config import config

logger = logging.getLogger(__name__)


def _fetch_tweet_via_syndication(tweet_id: str) -> List[str]:
    """Fetch public tweet text via Twitter syndication endpoint when API v2 tier restricts search/recent."""
    try:
        resp = requests.get(
            "https://cdn.syndication.twimg.com/tweet-result",
            params={"id": tweet_id, "token": "x"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            texts: List[str] = []
            if data.get("text"):
                texts.append(data["text"])
            parent = data.get("parent")
            if isinstance(parent, dict) and parent.get("text"):
                texts.append(parent["text"])
            quoted = data.get("quoted_tweet")
            if isinstance(quoted, dict) and quoted.get("text"):
                texts.append(quoted["text"])
            if texts:
                logger.info("Fetched %d tweet item(s) via syndication for tweet %s", len(texts), tweet_id)
                return texts
    except Exception as e:
        logger.debug("Syndication fallback failed for tweet %s: %s", tweet_id, e)
    return []


def fetch_twitter_comments(url: str) -> List[str]:
    """Fetch replies to a tweet using Twitter API v2 (with syndication fallback for Free tier).

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
    raw_bearer = config.TWITTER_BEARER_TOKEN

    if not raw_bearer:
        raise RuntimeError(
            "TWITTER_BEARER_TOKEN is not configured. "
            "Please set it in your .env file."
        )

    bearer = urllib.parse.unquote(raw_bearer)
    headers = {"Authorization": f"Bearer {bearer}"}
    params = {
        "query": f"conversation_id:{tweet_id}",
        "max_results": max(10, min(config.MAX_COMMENTS, 100)),
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
        syndicated = _fetch_tweet_via_syndication(tweet_id)
        if syndicated:
            return syndicated
        raise RuntimeError(f"Network error contacting Twitter API: {e}") from e

    if response.status_code in (401, 403, 429, 503):
        syndicated = _fetch_tweet_via_syndication(tweet_id)
        if syndicated:
            return syndicated
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
    if not tweets:
        tweets = _fetch_tweet_via_syndication(tweet_id)
    logger.info("Fetched %d Twitter replies for tweet %s", len(tweets), tweet_id)
    return tweets
