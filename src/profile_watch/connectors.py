"""
Public social profile resolution and activity ingestion.
Supports Reddit (via PRAW), YouTube (via YouTube Data API v3), and Twitter/X (via API v2).
"""

import re
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import requests
from src.config import config
from src.profile_watch.models import RawProfileItem, ResolvedProfile

logger = logging.getLogger(__name__)


def resolve_profile_url(raw_input: str) -> ResolvedProfile:
    """Parse and normalize a profile URL or handle across supported platforms."""
    clean = raw_input.strip()
    if not clean:
        raise ValueError("Profile URL or handle cannot be empty.")

    clean_lower = clean.lower()

    # 1. Reddit User Profile
    # Matches: reddit.com/user/NAME, reddit.com/u/NAME, u/NAME
    m_reddit = re.search(r"(?:reddit\.com/user/|reddit\.com/u/|^u/)([a-zA-Z0-9_\-]+)", clean, re.IGNORECASE)
    if m_reddit:
        handle = m_reddit.group(1)
        return ResolvedProfile(
            platform="Reddit",
            handle=handle,
            profile_url=f"https://www.reddit.com/user/{handle}",
        )

    # 2. YouTube Channel / Creator Profile
    # Matches: youtube.com/@HANDLE, youtube.com/channel/ID, youtube.com/c/NAME, @HANDLE
    if "youtube.com" in clean_lower or "youtu.be" in clean_lower or clean.startswith("@"):
        m_yt_handle = re.search(r"youtube\.com/@([a-zA-Z0-9_\.\-]+)", clean, re.IGNORECASE)
        if m_yt_handle:
            handle = m_yt_handle.group(1)
            return ResolvedProfile(
                platform="YouTube",
                handle=f"@{handle}",
                profile_url=f"https://www.youtube.com/@{handle}",
            )

        m_yt_channel = re.search(r"youtube\.com/channel/([a-zA-Z0-9_\-]+)", clean, re.IGNORECASE)
        if m_yt_channel:
            handle = m_yt_channel.group(1)
            return ResolvedProfile(
                platform="YouTube",
                handle=handle,
                profile_url=f"https://www.youtube.com/channel/{handle}",
            )

        m_yt_c = re.search(r"youtube\.com/c/([a-zA-Z0-9_\-]+)", clean, re.IGNORECASE)
        if m_yt_c:
            handle = m_yt_c.group(1)
            return ResolvedProfile(
                platform="YouTube",
                handle=handle,
                profile_url=f"https://www.youtube.com/c/{handle}",
            )

        if clean.startswith("@"):
            handle = clean.lstrip("@")
            return ResolvedProfile(
                platform="YouTube",
                handle=f"@{handle}",
                profile_url=f"https://www.youtube.com/@{handle}",
            )

    # 3. Twitter / X Profile
    # Matches: twitter.com/HANDLE, x.com/HANDLE
    m_twitter = re.search(r"(?:twitter\.com|x\.com)/([a-zA-Z0-9_]{1,15})(?:/.*)?$", clean, re.IGNORECASE)
    if m_twitter and m_twitter.group(1).lower() not in ("home", "explore", "search", "notifications", "messages"):
        handle = m_twitter.group(1)
        return ResolvedProfile(
            platform="Twitter",
            handle=f"@{handle}",
            profile_url=f"https://twitter.com/{handle}",
        )

    raise ValueError(
        f"Could not identify a supported public profile from: '{clean}'. "
        "Supported formats: "
        "Reddit (reddit.com/user/<handle>), "
        "YouTube (youtube.com/@<handle>), "
        "Twitter/X (x.com/<handle> or twitter.com/<handle>)."
    )


def fetch_reddit_profile_activity(handle: str, limit: int = 25) -> List[RawProfileItem]:
    """Fetch public submissions and comments for a Reddit user via PRAW."""
    client_id = config.REDDIT_CLIENT_ID
    client_secret = config.REDDIT_CLIENT_SECRET
    if not client_id or not client_secret:
        raise RuntimeError("Reddit credentials not configured in .env.")

    import praw
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=config.REDDIT_USER_AGENT,
    )

    clean_handle = handle.lstrip("u/").lstrip("/")
    redditor = reddit.redditor(clean_handle)

    items: List[RawProfileItem] = []

    # Recent Comments
    try:
        for c in redditor.comments.new(limit=limit):
            dt = datetime.fromtimestamp(c.created_utc, tz=timezone.utc).isoformat()
            parent = getattr(c, "parent_id", "") or ""
            act_type = "reply" if str(parent).startswith("t1_") else "comment"
            items.append(
                RawProfileItem(
                    item_id=f"reddit_comment_{c.id}",
                    text=c.body,
                    timestamp=dt,
                    permalink=f"https://reddit.com{c.permalink}",
                    media_url=None,
                    activity_type=act_type,
                    is_demo=False,
                )
            )
    except Exception as e:
        logger.warning("Error fetching comments for u/%s: %s", clean_handle, e)

    # Recent Submissions
    try:
        for s in redditor.submissions.new(limit=limit):
            dt = datetime.fromtimestamp(s.created_utc, tz=timezone.utc).isoformat()
            content = f"{s.title}\n{s.selftext}".strip()
            media = None
            if hasattr(s, "url") and any(str(s.url).endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                media = s.url
            items.append(
                RawProfileItem(
                    item_id=f"reddit_post_{s.id}",
                    text=content,
                    timestamp=dt,
                    permalink=f"https://reddit.com{s.permalink}",
                    media_url=media,
                    activity_type="post",
                    is_demo=False,
                )
            )
    except Exception as e:
        logger.warning("Error fetching submissions for u/%s: %s", clean_handle, e)

    items.sort(key=lambda x: x.timestamp, reverse=True)
    return items[:limit]


def fetch_youtube_profile_activity(handle: str, limit: int = 25) -> List[RawProfileItem]:
    """Fetch public uploads for a YouTube channel via YouTube Data API v3."""
    api_key = config.YOUTUBE_API_KEY
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is not configured in .env.")

    clean_handle = handle.lstrip("@")
    
    # 1. Resolve Channel to find Uploads Playlist
    ch_url = "https://www.googleapis.com/youtube/v3/channels"
    params = {
        "part": "contentDetails,snippet",
        "key": api_key,
    }
    if handle.startswith("UC") and len(handle) == 24:
        params["id"] = handle
    else:
        params["forHandle"] = clean_handle

    resp = requests.get(ch_url, params=params, timeout=12)
    if resp.status_code != 200:
        err_msg = resp.json().get("error", {}).get("message", resp.text[:200])
        raise RuntimeError(f"YouTube Channel API error: {err_msg}")

    data = resp.json()
    items = data.get("items", [])
    if not items:
        # Retry with forUsername if forHandle found nothing
        params.pop("forHandle", None)
        params["forUsername"] = clean_handle
        resp2 = requests.get(ch_url, params=params, timeout=12)
        if resp2.status_code == 200:
            items = resp2.json().get("items", [])

    if not items:
        raise ValueError(f"YouTube channel '{handle}' could not be resolved.")

    channel_info = items[0]
    uploads_id = channel_info.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not uploads_id:
        raise RuntimeError("Could not find uploads playlist for this YouTube channel.")

    # 2. Fetch Playlist Items
    pl_url = "https://www.googleapis.com/youtube/v3/playlistItems"
    pl_params = {
        "part": "snippet",
        "playlistId": uploads_id,
        "maxResults": min(limit, 50),
        "key": api_key,
    }
    pl_resp = requests.get(pl_url, params=pl_params, timeout=12)
    if pl_resp.status_code != 200:
        raise RuntimeError(f"YouTube Playlist API error: {pl_resp.text[:200]}")

    pl_data = pl_resp.json()
    obs_items: List[RawProfileItem] = []
    for item in pl_data.get("items", []):
        snippet = item.get("snippet", {})
        vid_id = snippet.get("resourceId", {}).get("videoId", item.get("id"))
        published_at = snippet.get("publishedAt", datetime.now(timezone.utc).isoformat())
        title = snippet.get("title", "")
        desc = snippet.get("description", "")
        text = f"{title}. {desc}".strip()
        thumbs = snippet.get("thumbnails", {}).get("high", {}).get("url")

        obs_items.append(
            RawProfileItem(
                item_id=f"yt_video_{vid_id}",
                text=text,
                timestamp=published_at,
                permalink=f"https://www.youtube.com/watch?v={vid_id}",
                media_url=thumbs,
                activity_type="post",
                is_demo=False,
            )
        )

    return obs_items[:limit]


def fetch_twitter_profile_activity(handle: str, limit: int = 25) -> List[RawProfileItem]:
    """Fetch recent tweets for a public Twitter/X user via API v2."""
    bearer = config.TWITTER_BEARER_TOKEN
    if not bearer:
        raise RuntimeError("TWITTER_BEARER_TOKEN is not configured in .env.")

    clean_handle = handle.lstrip("@")
    headers = {"Authorization": f"Bearer {bearer}"}

    # 1. Resolve User ID
    u_resp = requests.get(
        f"https://api.twitter.com/2/users/by/username/{clean_handle}",
        headers=headers,
        timeout=12,
    )
    if u_resp.status_code == 403:
        raise RuntimeError(
            "Twitter API returned 403 Forbidden. "
            "The Twitter API v2 Free tier restricts user profile tweet lookups."
        )
    if u_resp.status_code != 200:
        raise RuntimeError(f"Twitter API error: status {u_resp.status_code}")

    u_data = u_resp.json().get("data")
    if not u_data:
        raise ValueError(f"Twitter user @{clean_handle} not found.")

    user_id = u_data["id"]

    # 2. Fetch User Tweets
    t_resp = requests.get(
        f"https://api.twitter.com/2/users/{user_id}/tweets",
        headers=headers,
        params={
            "max_results": min(max(limit, 5), 100),
            "tweet.fields": "created_at,text",
        },
        timeout=12,
    )
    if t_resp.status_code != 200:
        raise RuntimeError(f"Twitter Tweets API error: status {t_resp.status_code}")

    tweets_data = t_resp.json().get("data", [])
    items: List[RawProfileItem] = []
    for tw in tweets_data:
        tw_text = tw["text"]
        act_type = "reply" if tw_text.strip().startswith("@") else "post"
        items.append(
            RawProfileItem(
                item_id=f"tweet_{tw['id']}",
                text=tw_text,
                timestamp=tw.get("created_at", datetime.now(timezone.utc).isoformat()),
                permalink=f"https://twitter.com/{clean_handle}/status/{tw['id']}",
                media_url=None,
                activity_type=act_type,
                is_demo=False,
            )
        )
    return items


from src.profile_watch.models import RawProfileItem, ResolvedProfile, RetrievalAudit

# ── Curated Multilingual Demo Profile Activity ───────────────────────
# Explicitly labeled DEMO DATA fallback when live API quotas are exhausted or restricted

DEMO_PROFILE_ACTIVITIES = {
    "Reddit": [
        RawProfileItem(
            item_id="demo_rd_1",
            text="Really solid discussion on the open source LLM benchmark. Appreciate the nuance here.",
            timestamp="2026-08-15T14:22:00Z",
            permalink="https://reddit.com/r/MachineLearning/comments/demo1",
            activity_type="post",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_2",
            text="You are an utterly pathetic clown who doesn't understand high school algebra.",
            timestamp="2026-09-02T09:15:00Z",
            permalink="https://reddit.com/r/technology/comments/demo2",
            activity_type="comment",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_3",
            text="bhai sach me dil jeet liya yaar, mast analysis hai totally agree with your points",
            timestamp="2026-09-12T11:04:00Z",
            permalink="https://reddit.com/r/india/comments/demo3",
            activity_type="reply",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_4",
            text="Get lost you absolute idiot, nobody asked for your degenerate garbage opinion.",
            timestamp="2026-09-28T16:30:00Z",
            permalink="https://reddit.com/r/news/comments/demo4",
            activity_type="reply",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_5",
            text="Stop spreading hatred against minorities or watch what happens to you.",
            timestamp="2026-10-04T17:45:00Z",
            permalink="https://reddit.com/r/politics/comments/demo5",
            activity_type="comment",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_6",
            text="Thanks for sharing the github repo, stars added and bookmarked for testing.",
            timestamp="2026-10-10T08:12:00Z",
            permalink="https://reddit.com/r/python/comments/demo6",
            activity_type="post",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_7",
            text="यह बेहद शर्मनाक और अस्वीकार्य आचरण है, सभ्य समाज में ऐसा नहीं होना चाहिए।",
            timestamp="2026-10-12T10:15:00Z",
            permalink="https://reddit.com/r/india/comments/demo7",
            activity_type="comment",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_8",
            text="bhai faltu me bakwaas mat karo chup chap niklo yahan se",
            timestamp="2026-10-13T14:40:00Z",
            permalink="https://reddit.com/r/askindia/comments/demo8",
            activity_type="reply",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_9",
            text="I agree with the first principle approach to multilingual evaluation metrics.",
            timestamp="2026-10-14T09:00:00Z",
            permalink="https://reddit.com/r/MachineLearning/comments/demo9",
            activity_type="comment",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_rd_10",
            text="You disgusting fraud, I will expose you publicly and make you pay.",
            timestamp="2026-10-14T11:20:00Z",
            permalink="https://reddit.com/r/technology/comments/demo10",
            activity_type="reply",
            is_demo=True,
        ),
    ],
    "YouTube": [
        RawProfileItem(
            item_id="demo_yt_1",
            text="Deep dive into multilingual transformer attention heads and cross-lingual transfer.",
            timestamp="2026-08-20T18:00:00Z",
            permalink="https://youtube.com/watch?v=demo_yt_1",
            activity_type="post",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_yt_2",
            text="Exposing corrupt fraudsters and traitors who destroy our culture.",
            timestamp="2026-09-10T19:30:00Z",
            permalink="https://youtube.com/watch?v=demo_yt_2",
            activity_type="post",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_yt_3",
            text="Watch out, I know where your studio is located and you will pay for this.",
            timestamp="2026-10-02T20:15:00Z",
            permalink="https://youtube.com/watch?v=demo_yt_3",
            activity_type="reply",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_yt_4",
            text="Super excited for the new PyTorch release! Check out the tutorial in the description.",
            timestamp="2026-10-08T12:00:00Z",
            permalink="https://youtube.com/watch?v=demo_yt_4",
            activity_type="post",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_yt_5",
            text="क्या शानदार वीडियो बनाया है भाई, पूरा कॉन्सेप्ट क्लियर हो गया!",
            timestamp="2026-10-11T15:00:00Z",
            permalink="https://youtube.com/watch?v=demo_yt_5",
            activity_type="comment",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_yt_6",
            text="tera dimaag kharab hai kya be aisi bakwaas faila raha hai",
            timestamp="2026-10-13T16:30:00Z",
            permalink="https://youtube.com/watch?v=demo_yt_6",
            activity_type="reply",
            is_demo=True,
        ),
    ],
    "Twitter": [
        RawProfileItem(
            item_id="demo_tw_1",
            text="Excited to publish our latest research on multilingual safety bounds!",
            timestamp="2026-08-18T10:00:00Z",
            permalink="https://twitter.com/demo/status/1",
            activity_type="post",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_tw_2",
            text="You worthless racist scum, go crawl back into your hole.",
            timestamp="2026-09-15T14:30:00Z",
            permalink="https://twitter.com/demo/status/2",
            activity_type="reply",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_tw_3",
            text="bhai kya bawaal project banaya hai, super proud of the team! 🚀",
            timestamp="2026-10-05T16:20:00Z",
            permalink="https://twitter.com/demo/status/3",
            activity_type="post",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_tw_4",
            text="Shut your disgusting mouth you stupid clown, you have zero credibility.",
            timestamp="2026-10-09T19:10:00Z",
            permalink="https://twitter.com/demo/status/4",
            activity_type="reply",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_tw_5",
            text="सच हमेशा सामने आता है, कोई भी झूठ ज्यादा दिन नहीं टिकता।",
            timestamp="2026-10-12T11:45:00Z",
            permalink="https://twitter.com/demo/status/5",
            activity_type="post",
            is_demo=True,
        ),
        RawProfileItem(
            item_id="demo_tw_6",
            text="tu to bilkul fraud nikla re sharm kar thodi",
            timestamp="2026-10-14T08:30:00Z",
            permalink="https://twitter.com/demo/status/6",
            activity_type="reply",
            is_demo=True,
        ),
    ],
}


def fetch_profile_activity(
    platform: str,
    handle: str,
    profile_url: str,
    limit: int = 25,
    demo_fallback: bool = True,
) -> Tuple[List[RawProfileItem], bool, Optional[str], RetrievalAudit]:
    """Dispatch activity ingestion to the corresponding platform connector.

    Returns
    -------
    items : List[RawProfileItem]
    is_demo : bool
    message : Optional[str]
    audit : RetrievalAudit
    """
    is_demo = False
    message = None
    rate_limited = False
    limitations = f"Public platform API limit requested: {limit} items."
    items: List[RawProfileItem] = []

    try:
        if platform == "Reddit":
            items = fetch_reddit_profile_activity(handle, limit=limit)
            limitations += " Reddit PRAW public user profile query: comments + submissions."
        elif platform == "YouTube":
            items = fetch_youtube_profile_activity(handle, limit=limit)
            limitations += " YouTube Data API v3 uploads playlist inspection."
        elif platform == "Twitter":
            items = fetch_twitter_profile_activity(handle, limit=limit)
            limitations += " Twitter API v2 user timeline endpoint."
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    except Exception as e:
        err_str = str(e)
        logger.warning("Error fetching live profile activity for %s (%s): %s", platform, handle, err_str)
        if "403" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
            rate_limited = True
        if demo_fallback:
            items = DEMO_PROFILE_ACTIVITIES.get(platform, DEMO_PROFILE_ACTIVITIES["Reddit"])[:limit]
            is_demo = True
            limitations = f"DEMO DATASET: Live {platform} API credentials unavailable or quota reached ({err_str[:120]})."
            message = (
                f"DEMO DATA: Live {platform} API credentials unavailable or restricted. "
                f"Loaded {len(items)} curated public observations for forensic analysis."
            )
        else:
            raise

    if not items and demo_fallback:
        items = DEMO_PROFILE_ACTIVITIES.get(platform, DEMO_PROFILE_ACTIVITIES["Reddit"])[:limit]
        is_demo = True
        limitations = f"DEMO DATASET: No public items returned from live {platform} endpoint."
        message = f"DEMO DATA: No public activity found for {platform} profile. Loaded {len(items)} sample items."

    timestamps = [it.timestamp for it in items if it.timestamp]
    earliest_ts = min(timestamps) if timestamps else None
    latest_ts = max(timestamps) if timestamps else None

    audit = RetrievalAudit(
        requested_limit=limit,
        retrieved_count=len(items),
        earliest_timestamp=earliest_ts,
        latest_timestamp=latest_ts,
        is_demo=is_demo,
        retrieval_limitations=limitations,
        rate_limit_encountered=rate_limited,
    )

    return items, is_demo, message, audit

