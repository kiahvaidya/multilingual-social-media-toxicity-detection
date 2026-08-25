import streamlit as st
import joblib
import re
import requests
import praw
import numpy as np
from googleapiclient.discovery import build


# ================= API KEYS ================= #

YOUTUBE_API_KEY = "AIzaSyCDhBg95V-UDoRuChcJwcMyYWnuEUn8-FU"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAOPv4wEAAAAArNVHceIkQlrWkyNdtvhH08nZgRA%3DZ4iqV40sT8N0HD1jxFXdVNoBLmoAh5b7laHtlQuCl5YvP0hoKT"
REDDIT_CLIENT_ID = "QQmegdOn_n5z_d3OBGqR5Q"
REDDIT_CLIENT_SECRET = "qeh851_bN1MOE6TPs1uKYuq-16cg8w"
REDDIT_USER_AGENT = "HateSpeechAnalyzerBot/0.1 by kv-05"


# ================= LOAD MODEL ================= #

model = joblib.load("toxic_model.pkl")
vectorizer = joblib.load("toxic_vectorizer.pkl")
labels = joblib.load("toxic_columns.pkl")


# ================= HINGLISH MAP ================= #

hinglish_map = {
    "chutiya":"idiot",
    "chutya":"idiot",
    "chodu":"idiot", 
    "mc":"motherfucker",
    "madarchod":"motherfucker",
    "bc":"fuck",
    "behenchod":"fuck",
    "bhosdi":"prostitute",
    "bhosdiwala":"prostitute",
    "bhosdika":"prostitute",
    "bhosdike":"prostitute",
    "bhadwa":"pimp",
    "bkl":"bastard",
    "gandu":"asshole",
    "gand":"asshole",   
    "harami":"bastard",
    "haramzada":"bastard",
    "lauder":"dickhead",
    "kamine":"bastard",
    "kutta":"dog",
    "nalla":"loser",
    "bakwas":"nonsense",
    "tatti":"shit",
    "lodu":"idiot",
    "lawde":"dick",
    "lund":"dick"
}


# ================= TEXT CLEANING ================= #

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # Hinglish normalization
    for k,v in hinglish_map.items():
        text = text.replace(k,v)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    return text


# ================= PREDICTION ================= #

def predict_comment(text):

    clean = preprocess_text(text)

    vec = vectorizer.transform([clean])

    try:
        probs = model.predict_proba(vec)[0]
    except:
        probs = model.predict(vec)[0]

    predictions = {}

    for i,label in enumerate(labels):

        if i < len(probs):
            predictions[label] = float(round(probs[i],3))
        else:
            predictions[label] = 0.0

    toxic_score = max(predictions.values())

    if toxic_score > 0.7:
        intensity = "High"
    elif toxic_score > 0.4:
        intensity = "Moderate"
    else:
        intensity = "Low"

    return predictions,toxic_score,intensity


# ================= YOUTUBE ================= #

def fetch_youtube_comments(url):

    video_id = url.split("v=")[-1].split("&")[0]

    youtube = build("youtube","v3",developerKey=YOUTUBE_API_KEY)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=10,
        textFormat="plainText"
    )

    response = request.execute()

    comments = []

    for item in response.get("items",[]):
        comments.append(
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        )

    return comments


# ================= REDDIT ================= #

def fetch_reddit_comments(url):

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    submission = reddit.submission(url=url)

    submission.comments.replace_more(limit=0)

    return [c.body for c in submission.comments[:10]]


# ================= TWITTER ================= #

def fetch_twitter_comments(url):

    match = re.search(r"status/(\d+)", url)

    if not match:
        return []

    tweet_id = match.group(1)

    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

    params = {
        "query": f"conversation_id:{tweet_id}",
        "max_results": 10,
        "tweet.fields": "text"
    }

    response = requests.get(
        "https://api.twitter.com/2/tweets/search/recent",
        headers=headers,
        params=params
    )

    if response.status_code != 200:
        return []

    data = response.json()

    return [tweet["text"] for tweet in data.get("data",[])]


# ================= STREAMLIT UI ================= #

st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🧠",
    layout="wide"
)

st.title("💬 Advanced Hate Speech & Toxicity Detector")

st.markdown(
"Analyze text or fetch comments from **YouTube**, **Reddit**, or **Twitter (X)**"
)

choice = st.sidebar.radio(
"Choose Input Type",
["📝 Custom Text","📺 YouTube","👽 Reddit","🐦 Twitter (X)"]
)


# ================= CUSTOM TEXT ================= #

if choice == "📝 Custom Text":

    text = st.text_area("Type or paste text",height=150)

    if st.button("Analyze"):

        if text.strip():

            preds,score,intensity = predict_comment(text)

            if score > 0.5:
                st.error("⚠️ Toxic Content Detected")
            else:
                st.success("✅ Safe Comment")

            st.write("### Toxicity Breakdown")

            for k,v in preds.items():
                st.write(f"{k}: {v}")

            st.progress(score)

            st.write("Intensity:",intensity)

        else:
            st.warning("Please enter text")


# ================= SOCIAL MEDIA ================= #

else:

    url = st.text_input("Enter URL")

    if st.button("Fetch & Analyze"):

        with st.spinner("Fetching comments..."):

            if "YouTube" in choice:
                comments = fetch_youtube_comments(url)

            elif "Reddit" in choice:
                comments = fetch_reddit_comments(url)

            else:
                comments = fetch_twitter_comments(url)

        if not comments:
            st.error("No comments found")

        else:

            st.success(f"{len(comments)} comments fetched")

            for c in comments:

                preds,score,intensity = predict_comment(c)

                with st.expander(c[:80]+"..."):

                    if score > 0.5:
                        st.error("⚠️ Toxic")
                    else:
                        st.success("✅ Safe")

                    for k,v in preds.items():
                        st.write(f"{k}: {v}")

                    st.progress(score)

                    st.write("Intensity:",intensity)


st.markdown("---")