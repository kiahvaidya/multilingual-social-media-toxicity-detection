import urllib.request

urls = [
    "https://raw.githubusercontent.com/bhargavyagnik/Hateful_speech_Hinglish_social_media_Paper/master/Hate_speech_dataset.csv",
    "https://raw.githubusercontent.com/hmMythreya/hatespeechdetection/main/Hinglish.csv"
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read(2048)
            print(f"SUCCESS: {url}")
            print("Snippet:\n", data[:400].decode('utf-8', errors='ignore'))
            print("=" * 60)
    except Exception as e:
        print(f"FAIL: {url} -> {e}")
