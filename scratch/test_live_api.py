import urllib.request
import json
import sys

# Set utf-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

def test_api():
    # 1. Benign
    data_benign = json.dumps({"text": "Have a wonderful day and good luck with your work!"}).encode()
    req1 = urllib.request.Request("http://127.0.0.1:8000/api/analyze/text", data=data_benign, headers={"Content-Type": "application/json"})
    res1 = json.loads(urllib.request.urlopen(req1, timeout=10).read().decode())
    print("BENIGN TEST:")
    print("  Risk Level :", res1["risk_level"])
    print("  Badge Icon :", res1["badge_icon"])
    print("  Confidence :", res1["confidence"])
    print("  Explanation:", res1["explanation"])

    # 2. Toxic
    data_toxic = json.dumps({"text": "You are a stupid idiot and I hate you!"}).encode()
    req2 = urllib.request.Request("http://127.0.0.1:8000/api/analyze/text", data=data_toxic, headers={"Content-Type": "application/json"})
    res2 = json.loads(urllib.request.urlopen(req2, timeout=10).read().decode())
    print("\nTOXIC TEST:")
    print("  Risk Level :", res2["risk_level"])
    print("  Badge Icon :", res2["badge_icon"])
    print("  Confidence :", res2["confidence"])
    print("  Explanation:", res2["explanation"])

    # 3. Social endpoint check
    data_social = json.dumps({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "platform": "youtube", "max_comments": 5, "demo_fallback": True}).encode()
    req3 = urllib.request.Request("http://127.0.0.1:8000/api/analyze/social", data=data_social, headers={"Content-Type": "application/json"})
    res3 = json.loads(urllib.request.urlopen(req3, timeout=10).read().decode())
    print("\nSOCIAL INGESTION TEST:")
    print("  Platform   :", res3["platform"])
    print("  Total      :", res3["total_comments"])
    print("  High Risk  :", res3["high_risk_count"])
    print("  Safe       :", res3["safe_count"])
    print("  Review     :", res3["review_count"])

if __name__ == "__main__":
    test_api()
