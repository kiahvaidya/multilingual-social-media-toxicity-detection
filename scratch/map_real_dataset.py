import sys
import io
import urllib.request
import pandas as pd
import re

sys.stdout.reconfigure(encoding='utf-8')

url = "https://raw.githubusercontent.com/hmMythreya/hatespeechdetection/main/Hinglish.csv"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    df = pd.read_csv(io.StringIO(resp.read().decode('utf-8', errors='ignore')))

# Inspect columns and values
print("Dataset columns:", df.columns.tolist())

# Function to detect script
def detect_script(text):
    text_str = str(text)
    devanagari = len(re.findall(r'[\u0900-\u097F]', text_str))
    latin = len(re.findall(r'[a-zA-Z]', text_str))
    
    if devanagari > 3:
        return "hindi_devanagari"
    elif latin > 3:
        # Check if hinglish or pure english
        # Simple heuristic or check common hindi romanized words
        hinglish_words = ["hai", "hoon", "bhai", "kya", "mera", "teri", "wale", "karo", "teree", "chutiya", "gand", "desh", "modi", "bjp", "log"]
        words = text_str.lower().split()
        if any(w in words for w in hinglish_words):
            return "hinglish"
        else:
            return "english"
    else:
        return "other"

df['script_cat'] = df['Text'].apply(detect_script)
print("\nScript Category Breakdown:")
print(df['script_cat'].value_counts())

print("\nSample Hindi Devanagari:")
for t in df[df['script_cat'] == 'hindi_devanagari']['Text'].head(5):
    print("  -", t.strip())

print("\nSample Hinglish:")
for t in df[df['script_cat'] == 'hinglish']['Text'].head(5):
    print("  -", t.strip())

print("\nSample English:")
for t in df[df['script_cat'] == 'english']['Text'].head(5):
    print("  -", t.strip())
