import sys
import io
import urllib.request
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

url1 = "https://raw.githubusercontent.com/bhargavyagnik/Hateful_speech_Hinglish_social_media_Paper/master/Hate_speech_dataset.csv"
url2 = "https://raw.githubusercontent.com/hmMythreya/hatespeechdetection/main/Hinglish.csv"

print("--- Loading Dataset 1 (Hinglish Social Media Paper Dataset) ---")
req1 = urllib.request.Request(url1, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req1) as resp:
    df1 = pd.read_csv(io.StringIO(resp.read().decode('utf-8', errors='ignore')))

print("Dataset 1 Shape:", df1.shape)
print("Dataset 1 Columns:", df1.columns.tolist())
print("Dataset 1 Value Counts for 'label':")
print(df1['label'].value_counts(dropna=False))
print("\nSample comments:")
for idx, row in df1.head(10).iterrows():
    print(f"[{row['label']}] {row['comment']}")

print("\n" + "=" * 60 + "\n")

print("--- Loading Dataset 2 (Hinglish Hate Speech Dataset) ---")
req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req2) as resp:
    df2 = pd.read_csv(io.StringIO(resp.read().decode('utf-8', errors='ignore')))

print("Dataset 2 Shape:", df2.shape)
print("Dataset 2 Columns:", df2.columns.tolist())
if 'label' in df2.columns:
    print(df2['label'].value_counts(dropna=False))
elif 'Label' in df2.columns:
    print(df2['Label'].value_counts(dropna=False))
print("\nSample comments:")
for idx, row in df2.head(10).iterrows():
    print(dict(row))
