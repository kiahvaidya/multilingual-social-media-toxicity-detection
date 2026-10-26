"""
Audit script for TRAC-1, HASOC, and Jigsaw datasets to inspect original schemas, labels, and exact sample distributions.
"""
import pandas as pd
import io
import urllib.request

print("--- Inspecting HASOC 2019 / 2020 / 2021 & TRAC Datasets ---")

# TRAC / HASOC dataset URL
url = "https://raw.githubusercontent.com/hmMythreya/hatespeechdetection/main/Hinglish.csv"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    df_trac = pd.read_csv(io.StringIO(resp.read().decode('utf-8', errors='ignore')))

print("TRAC/HASOC dataset shape:", df_trac.shape)
print("TRAC/HASOC columns:", df_trac.columns.tolist())
print("TRAC/HASOC label distribution:")
print(df_trac['Aggression-level'].value_counts(dropna=False))

# Also check HASOC 2020 / 2021 original headers if available
hasoc_url = "https://raw.githubusercontent.com/l3s/hasoc-2020/master/hasoc_2020_de-en-hi_dataset/hasoc_2020_hi_train-dataset.csv"
try:
    req2 = urllib.request.Request(hasoc_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req2) as resp:
        df_hasoc = pd.read_csv(io.StringIO(resp.read().decode('utf-8', errors='ignore')))
    print("\nHASOC 2020 Hindi shape:", df_hasoc.shape)
    print("HASOC 2020 Hindi columns:", df_hasoc.columns.tolist())
    print("HASOC 2020 Task 1 distribution:")
    print(df_hasoc['task_1'].value_counts())
    print("HASOC 2020 Task 2 distribution:")
    print(df_hasoc['task_2'].value_counts())
except Exception as e:
    print("HASOC URL test:", e)
