import sys
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

df = pd.read_csv("artifacts/adversarial_validation_results.csv")
fps = df[df["is_false_positive"] == True]
fns = df[df["is_false_negative"] == True]

print("=" * 60)
print(f"TOTAL SAMPLES: {len(df)}")
print(f"FALSE POSITIVES: {len(fps)}")
print(f"FALSE NEGATIVES: {len(fns)}")
print("=" * 60)

print("\n--- FALSE POSITIVE DISTRIBUTION BY PRIMARY CATEGORY ---")
print(fps["primary_category"].value_counts())

print("\n--- TOP FALSE POSITIVE EXAMPLES ---")
for _, r in fps.head(12).iterrows():
    flags = [col for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
             if r[f"pred_{col}"] == 1 and r[f"expected_{col}"] == 0]
    print(f"[{r['id']}] ({r['primary_category']}) \"{r['text']}\"")
    print(f"    Triggered Flags: {flags} | Max Prob: {r['max_toxicity_prob']:.4f}")

print("\n--- FALSE NEGATIVE DISTRIBUTION BY THREAT SUBTYPE ---")
threat_fns = fns[fns["expected_threat"] == 1]
print(threat_fns["threat_subtype"].value_counts())

print("\n--- TOP THREAT FALSE NEGATIVE EXAMPLES ---")
for _, r in threat_fns.head(10).iterrows():
    print(f"[{r['id']}] ({r['threat_subtype']}) \"{r['text']}\"")
    print(f"    Threat Prob: {r['prob_threat']:.4f} (Thresh: 0.40) | Toxic Prob: {r['prob_toxic']:.4f}")

print("\n--- FALSE NEGATIVE DISTRIBUTION BY IDENTITY SUBTYPE ---")
id_fns = fns[fns["expected_identity_hate"] == 1]
print(id_fns["identity_subtype"].value_counts())

print("\n--- TOP IDENTITY-HATE FALSE NEGATIVE EXAMPLES ---")
for _, r in id_fns.head(10).iterrows():
    print(f"[{r['id']}] ({r['identity_subtype']}) \"{r['text']}\"")
    print(f"    Id Hate Prob: {r['prob_identity_hate']:.4f} (Thresh: 0.55) | Toxic Prob: {r['prob_toxic']:.4f} | Insult Prob: {r['prob_insult']:.4f}")
