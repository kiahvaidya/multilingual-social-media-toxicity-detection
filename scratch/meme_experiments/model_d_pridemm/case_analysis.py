import json
from pathlib import Path
import pandas as pd

p = Path("scratch/meme_experiments/model_d_pridemm/multi3hate_predictions.csv")
df = pd.read_csv(p)

# Check distribution of Model B toxic prob on Multi3Hate
print("Model B toxic prob summary:")
print(df["model_b_toxic_prob"].describe())

# A. Contextual Inversion: Text looks toxic (model_b >= 0.45), but GT == 0 (benign meme)
sub_ci = df[(df["model_b_toxic_prob"] >= 0.45) & (df["IN"] == 0)]

# B. Emergent Hate: Text looks benign (lowest tertile of model_b toxic prob), but meme is hateful (GT == 1)
# Find lowest Model B toxic scores among GT==1
sub_eh = df[df["IN"] == 1].sort_values("model_b_toxic_prob", ascending=True)

# C. Explicit Hate: Text looks toxic (model_b >= 0.45) and GT == 1
sub_xh = df[(df["model_b_toxic_prob"] >= 0.45) & (df["IN"] == 1)]

# D. Benign: Both text and meme benign (model_b lowest among GT==0)
sub_bn = df[df["IN"] == 0].sort_values("model_b_toxic_prob", ascending=True)

def extract_samples(subset, n=3):
    res = []
    for _, r in subset.head(n).iterrows():
        res.append({
            "image_id": int(r["meme_id"]),
            "language": str(r["language"]),
            "text": str(r["text"]).encode("ascii", "backslashreplace").decode("ascii")[:120],
            "ground_truth": int(r["IN"]),
            "text_only_score": round(float(r["model_b_toxic_prob"]), 4),
            "multimodal_score": round(float(r["model_d_prob"]), 4),
            "prediction": int(r["model_d_pred"])
        })
    return res

results = {
    "contextual_inversion": {
        "description": "Text looks toxic/hostile but meme context is benign (GT=0)",
        "count": len(sub_ci),
        "samples": extract_samples(sub_ci)
    },
    "emergent_hate": {
        "description": "Text looks benign/innocuous (low text toxicity score) but meme is hateful (GT=1)",
        "count": len(sub_eh),
        "samples": extract_samples(sub_eh)
    },
    "explicit_hate": {
        "description": "Both modalities indicate hate (high text toxicity and GT=1)",
        "count": len(sub_xh),
        "samples": extract_samples(sub_xh)
    },
    "benign": {
        "description": "Both modalities are benign (low text toxicity and GT=0)",
        "count": len(sub_bn),
        "samples": extract_samples(sub_bn)
    }
}

with open("scratch/meme_experiments/model_d_pridemm/case_studies.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved case_studies.json successfully!")
