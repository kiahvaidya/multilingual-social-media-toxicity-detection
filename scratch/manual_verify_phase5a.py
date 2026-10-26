import sys
from pathlib import Path

# Force UTF-8 stdout
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import load_muril_model, predict_comment_muril

model, tokenizer, labels, thresholds, device = load_muril_model()

test_cases = [
    ("English Safe", "Thank you so much for this helpful and comprehensive tutorial!"),
    ("English Toxic / Insult", "You are an absolute idiot, a complete disgrace, and a total waste of space."),
    ("English Threat", "I know where you live and I am going to kill you tonight."),
    ("Hinglish Safe", "Bhai bohot accha explain kiya hai, maza aa gaya!"),
    ("Hinglish Toxic", "Tu chutiya hai behenchod nikal yahan se."),
    ("Hindi Safe", "यह बहुत ही सुंदर और उपयोगी जानकारी है, धन्यवाद।"),
    ("Hindi Toxic / Insult", "तू सबसे बड़ा कुत्ता और मूर्ख है।"),
    ("Hindi Threat", "तुझे जिंदा नहीं छोडूंगा, मार डालूंगा।"),
]

print("=" * 95)
print("PHASE 5A MANUAL VERIFICATION: REPRESENTATIVE MULTILINGUAL PREDICTIONS")
print("=" * 95)

for category, text in test_cases:
    preds, score, intensity, flags = predict_comment_muril(text, model, tokenizer, labels, device, thresholds)
    flagged = [k for k, v in flags.items() if v]
    flag_str = ", ".join(flagged) if flagged else "NONE (Safe)"
    print(f"\n[Category]: {category}")
    print(f"[Input Text]: \"{text}\"")
    print(f"[Intensity]: {intensity} (Max Score: {score:.4f})")
    print(f"[Flagged Labels (vs Phase 4C Thresholds)]: {flag_str}")
    print("[Per-Label Probabilities]:")
    for k, v in preds.items():
        th = thresholds[k]
        status = "ALERT" if flags[k] else "OK"
        print(f"   - {k:<15}: {v:.4f} (Threshold: {th:.2f}) [{status}]")
    print("-" * 95)
