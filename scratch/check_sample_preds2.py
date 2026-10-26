import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.model import load_muril_model, predict_comment_muril

model, tokenizer, labels, thresholds, device = load_muril_model()

samples = [
    ("en_safe", "Thank you so much for your help and kindness."),
    ("en_toxic", "You are an idiot and nobody likes you."),
    ("en_threat", "I will kill you and burn your house down."),
    ("hg_safe", "aapka bahut shukriya, bahut accha laga"),
    ("hg_toxic", "tu kutta hai mc bhosdike nikal"),
    ("hi_safe", "नमस्ते, आपका दिन शुभ और मंगलमय हो।"),
    ("hi_toxic", "तू महामूर्ख और हरामी है, तुझे मार डालूंगा।"),
]

for tag, text in samples:
    preds, score, intensity, flags = predict_comment_muril(text, model, tokenizer, labels, device, thresholds)
    print(f"[{tag}] '{text}' -> toxic: {preds['toxic']:.4f}, max: {score:.4f}, flags: {flags}")
