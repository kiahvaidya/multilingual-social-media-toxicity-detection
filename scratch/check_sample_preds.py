import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.model import load_muril_model, predict_comment_muril

model, tokenizer, labels, thresholds, device = load_muril_model()
text_hi = "यह बहुत ही सुंदर और ज्ञानवर्धक वीडियो है, आपका बहुत धन्यवाद।"
text_hg = "Bhai bohot accha video banaya hai, thanks for sharing!"

preds_hi, score_hi, int_hi, flags_hi = predict_comment_muril(text_hi, model, tokenizer, labels, device, thresholds)
preds_hg, score_hg, int_hg, flags_hg = predict_comment_muril(text_hg, model, tokenizer, labels, device, thresholds)

print("Hindi preds:", preds_hi)
print("Hindi flags:", flags_hi)
print("Hinglish preds:", preds_hg)
print("Hinglish flags:", flags_hg)
