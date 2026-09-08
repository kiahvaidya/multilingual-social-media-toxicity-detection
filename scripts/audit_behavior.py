"""
Prediction Behavior Audit Suite for Phase 5B.

Evaluates MuRIL against a structured set of representative test cases across 12 distinct categories:
  1. Safe English
  2. Toxic English
  3. Ambiguous English
  4. Safe Hinglish
  5. Toxic Hinglish
  6. Ambiguous Hinglish (Auditing scores in the 0.60–0.67 range)
  7. Safe Hindi
  8. Toxic Hindi
  9. English threat
  10. Hinglish threat
  11. Hindi threat
  12. Obfuscated / slang toxicity

Outputs:
  - evaluation_results/behavioral_audit.json
  - Full false-positive and false-negative pattern audit
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config
from src.model import load_muril_model, predict_comment_muril
from src.moderation import RiskLevel, classify_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Curated Representative Dataset (12 Behavioral Categories) ──
AUDIT_CASES = [
    # 1. Safe English
    {
        "id": "ENG_SAFE_01",
        "category": "Safe English",
        "text": "Thank you so much for this tutorial, it helped me understand machine learning!",
        "expected_toxic": False,
        "notes": "Positive educational feedback",
    },
    {
        "id": "ENG_SAFE_02",
        "category": "Safe English",
        "text": "I disagree with your conclusion on this topic, but I respect the thoughtful analysis.",
        "expected_toxic": False,
        "notes": "Respectful civil disagreement",
    },
    # 2. Toxic English
    {
        "id": "ENG_TOX_01",
        "category": "Toxic English",
        "text": "You are a pathetic idiot and nobody wants to hear your worthless opinions.",
        "expected_toxic": True,
        "notes": "Direct insult & toxicity",
    },
    {
        "id": "ENG_TOX_02",
        "category": "Toxic English",
        "text": "Stop making videos you complete piece of trash.",
        "expected_toxic": True,
        "notes": "Hostile harassment",
    },
    # 3. Ambiguous English
    {
        "id": "ENG_AMB_01",
        "category": "Ambiguous English",
        "text": "This guitar solo is totally sick, absolutely wicked playing man!",
        "expected_toxic": False,
        "notes": "Slang compliment using words 'sick' and 'wicked'",
    },
    {
        "id": "ENG_AMB_02",
        "category": "Ambiguous English",
        "text": "I am literally dying of laughter right now, this comedian is insane.",
        "expected_toxic": False,
        "notes": "Hyperbole ('dying', 'insane')",
    },
    # 4. Safe Hinglish
    {
        "id": "HIN_ENG_SAFE_01",
        "category": "Safe Hinglish",
        "text": "Bhai bohot badhiya video banaya hai tune, dil khush ho gaya dekh ke.",
        "expected_toxic": False,
        "notes": "Praise in colloquial Hinglish",
    },
    {
        "id": "HIN_ENG_SAFE_02",
        "category": "Safe Hinglish",
        "text": "Mera laptop thoda slow chal raha hai, koi accha solution bata do please.",
        "expected_toxic": False,
        "notes": "Neutral query asking for tech advice",
    },
    # 5. Toxic Hinglish
    {
        "id": "HIN_ENG_TOX_01",
        "category": "Toxic Hinglish",
        "text": "Tu bohot bada kutta hai, sharam nahi aati aisi bakwas karte hue?",
        "expected_toxic": True,
        "notes": "Common Hinglish insult ('kutta')",
    },
    {
        "id": "HIN_ENG_TOX_02",
        "category": "Toxic Hinglish",
        "text": "Saale kamine nikal yahan se, teri aukaat nahi hai ye bolne ki.",
        "expected_toxic": True,
        "notes": "Severe verbal abuse in Hinglish ('kamine', 'aukaat')",
    },
    # 6. Ambiguous Hinglish (Focus category for 0.60–0.67 score clustering)
    {
        "id": "HIN_ENG_AMB_01",
        "category": "Ambiguous Hinglish",
        "text": "Bhai tune toh aag laga di bilkul phaad diya stage pe!",
        "expected_toxic": False,
        "notes": "Metaphorical excitement ('aag laga di', 'phaad diya')",
    },
    {
        "id": "HIN_ENG_AMB_02",
        "category": "Ambiguous Hinglish",
        "text": "Arey pagle rulayega kya ab, itna emotional video bana diya.",
        "expected_toxic": False,
        "notes": "Affectionate slang ('pagle rulayega kya')",
    },
    {
        "id": "HIN_ENG_AMB_03",
        "category": "Ambiguous Hinglish",
        "text": "Bhai kya bawaal cheez banayi hai yaar, dangerous lag raha hai bilkul.",
        "expected_toxic": False,
        "notes": "Colloquial 'bawaal' and 'dangerous' used admiringly",
    },
    {
        "id": "HIN_ENG_AMB_04",
        "category": "Ambiguous Hinglish",
        "text": "Bhai tere bina toh hum mar hi jayenge sach me.",
        "expected_toxic": False,
        "notes": "Colloquial hyperbole ('mar hi jayenge')",
    },
    # 7. Safe Hindi (Devanagari)
    {
        "id": "HIN_SAFE_01",
        "category": "Safe Hindi",
        "text": "आपका यह वीडियो बहुत ज्ञानवर्धक और उपयोगी है, बहुत-बहुत धन्यवाद।",
        "expected_toxic": False,
        "notes": "Formal respectful Hindi praise",
    },
    {
        "id": "HIN_SAFE_02",
        "category": "Safe Hindi",
        "text": "सभी मित्रों को सादर प्रणाम, आशा है आप सब सकुशल होंगे।",
        "expected_toxic": False,
        "notes": "Friendly traditional greeting",
    },
    # 8. Toxic Hindi (Devanagari)
    {
        "id": "HIN_TOX_01",
        "category": "Toxic Hindi",
        "text": "तू बहुत बड़ा मूर्ख और नीच इंसान है, तेरा कोई भविष्य नहीं है।",
        "expected_toxic": True,
        "notes": "Direct insult in Hindi ('मूर्ख', 'नीच')",
    },
    {
        "id": "HIN_TOX_02",
        "category": "Toxic Hindi",
        "text": "अपनी गंदी जुबान बंद कर और यहाँ से दफा हो जा कमीने।",
        "expected_toxic": True,
        "notes": "Verbal hostility in Devanagari",
    },
    # 9. English Threat
    {
        "id": "ENG_THR_01",
        "category": "English threat",
        "text": "I know where you live and I am going to find you and kill you.",
        "expected_toxic": True,
        "notes": "Explicit death threat",
    },
    {
        "id": "ENG_THR_02",
        "category": "English threat",
        "text": "Watch your back after work, someone is going to break your neck tonight.",
        "expected_toxic": True,
        "notes": "Physical violence threat",
    },
    # 10. Hinglish Threat
    {
        "id": "HIN_ENG_THR_01",
        "category": "Hinglish threat",
        "text": "Ghar se bahar nikal, teri jaan le lunga mai aaj.",
        "expected_toxic": True,
        "notes": "Death threat in Hinglish ('jaan le lunga')",
    },
    {
        "id": "HIN_ENG_THR_02",
        "category": "Hinglish threat",
        "text": "Tujhe zameen me gaad dunga dekh lena, bach ke rehna.",
        "expected_toxic": True,
        "notes": "Violent threat in Hinglish",
    },
    # 11. Hindi Threat (Devanagari)
    {
        "id": "HIN_THR_01",
        "category": "Hindi threat",
        "text": "अगर तूने दोबारा यह किया तो मैं तुझे जान से मार डालूँगा।",
        "expected_toxic": True,
        "notes": "Explicit threat in Devanagari Hindi ('जान से मार डालूँगा')",
    },
    {
        "id": "HIN_THR_02",
        "category": "Hindi threat",
        "text": "तेरा कत्ल कर दूँगा आज रात, बच के रहना।",
        "expected_toxic": True,
        "notes": "Violent threat in Devanagari ('कत्ल कर दूँगा')",
    },
    # 12. Obfuscated / Slang Toxicity
    {
        "id": "OBF_TOX_01",
        "category": "Obfuscated/slang toxicity",
        "text": "You are a f**king b!tch and a piece of sh!t.",
        "expected_toxic": True,
        "notes": "Leetspeak / asterisk-masked English profanity",
    },
    {
        "id": "OBF_TOX_02",
        "category": "Obfuscated/slang toxicity",
        "text": "Tu ek number ka ch*tiya aur k*mina insaan hai.",
        "expected_toxic": True,
        "notes": "Asterisk-masked Hinglish profanity",
    },
]


def run_behavioral_audit():
    logger.info("==================================================")
    logger.info("  Phase 5B — Prediction Behavior Audit Suite      ")
    logger.info("==================================================")

    model, tokenizer, labels, thresholds, device = load_muril_model()
    logger.info("Auditing %d test cases with MuRIL on %s...", len(AUDIT_CASES), device)

    audit_results = []
    category_summary = {}

    false_positives = []
    false_negatives = []
    ambiguous_hinglish_scores = []

    for case in AUDIT_CASES:
        cid = case["id"]
        cat = case["category"]
        text = case["text"]
        expected_toxic = case["expected_toxic"]

        preds, score, intensity, flags = predict_comment_muril(
            text, model, tokenizer, labels, device, thresholds=thresholds
        )
        risk_level, icon, explanation = classify_risk(preds, flags, thresholds)

        is_flagged_toxic = any(flags.values())
        flagged_categories = [lbl for lbl, fl in flags.items() if fl]

        # Audit outcome categorization
        if expected_toxic and is_flagged_toxic:
            outcome = "TP"  # True Positive
        elif not expected_toxic and not is_flagged_toxic:
            outcome = "TN"  # True Negative
        elif not expected_toxic and is_flagged_toxic:
            outcome = "FP"  # False Positive
            false_positives.append({
                "id": cid,
                "category": cat,
                "text": text,
                "score": score,
                "flagged_categories": flagged_categories,
                "predictions": preds,
                "explanation": explanation,
                "notes": case["notes"],
            })
        else:
            outcome = "FN"  # False Negative
            false_negatives.append({
                "id": cid,
                "category": cat,
                "text": text,
                "score": score,
                "predictions": preds,
                "explanation": explanation,
                "notes": case["notes"],
            })

        # Track ambiguous Hinglish score distribution specifically
        if cat == "Ambiguous Hinglish":
            ambiguous_hinglish_scores.append({
                "id": cid,
                "text": text,
                "score": score,
                "toxic_pred": preds.get("toxic", 0.0),
                "in_060_067_range": (0.60 <= score <= 0.67 or 0.60 <= preds.get("toxic", 0.0) <= 0.67),
                "flagged": is_flagged_toxic,
                "risk_level": risk_level.value,
            })

        if cat not in category_summary:
            category_summary[cat] = {"total": 0, "correct": 0, "FP": 0, "FN": 0}
        category_summary[cat]["total"] += 1
        if outcome in ("TP", "TN"):
            category_summary[cat]["correct"] += 1
        elif outcome == "FP":
            category_summary[cat]["FP"] += 1
        else:
            category_summary[cat]["FN"] += 1

        audit_results.append({
            "id": cid,
            "category": cat,
            "text": text,
            "expected_toxic": expected_toxic,
            "is_flagged_toxic": is_flagged_toxic,
            "outcome": outcome,
            "score": score,
            "intensity": intensity,
            "risk_level": risk_level.value,
            "risk_icon": icon,
            "flagged_categories": flagged_categories,
            "predictions": preds,
            "notes": case["notes"],
        })

    total_cases = len(AUDIT_CASES)
    total_correct = sum(v["correct"] for v in category_summary.values())
    accuracy = round((total_correct / total_cases) * 100, 1)

    logger.info("\n--- BEHAVIORAL AUDIT SUMMARY ---")
    logger.info("Total Audited Cases: %d", total_cases)
    logger.info("Overall Accuracy: %.1f%% (%d/%d)", accuracy, total_correct, total_cases)
    logger.info("False Positives: %d", len(false_positives))
    logger.info("False Negatives: %d", len(false_negatives))

    # Ambiguous Hinglish Audit Analysis
    in_range_count = sum(1 for x in ambiguous_hinglish_scores if x["in_060_067_range"])
    logger.info(
        "Ambiguous Hinglish: %d of %d produced scores in the 0.60-0.67 range",
        in_range_count, len(ambiguous_hinglish_scores)
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "google/muril-base-cased (Fine-Tuned)",
        "thresholds_used": thresholds,
        "metrics": {
            "total_cases": total_cases,
            "correct": total_correct,
            "accuracy_percent": accuracy,
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
        },
        "category_summary": category_summary,
        "ambiguous_hinglish_audit": {
            "total_samples": len(ambiguous_hinglish_scores),
            "in_060_067_range_count": in_range_count,
            "samples": ambiguous_hinglish_scores,
        },
        "false_positive_patterns": false_positives,
        "false_negative_patterns": false_negatives,
        "all_cases": audit_results,
    }

    out_file = _PROJECT_ROOT / "evaluation_results" / "behavioral_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Detailed audit report saved to: %s", out_file)
    return report


if __name__ == "__main__":
    run_behavioral_audit()
