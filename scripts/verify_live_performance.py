"""
Live Performance and Robustness Verification Script.
Tests the complete end-to-end stack:
  Preprocessing (Deobfuscator) -> MuRIL Inference -> Moderation Risk Tier -> JEV Ambiguity Layer
"""

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import load_muril_model, predict_comment_muril
from src.moderation import moderate_comment_with_context, RiskLevel
from src.jev import MockJEVDecisionProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEST_CASES = [
    # ── Category A: Standard Toxic Comments ──
    {
        "id": "A1_STD_TOXIC_ENG",
        "category": "Direct English Toxicity",
        "text": "You are a complete idiot, get out of here and never come back.",
        "expected_risk": [RiskLevel.HIGH_RISK, RiskLevel.REVIEW],
        "must_not_be": RiskLevel.SAFE,
    },
    {
        "id": "A2_STD_TOXIC_HINGLISH",
        "category": "Direct Hinglish Toxicity",
        "text": "Tu bohot bada chutiya hai, sharam kar thodi.",
        "expected_risk": [RiskLevel.HIGH_RISK, RiskLevel.REVIEW],
        "must_not_be": RiskLevel.SAFE,
    },
    {
        "id": "A3_STD_TOXIC_HINDI",
        "category": "Direct Hindi Threat",
        "text": "अगर तूने दोबारा यह किया तो मैं तुझे जान से मार डालूँगा।",
        "expected_risk": [RiskLevel.HIGH_RISK],
        "must_not_be": RiskLevel.SAFE,
    },

    # ── Category B: Adversarial & Obfuscated Attacks ──
    {
        "id": "B1_OBF_HOMOGLYPH",
        "category": "Adversarial Homoglyphs (Cyrillic)",
        "text": "You are рathetic and a disgusting failure.",  # Cyrillic 'р'
        "expected_risk": [RiskLevel.HIGH_RISK, RiskLevel.REVIEW],
        "must_not_be": RiskLevel.SAFE,
    },
    {
        "id": "B2_OBF_LEETSPEAK",
        "category": "Adversarial Leetspeak",
        "text": "Go k!ll y0ur$elf right now, you 1diot.",
        "expected_risk": [RiskLevel.HIGH_RISK],
        "must_not_be": RiskLevel.SAFE,
    },
    {
        "id": "B3_OBF_SPACED",
        "category": "Intra-word Spacing & Punctuation",
        "text": "They will k . i . l . l you and f.u.c.k your family.",
        "expected_risk": [RiskLevel.HIGH_RISK],
        "must_not_be": RiskLevel.SAFE,
    },
    {
        "id": "B4_OBF_MASKED",
        "category": "Asterisk-Masked Hinglish Slurs",
        "text": "Tu ek number ka ch**tiya aur bh**dike insaan hai.",
        "expected_risk": [RiskLevel.HIGH_RISK, RiskLevel.REVIEW],
        "must_not_be": RiskLevel.SAFE,
    },

    # ── Category C: Quoted Speech & Counter-Speech (Victim Protection) ──
    {
        "id": "C1_COUNTER_SPEECH_QUOTE",
        "category": "Quoted Harassment Report",
        "text": 'Someone sent me "go kill yourself" and it broke my heart, this is horrible.',
        "expected_risk": [RiskLevel.SAFE, RiskLevel.REVIEW],
        "must_not_be": RiskLevel.HIGH_RISK,  # Must NOT falsely ban victim
    },
    {
        "id": "C2_COUNTER_SPEECH_INTERVENTION",
        "category": "Anti-Harassment Intervention",
        "text": "Stop calling people chutiya, show some basic respect to others.",
        "expected_risk": [RiskLevel.SAFE, RiskLevel.REVIEW],
        "must_not_be": RiskLevel.HIGH_RISK,
    },

    # ── Category D: Sarcastic Mockery & Pragmatic Irony ──
    {
        "id": "D1_SARCASM_MOCKERY",
        "category": "Sarcastic Mockery",
        "text": "Great job breaking the production server on Friday, you absolute genius.",
        "expected_risk": [RiskLevel.REVIEW, RiskLevel.HIGH_RISK],
        "must_not_be": RiskLevel.SAFE,  # Must NOT silently bypass as Safe
    },
    {
        "id": "D2_SARCASM_HINGLISH",
        "category": "Sarcastic Hinglish Mockery",
        "text": "Shabash, pura kaam kharab kar diya mahashay, kya baat hai.",
        "expected_risk": [RiskLevel.REVIEW, RiskLevel.HIGH_RISK],
        "must_not_be": RiskLevel.SAFE,
    },

    # ── Category E: Innocuous Slang & Celebratory Swearing ──
    {
        "id": "E1_PRAISE_SLANG",
        "category": "Enthusiastic Slang Metaphor",
        "text": "Bro you absolutely killed it on stage tonight, pure fire!",
        "expected_risk": [RiskLevel.SAFE],
        "must_not_be": RiskLevel.HIGH_RISK,
    },
    {
        "id": "E2_CELEBRATORY_SWEARING",
        "category": "Cathartic Celebratory Swearing",
        "text": "Fuck yeah, we finally passed the toughest exam!",
        "expected_risk": [RiskLevel.SAFE],
        "must_not_be": RiskLevel.HIGH_RISK,
    },
    {
        "id": "E3_HINGLISH_BANTER",
        "category": "Friendly Hinglish Banter",
        "text": "Job lag gayi teri bc, party kab de raha hai bhai?",
        "expected_risk": [RiskLevel.SAFE],
        "must_not_be": RiskLevel.HIGH_RISK,
    },
    {
        "id": "E4_SAFE_CLEAN",
        "category": "Clean Educational Feedback",
        "text": "Thank you so much for this tutorial, it really helped me understand everything.",
        "expected_risk": [RiskLevel.SAFE],
        "must_not_be": RiskLevel.HIGH_RISK,
    },
]

def main():
    logger.info("Initializing MuRIL multi-label model for live performance verification...")
    model, tokenizer, labels, thresholds, device = load_muril_model()
    mock_jev = MockJEVDecisionProvider()

    passed_count = 0
    total_count = len(TEST_CASES)
    detailed_results = []

    print("\n" + "=" * 90)
    print("LIVE PIPELINE PERFORMANCE VERIFICATION RESULTS")
    print("=" * 90)

    for tc in TEST_CASES:
        preds, score, intensity, flags = predict_comment_muril(
            tc["text"], model, tokenizer, labels, device, thresholds=thresholds
        )

        mod_result = moderate_comment_with_context(
            text=tc["text"],
            predictions=preds,
            is_toxic_flags=flags,
            thresholds=thresholds,
            jev_provider=mock_jev,
        )

        assigned_tier = mod_result.final_risk_level
        passed = (assigned_tier in tc["expected_risk"]) and (assigned_tier != tc["must_not_be"])

        if passed:
            passed_count += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        result_entry = {
            "id": tc["id"],
            "category": tc["category"],
            "text": tc["text"],
            "score": round(score, 3),
            "tier": assigned_tier.value,
            "badge": mod_result.final_badge_icon,
            "jev_invoked": mod_result.jev_invoked,
            "verdict": verdict,
            "explanation": mod_result.final_explanation,
        }
        detailed_results.append(result_entry)

        print(f"[{verdict}] {tc['category']:<36} | Tier: {mod_result.final_badge_icon} {assigned_tier.value:<9} | Max Score: {score:.3f}")
        print(f"       Text: \"{tc['text']}\"")
        print(f"       Reason: {mod_result.final_explanation[:90]}...\n")

    print("=" * 90)
    print(f"VERIFICATION SUMMARY: {passed_count}/{total_count} PASSED ({passed_count / total_count * 100:.1f}%)")
    print("=" * 90 + "\n")

    return 0 if passed_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())

