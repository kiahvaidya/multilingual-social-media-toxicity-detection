"""
Tests for World-Class Content Moderation Enhancements:
  - Module 1: UTS #39 Unicode Homoglyphs, Masked Profanity & Adversarial Leetspeak Normalizer
  - Module 2: Quoted Speech & Counter-Speech Attribution Parser and Sarcasm Heuristics
  - Module 3: Sarcastic Mockery & Counter-Speech JEV Contextual Adjudication
"""

import pytest

from src.preprocessing import clean_text_for_transformer, deobfuscate_text, preprocess_text
from src.moderation import (
    RiskLevel,
    classify_risk,
    should_invoke_jev,
    detect_counter_speech_context,
    detect_sarcasm_indicators,
    moderate_comment_with_context,
)
from src.jev import (
    JEVDecision,
    JEVIntent,
    MockJEVDecisionProvider,
)
from src.config import config


# ============================================================================
# MODULE 1 TESTS: Homoglyphs, Masked Profanity & Leetspeak Deobfuscation
# ============================================================================

class TestModule1Deobfuscation:
    """Verifies robustness against adversarial evasions (Google Jigsaw Perspective paradigm)."""

    def test_unicode_uts39_homoglyphs(self):
        # Cyrillic 'р' (U+0440) lookalike for Latin 'p'
        assert deobfuscate_text("рathetic") == "pathetic"
        # Cyrillic 'а', 'с', 'е', 'о'
        assert deobfuscate_text("\u0430\u0441\u0435\u043E") == "aceo"
        # Cyrillic 'о' in English word
        assert deobfuscate_text("idi\u043Et") == "idiot"

    def test_intra_word_noise_and_spacing_collapse(self):
        # Punctuation-separated letters
        assert deobfuscate_text("k . i . l . l") == "kill"
        assert deobfuscate_text("f.u.c.k") == "fuck"
        assert deobfuscate_text("s-h-i-t") == "shit"
        assert deobfuscate_text("b_h_o_s_d_i_k_e") == "bhosdike"

    def test_targeted_spaced_evasions(self):
        assert deobfuscate_text("k i l l") == "kill"
        assert deobfuscate_text("s h i t") == "shit"
        assert deobfuscate_text("c h u t i y a") == "chutiya"

    def test_masked_profanity_expansion(self):
        # English masked terms
        assert deobfuscate_text("f**k") == "fuck"
        assert deobfuscate_text("b!tch") == "bitch"
        assert deobfuscate_text("sh!t") == "shit"
        assert deobfuscate_text("a$$hole") == "asshole"
        assert deobfuscate_text("k!ll") == "kill"

        # Hinglish masked terms
        assert deobfuscate_text("ch**tiya") == "chutiya"
        assert deobfuscate_text("m**archod") == "madarchod"
        assert deobfuscate_text("bh**dike") == "bhosdike"

    def test_leetspeak_symbol_substitution_within_words(self):
        # '!' -> 'i' inside words
        assert deobfuscate_text("k!ll") == "kill"
        assert deobfuscate_text("sh!rt") == "shirt"

        # '$' -> 's' inside words
        assert deobfuscate_text("y0ur$elf") == "yourself"

        # '@' -> 'a' inside words or leading words
        assert deobfuscate_text("@sshole") == "asshole"

        # Digits inside words (0->o, 1->i, 3->e, etc.)
        assert deobfuscate_text("1diot") == "idiot"
        assert deobfuscate_text("l0ser") == "loser"
        assert deobfuscate_text("c00l") == "cool"
        assert deobfuscate_text("m0therfuck3r") == "motherfucker"

    def test_preserves_legitimate_numbers_and_punctuation(self):
        # Currency and pure numbers must NOT be corrupted
        assert deobfuscate_text("This costs $100 and that is 500 dollars") == "This costs $100 and that is 500 dollars"
        # Trailing sentence exclamation marks must be preserved
        assert deobfuscate_text("Hello world!") == "Hello world!"
        # Devanagari script must be preserved intact
        hindi_text = "यह एक बहुत अच्छा गाना है"
        assert deobfuscate_text(hindi_text) == hindi_text

    def test_clean_text_for_transformer_integration(self):
        # clean_text_for_transformer applies deobfuscation
        raw = "Check this out: рathetic f.u.c.k and k!ll y0ur$elf @user http://example.com"
        cleaned = clean_text_for_transformer(raw)
        assert "pathetic" in cleaned
        assert "fuck" in cleaned
        assert "kill yourself" in cleaned
        assert "http" not in cleaned
        assert "@user" not in cleaned


# ============================================================================
# MODULE 2 TESTS: Counter-Speech Attribution & Sarcasm Heuristics
# ============================================================================

class TestModule2ModerationAttribution:
    """Verifies counter-speech attribution and pragmatic irony / sarcasm detection."""

    def test_counter_speech_quoted_abuse_detection(self):
        text1 = 'He called me an "idiot" and that was horrible'
        is_cs1, reason1 = detect_counter_speech_context(text1)
        assert is_cs1 is True
        assert "Counter-speech or victim reporting" in reason1

        text2 = 'Someone sent me "go kill yourself" and it broke my heart'
        is_cs2, reason2 = detect_counter_speech_context(text2)
        assert is_cs2 is True

        text3 = 'Saying "all immigrants are parasites" is disgusting hate speech'
        is_cs3, reason3 = detect_counter_speech_context(text3)
        assert is_cs3 is True

    def test_corrective_counter_speech_detection(self):
        text = "Stop calling people chutiya, show some respect"
        is_cs, reason = detect_counter_speech_context(text)
        assert is_cs is True
        assert "counter-speech" in reason.lower()

    def test_direct_abuse_does_not_trigger_counter_speech(self):
        text = "You are a disgusting idiot and a chutiya"
        is_cs, _ = detect_counter_speech_context(text)
        assert is_cs is False

    def test_sarcasm_lexical_contradiction_detection(self):
        text1 = "Great job breaking the production server, you absolute genius"
        is_sarc1, reason1 = detect_sarcasm_indicators(text1)
        assert is_sarc1 is True
        assert "Lexical contradiction" in reason1

        text2 = "Brilliant work losing all our customer data, bravo"
        is_sarc2, _ = detect_sarcasm_indicators(text2)
        assert is_sarc2 is True

        text3 = "Shabash, pura kaam kharab kar diya"
        is_sarc3, _ = detect_sarcasm_indicators(text3)
        assert is_sarc3 is True

    def test_genuine_praise_does_not_trigger_sarcasm(self):
        text = "Great job on the new release, you guys are absolute geniuses!"
        is_sarc, _ = detect_sarcasm_indicators(text)
        assert is_sarc is False

    def test_classify_risk_downgrades_counter_speech_to_review(self):
        # A victim reporting severe abuse would trigger severe_toxic if evaluated purely lexically
        predictions = {"toxic": 0.82, "insult": 0.78, "severe_toxic": 0.65, "threat": 0.10, "obscene": 0.30, "identity_hate": 0.10}
        flags = {"toxic": True, "insult": True, "severe_toxic": True, "threat": False, "obscene": False, "identity_hate": False}
        thresholds = config.FROZEN_MURIL_THRESHOLDS

        # Without counter-speech context, severe_toxic flags High Risk
        risk_raw, icon_raw, _ = classify_risk(predictions, flags, thresholds, text="You are a severe toxic idiot")
        assert risk_raw == RiskLevel.HIGH_RISK
        assert icon_raw == "🔴"

        # With counter-speech reporting context, it routes to Review with explanation
        reporting_text = 'Someone messaged me "severe toxic idiot" and it was horrible and unacceptable'
        risk_cs, icon_cs, expl_cs = classify_risk(predictions, flags, thresholds, text=reporting_text)
        assert risk_cs == RiskLevel.REVIEW
        assert icon_cs == "🟡"
        assert "Counter-speech / victim reporting" in expl_cs

    def test_classify_risk_elevates_sarcasm_to_review(self):
        # Sarcastic text with sub-threshold toxicity scores (~0.35) that would otherwise be classified Safe
        predictions = {"toxic": 0.35, "insult": 0.32, "severe_toxic": 0.05, "threat": 0.02, "obscene": 0.05, "identity_hate": 0.02}
        flags = {k: False for k in predictions}
        thresholds = config.FROZEN_MURIL_THRESHOLDS

        sarcastic_text = "Great job breaking the production server, you absolute genius"
        risk, icon, expl = classify_risk(predictions, flags, thresholds, text=sarcastic_text)
        assert risk == RiskLevel.REVIEW
        assert icon == "🟡"
        assert "sarcastic mockery" in expl.lower()

    def test_should_invoke_jev_triggers_for_counter_speech_and_sarcasm(self):
        predictions = {"toxic": 0.50, "insult": 0.40, "severe_toxic": 0.10, "threat": 0.05, "obscene": 0.10, "identity_hate": 0.05}
        flags = {"toxic": True, "insult": False, "severe_toxic": False, "threat": False, "obscene": False, "identity_hate": False}
        thresholds = config.FROZEN_MURIL_THRESHOLDS

        # Counter-speech triggers JEV
        cs_text = 'He called me an "idiot" and that was horrible'
        should_cs, reason_cs = should_invoke_jev(cs_text, predictions, flags, RiskLevel.REVIEW, thresholds)
        assert should_cs is True
        assert "Counter-speech" in reason_cs

        # Sarcasm triggers JEV
        sarc_text = "Great job breaking the entire database, absolute genius"
        should_sarc, reason_sarc = should_invoke_jev(sarc_text, predictions, flags, RiskLevel.REVIEW, thresholds)
        assert should_sarc is True
        assert "sarcastic mockery" in reason_sarc.lower()


# ============================================================================
# MODULE 3 TESTS: JEV Intents & Contextual Moderation Adjudication
# ============================================================================

class TestModule3JEVAdjudication:
    """Verifies JEV intent handling and MockJEVDecisionProvider adjudication."""

    def test_jev_intent_enum_and_parsing(self):
        assert JEVIntent.SARCASTIC_MOCKERY == "sarcastic_mockery"
        assert JEVIntent.COUNTER_SPEECH_OR_REPORTING == "counter_speech_or_reporting"

        # Parsing aliases
        assert JEVIntent.from_raw("sarcastic_mockery") == JEVIntent.SARCASTIC_MOCKERY
        assert JEVIntent.from_raw("sarcasm") == JEVIntent.SARCASTIC_MOCKERY
        assert JEVIntent.from_raw("counter_speech") == JEVIntent.COUNTER_SPEECH_OR_REPORTING
        assert JEVIntent.from_raw("reporting_abuse") == JEVIntent.COUNTER_SPEECH_OR_REPORTING
        assert JEVIntent.from_raw("quoted_abuse") == JEVIntent.COUNTER_SPEECH_OR_REPORTING

    def test_mock_jev_adjudicates_counter_speech_to_safe(self):
        provider = MockJEVDecisionProvider()
        text = 'He called me an "idiot" and that was horrible'
        muril_scores = {"toxic": 0.55, "insult": 0.52, "severe_toxic": 0.10, "threat": 0.02, "obscene": 0.10, "identity_hate": 0.02}
        muril_flags = {"toxic": True, "insult": True, "severe_toxic": False, "threat": False, "obscene": False, "identity_hate": False}

        decision = provider.assess_context(text, muril_scores, muril_flags, RiskLevel.REVIEW)
        assert decision.intent == JEVIntent.COUNTER_SPEECH_OR_REPORTING
        assert decision.contextual_toxicity is False
        assert decision.moderation == RiskLevel.SAFE
        assert decision.human_review_required is False

    def test_mock_jev_adjudicates_sarcastic_mockery_to_review(self):
        provider = MockJEVDecisionProvider()
        text = "Great job breaking the production server, you absolute genius"
        muril_scores = {"toxic": 0.36, "insult": 0.30, "severe_toxic": 0.05, "threat": 0.02, "obscene": 0.05, "identity_hate": 0.02}
        muril_flags = {k: False for k in muril_scores}

        decision = provider.assess_context(text, muril_scores, muril_flags, RiskLevel.REVIEW)
        assert decision.intent == JEVIntent.SARCASTIC_MOCKERY
        assert decision.contextual_toxicity is True
        assert decision.moderation == RiskLevel.REVIEW
        assert decision.human_review_required is True

    def test_end_to_end_contextual_moderation_with_counter_speech(self):
        provider = MockJEVDecisionProvider()
        text = 'Someone said "you are a chutiya" and it was disgusting and hurtful'
        predictions = {"toxic": 0.70, "insult": 0.65, "severe_toxic": 0.20, "threat": 0.05, "obscene": 0.40, "identity_hate": 0.05}
        flags = {"toxic": True, "insult": True, "severe_toxic": False, "threat": False, "obscene": False, "identity_hate": False}
        thresholds = config.FROZEN_MURIL_THRESHOLDS

        result = moderate_comment_with_context(
            text=text,
            predictions=predictions,
            is_toxic_flags=flags,
            thresholds=thresholds,
            jev_provider=provider,
        )

        assert result.base_risk_level == RiskLevel.REVIEW
        assert result.is_ambiguous is True
        assert result.jev_invoked is True
        # JEV confirms counter-speech and clears false positive to Safe
        assert result.final_risk_level == RiskLevel.SAFE
        assert result.final_badge_icon == "🟢"
        assert result.jev_decision.intent == JEVIntent.COUNTER_SPEECH_OR_REPORTING

    def test_end_to_end_contextual_moderation_with_praise_and_profanity(self):
        provider = MockJEVDecisionProvider()
        text = "HEY MAN U ARE THE FUCKING GOAT"
        # The swear word triggers toxic and obscene probabilities
        predictions = {"toxic": 0.78, "insult": 0.40, "severe_toxic": 0.10, "threat": 0.05, "obscene": 0.82, "identity_hate": 0.05}
        flags = {"toxic": True, "insult": False, "severe_toxic": False, "threat": False, "obscene": True, "identity_hate": False}
        thresholds = config.FROZEN_MURIL_THRESHOLDS

        result = moderate_comment_with_context(
            text=text,
            predictions=predictions,
            is_toxic_flags=flags,
            thresholds=thresholds,
            jev_provider=provider,
        )

        # Base moderation catches the praise context and assigns Safe
        assert result.base_risk_level in (RiskLevel.REVIEW, RiskLevel.SAFE)
        assert result.is_ambiguous is True
        assert result.jev_invoked is True
        # JEV confirms non-toxic praise / slang and resolves to Safe
        assert result.final_risk_level == RiskLevel.SAFE
        assert result.final_badge_icon == "🟢"
        assert result.jev_decision.intent == JEVIntent.NON_TOXIC_PRAISE_OR_EXCITEMENT


# ============================================================================
# MODULE 4 TESTS: Word Class Systems & Syntactic Constituency Parsing
# ============================================================================

class TestModule4WordClassSyntacticEngine:
    """Verifies syntactic role decomposition and word class constituency parsing."""

    def test_intensified_praise_syntactic_decomposition(self):
        from src.moderation import analyze_syntactic_intent

        cases = [
            ("HEY MAN U ARE THE FUCKING GOAT", "goat", True, False),
            ("you are a fucking wizard", "wizard", True, False),
            ("she is bloody brilliant", "brilliant", True, False),
            ("this track is fucking fire", "fire", True, False),
            ("he is an absolute beast", "beast", True, False),
            ("bro you are insanely fucking good", "good", True, False),
            ("this code is clean as fuck", "clean", True, False),
            ("bhai tu katai bawaal hai", "bawaal", True, False),
            ("she fucking killed it", "killed it", True, False),
            ("u rock man", "you rock", True, False),
        ]

        for text, expected_head, exp_praise, exp_insult in cases:
            res = analyze_syntactic_intent(text)
            assert res.is_intensified_praise is exp_praise, f"Failed praise for '{text}'"
            assert res.is_targeted_insult is exp_insult, f"Unexpected insult for '{text}'"
            assert res.praise_headword is not None, f"Expected praise headword for '{text}'"

    def test_targeted_insult_syntactic_decomposition(self):
        from src.moderation import analyze_syntactic_intent

        cases = [
            ("you are a fucking idiot", "idiot", False, True),
            ("u fucking clown", "clown", False, True),
            ("he is a pathetic loser", "loser", False, True),
            ("she is a disgusting bitch", "bitch", False, True),
            ("tu saala kutta hai", "kutta", False, True),
            ("you are an absolute moron", "moron", False, True),
        ]

        for text, expected_head, exp_praise, exp_insult in cases:
            res = analyze_syntactic_intent(text)
            assert res.is_targeted_insult is exp_insult, f"Failed insult for '{text}'"
            assert res.is_intensified_praise is exp_praise, f"Unexpected praise for '{text}'"
            assert res.insult_headword is not None, f"Expected insult headword for '{text}'"

    def test_untargeted_profanity_syntactic_decomposition(self):
        from src.moderation import analyze_syntactic_intent

        cases = [
            "what the fuck is going on",
            "holy shit that hurt",
            "damn it is raining again",
            "oh fuck I forgot my keys",
        ]

        for text in cases:
            res = analyze_syntactic_intent(text)
            assert res.is_untargeted_profanity is True, f"Failed untargeted profanity for '{text}'"
            assert res.is_targeted_insult is False
            assert res.is_intensified_praise is False

    def test_classify_risk_word_class_resolutions(self):
        thresholds = config.FROZEN_MURIL_THRESHOLDS

        # 1. Intensified Praise: expletive intensifiers + positive headwords -> SAFE
        praise_preds = {"toxic": 0.78, "obscene": 0.82, "insult": 0.40, "severe_toxic": 0.10, "threat": 0.05, "identity_hate": 0.05}
        praise_flags = {"toxic": True, "obscene": True, "insult": False, "severe_toxic": False, "threat": False, "identity_hate": False}

        risk1, icon1, expl1 = classify_risk(praise_preds, praise_flags, thresholds, text="HEY MAN U ARE THE FUCKING GOAT")
        assert risk1 == RiskLevel.SAFE
        assert icon1 == "🟢"
        assert "Enthusiastic praise" in expl1

        risk2, icon2, expl2 = classify_risk(praise_preds, praise_flags, thresholds, text="you are a fucking wizard")
        assert risk2 == RiskLevel.SAFE
        assert icon2 == "🟢"

        # 2. Targeted Insult: expletive + insult headword + target -> HIGH_RISK
        insult_preds = {"toxic": 0.85, "obscene": 0.80, "insult": 0.82, "severe_toxic": 0.20, "threat": 0.05, "identity_hate": 0.05}
        insult_flags = {"toxic": True, "obscene": True, "insult": True, "severe_toxic": False, "threat": False, "identity_hate": False}

        risk_insult, icon_insult, expl_insult = classify_risk(insult_preds, insult_flags, thresholds, text="you are a fucking idiot")
        assert risk_insult == RiskLevel.HIGH_RISK
        assert icon_insult == "🔴"
        assert "Targeted personal insult" in expl_insult

        # 3. Untargeted Profanity: expletive without insult/threat -> REVIEW (not High Risk)
        untargeted_preds = {"toxic": 0.55, "obscene": 0.80, "insult": 0.15, "severe_toxic": 0.05, "threat": 0.02, "identity_hate": 0.02}
        untargeted_flags = {"toxic": True, "obscene": True, "insult": False, "severe_toxic": False, "threat": False, "identity_hate": False}

        risk_un, icon_un, expl_un = classify_risk(untargeted_preds, untargeted_flags, thresholds, text="what the fuck is going on")
        assert risk_un == RiskLevel.REVIEW
        assert icon_un == "🟡"
        assert "Untargeted colloquial profanity" in expl_un
