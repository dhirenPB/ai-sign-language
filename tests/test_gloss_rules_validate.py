import unittest

from isl_gloss.models import Candidate
from isl_gloss.prepare import prepare_chunk
from isl_gloss.rules import RULES, RULES_BY_ID, RuleContext, RuleEngine, RulePassError, skipped_words_from_spans
from isl_gloss.validate import validate_gloss
from tests.gloss_helpers import chunk


def ctx(text="", skipped=()):
    return RuleContext(source_text=text, skipped_words=frozenset(skipped))


def apply(tokens, text="", skipped=(), enable=(), disable=()):
    return RuleEngine(enable, disable).apply(tokens, ctx(text, skipped))


class PrepareTests(unittest.TestCase):
    def test_skipped_and_excluded_spans_are_cut_but_recorded(self):
        text = "Hmm, that is true about the Jane tradition."
        c = chunk("c1", text, skip_spans=[
            {"start": 0, "end": 3, "text": "Hmm", "reason": "expression_cue"},
            {"start": text.index("Jane"), "end": text.index("Jane") + 4, "text": "Jane", "reason": "skip_term"},
        ])
        p = prepare_chunk(c)
        self.assertEqual(p.llm_text, "that is true about the tradition.")
        self.assertEqual([(s["text"], s["reason"]) for s in p.skipped], [("Hmm", "expression_cue"), ("Jane", "skip_term")])
        for s in p.skipped:  # positions still point into the original chunk text
            self.assertEqual(text[s["start"]:s["end"]], s["text"])

    def test_whole_chunk_skipped_is_not_translatable(self):
        c = chunk("c1", "Jya tu tya hu.", skip_spans=[{"start": 0, "end": 13, "text": "Jya tu tya hu", "reason": "skip_non_english"}])
        self.assertFalse(prepare_chunk(c).translatable)

    def test_unclear_audio_is_excluded_and_noted(self):
        c = chunk("c1", "[inaudible] and then we deployed it.",
                  excluded_spans=[{"start": 0, "end": 11, "text": "[inaudible]", "reason": "unclear_audio"}])
        p = prepare_chunk(c)
        self.assertEqual(p.llm_text, "and then we deployed it.")
        self.assertTrue(p.unclear_audio_excluded)

    def test_overlapping_spans_merge_and_mismatch_is_an_error(self):
        c = chunk("c1", "alpha beta gamma delta", skip_spans=[
            {"start": 0, "end": 10, "text": "alpha beta", "reason": "a"},
            {"start": 6, "end": 16, "text": "beta gamma", "reason": "b"}])
        p = prepare_chunk(c)
        self.assertEqual(p.skipped[0]["text"], "alpha beta gamma")
        self.assertEqual(p.skipped[0]["reason"], "a+b")
        self.assertEqual(p.llm_text, "delta")
        with self.assertRaises(ValueError):
            prepare_chunk(chunk("c2", "alpha beta", skip_spans=[{"start": 0, "end": 5, "text": "WRONG", "reason": "x"}]))


class RuleTests(unittest.TestCase):
    def test_format_rule_normalises_tokens(self):
        final, applied = apply(["me", "go.", "thank you", "  ", "fs:Sanket", "!!"])
        self.assertEqual(final, ["ME", "GO", "THANK_YOU", "FS:SANKET"])
        self.assertEqual([a["rule_id"] for a in applied], ["R-FMT-01"])
        self.assertEqual(applied[0]["before"][0], "me")

    def test_rules_do_not_apply_to_clean_input(self):
        final, applied = apply(["ME", "SCHOOL", "GO"])
        self.assertEqual((final, applied), (["ME", "SCHOOL", "GO"], []))

    def test_articles_removed_but_spelled_letters_kept(self):
        self.assertEqual(apply(["THE", "BOOK", "AN", "APPLE"])[0], ["BOOK", "APPLE"])
        self.assertEqual(apply(["A", "BOOK"])[0], ["BOOK"])
        self.assertEqual(apply(["M", "A", "R", "I", "A"])[0], ["M", "A", "R", "I", "A"])

    def test_skip_rule_removes_only_words_the_model_did_not_see(self):
        spans = [{"text": "Hmm"}, {"text": "Music"}]
        words = skipped_words_from_spans(spans, "I love music")
        self.assertEqual(words, frozenset({"HMM"}))  # MUSIC survives: it is a real word in the remaining text
        final, applied = apply(["HMM", "I", "LOVE", "MUSIC", "FS:HMM"], skipped=words)
        self.assertEqual(final, ["I", "LOVE", "MUSIC"])
        self.assertEqual(applied[0]["rule_id"], "R-SKIP-01")

    def test_risky_rules_are_off_by_default(self):
        self.assertEqual(apply(["HE", "IS", "TALL"])[0], ["HE", "IS", "TALL"])
        self.assertEqual(apply(["WHERE", "YOU", "GO"], text="Where are you going?")[0], ["WHERE", "YOU", "GO"])
        self.assertEqual(apply(["I", "MEET", "TEACHER", "YESTERDAY"])[0], ["I", "MEET", "TEACHER", "YESTERDAY"])
        self.assertFalse(RULES_BY_ID["R-COP-01"].enabled_by_default)

    def test_copula_rule_when_enabled(self):
        final, applied = apply(["HE", "IS", "TALL"], enable=("R-COP-01",))
        self.assertEqual(final, ["HE", "TALL"])
        self.assertEqual(applied[0]["rule_id"], "R-COP-01")

    def test_wh_rule_needs_a_question_and_a_single_wh_word(self):
        on = dict(enable=("R-ORD-WH-01",))
        self.assertEqual(apply(["WHERE", "YOU", "GO"], text="Where are you going?", **on)[0], ["YOU", "GO", "WHERE"])
        self.assertEqual(apply(["WHERE", "YOU", "GO"], text="I know where you go.", **on)[0], ["WHERE", "YOU", "GO"])
        self.assertEqual(apply(["WHAT", "WHERE", "GO"], text="What? Where?", **on)[0], ["WHAT", "WHERE", "GO"])
        self.assertEqual(apply(["YOU", "GO", "WHERE"], text="Where do you go?", **on)[0], ["YOU", "GO", "WHERE"])

    def test_time_rule_moves_a_single_time_word_to_front(self):
        on = dict(enable=("R-ORD-TIME-01",))
        self.assertEqual(apply(["I", "MEET", "TEACHER", "YESTERDAY"], **on)[0], ["YESTERDAY", "I", "MEET", "TEACHER"])
        self.assertEqual(apply(["TODAY", "I", "GO", "TOMORROW"], **on)[0], ["TODAY", "I", "GO", "TOMORROW"])

    def test_deterministic_and_idempotent(self):
        tokens = ["the", "Book", "Is", "good."]
        eng = RuleEngine(enable=("R-COP-01",))
        once, _ = eng.apply(tokens, ctx())
        twice, applied_again = eng.apply(once, ctx())
        self.assertEqual(once, twice)
        self.assertEqual(applied_again, [])
        self.assertEqual(eng.apply(tokens, ctx())[0], once)

    def test_unknown_rule_id_rejected_and_rule_errors_are_wrapped(self):
        with self.assertRaises(ValueError):
            RuleEngine(enable=("R-NOPE-99",))
        eng = RuleEngine()
        eng.active = [type(RULES[0])(**{**RULES[0].__dict__, "transform": lambda t, c: 1 / 0})]
        with self.assertRaises(RulePassError) as cm:
            eng.apply(["A"], ctx())
        self.assertEqual(cm.exception.rule_id, "R-FMT-01")

    def test_every_rule_documents_itself(self):
        for r in RULES:
            self.assertTrue(r.description and r.assumptions and r.limitation and r.version, r.rule_id)
        self.assertEqual(len({r.rule_id for r in RULES}), len(RULES))


def validate(text, tokens, uncertain=False, notes="", skipped=None, raw=None, applied=(), completed=True, vocab=None):
    c = chunk("c1", text, skip_spans=skipped or [])
    prep = prepare_chunk(c)
    cand = Candidate(tokens=raw or tokens, uncertain=uncertain, notes=notes)
    return validate_gloss(prep, cand, tokens, list(applied), completed, vocab)


def codes(result):
    return {w["code"] for w in result["warnings"]}


class ValidationTests(unittest.TestCase):
    def test_clean_gloss_passes(self):
        r = validate("The teacher gave us a book.", ["TEACHER", "ME", "BOOK", "GIVE"])
        self.assertEqual(r["status"], "pass", r)

    def test_empty_gloss_fails(self):
        r = validate("The teacher gave us a book.", [])
        self.assertEqual(r["status"], "failed")
        self.assertIn("empty_gloss", codes(r))

    def test_lost_negation_number_and_name_need_review(self):
        r = validate("Maria did not pay 50 rupees to Mohan.", ["FS:MARIA", "PAY", "RUPEES"])
        self.assertEqual(r["status"], "review_required")
        self.assertTrue({"negation_may_be_lost", "number_not_preserved", "name_may_be_missing"} <= codes(r))

    def test_preserved_negation_number_and_name_do_not_warn(self):
        r = validate("Maria did not pay 50 rupees to Mohan.", ["FS:MARIA", "FS:MOHAN", "50", "RUPEES", "PAY", "NOT"])
        self.assertTrue(codes(r).isdisjoint({"negation_may_be_lost", "number_not_preserved", "name_may_be_missing"}), r)

    def test_multi_word_fingerspelled_name_counts_as_present(self):
        r = validate("Professor Jonathan Young invited us.", ["PROFESSOR", "FS:JONATHAN_YOUNG", "INVITE"])
        self.assertNotIn("name_may_be_missing", codes(r))

    def test_capitalised_conjunctions_are_not_mistaken_for_names(self):
        r = validate("Because Though the water had dried up it left an image.", ["WATER", "DRY_UP", "IMAGE", "LEAVE"])
        self.assertNotIn("name_may_be_missing", codes(r))

    def test_uncertain_and_unclear_audio_need_review(self):
        self.assertIn("llm_marked_uncertain", codes(validate("Sentence one two three.", ["A1"], uncertain=True, notes="idiom")))
        r = validate("[inaudible] and then we deployed it.", ["DEPLOY"], skipped=None)
        self.assertNotIn("unclear_audio_excluded", codes(r))  # only when the chunk really carried the marker
        c = chunk("c1", "[inaudible] and then we deployed it.",
                  excluded_spans=[{"start": 0, "end": 11, "text": "[inaudible]", "reason": "unclear_audio"}])
        r = validate_gloss(prepare_chunk(c), Candidate(["DEPLOY"]), ["DEPLOY"], [], True)
        self.assertIn("unclear_audio_excluded", codes(r))
        self.assertEqual(r["status"], "review_required")

    def test_length_ratio_flags_omission_and_addition(self):
        text = "Every action leaves a lasting impression inside the human mind."
        self.assertIn("possible_omission", codes(validate(text, ["ACTION"])))
        self.assertIn("possible_addition", codes(validate(text, [f"T{i}" for i in range(30)])))

    def test_vocabulary_flags_unknown_tokens_but_not_fingerspelling(self):
        r = validate("Hello Sanket", ["HELLO", "FS:SANKET", "ZORP"], vocab=frozenset({"HELLO"}))
        self.assertIn("unknown_tokens", codes(r))
        self.assertIn("ZORP", [w for w in r["warnings"] if w["code"] == "unknown_tokens"][0]["message"])
        self.assertNotIn("FS:SANKET", [w for w in r["warnings"] if w["code"] == "unknown_tokens"][0]["message"])

    def test_format_prose_and_rule_failure(self):
        r = validate("Hello there friend today.", ["HELLO"], raw=["hello there friend"])
        self.assertIn("prose_in_gloss", codes(r))
        r = validate("Hello there friend today.", ["HELLO", "THERE"], completed=False)
        self.assertEqual(r["status"], "failed")
        self.assertIn("rule_pass_failed", codes(r))

    def test_rule_changes_are_reported(self):
        r = validate("The book is good.", ["BOOK", "GOOD"], applied=[{"rule_id": "R-ART-01"}])
        self.assertIn("rules_changed_output", codes(r))
        self.assertEqual(r["status"], "pass")


if __name__ == "__main__":
    unittest.main()
