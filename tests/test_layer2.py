"""Layer 2: spaCy adapter, candidate comparison, selection policy and pipeline integration.

Tests marked SPACY need spaCy + en_core_web_sm and are skipped otherwise (run them with the project's .venv python).
Expected spaCy outputs come from tests/fixtures/spacy_reference_outputs.json: outputs actually observed from the UNMODIFIED
reference code. They are parity data, NOT ground truth about correct ISL."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from isl_gloss.layer2 import Layer2Config, SelectionPolicy, SpacyTranslator, compare_candidates, run_layer2, select
from isl_gloss.layer2.candidate_comparator import check_llm, check_spacy, semantic_warnings, words_of
from isl_gloss.layer2.spacy_translator import DependencyUnavailable, SpacyResult

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "spacy_reference_outputs.json").read_text(encoding="utf-8"))["outputs"]


def _have_spacy():
    try:
        SpacyTranslator().available() and None
        return SpacyTranslator().available()[0]
    except Exception:
        return False


HAVE_SPACY = _have_spacy()


def l1_row(sid, text, tokens, status="translated", needs=False, notes="", echo="same", group="g1", **extra):
    gloss = " ".join(map(str, tokens)) if tokens is not None else None
    return {"sentence_id": sid, "group_id": group, "source_text": text, "llm_input_text": text, "translation_status": status,
            "candidate_gloss": gloss, "candidate_tokens": tokens, "final_gloss": gloss, "final_tokens": tokens,
            "uncertainty_notes": notes, "needs_human_review": needs, "error": None if status == "translated" else {"kind": "x", "message": "boom"},
            "llm": {"needs_human_review": needs, "echoed_source_text": text if echo == "same" else echo},
            "validation": {"status": "pass", "warnings": []}, "speaker_id": "spk1", "start_timestamp": 12.5, "end_timestamp": 14.0,
            "turn_index": 3, "metadata": {"prompt_version": "7.0"}, **extra}


def fake(mapping=None, fail=()):
    """A spaCy backend double. Returns lower-case tokens from `mapping`, raises for texts in `fail`."""
    def run(text):
        if text in fail:
            raise RuntimeError("synthetic parser failure")
        return (mapping or {}).get(text, text.lower().replace(".", "").split())
    return SpacyTranslator(run)


def write_l1(d, rows, name="results_prompt_7.0.json"):
    p = Path(d) / "layer1" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"prompt_version": "7.0", "config": {"model": "m"}, "complete": True, "results": rows}), encoding="utf-8")
    return p


class AdapterTests(unittest.TestCase):
    def test_ids_and_metadata_preserved_and_no_llm_text_used(self):
        r = fake().translate("s-1", "We go home.", group_id="g9", metadata={"start_timestamp": 1.5, "speaker_id": "a"})
        d = r.to_dict()
        self.assertEqual((d["sentence_id"], d["group_id"], d["source_text"], d["start_timestamp"], d["speaker_id"]), ("s-1", "g9", "We go home.", 1.5, "a"))
        self.assertEqual((r.spacy_status, r.spacy_gloss, r.spacy_tokens), ("ok", "we go home", ["we", "go", "home"]))

    def test_empty_and_blank_input(self):
        for text in ("", "   ", None):
            r = fake().translate("s", text)
            self.assertEqual((r.spacy_status, r.spacy_error["kind"]), ("empty", "empty_source"))

    def test_translator_exception_becomes_a_recorded_error(self):
        r = fake(fail={"Boom."}).translate("s-2", "Boom.")
        self.assertEqual(r.spacy_status, "error")
        self.assertEqual((r.spacy_error["kind"], r.spacy_error["sentence_id"]), ("RuntimeError", "s-2"))

    def test_malformed_backend_output(self):
        r = SpacyTranslator(lambda t: "not a list").translate("s", "Hi.")
        self.assertEqual((r.spacy_status, r.spacy_error["kind"]), ("error", "malformed_output"))

    def test_empty_backend_output(self):
        r = SpacyTranslator(lambda t: []).translate("s", "Hi.")
        self.assertEqual((r.spacy_status, r.spacy_error["kind"]), ("empty", "empty_output"))

    def test_missing_dependency_is_reported_not_raised(self):
        import isl_gloss.layer2 as pkg
        name = "isl_gloss.layer2.spacy_reference"
        saved_mod, saved_attr = sys.modules.pop(name, None), pkg.__dict__.pop("spacy_reference", None)
        sys.modules[name] = None  # makes `from . import spacy_reference` raise ImportError, as if spaCy/nltk were absent
        try:
            t = SpacyTranslator()
            r = t.translate("s", "We go home.")
            ok, why = t.available()
        finally:
            del sys.modules[name]
            if saved_mod is not None:
                sys.modules[name] = saved_mod
            if saved_attr is not None:
                pkg.spacy_reference = saved_attr
        self.assertEqual((r.spacy_status, r.spacy_error["kind"]), ("unavailable", "dependency_missing"))
        self.assertIn("python -m spacy download en_core_web_sm", r.spacy_error["setup"])
        self.assertFalse(ok)

    def test_missing_model_is_reported(self):
        t = SpacyTranslator()

        def boom():
            raise DependencyUnavailable("spaCy model en_core_web_sm is not available")
        t._backend = boom
        r = t.translate("s", "Hello there.")
        self.assertEqual(r.spacy_status, "unavailable")


@unittest.skipUnless(HAVE_SPACY, "SPACY: spaCy / en_core_web_sm not installed")
class ReferenceParityTests(unittest.TestCase):
    """The integrated module must reproduce the outputs recorded from the unmodified upstream code."""

    @classmethod
    def setUpClass(cls):
        cls.t = SpacyTranslator()

    def test_all_recorded_outputs_match(self):
        for text, expected in FIXTURE.items():
            if expected.startswith("ERR:"):
                continue
            r = self.t.translate("x", text)
            self.assertEqual(r.spacy_gloss or "", expected, text)

    def test_categories_run_and_match_fixture(self):
        cases = {"simple": "The cat sat on the mat.", "date": "I met him on 5 May 2021.", "negation": "We will not go outside.",
                 "question": "Where is Sanket going?", "numbers": "She bought 3 apples and 2 oranges.",
                 "names": "Maria Gonzalez will join us from Mumbai.", "conjunction": "and then we left.",
                 "compound_np": "The new computer science department opened.", "pronoun": "John told Peter that he would be late.",
                 "conditional": "If it rains we stay home.", "two_dates": "We met on Monday and on Tuesday 5 June."}
        for name, text in cases.items():
            r = self.t.translate(name, text)
            self.assertEqual(r.spacy_status, "ok", name)
            self.assertEqual(r.spacy_gloss, FIXTURE[text], name)

    def test_question_word_kept_by_reference(self):
        self.assertIn("where", self.t.translate("q", "Where is Sanket going?").spacy_tokens)

    def test_ids_and_timestamps_survive_a_real_translation(self):
        r = self.t.translate("id-7", "We go home.", group_id="gg", metadata={"start_timestamp": 3.25})
        self.assertEqual((r.sentence_id, r.group_id, r.to_dict()["start_timestamp"]), ("id-7", "gg", 3.25))

    def test_whitespace_only_is_reported_empty_where_upstream_raises(self):
        self.assertEqual(self.t.translate("s", "   ").spacy_status, "empty")

    def test_known_reference_defects_are_preserved_not_fixed(self):
        # documented limitations of the baseline: duplicate 'whenever', 'the' kept inside merged noun chunks
        self.assertEqual(self.t.translate("s", "Whenever we get angry, it leaves an impression.").spacy_tokens.count("whenever"), 2)
        self.assertIn("the cat", self.t.translate("s", "The cat sat on the mat.").spacy_tokens)


class ComparatorTests(unittest.TestCase):
    def cmp(self, text, llm_tokens, spacy_tokens, **kw):
        row = l1_row("s1", text, llm_tokens, **kw)
        sp = SpacyResult("s1", text, " ".join(spacy_tokens) if spacy_tokens else "", spacy_tokens, "ok" if spacy_tokens else "empty")
        return compare_candidates("s1", text, row, sp)

    def test_structural_llm_cases(self):
        self.assertTrue(check_llm(l1_row("s1", "Hi.", ["HI"]), "s1", "Hi.").usable)
        for row, code in ((l1_row("s1", "Hi.", []), "empty_output"), (l1_row("s1", "Hi.", None, status="failed"), "translator_failed"),
                          (l1_row("s1", "Hi.", ["HI"]), "sentence_id_mismatch"), (l1_row("s1", "Hi.", ["HI"], echo="Other."), "source_text_mismatch"),
                          (l1_row("s1", "Hi.", ["HI", 3]), "malformed_output"), ({}, "missing_output")):
            exp = "s2" if code == "sentence_id_mismatch" else "s1"
            c = check_llm(row, exp, "Hi.")
            self.assertFalse(c.usable, code)
            self.assertIn(code, [i["code"] for i in c.structural_issues])

    def test_structural_spacy_cases(self):
        ok = SpacyResult("s1", "Hi.", "hi", ["hi"], "ok")
        self.assertTrue(check_spacy(ok, "s1", "Hi.").usable)
        for res, code in ((SpacyResult("s1", "Hi.", None, None, "error", {"kind": "X", "message": "m"}), "translator_failed"),
                          (SpacyResult("s1", "Hi.", None, None, "unavailable", {"message": "no spacy"}), "translator_unavailable"),
                          (SpacyResult("s9", "Hi.", "hi", ["hi"], "ok"), "sentence_id_mismatch"),
                          (SpacyResult("s1", "Different.", "hi", ["hi"], "ok"), "source_text_mismatch"),
                          (SpacyResult("s1", "Hi.", "", [], "empty", {"kind": "empty_output"}), "empty_output"), (None, "missing_output")):
            c = check_spacy(res, "s1", "Hi.")
            self.assertFalse(c.usable, code)
            self.assertIn(code, [i["code"] for i in c.structural_issues])

    def test_candidates_are_never_modified(self):
        row = l1_row("s1", "We will not go.", ["WE", "GO"])
        sp = SpacyResult("s1", "We will not go.", "we go will", ["we", "go", "will"], "ok")
        before = (copy.deepcopy(row), copy.deepcopy(sp.to_dict()))
        c = compare_candidates("s1", "We will not go.", row, sp)
        self.assertEqual((row, sp.to_dict()), before)
        self.assertEqual(c.llm.gloss, "WE GO")  # still without NOT: reported, not repaired
        self.assertIn("possible_negation_not_represented", [w["code"] for w in c.llm.warnings])

    def test_semantic_warnings_are_conservative(self):
        codes = lambda t, g: [w["code"] for w in semantic_warnings(t, g) if w["severity"] == "warning"]
        self.assertEqual(codes("We will not go.", "WE NOT GO"), [])
        self.assertEqual(codes("We will not go.", "we go will not"), [])
        self.assertIn("possible_negation_not_represented", codes("I do not like it.", "I LIKE"))
        self.assertEqual(codes("The cat sat on the mat.", "CAT SIT MAT"), [])  # missing function words are NOT warnings
        self.assertIn("possible_condition_not_represented", codes("If it rains we stay.", "RAIN WE STAY"))
        self.assertEqual(codes("If it rains we stay.", "IF RAIN WE STAY"), [])
        self.assertIn("number_not_found", codes("She has 3 apples.", "SHE HAVE APPLE"))
        self.assertEqual(codes("She has 3 apples.", "SHE 3 APPLE HAVE"), [])
        self.assertIn("possible_time_expression_not_represented", codes("We left yesterday.", "WE LEAVE"))
        self.assertIn("possible_question_word_not_represented", codes("Why do we get angry?", "WE ANGRY"))
        self.assertIn("possible_cause_marker_not_represented", codes("We stayed because it rained.", "WE STAY RAIN"))
        self.assertEqual(codes("Maria bought a laptop.", "FS:MARIA BUY LAPTOP"), [])
        self.assertIn("possible_name_not_found", codes("We met Maria today.", "WE MEET TODAY"))

    def test_capitalised_ordinary_words_are_not_treated_as_names(self):
        self.assertNotIn("possible_name_not_found", [w["code"] for w in semantic_warnings("Hello. Today They are happy, Hi.", "HAPPY")])

    def test_yes_no_question_is_info_only(self):
        w = semantic_warnings("Did you finish the report?", "YOU FINISH REPORT")
        self.assertEqual([x["severity"] for x in w if x["feature"] == "question"], ["info"])

    def test_fs_notation_is_flagged_not_converted(self):
        c = check_llm(l1_row("s1", "Jonathan Young spoke.", ["FS:JONATHAN_YOUNG", "SPEAK"]), "s1", "Jonathan Young spoke.")
        self.assertIn("fs_token_joins_several_words", [w["code"] for w in c.warnings])
        self.assertEqual(c.tokens, ["FS:JONATHAN_YOUNG", "SPEAK"])
        bad = check_llm(l1_row("s1", "Hi.", ["hello", "WORLD"]), "s1", "Hi.")
        self.assertIn("token_not_in_project_format", [w["code"] for w in bad.warnings])

    def test_difference_is_descriptive(self):
        c = self.cmp("We go home.", ["WE", "GO", "HOME"], ["we", "home", "go"])
        d = c.difference
        self.assertTrue(d["same_words_different_order"] and not d["identical_after_normalisation"])
        c2 = self.cmp("We go home.", ["WE", "GO", "HOME"], ["we", "go"])
        self.assertEqual((c2.difference["only_in_llm"], c2.difference["only_in_spacy"]), (["HOME"], []))
        self.assertIn("not a similarity score", c2.difference["note"])

    def test_multiword_spacy_token_is_info(self):
        sp = SpacyResult("s1", "Hi.", "the cat sit", ["the cat", "sit"], "ok")
        c = check_spacy(sp, "s1", "Hi.")
        self.assertIn("multiword_token", [w["code"] for w in c.warnings if w["severity"] == "info"])

    def test_words_of_normalisation(self):
        self.assertEqual(words_of("FS:MARIA DRY_UP STUDENT-PL"), ["MARIA", "DRY", "UP", "STUDENT", "PL"])
        self.assertEqual(words_of(None), [])


class SelectionTests(unittest.TestCase):
    """Policy: the LLM is the primary candidate; spaCy is an independent alternative used as a fallback only after a stricter check."""

    def cmp(self, llm, spacy, text="We go home.", llm_status="translated", abstain=None):
        needs = (llm == []) if abstain is None else abstain
        row = l1_row("s1", text, llm, status=llm_status, needs=needs, notes="why" if needs else "")
        sp = (SpacyResult("s1", text, " ".join(spacy), spacy, "ok") if spacy else
              SpacyResult("s1", text, None, None, "error", {"kind": "E", "message": "m"}))
        return compare_candidates("s1", text, row, sp)

    def test_llm_is_primary_by_policy_default_when_both_usable(self):
        s = select(self.cmp(["WE", "GO", "HOME"], ["we", "home", "go"]))
        self.assertEqual((s.selection_status, s.selected_candidate, s.rule_id, s.selection_basis), ("llm_primary", "LLM", "R-SEL-06", "policy_default_primary"))
        self.assertEqual((s.primary_candidate, s.alternative_candidate), ("LLM", "SPACY"))
        self.assertIn("POLICY DEFAULT", s.selection_reason)
        self.assertIn("not because it was judged better", s.selection_reason)

    def test_length_and_token_count_never_decide(self):
        for spacy in (["we"], ["we", "go", "home", "now", "quickly", "again"]):
            s = select(self.cmp(["WE", "GO", "HOME"], spacy))
            self.assertEqual((s.selected_candidate, s.selection_basis), ("LLM", "policy_default_primary"))

    def test_only_llm_usable_keeps_llm_and_records_the_alternative_failure(self):
        s = select(self.cmp(["WE", "GO", "HOME"], None))
        self.assertEqual((s.selection_status, s.selected_candidate, s.rule_id), ("llm_primary", "LLM", "R-SEL-06"))
        self.assertIs(s.alternative_usable_as_fallback, False)
        self.assertIn("not usable", s.selection_reason)

    def test_llm_failure_with_a_clean_spacy_candidate_uses_spacy_as_a_checked_fallback(self):
        s = select(self.cmp(None, ["we", "go", "home"], llm_status="failed"))
        self.assertEqual((s.selection_status, s.selected_candidate, s.rule_id, s.selection_basis), ("spacy_fallback_selected", "SPACY", "R-SEL-02", "technical_fallback"))
        self.assertTrue(s.needs_human_review)
        self.assertIs(s.alternative_usable_as_fallback, True)

    def test_empty_llm_gloss_without_a_review_flag_is_a_failure_not_an_abstention(self):
        s = select(self.cmp([], ["we", "go", "home"], abstain=False))
        self.assertEqual((s.selection_status, s.selected_candidate), ("spacy_fallback_selected", "SPACY"))

    def test_llm_abstention_offers_spacy_but_does_not_substitute_it(self):
        s = select(self.cmp([], ["we", "go", "home"]))
        self.assertEqual((s.selection_status, s.selected_candidate, s.rule_id, s.needs_human_review), ("llm_abstained_alternative_available", None, "R-SEL-07", True))
        self.assertIs(s.alternative_usable_as_fallback, True)
        self.assertIn("OFFERED", s.selection_reason)

    def test_abstention_substitution_is_a_switch_that_is_off_by_default(self):
        s = select(self.cmp([], ["we", "go", "home"]), SelectionPolicy(fallback_on_llm_abstention=True))
        self.assertEqual((s.selection_status, s.selected_candidate), ("spacy_fallback_selected", "SPACY"))

    def test_fallback_requires_the_stricter_check_not_just_a_usable_candidate(self):
        # structurally usable but raises a possible-omission warning (no negation word) -> rejected as a stand-in
        c = self.cmp(None, ["we", "go", "outside"], "We will not go outside.", llm_status="failed")
        self.assertTrue(c.spacy.usable)
        s = select(c)
        self.assertEqual((s.selection_status, s.selected_candidate, s.rule_id), ("llm_failed_alternative_rejected", None, "R-SEL-08"))
        self.assertIn("possible_negation_not_represented", s.selection_reason)
        self.assertIs(s.alternative_usable_as_fallback, False)
        # a punctuation-only token is also rejected
        c2 = self.cmp(None, ["we", "go", "."], llm_status="failed")
        self.assertEqual(select(c2).selection_status, "llm_failed_alternative_rejected")
        self.assertIn("punctuation-only token", select(c2).alternative_check_reasons[0])

    def test_fallback_can_be_switched_off(self):
        s = select(self.cmp(None, ["we", "go", "home"], llm_status="failed"), SelectionPolicy(spacy_fallback_when_llm_fails=False))
        self.assertEqual((s.selection_status, s.selected_candidate), ("llm_failed_alternative_rejected", None))

    def test_both_invalid_requires_review(self):
        s = select(self.cmp(None, None, llm_status="failed"))
        self.assertEqual((s.selection_status, s.selected_candidate, s.needs_human_review), ("both_invalid", None, True))

    def test_identical_candidates_agree_but_are_not_called_verified(self):
        s = select(self.cmp(["FS:MARIA", "GO"], ["maria", "go"], "Maria goes."))
        self.assertEqual((s.selection_status, s.rule_id, s.selected_candidate, s.selection_basis), ("candidates_agree", "R-SEL-03", "LLM", "agreement"))
        self.assertIn("not proof", s.selection_reason)
        self.assertEqual(s.verification, "technical_checks_only_not_linguistically_verified")

    def test_shared_omission_is_flagged_for_review_not_resolved(self):
        c = self.cmp(["WE", "GO"], ["we", "go"], "We will not go.")
        self.assertEqual(c.shared_warnings, ["possible_negation_not_represented"])
        s = select(c)
        self.assertTrue(s.needs_human_review)
        self.assertIn("possible_negation_not_represented", s.selection_reason)

    def test_llm_primary_is_not_flagged_merely_because_spacy_differs(self):
        s = select(self.cmp(["WE", "GO", "HOME"], ["we", "home", "go"]))
        self.assertFalse(s.needs_human_review)

    def test_llm_own_flag_or_warning_keeps_it_flagged(self):
        flagged = select(self.cmp(["WE", "GO"], ["we", "go", "now"], abstain=True))
        self.assertTrue(flagged.needs_human_review)
        warned = select(self.cmp(["SHE", "BUY", "APPLE"], ["she", "3", "apple", "buy"], "She bought 3 apples."))
        self.assertTrue(warned.needs_human_review)

    def test_check_based_preference_is_off_by_default_and_bounded_when_on(self):
        c = self.cmp(["SHE", "BUY", "APPLE"], ["she", "3", "apple", "buy"], "She bought 3 apples.")
        self.assertEqual(select(c).selected_candidate, "LLM")
        s = select(c, SelectionPolicy(allow_check_based_preference=True))
        self.assertEqual((s.selection_status, s.selected_candidate, s.needs_human_review, s.rule_id, s.selection_basis),
                         ("selected_by_check", "SPACY", True, "R-SEL-05", "check_based"))
        self.assertEqual(select(c, SelectionPolicy(allow_check_based_preference=True, preference_codes=())).selected_candidate, "LLM")

    def test_no_source_text(self):
        row = l1_row("s1", "", None, status="skipped_no_translatable_text")
        c = compare_candidates("s1", "", row, fake().translate("s1", ""))
        s = select(c)
        self.assertEqual((s.selection_status, s.needs_human_review), ("no_source_text", False))

    def test_policy_is_serialised_with_the_primary_candidate(self):
        d = SelectionPolicy().to_dict()
        self.assertEqual((d["primary_candidate"], d["fallback_on_llm_abstention"], d["allow_check_based_preference"]), ("LLM", False, False))


class PipelineTests(unittest.TestCase):
    ROWS = lambda self: [l1_row("a-s1", "We go home.", ["WE", "GO", "HOME"], group="g1"),
                         l1_row("a-s2", "Boom.", ["BOOM"], group="g1"),
                         l1_row("a-s3", "Xq zr.", [], needs=True, notes="garbled", group="g2"),
                         l1_row("a-s4", "Fail here.", None, status="failed", group="g2")]

    def run_it(self, d, translator=None, cfg=None, rows=None):
        p = write_l1(d, rows or self.ROWS())
        return p, run_layer2(p, Path(d) / "layer2", cfg, translator or fake(fail={"Boom.", "Fail here."}))

    def test_one_fails_other_succeeds_and_both_fail(self):
        with tempfile.TemporaryDirectory() as d:
            _, res = self.run_it(d)
        by = {r["sentence_id"]: r for r in res["rows"]}
        self.assertEqual((by["a-s1"]["selection_status"], by["a-s1"]["spacy_status"]), ("candidates_agree", "ok"))  # fake spaCy gives the same words
        self.assertEqual((by["a-s2"]["spacy_status"], by["a-s2"]["selected_candidate"], by["a-s2"]["llm_gloss"]), ("error", "LLM", "BOOM"))  # spaCy failure keeps the LLM result
        self.assertEqual(by["a-s2"]["selection_status"], "llm_primary")
        self.assertEqual((by["a-s3"]["selection_status"], by["a-s3"]["selected_candidate"]), ("llm_abstained_alternative_available", None))  # LLM declined: spaCy offered, not substituted
        self.assertEqual((by["a-s3"]["spacy_gloss"], by["a-s3"]["alternative_usable_as_fallback"]), ("xq zr", True))  # the alternative is kept and was checked
        self.assertEqual((by["a-s4"]["selection_status"], by["a-s4"]["needs_human_review"]), ("both_invalid", True))
        self.assertEqual(len(res["rows"]), 4)

    def test_ids_groups_timestamps_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            _, res = self.run_it(d)
        r = res["rows"][0]
        self.assertEqual((r["sentence_id"], r["group_id"], r["start_timestamp"], r["end_timestamp"], r["speaker_id"], r["turn_index"]),
                         ("a-s1", "g1", 12.5, 14.0, "spk1", 3))
        self.assertEqual([x["sentence_id"] for x in res["rows"]], ["a-s1", "a-s2", "a-s3", "a-s4"])  # order kept

    def test_layer1_file_untouched_and_no_llm_requests(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_l1(d, self.ROWS())
            before = hashlib.sha256(p.read_bytes()).hexdigest()
            res = run_layer2(p, Path(d) / "layer2", translator=fake())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), before)
        m = res["manifest"]
        self.assertTrue(m["layer1_input"]["unchanged"])
        self.assertEqual(m["llm_requests_made_by_layer2"], 0)
        self.assertEqual(m["review_status"], "unreviewed")

    def test_rerun_is_idempotent_with_no_duplicate_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_l1(d, self.ROWS())
            out = Path(d) / "layer2"
            run_layer2(p, out, translator=fake())
            first = {f.name: f.read_bytes() for f in out.iterdir() if f.name != "run_manifest.json"}
            run_layer2(p, out, translator=fake())
            second = {f.name: f.read_bytes() for f in out.iterdir() if f.name != "run_manifest.json"}
            self.assertEqual(first, second)
            self.assertFalse([f for f in out.iterdir() if f.name.endswith(".tmp")])

    def test_output_folder_must_differ_from_layer1_folder(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_l1(d, self.ROWS())
            with self.assertRaises(ValueError):
                run_layer2(p, p.parent, translator=fake())

    def test_disabled_runs_nothing_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_l1(d, self.ROWS())
            res = run_layer2(p, Path(d) / "layer2", Layer2Config(enabled=False), fake())
            self.assertFalse(res["enabled"])
            self.assertFalse((Path(d) / "layer2").exists())

    def test_locked_output_file_stops_the_run_before_anything_is_changed(self):
        from unittest import mock
        import isl_gloss.layer2.pipeline as pipe
        real_open = open

        def locked_open(path, mode="r", *a, **k):
            if str(path).endswith("candidates_side_by_side.csv") and "a" in mode:
                raise PermissionError(13, "Access is denied")  # what Windows says when another program has the file open
            return real_open(path, mode, *a, **k)

        with tempfile.TemporaryDirectory() as d:
            p = write_l1(d, self.ROWS())
            out = Path(d) / "layer2"
            run_layer2(p, out, translator=fake())
            before = {f.name: f.read_bytes() for f in out.iterdir()}
            with mock.patch.object(pipe, "open", locked_open, create=True):
                with self.assertRaises(OSError) as cm:
                    run_layer2(p, out, translator=fake(fail={"We go home."}))  # would change the content if it were written
            self.assertIn("candidates_side_by_side.csv", str(cm.exception))
            self.assertIn("nothing was changed", str(cm.exception))
            self.assertEqual(before, {f.name: f.read_bytes() for f in out.iterdir()})

    def test_duplicate_and_bad_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.ROWS() + [l1_row("a-s1", "Dup.", ["DUP"])]
            with self.assertRaises(ValueError):
                run_layer2(write_l1(d, rows), Path(d) / "o", translator=fake())
            bad = Path(d) / "bad.json"
            bad.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                run_layer2(bad, Path(d) / "o2", translator=fake())

    def test_artifacts_and_reviewer_fields(self):
        with tempfile.TemporaryDirectory() as d:
            _, res = self.run_it(d)
            out = Path(d) / "layer2"
            for name in ("spacy_raw_outputs.json", "candidates_side_by_side.json", "candidates_side_by_side.csv", "selection_decisions.csv",
                         "warnings.csv", "summary.json", "summary_report.md", "run_manifest.json"):
                self.assertTrue((out / name).is_file(), name)
            header = (out / "candidates_side_by_side.csv").read_text(encoding="utf-8-sig").splitlines()[0]
            for col in ("preferred_candidate (LLM/SPACY/BOTH_ACCEPTABLE/NEITHER_ACCEPTABLE/UNCERTAIN)", "meaning_preserved_llm",
                        "linguistically_acceptable_spacy", "missing_or_introduced_meaning", "reviewer_comments", "llm_gloss", "spacy_gloss",
                        "selection_status", "needs_human_review"):
                self.assertIn(col, header)
            report = (out / "summary_report.md").read_text(encoding="utf-8")
            self.assertIn("UNREVIEWED", report)
            self.assertIn("does **not** show that either candidate is correct", report)
        self.assertFalse(res["summary"]["linguistic_accuracy_established"])

    def test_summary_counts(self):
        with tempfile.TemporaryDirectory() as d:
            _, res = self.run_it(d)
        s = res["summary"]
        self.assertEqual((s["sentences"], s["spacy"]["success"], s["spacy"]["failed"], s["llm"]["empty"], s["llm"]["failed"]), (4, 2, 2, 1, 1))
        self.assertEqual(s["selection"]["status_counts"], {"both_invalid": 1, "candidates_agree": 1, "llm_abstained_alternative_available": 1, "llm_primary": 1})
        sel = s["selection"]
        self.assertEqual((sel["carried_forward"], sel["spacy_selected"], sel["llm_primary_by_policy_default"], sel["evidence_based_selections"]), (2, 0, 1, 0))
        self.assertEqual(sel["llm_declined_spacy_offered_not_substituted"], 1)
        self.assertEqual(s["structural_validation_failures"]["spacy"], 2)

    def test_final_field_option(self):
        rows = self.ROWS()
        rows[0]["final_tokens"], rows[0]["final_gloss"] = ["WE", "GO"], "WE GO"
        with tempfile.TemporaryDirectory() as d:
            _, res = self.run_it(d, cfg=Layer2Config(llm_field="final_tokens"), rows=rows)
        self.assertEqual(res["rows"][0]["llm_gloss"], "WE GO")
        with self.assertRaises(ValueError):
            Layer2Config(llm_field="nope")

    def test_conditionals_pronouns_and_numbers_flow_through(self):
        rows = [l1_row("b-s1", "If it rains we stay home.", ["RAIN", "WE", "STAY", "HOME"]),
                l1_row("b-s2", "John told Peter that he would be late.", ["JOHN", "TELL", "PETER", "HE", "LATE"]),
                l1_row("b-s3", "She bought 3 apples.", ["SHE", "BUY", "APPLE"])]
        with tempfile.TemporaryDirectory() as d:
            _, res = self.run_it(d, translator=fake(), rows=rows)
        codes = {r["sentence_id"]: [w["code"] for w in r["llm_warnings"]] for r in res["rows"]}
        self.assertIn("possible_condition_not_represented", codes["b-s1"])
        self.assertIn("number_not_found", codes["b-s3"])
        self.assertTrue(all(r["selection_status"] == "llm_primary" for r in res["rows"]))
        self.assertTrue(all(r["needs_human_review"] for r in res["rows"] if r["llm_warnings"] and any(w["severity"] == "warning" for w in r["llm_warnings"])))


@unittest.skipUnless(HAVE_SPACY, "SPACY: spaCy / en_core_web_sm not installed")
class RealSpacyPipelineTests(unittest.TestCase):
    def test_end_to_end_with_real_spacy(self):
        rows = [l1_row("c-s1", "We will not go outside.", ["WE", "NOT", "GO", "OUTSIDE"]),
                l1_row("c-s2", "Where is Sanket going?", ["WHERE", "FS:SANKET", "GO"])]
        with tempfile.TemporaryDirectory() as d:
            p = write_l1(d, rows)
            res = run_layer2(p, Path(d) / "o")
        self.assertEqual([r["spacy_gloss"] for r in res["rows"]], [FIXTURE["We will not go outside."], FIXTURE["Where is Sanket going?"]])
        self.assertTrue(all((r["selection_status"], r["selected_candidate"], r["selection_basis"]) == ("llm_primary", "LLM", "policy_default_primary")
                            for r in res["rows"]))


class IndependenceTests(unittest.TestCase):
    def test_layer1_does_not_import_layer2(self):
        code = "import sys, isl_gloss.translator, isl_gloss.prompt, isl_gloss.llm; print(any(m.startswith('isl_gloss.layer2') for m in sys.modules))"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parent.parent,
                             env={**__import__("os").environ, "PYTHONPATH": "src"})
        self.assertEqual(out.stdout.strip(), "False", out.stderr)


if __name__ == "__main__":
    unittest.main()
