"""Prompt 8.0, reference retrieval, the bounded empty-gloss retry, the empty-gloss validation and the diagnostic scoring.

Every reference entry below is a SYNTHETIC TEST FIXTURE (reviewer 'TEST FIXTURE'). None of it is linguistic data and none
of it is ISL-reviewed."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from isl_gloss.config import TranslatorConfig
from isl_gloss.prompt import DEFAULT_PROMPT_VERSION, PROMPTS, get_prompt
from isl_gloss.references import ReferenceSet, detect_features, load_references
from isl_gloss.scoring import score_row, score_rows
from isl_gloss.translator import Translator
from tests.gloss_helpers import FakeClient, chunk, make_row, parse_user, results_json

FIXTURE = {
    "format_version": "1.0",
    "notation_conventions": [
        {"id": "conv-1", "statement": "TEST FIXTURE: write multiword names word by word.", "reviewed_by": "TEST FIXTURE",
         "reviewed_on": "2000-01-01", "source": "synthetic"}],
    "vocabulary": [
        {"id": "voc-1", "english": "garden", "gloss": "GARDEN_SIGN", "reviewed_by": "TEST FIXTURE", "reviewed_on": "2000-01-01",
         "source": "synthetic"}],
    "sentence_pairs": [
        {"id": "pair-1", "english": "If it rains we stay home.", "gloss": ["IF", "RAIN", "WE", "HOME"], "tags": ["condition"],
         "reviewed_by": "TEST FIXTURE", "reviewed_on": "2000-01-01", "source": "synthetic"},
        {"id": "pair-2", "english": "The market opens early.", "gloss": ["MARKET", "OPEN", "EARLY"], "tags": [],
         "reviewed_by": "TEST FIXTURE", "reviewed_on": "2000-01-01", "source": "synthetic"},
        {"id": "pair-bad", "english": "No reviewer here.", "gloss": ["X"], "tags": [], "source": "synthetic"}],
}


def write_fixture(d, data=FIXTURE):
    p = Path(d) / "refs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def make(client, **cfg):
    base = TranslatorConfig(prompt_version="8.0", **cfg)
    return Translator(base, client)


def row_reply(fn):
    def reply(user):
        return results_json([fn(label, text, user) for label, text in parse_user(user)[1]])
    return reply


class Prompt8Tests(unittest.TestCase):
    def test_registered_and_not_default(self):
        self.assertIn("8.0", PROMPTS)
        self.assertEqual(DEFAULT_PROMPT_VERSION, "7.0")  # not switched without evidence
        self.assertTrue(get_prompt("8.0").supports_references)
        self.assertFalse(get_prompt("7.0").supports_references)

    def test_earlier_prompts_unchanged(self):
        self.assertEqual(len(get_prompt("5.0").build_system(None)), 6697)
        self.assertEqual(len(get_prompt("6.0").build_system(None)), 7938)
        self.assertEqual(len(get_prompt("7.0").build_system(None)), 10928)

    def test_same_schema_fields_as_7(self):
        self.assertEqual(get_prompt("8.0").schema_for(["a"]), get_prompt("7.0").schema_for(["a"]))

    def test_text_covers_the_requested_rules(self):
        text = get_prompt("8.0").build_system(None)
        for needle in ("FS:", "Keep a pronoun as a pronoun", "never a reason for an empty gloss", "DRY_UP", "BIRTH_ANNIVERSARY",
                       "Never silently drop", "genuinely unintelligible", "Final self-check", "does not guarantee linguistic correctness"):
            self.assertIn(needle, text)
        self.assertNotIn("{vocab}", text)

    def test_user_message_blocks_only_for_supporting_prompts(self):
        items = [("s1", "Hello there.")]
        v8 = get_prompt("8.0").build_user([], items, None, reference_block="Reference entries (x)", retry_note="again")
        self.assertIn("Reference entries (x)", v8)
        self.assertTrue(v8.endswith("Retry note: again"))
        v7 = get_prompt("7.0").build_user([], items, None, reference_block="Reference entries (x)", retry_note="again")
        self.assertEqual(v7, get_prompt("7.0").build_user([], items, None))


class ReferenceTests(unittest.TestCase):
    def test_shipped_template_is_empty_and_loads(self):
        rs = load_references("references/reference_set.json")
        self.assertEqual((len(rs), rs.rejected), (0, []))

    def test_unreviewed_entry_is_rejected_not_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            rs = load_references(write_fixture(d))
        self.assertEqual([r["id"] for r in rs.rejected], ["pair-bad"])
        self.assertNotIn("pair-bad", [e["id"] for e in rs.pairs])

    def test_bad_version_and_bad_gloss(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_references(write_fixture(d, {**FIXTURE, "format_version": "9"}))
            data = json.loads(json.dumps(FIXTURE))
            data["vocabulary"][0]["gloss"] = "lower case"
            rs = load_references(write_fixture(d, data))
        self.assertEqual(rs.rejected[0]["id"], "voc-1")

    def test_retrieval_is_relevant_and_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            rs = load_references(write_fixture(d))
        got = rs.retrieve(["If the garden is dry we wait."])
        self.assertEqual([e["id"] for e in got["vocabulary"]], ["voc-1"])
        self.assertEqual([e["id"] for e in got["pairs"]], ["pair-1"])  # condition-tagged pair; the market pair is irrelevant
        self.assertEqual(got, rs.retrieve(["If the garden is dry we wait."]))
        self.assertEqual(rs.retrieve(["Nothing related appears here"])["vocabulary"], [])

    def test_empty_set_sends_nothing(self):
        rs = ReferenceSet()
        self.assertEqual(rs.format_block(rs.retrieve(["If it rains."])), "")

    def test_cap_is_respected(self):
        pairs = [{"id": f"p{i}", "english": "If rain falls", "gloss": ["IF"], "tags": ["condition"], "reviewed_by": "t",
                  "reviewed_on": "2000-01-01", "source": "s"} for i in range(9)]
        rs = ReferenceSet(pairs=pairs)
        self.assertEqual(len(rs.retrieve(["If it rains."], max_entries=3)["pairs"]), 3)

    def test_features(self):
        f = detect_features(["If it rains, he is not coming?", "I have 5 apples."])
        self.assertTrue({"condition", "negation", "question", "number", "pronoun", "context"} <= f)


class TranslatorV8Tests(unittest.TestCase):
    def test_reference_block_reaches_the_request_and_is_traced(self):
        with tempfile.TemporaryDirectory() as d:
            t = make(FakeClient(), references_file=str(write_fixture(d)))
            r = t.run([chunk("a-s0001", "If the garden is dry we wait.")]).results[0]
        user = t.client.calls[0]["user"]
        self.assertIn("Reference entries (reviewed", user)
        self.assertIn("GARDEN_SIGN", user)
        codes = [w["code"] for w in r.validation["warnings"]]
        self.assertIn("reference_entries_supplied", codes)
        self.assertNotIn("pair-bad", user)
        self.assertEqual(t.metadata()["llm_config"]["reference_entries_loaded"], 4)

    def test_no_references_means_no_block(self):
        t = make(FakeClient())
        t.run([chunk("a-s0001", "Hello there.")])
        self.assertNotIn("Reference entries", t.client.calls[0]["user"])

    def test_references_ignored_by_prompts_that_do_not_support_them(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = TranslatorConfig(prompt_version="7.0", references_file=str(write_fixture(d)))
            t = Translator(cfg, FakeClient())
            t.run([chunk("a-s0001", "If the garden is dry we wait.")])
        self.assertNotIn("Reference entries", t.client.calls[0]["user"])
        self.assertNotIn("reference_entries_loaded", t.metadata()["llm_config"])

    def test_empty_gloss_gets_one_targeted_retry_and_recovers(self):
        def first(label, text, user):
            return make_row(label, text, [], needs=True, notes="pronoun is ambiguous")

        def second(label, text, user):
            return make_row(label, text, ["SHE", "TELL", "HER"], needs=True, notes="pronoun ambiguous: SHE or HER?")

        client = FakeClient([row_reply(first), row_reply(second)])
        t = make(client)
        r = t.run([chunk("a-s0001", "She told her.")]).results[0]
        self.assertEqual(client.requests_made, 2)
        self.assertIn("Retry note:", client.calls[1]["user"])
        self.assertIn("a-s0001", client.calls[1]["user"].split("Retry note:")[1])
        self.assertEqual(r.candidate.tokens, ["SHE", "TELL", "HER"])
        self.assertIn("empty_gloss_retried", [w["code"] for w in r.validation["warnings"]])
        self.assertEqual((t.stats["empty_gloss_retries"], t.stats["empty_gloss_recovered"]), (1, 1))

    def test_retry_is_bounded_to_one(self):
        def empty(label, text, user):
            return make_row(label, text, [], needs=True, notes="cannot tell")

        client = FakeClient([row_reply(empty), row_reply(empty), row_reply(empty)])
        t = make(client)
        r = t.run([chunk("a-s0001", "Xq zr plm.")]).results[0]
        self.assertEqual(client.requests_made, 2)  # the original request and exactly one retry
        self.assertEqual(r.candidate.tokens, [])
        self.assertEqual(r.validation["status"], "review_required")  # still reported honestly, not invented

    def test_unintelligible_stays_empty_without_inventing(self):
        def empty(label, text, user):
            return make_row(label, text, [], needs=True, notes="garbled ASR fragment")

        t = make(FakeClient([row_reply(empty), row_reply(empty)]))
        r = t.run([chunk("a-s0001", "Xq zr plm.")]).results[0]
        self.assertEqual(r.candidate.tokens, [])
        self.assertEqual(r.candidate.notes, "garbled ASR fragment")

    def test_retry_off_and_not_applied_to_older_prompts(self):
        def empty(label, text, user):
            return make_row(label, text, [], needs=True, notes="x")

        client = FakeClient([row_reply(empty)])
        make(client, empty_gloss_retry=False).run([chunk("a-s0001", "Hello there.")])
        self.assertEqual(client.requests_made, 1)
        client7 = FakeClient([row_reply(empty)])
        Translator(TranslatorConfig(prompt_version="7.0"), client7).run([chunk("a-s0001", "Hello there.")])
        self.assertEqual(client7.requests_made, 1)

    def test_only_empty_sentences_are_retried(self):
        def first(label, text, user):
            return make_row(label, text, [] if label.endswith("2") else ["OK"], needs=label.endswith("2"), notes="n" * label.endswith("2"))

        def second(label, text, user):
            return make_row(label, text, ["BEST", "GUESS"], needs=True, notes="unsure")

        client = FakeClient([row_reply(first), row_reply(second)])
        t = make(client)
        res = t.run([chunk("a-s0001", "One."), chunk("a-s0002", "Two.")]).results
        retry_items = parse_user(client.calls[1]["user"])[1]
        self.assertEqual([label for label, _ in retry_items], ["a-s0002"])
        self.assertEqual([r.candidate.tokens for r in res], [["OK"], ["BEST", "GUESS"]])

    def test_usage_is_recorded(self):
        t = make(FakeClient())
        t.run([chunk("a-s0001", "Hello there.")])
        self.assertEqual(t.stats["llm_calls"], 1)
        self.assertGreaterEqual(t.stats["latency_seconds"], 0)


class ValidationTests(unittest.TestCase):
    def test_empty_gloss_without_reason_fails(self):
        t = make(FakeClient([row_reply(lambda l, s, u: make_row(l, s, [], needs=True, notes="")),
                             row_reply(lambda l, s, u: make_row(l, s, [], needs=True, notes=""))]))
        r = t.run([chunk("a-s0001", "Hello there.")]).results[0]
        self.assertIn("empty_gloss_without_reason", [w["code"] for w in r.validation["warnings"]])
        self.assertEqual(r.validation["status"], "failed")

    def test_empty_gloss_with_reason_is_review_not_failure(self):
        t = make(FakeClient([row_reply(lambda l, s, u: make_row(l, s, [], needs=True, notes="garbled")),
                             row_reply(lambda l, s, u: make_row(l, s, [], needs=True, notes="garbled"))]))
        r = t.run([chunk("a-s0001", "Xq zr.")]).results[0]
        self.assertNotIn("empty_gloss_without_reason", [w["code"] for w in r.validation["warnings"]])
        self.assertEqual(r.validation["status"], "review_required")


# --- scoring -------------------------------------------------------------------------------------------------------
ANN = {
    "unintelligible": ["u-s1"], "expected_review": ["amb-s1"], "review_optional": [],
    "named_entities": {"n-s1": ["JONATHAN", "YOUNG"]}, "not_names": {"n-s1": ["TODAY"]},
    "terms_not_to_paraphrase": {"t-s1": ["JANMOTSAV"]}, "pronouns_to_keep": {"p-s1": [["HE"]]},
    "expected_markers": {"c-s1": {"condition": ["IF"]}},
}


def srow(sid, text, tokens, needs=False, notes="", status="translated"):
    return {"sentence_id": sid, "source_text": text, "llm_input_text": text, "translation_status": status,
            "candidate_tokens": tokens, "uncertainty_notes": notes, "llm": {"needs_human_review": needs},
            "validation": {"warnings": []}}


class ScoringTests(unittest.TestCase):
    def issues(self, row):
        return score_row(row, ANN, set())

    def test_empty_gloss_for_understandable_vs_unintelligible(self):
        self.assertIn("empty_gloss_for_understandable_sentence", self.issues(srow("x-s1", "Hello.", [], True, "why")))
        self.assertEqual(self.issues(srow("u-s1", "Xq.", [], True, "garbled")), ["unintelligible_empty_flagged_with_reason"])
        self.assertIn("unintelligible_input_given_an_invented_gloss", self.issues(srow("u-s1", "Xq.", ["HELLO"])))

    def test_name_formatting(self):
        text = "Professor Jonathan Young spoke today."
        self.assertEqual([i for i in self.issues(srow("n-s1", text, ["PROFESSOR", "FS:JONATHAN", "FS:YOUNG", "SPEAK", "TODAY"]))
                          if i.startswith(("name", "fs_"))], [])
        bad = self.issues(srow("n-s1", text, ["PROFESSOR_JONATHAN_YOUNG", "SPEAK", "TODAY"]))
        self.assertIn("name_in_compound_label", bad)
        self.assertIn("name_missing", self.issues(srow("n-s1", text, ["SPEAK", "TODAY"])))
        self.assertIn("name_false_positive", self.issues(srow("n-s1", text, ["FS:JONATHAN", "FS:YOUNG", "FS:TODAY"])))
        self.assertIn("fs_token_joins_several_words", self.issues(srow("n-s1", text, ["FS:JONATHAN_YOUNG"])))

    def test_markers_pronouns_paraphrase_and_flags(self):
        self.assertIn("marker_missing:condition", self.issues(srow("c-s1", "If it rains we stay.", ["RAIN", "WE", "STAY"])))
        self.assertIn("pronoun_not_preserved", self.issues(srow("p-s1", "He left.", ["BOY", "LEAVE"])))
        self.assertIn("culturally_specific_term_paraphrased_or_dropped",
                      self.issues(srow("t-s1", "We celebrate Janmotsav.", ["WE", "CELEBRATE", "BIRTH_ANNIVERSARY"])))
        self.assertIn("review_flag_possibly_unnecessary", self.issues(srow("z-s1", "We stay.", ["WE", "STAY"], True, "n")))
        self.assertIn("expected_review_not_flagged", self.issues(srow("amb-s1", "We stay.", ["WE", "STAY"])))

    def test_summary_has_every_category_and_is_labelled_provisional(self):
        s = score_rows([srow("x-s1", "We stay.", ["WE", "STAY"])], ANN, usage={"recorded": False})
        self.assertIn("PROVISIONAL", s["annotations"])
        for n in ("1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_", "9_", "10_"):
            self.assertTrue(any(k.startswith(n) for k in s["categories"]), n)
        self.assertEqual(s["usage"], {"recorded": False})

    def test_missing_results_are_counted(self):
        s = score_rows([srow("x-s1", "We stay.", None, status="failed")], ANN)
        self.assertEqual(s["categories"]["10_alignment_problems"], 1)


if __name__ == "__main__":
    unittest.main()
