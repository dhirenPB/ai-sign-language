import json
import tempfile
import unittest
from pathlib import Path

from isl_gloss.config import TranslatorConfig
from isl_gloss.llm import AuthError, ModelUnavailable, QuotaExhausted, TransientError
from isl_gloss.models import FAILED, NOT_ATTEMPTED, SKIPPED, TRANSLATED
from isl_gloss.store import ResultStore
from isl_gloss.translator import Translator
from tests.gloss_helpers import FakeClient, chunk, echo_reply, make_row, parse_user, reply_from, results_json


def make(client, **cfg_kw):
    cfg = TranslatorConfig(**cfg_kw)
    return Translator(cfg, client)


def sentences(n):
    return [chunk(f"c{i:03d}", f"This is sentence number {i}.") for i in range(n)]


def reply_with(gloss_by_text):
    """Scripted model reply: look up each sentence's text and return the prepared gloss for it."""
    return reply_from(lambda label, text: gloss_by_text[text])


class InputFormatTests(unittest.TestCase):
    def test_model_receives_only_id_labelled_sentences(self):
        client = FakeClient()
        c = chunk("talk-0001", "We study hard today.", speaker_id="Speaker 1", turn_index=3,
                  start_time="00:00:01.000", end_time="00:00:02.000")
        d = chunk("talk-0002", "It is not easy.", speaker_id="Speaker 2")
        make(client).run([c, d])
        self.assertEqual(client.calls[0]["user"], "[talk-0001] We study hard today.\n[talk-0002] It is not easy.")
        for leaked in ("Speaker", "00:00", "{", "turn", "Title"):  # ids are the only metadata
            self.assertNotIn(leaked, client.calls[0]["user"])

    def test_previous_prompt_5_0_still_sends_numbered_sentences(self):
        client = FakeClient()
        make(client, prompt_version="5.0").run([chunk("talk-0001", "We study hard today."), chunk("talk-0002", "It is not easy.")])
        self.assertEqual(client.calls[0]["user"], "1. We study hard today.\n2. It is not easy.")
        self.assertIn("ISL GRAMMAR CONVENTIONS", client.calls[0]["system"])  # the frozen previous prompt
        self.assertEqual(client.calls[0]["schema"]["properties"]["results"]["items"]["required"][0], "n")

    def test_no_context_by_default_and_context_is_a_plain_labelled_block_when_enabled(self):
        client = FakeClient()
        make(client, batch_size=2).run(sentences(4))
        self.assertTrue(all("Earlier speech" not in c["user"] for c in client.calls))

        client = FakeClient()
        make(client, batch_size=2, context_chunks=2).run(sentences(4))
        ctx, items = parse_user(client.calls[1]["user"])
        self.assertEqual(ctx, ["This is sentence number 0.", "This is sentence number 1."])  # no ids on context lines
        self.assertEqual(items, [("c002", "This is sentence number 2."), ("c003", "This is sentence number 3.")])
        self.assertIn("Earlier speech", client.calls[1]["system"])

    def test_the_reply_schema_pins_the_sentence_ids_of_this_request(self):
        client = FakeClient()
        make(client, batch_size=3).run(sentences(5))
        enums = [c["schema"]["properties"]["results"]["items"]["properties"]["sentence_id"]["enum"] for c in client.calls]
        self.assertEqual(enums, [["c000", "c001", "c002"], ["c003", "c004"]])


class SequentialHybridTests(unittest.TestCase):
    def test_rule_pass_runs_on_every_candidate_and_both_outputs_are_kept(self):
        client = FakeClient([reply_with({"The book is good.": ["the", "Book", "good."], "I go to school.": ["SCHOOL", "GO"]})])
        out = make(client).run([chunk("c1", "The book is good."), chunk("c2", "I go to school.")])
        r1, r2 = out.results
        self.assertEqual(r1.candidate.tokens, ["the", "Book", "good."])  # untouched LLM candidate
        self.assertEqual(r1.final_tokens, ["BOOK", "GOOD"])
        self.assertEqual({a["rule_id"] for a in r1.applied_rules}, {"R-FMT-01", "R-ART-01"})
        self.assertEqual(r2.final_tokens, ["SCHOOL", "GO"])  # rule pass ran; nothing needed changing
        self.assertTrue(r1.rule_pass_completed and r2.rule_pass_completed)
        self.assertEqual(r2.applied_rules, [])

    def test_rule_pass_still_runs_when_validation_warns(self):
        client = FakeClient([reply_with({"I did not see the 50 rupees.": ["the", "x"]})])
        r = make(client).run([chunk("c1", "I did not see the 50 rupees.")]).results[0]
        self.assertEqual(r.final_tokens, ["X"])
        self.assertEqual(r.validation["status"], "review_required")
        self.assertTrue(r.rule_pass_completed)

    def test_result_follows_the_contract(self):
        t = make(FakeClient())
        d = t.run([chunk("c1", "We study hard.", speaker_id="Speaker 1", start_time="00:00:01.000")]).results[0].to_dict(t.metadata())
        for key in ("sentence_id", "source_text", "candidate_gloss", "uncertainty_notes", "needs_human_review", "final_gloss",
                    "applied_rules", "validation", "metadata", "review_status", "translation_status"):
            self.assertIn(key, d)
        self.assertEqual(d["review_status"], "unreviewed")
        self.assertIn(d["validation"]["status"], ("pass", "review_required", "failed"))
        self.assertEqual(set(d["metadata"]) >= {"model", "prompt_version", "rule_set_version", "llm_config"}, True)
        self.assertEqual((d["speaker_id"], d["start_timestamp"], d["sentence_id"]), ("Speaker 1", "00:00:01.000", "c1"))

    def test_the_configuration_used_is_recorded_for_fair_comparison(self):
        t = make(FakeClient(), reasoning_effort="medium")
        meta = t.metadata()
        self.assertEqual(meta["llm_config"], {
            "provider": "groq", "model": "openai/gpt-oss-120b", "reasoning_effort": "medium", "temperature": 0.0,
            "max_output_tokens": 8192, "prompt_version": "7.0", "thinking_budget": None})
        self.assertEqual(meta["prompt_version"], "7.0")
        d = t.run([chunk("c1", "We study hard.")]).results[0].to_dict(meta)
        self.assertEqual(d["metadata"]["llm_config"]["reasoning_effort"], "medium")


class SkipTests(unittest.TestCase):
    def test_skipped_text_never_reaches_the_llm(self):
        text = "Hmm, we study the Jane tradition."
        j = text.index("Jane")
        c = chunk("c2", text, skip_spans=[{"start": 0, "end": 3, "text": "Hmm", "reason": "expression_cue"},
                                          {"start": j, "end": j + 4, "text": "Jane", "reason": "skip_term"}])
        client = FakeClient()
        r = make(client).run([c]).results[0]
        self.assertEqual(r.status, TRANSLATED)
        self.assertEqual(client.calls[0]["user"], "[c2] we study the tradition.")
        self.assertEqual([s["text"] for s in r.to_dict({})["skipped_spans"]], ["Hmm", "Jane"])

    def test_bad_skip_span_fails_the_chunk_cleanly_without_a_request(self):
        c = chunk("c1", "We study hard.", skip_spans=[{"start": 0, "end": 5, "text": "WRONG", "reason": "x"}])
        client = FakeClient()
        out = make(client).run([c])
        self.assertEqual(out.results[0].status, FAILED)
        self.assertEqual(out.results[0].error["kind"], "bad_skip_span")
        self.assertEqual(client.calls, [])

    def test_model_bringing_back_a_skipped_word_is_cleaned_by_rule(self):
        c = chunk("c1", "Hmm, we study hard.", skip_spans=[{"start": 0, "end": 3, "text": "Hmm", "reason": "expression_cue"}])
        client = FakeClient([reply_with({"we study hard.": ["HMM", "WE", "STUDY", "HARD"]})])
        r = make(client).run([c]).results[0]
        self.assertEqual(r.final_tokens, ["WE", "STUDY", "HARD"])
        self.assertIn("R-SKIP-01", [a["rule_id"] for a in r.applied_rules])

    def test_fully_skipped_chunk_makes_no_request(self):
        c = chunk("c1", "Jya tu tya hu.", skip_spans=[{"start": 0, "end": 13, "text": "Jya tu tya hu", "reason": "skip_non_english"}])
        client = FakeClient()
        r = make(client).run([c]).results[0]
        self.assertEqual(r.status, SKIPPED)
        self.assertEqual(client.calls, [])
        self.assertEqual(r.validation["status"], "pass")

    def test_unclear_audio_is_not_translated_and_goes_to_review(self):
        c = chunk("c1", "[inaudible] and then we deployed it.",
                  excluded_spans=[{"start": 0, "end": 11, "text": "[inaudible]", "reason": "unclear_audio"}])
        client = FakeClient()
        r = make(client).run([c]).results[0]
        self.assertEqual(client.calls[0]["user"], "[c1] and then we deployed it.")
        self.assertEqual(r.validation["status"], "review_required")


class BatchingAndRepairTests(unittest.TestCase):
    def test_batches_keep_order_and_respect_size(self):
        client = FakeClient()
        out = make(client, batch_size=4).run(sentences(10))
        self.assertEqual(len(client.calls), 3)
        self.assertEqual([r.prepared.source.chunk_id for r in out.results], [f"c{i:03d}" for i in range(10)])
        self.assertTrue(all(r.status == TRANSLATED for r in out.results))
        self.assertEqual([r.candidate.tokens[-1] for r in out.results], [str(i) for i in range(10)])  # right answer, right chunk

    def test_results_are_matched_by_sentence_id_even_if_returned_out_of_order(self):
        def shuffled(user):
            _, items = parse_user(user)
            rows = [make_row(label, text, [text.upper().strip(".").split()[-1]]) for label, text in items]
            return results_json(list(reversed(rows)))
        out = make(FakeClient([shuffled]), batch_size=3).run(sentences(3))
        self.assertEqual([r.candidate.tokens for r in out.results], [["0"], ["1"], ["2"]])
        self.assertTrue(all("results_out_of_order" in [w["code"] for w in r.validation["warnings"]] for r in out.results))

    def test_missing_result_is_requested_once_more_with_the_same_ids(self):
        def drops_last(user):
            _, items = parse_user(user)
            return echo_reply("\n".join(f"[{label}] {text}" for label, text in items[:-1]))
        client = FakeClient([drops_last])
        out = make(client, batch_size=3).run(sentences(3))
        self.assertEqual(len(client.calls), 2)
        ctx, items = parse_user(client.calls[1]["user"])
        self.assertEqual(items, [("c002", "This is sentence number 2.")])  # ids are stable: no renumbering
        self.assertEqual(ctx, ["This is sentence number 0.", "This is sentence number 1."])  # the rest, read-only
        self.assertTrue(all(r.status == TRANSLATED for r in out.results))

    def test_malformed_json_splits_then_fails_without_inventing_gloss(self):
        client = FakeClient(["this is not json"] * 20)
        out = make(client, batch_size=4).run(sentences(4))
        self.assertTrue(all(r.status == FAILED and r.candidate is None and r.final_tokens == [] for r in out.results))
        self.assertEqual(out.results[0].error["kind"], "malformed_response")
        self.assertLessEqual(len(client.calls), 15)  # bounded

    def test_malformed_then_valid_after_split(self):
        client = FakeClient(["```oops```"])  # first call garbage, the halves get proper echo replies
        out = make(client, batch_size=4).run(sentences(4))
        self.assertTrue(all(r.status == TRANSLATED for r in out.results))
        self.assertEqual(len(client.calls), 3)

    def test_fenced_json_is_repaired(self):
        client = FakeClient(["```json\n" + echo_reply("[c000] Hi there.") + "\n```"])
        out = make(client).run([chunk("c000", "Hi there.")])
        self.assertEqual(out.results[0].status, TRANSLATED)

    def test_string_gloss_is_tolerated(self):
        row = {"sentence_id": "c000", "source_text": "I go to school.", "candidate_gloss": "ME GO SCHOOL", "uncertain_parts": [],
               "uncertainty_notes": "", "needs_human_review": False}
        out = make(FakeClient([results_json([row])])).run([chunk("c000", "I go to school.")])
        self.assertEqual(out.results[0].final_tokens, ["ME", "GO", "SCHOOL"])

    def test_an_unknown_or_mistyped_sentence_id_is_never_accepted_as_a_match(self):
        for bad in ("c0001", "C000", "", None, True, 0):
            row = {"sentence_id": bad, "source_text": "I go to school.", "candidate_gloss": ["X"], "uncertain_parts": [],
                   "uncertainty_notes": "", "needs_human_review": False}
            out = make(FakeClient([results_json([row])] * 10)).run([chunk("c000", "I go to school.")])
            self.assertEqual(out.results[0].status, FAILED, bad)
            self.assertIsNone(out.results[0].candidate)


class FailureTests(unittest.TestCase):
    def test_llm_failure_is_explicit_and_no_other_translator_is_used(self):
        client = FakeClient([TransientError("network down")] * 3)
        out = make(client, batch_size=1).run(sentences(3))
        self.assertTrue(all(r.status == FAILED for r in out.results))
        self.assertTrue(all(r.final_tokens == [] and not r.rule_pass_completed for r in out.results))
        self.assertEqual(out.results[0].error["kind"], "transient_error")
        self.assertFalse(out.complete)
        self.assertEqual(out.stop_reason, "too_many_failed_requests")

    def test_quota_exhaustion_stops_keeps_progress_and_resumes_from_cache(self):
        with tempfile.TemporaryDirectory() as d:
            store_path = Path(d) / "cache.jsonl"
            first = FakeClient([echo_reply, QuotaExhausted("daily quota")])
            out = Translator(TranslatorConfig(batch_size=2), first, ResultStore(store_path)).run(sentences(6))
            self.assertFalse(out.complete)
            self.assertIn("quota_exhausted", out.stop_reason)
            self.assertEqual([r.status for r in out.results], [TRANSLATED] * 2 + [NOT_ATTEMPTED] * 4)

            second = FakeClient()
            out2 = Translator(TranslatorConfig(batch_size=2), second, ResultStore(store_path)).run(sentences(6))
            self.assertTrue(out2.complete)
            self.assertEqual(len(second.calls), 2)  # only the 4 missing chunks (2 batches)
            self.assertEqual(out2.counts()["cache_hits"], 2)

    def test_unavailable_model_stops_the_run_instead_of_failing_every_batch(self):
        client = FakeClient([ModelUnavailable("gone")])
        out = make(client, batch_size=2).run(sentences(6))
        self.assertFalse(out.complete)
        self.assertIn("model_unavailable", out.stop_reason)
        self.assertEqual(len(client.calls), 1)

    def test_auth_error_stops_the_run(self):
        out = make(FakeClient([AuthError("bad key")]), batch_size=2).run(sentences(6))
        self.assertFalse(out.complete)
        self.assertTrue(all(r.status == NOT_ATTEMPTED for r in out.results))

    def test_rule_pass_error_preserves_candidate_and_fails_validation(self):
        t = make(FakeClient())
        t.engine.active[0] = type(t.engine.active[0])(**{**t.engine.active[0].__dict__, "transform": lambda tk, c: 1 / 0})
        r = t.run([chunk("c1", "We study hard today.")]).results[0]
        self.assertEqual(r.final_tokens, r.candidate.tokens)
        self.assertFalse(r.rule_pass_completed)
        self.assertEqual(r.validation["status"], "failed")
        self.assertEqual(r.error["kind"], "rule_pass_failed")

    def test_dry_run_calls_nothing(self):
        client = FakeClient()
        out = make(client, batch_size=3).run(sentences(7), dry_run=True)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(out.dry_run_requests), 3)
        self.assertEqual(out.dry_run_requests[0]["user_payload"].split("\n")[0], "[c000] This is sentence number 0.")
        self.assertEqual(out.dry_run_requests[0]["prompt_version"], "7.0")
        self.assertTrue(all(r.status == NOT_ATTEMPTED for r in out.results))

    def test_rule_changes_take_effect_without_new_llm_calls(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "c.jsonl"
            reply = reply_with({"He is tall.": ["HE", "IS", "TALL"]})
            Translator(TranslatorConfig(), FakeClient([reply]), ResultStore(cache)).run([chunk("c1", "He is tall.")])
            client = FakeClient()
            t = Translator(TranslatorConfig(enable_rules=("R-COP-01",)), client, ResultStore(cache))
            r = t.run([chunk("c1", "He is tall.")]).results[0]
            self.assertEqual(client.calls, [])
            self.assertEqual(r.final_tokens, ["HE", "TALL"])
            self.assertEqual(r.candidate.tokens, ["HE", "IS", "TALL"])


if __name__ == "__main__":
    unittest.main()
