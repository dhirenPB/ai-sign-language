import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from isl_gloss.cli import main
from isl_gloss.compare import run_compare
from isl_gloss.config import TranslatorConfig
from isl_gloss.grouping import group_sentences
from isl_gloss.llm import TransientError
from isl_gloss.models import FAILED, NOT_ATTEMPTED, SKIPPED, TRANSLATED
from isl_gloss.prepare import chunks_from_doc, chunks_from_sentences
from isl_gloss.sentences import build_sentences
from isl_gloss.translator import Translator
from isl_gloss.validate import validate_assembly
from tests.gloss_helpers import FakeClient, MockGemini, echo_reply, make_row, parse_user, results_json
from tests.test_gloss_sentences_grouping import cleaned, mk_doc

KEY = "test-key-123456"


def pipeline(doc, client, cfg=None, **tr_kw):
    cfg = cfg or TranslatorConfig()
    sentences = build_sentences(doc, cfg)
    groups = group_sentences(sentences, cfg)
    tr = Translator(cfg, client, **tr_kw)
    return sentences, groups, tr.run(chunks_from_sentences(sentences), groups=groups), tr


class GroupRequestTests(unittest.TestCase):
    PRONOUNS = "Speaker 1: Maria bought a new laptop yesterday. She says it is very fast. It replaced the old one that kept crashing."

    def test_related_sentences_go_to_the_llm_together_in_one_request_but_numbered_separately(self):
        client = FakeClient()
        s, g, out, _ = pipeline(cleaned(self.PRONOUNS), client)
        self.assertEqual(len(g), 1)
        self.assertEqual(len(client.calls), 1)  # the whole group in one request: the model can resolve 'She' and 'it'
        ctx, items = parse_user(client.calls[0]["user"])
        self.assertEqual([t for _, t in items], [x.source_text for x in s])
        self.assertEqual([i for i, _ in items], [x.sentence_id for x in s])  # each sentence labelled with its own id
        self.assertEqual(ctx, [])
        self.assertTrue(all(r.status == TRANSLATED for r in out.results))
        self.assertIn("contextual group", client.calls[0]["system"])
        self.assertIn("return a separate gloss result for every target sentence", client.calls[0]["system"])

    def test_one_request_per_group_and_context_blocks_are_read_only(self):
        doc = mk_doc(["We opened the shop,", "and sales grew fast.", "A new topic starts here today."], turns=[0, 1, 2])
        client = FakeClient()
        s, g, out, _ = pipeline(doc, client)
        self.assertEqual(len(g), 3)
        self.assertEqual(len(client.calls), 3)
        ctx0, items0 = parse_user(client.calls[0]["user"])
        self.assertEqual(items0, [(s[0].sentence_id, "We opened the shop,")])
        self.assertIn("Following speech", client.calls[0]["user"])  # next sentence given as context for the open-ended one
        self.assertIn("and sales grew fast.", client.calls[0]["user"].split("Following speech")[1])
        ctx1, items1 = parse_user(client.calls[1]["user"])
        self.assertEqual(ctx1, ["We opened the shop,"])  # 'and sales...' depends on it
        self.assertEqual(items1, [(s[1].sentence_id, "and sales grew fast.")])
        self.assertEqual(len(out.results), 3)  # context sentences did not create extra results
        self.assertEqual([r.candidate.tokens for r in out.results][1], ["AND", "SALES", "GREW", "FAST"])

    def test_skipped_words_are_cut_from_group_text_and_from_context(self):
        doc = mk_doc(["We study the Jane tradition,", "and it is old."], turns=[0, 1])
        j = doc["chunks"][0]["text"].index("Jane")
        doc["chunks"][0]["skip_translation_spans"] = [{"start": j, "end": j + 4, "text": "Jane", "reason": "skip_term"}]
        client = FakeClient()
        pipeline(doc, client)
        self.assertTrue(all("Jane" not in c["user"] for c in client.calls))  # not as a sentence, not as context


class MappingAndOrderTests(unittest.TestCase):
    def test_sentence_ids_map_to_the_right_gloss_even_if_the_model_reorders_its_reply(self):
        def shuffled(user):
            _, items = parse_user(user)
            rows = [make_row(label, t, [t.split()[0].upper().strip(".")], notes=f"note {i}") for i, (label, t) in enumerate(items, 1)]
            return results_json(list(reversed(rows)))
        doc = mk_doc(["Alpha opens the day.", "Beta follows closely.", "Gamma changes the plan."])
        s, g, out, _ = pipeline(doc, FakeClient([shuffled]))
        self.assertEqual([r.candidate.tokens for r in out.results], [["ALPHA"], ["BETA"], ["GAMMA"]])
        self.assertEqual([r.candidate.notes for r in out.results], ["note 1", "note 2", "note 3"])
        d = [r.to_dict({}) for r in out.results]
        self.assertEqual([x["sentence_id"] for x in d], [x.sentence_id for x in s])
        self.assertEqual([x["uncertainty_notes"] for x in d], ["note 1", "note 2", "note 3"])
        self.assertTrue(all(any(w["code"] == "results_out_of_order" for w in x["validation"]["warnings"]) for x in d))

    def test_duplicate_stray_and_missing_results_are_detected_not_hidden(self):
        def messy(user):
            _, items = parse_user(user)
            (l1, t1), (l2, t2), (l3, t3) = items
            rows = [make_row(l1, t1, ["ONE"]), make_row(l1, t1, ["DUPLICATE"]), make_row("not-a-real-id", "x", ["STRAY"]),
                    make_row(l3, t3, ["THREE"])]
            return results_json(rows)
        doc = mk_doc(["First statement here.", "Second statement here.", "Third statement here."])
        client = FakeClient([messy])
        s, g, out, _ = pipeline(doc, client)
        r1, r2, r3 = out.results
        self.assertEqual(r1.candidate.tokens, ["ONE"])  # first valid result wins
        self.assertIn("duplicate_result_for_sentence", [w["code"] for w in r1.validation["warnings"]])
        self.assertIn("stray_results_ignored", [w["code"] for w in r3.validation["warnings"]])
        self.assertEqual(r1.validation["status"], "review_required")
        self.assertEqual(r2.status, TRANSLATED)  # the missing one was re-requested (second call, echo reply)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(parse_user(client.calls[1]["user"])[1], [(s[1].sentence_id, "Second statement here.")])

    def test_chronological_order_is_preserved_across_many_groups(self):
        texts = [f"Independent statement number {i} about nothing." for i in range(40)]
        times = [(f"00:00:{i:02d}.000", f"00:00:{i:02d}.500") for i in range(40)]
        cfg = TranslatorConfig(group_max_sentences=5)
        s, g, out, _ = pipeline(mk_doc(texts, times=times), FakeClient(), cfg)
        self.assertEqual(len(g), 8)
        self.assertEqual([r.prepared.source.chunk_id for r in out.results], [x.sentence_id for x in s])
        self.assertEqual([r.to_dict({})["start_timestamp"] for r in out.results], [t[0] for t in times])
        self.assertEqual([r.candidate.tokens[-3] for r in out.results], [str(i) for i in range(40)])
        self.assertEqual([r.group_id for r in out.results], [gr.group_id for gr in g for _ in gr.sentence_ids])
        self.assertTrue(out.assembly["passed"], out.assembly)
        self.assertTrue(out.complete)


class PropagationTests(unittest.TestCase):
    def test_ids_speakers_timestamps_flow_through_llm_rules_and_validation(self):
        srt = ("1\n00:00:01,000 --> 00:00:03,000\nSpeaker 1: Maria did not pay 50 rupees. She left.\n\n"
               "2\n00:00:04,000 --> 00:00:05,000\nSpeaker 2: Hmm, the old one broke.\n")
        doc = cleaned(srt, "talk.srt")
        reply = lambda user: results_json([make_row(label, text, ["the", "Maria", "left."], needs=i == 1,
                                                    notes="who is she?" if i == 1 else "")
                                           for i, (label, text) in enumerate(parse_user(user)[1])])
        client = FakeClient([reply, reply])
        s, g, out, tr = pipeline(doc, client)
        rows = [r.to_dict(tr.metadata()) for r in out.results]
        self.assertEqual([r["sentence_id"] for r in rows], [x.sentence_id for x in s])
        self.assertTrue(all(r["group_id"] for r in rows))
        first, second = rows[0], rows[1]
        self.assertEqual((first["speaker_id"], first["start_timestamp"], first["end_timestamp"]), ("Speaker 1", "00:00:01.000", "00:00:03.000"))
        self.assertEqual(rows[2]["speaker_id"], "Speaker 2")
        self.assertEqual(first["candidate_tokens"], ["the", "Maria", "left."])  # LLM candidate preserved
        self.assertEqual(first["final_tokens"], ["MARIA", "LEFT"])  # rule pass applied, with ids intact
        self.assertEqual([a["rule_id"] for a in first["applied_rules"]], ["R-FMT-01", "R-ART-01"])
        self.assertIn("number_not_preserved", [w["code"] for w in first["validation"]["warnings"]])  # validation saw sentence 1's own text
        self.assertEqual(second["uncertainty_notes"], "who is she?")
        self.assertIn("llm_marked_uncertain", [w["code"] for w in second["validation"]["warnings"]])
        self.assertEqual(first["source_metadata"]["parent_chunk_id"], "talk-0001")
        self.assertTrue(out.assembly["passed"])
        self.assertEqual(rows[0]["metadata"]["prompt_version"], "7.0")


class FailureTests(unittest.TestCase):
    def test_a_failed_group_never_drops_its_sentences_and_other_groups_still_run(self):
        doc = mk_doc(["First group sentence.", "Second group sentence.", "Third group sentence."], turns=[0, 1, 2])
        client = FakeClient([echo_reply, TransientError("network down"), echo_reply])
        s, g, out, _ = pipeline(doc, client)
        self.assertEqual([r.status for r in out.results], [TRANSLATED, FAILED, TRANSLATED])
        self.assertEqual(len(out.results), 3)  # nothing silently discarded
        self.assertEqual(out.results[1].error["kind"], "transient_error")
        self.assertEqual([r.to_dict({})["sentence_id"] for r in out.results], [x.sentence_id for x in s])
        self.assertTrue(out.assembly["passed"])  # results still align with sentences and groups

    def test_unattempted_sentences_stay_in_the_output_when_the_run_stops(self):
        from isl_gloss.llm import QuotaExhausted
        doc = mk_doc([f"Group {i} sentence." for i in range(4)], turns=[0, 1, 2, 3])
        s, g, out, _ = pipeline(doc, FakeClient([echo_reply, QuotaExhausted("daily")]))
        self.assertEqual([r.status for r in out.results], [TRANSLATED, NOT_ATTEMPTED, NOT_ATTEMPTED, NOT_ATTEMPTED])
        self.assertFalse(out.complete)
        self.assertEqual(len(out.results), 4)

    def test_groups_that_do_not_match_the_sentences_are_rejected_or_reported(self):
        doc = mk_doc(["One thing here.", "Another thing here."], turns=[0, 1])
        cfg = TranslatorConfig()
        s = build_sentences(doc, cfg)
        groups = group_sentences(s, cfg)
        tr = Translator(cfg, FakeClient())
        out = tr.run(chunks_from_sentences(s), groups=groups[:1])  # a group is missing
        self.assertFalse(out.assembly["passed"])
        self.assertIn("group_coverage", [e["code"] for e in out.assembly["errors"]])
        self.assertFalse(out.complete)  # the uncovered sentence was not attempted, and that is visible
        groups[0].sentence_ids.append("t-s9999")
        with self.assertRaises(ValueError):
            Translator(cfg, FakeClient()).run(chunks_from_sentences(s), groups=groups)


class AssemblyValidationTests(unittest.TestCase):
    ids = ["a", "b", "c"]

    def codes(self, **kw):
        return {e["code"] for e in validate_assembly(**kw)["errors"]}

    def test_clean_run_passes(self):
        r = validate_assembly(self.ids, [("g1", ["a", "b"]), ("g2", ["c"])], self.ids, ["00:00:01.000", None, "00:00:03.000"])
        self.assertEqual((r["passed"], r["errors"], r["warnings"]), (True, [], []))

    def test_detects_missing_duplicate_unexpected_and_reordered_results(self):
        self.assertIn("missing_result", self.codes(sentence_ids=self.ids, groups=None, result_ids=["a", "b"]))
        self.assertIn("duplicate_result", self.codes(sentence_ids=self.ids, groups=None, result_ids=["a", "b", "b", "c"]))
        self.assertIn("unexpected_result", self.codes(sentence_ids=self.ids, groups=None, result_ids=["a", "b", "c", "z"]))
        self.assertIn("order_changed", self.codes(sentence_ids=self.ids, groups=None, result_ids=["a", "c", "b"]))
        self.assertIn("duplicate_sentence_id", self.codes(sentence_ids=["a", "a"], groups=None, result_ids=["a", "a"]))

    def test_detects_bad_groups(self):
        self.assertIn("group_coverage", self.codes(sentence_ids=self.ids, groups=[("g1", ["a", "b"])], result_ids=self.ids))
        self.assertIn("group_coverage", self.codes(sentence_ids=self.ids, groups=[("g1", ["a", "b"]), ("g2", ["b", "c"])], result_ids=self.ids))
        self.assertIn("group_order", self.codes(sentence_ids=self.ids, groups=[("g1", ["b", "a"]), ("g2", ["c"])], result_ids=self.ids))
        self.assertIn("duplicate_group_id", self.codes(sentence_ids=self.ids, groups=[("g", ["a"]), ("g", ["b", "c"])], result_ids=self.ids))
        self.assertIn("empty_group", self.codes(sentence_ids=self.ids, groups=[("g1", []), ("g2", self.ids)], result_ids=self.ids))

    def test_timestamps_going_backwards_warn(self):
        r = validate_assembly(self.ids, None, self.ids, ["00:00:05.000", "00:00:01.000", None])
        self.assertTrue(r["passed"])
        self.assertEqual(r["warnings"][0]["code"], "timestamps_out_of_order")


class CacheAndContextTests(unittest.TestCase):
    def test_same_sentence_in_a_different_context_is_not_served_from_the_cache(self):
        with tempfile.TemporaryDirectory() as d:
            from isl_gloss.store import ResultStore
            cache = Path(d) / "c.jsonl"
            cfg = TranslatorConfig()
            a = build_sentences(mk_doc(["It was old.", "We kept it."]), cfg)
            b = build_sentences(mk_doc(["It was new.", "We kept it."]), cfg)
            first, second = FakeClient(), FakeClient()
            Translator(cfg, first, ResultStore(cache)).run(chunks_from_sentences(a), groups=group_sentences(a, cfg))
            Translator(cfg, second, ResultStore(cache)).run(chunks_from_sentences(b), groups=group_sentences(b, cfg))
            self.assertEqual(len(first.calls), 1)
            self.assertEqual(len(second.calls), 1)  # different group text => fresh request
            third = FakeClient()
            Translator(cfg, third, ResultStore(cache)).run(chunks_from_sentences(a), groups=group_sentences(a, cfg))
            self.assertEqual(third.calls, [])  # identical group => cache


class CompareTests(unittest.TestCase):
    SPEC = {"passages": [
        {"id": "p1", "description": "d", "look_for": "l",
         "text": "Speaker 1: Maria bought a laptop. She says it is fast."},
        {"id": "p2", "text": "Speaker 1: The monsoon arrives in June.\n\nSpeaker 1: Our library opens next week."}]}

    def test_dry_run_reports_request_counts_without_a_client(self):
        with tempfile.TemporaryDirectory() as d:
            s = run_compare(self.SPEC, TranslatorConfig(), None, Path(d), dry_run=True)
            self.assertEqual(s["requests"], {"direct": 2, "grouped": 3})
            self.assertFalse(list(Path(d).iterdir()))

    def test_comparison_writes_sheet_and_flags_regressions(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeClient()
            cfg = TranslatorConfig()
            s = run_compare(self.SPEC, cfg, lambda mode: Translator(cfg, client), Path(d))
            self.assertEqual(s["chunks_compared"], 4)
            self.assertEqual(s["chunks_with_different_gloss"], 0)  # echo model: grouping cannot change it
            self.assertEqual(s["automatic_regressions"], [])
            self.assertEqual(s["totals"]["direct"]["translated"], s["totals"]["grouped"]["translated"])
            self.assertTrue(s["totals"]["grouped"]["assembly_passed"])
            self.assertEqual((s["totals"]["direct"]["requests"], s["totals"]["grouped"]["requests"]), (2, 3))
            sheet = (Path(d) / "comparison_sheet.csv").read_text(encoding="utf-8-sig")
            for col in ("direct_gloss", "grouped_gloss", "better_version", "wrong_pronoun_reference"):
                self.assertIn(col, sheet)

    def test_a_worse_grouped_run_is_reported_as_a_possible_regression(self):
        class DropsNegationWhenGrouped(FakeClient):
            def complete(self, system, user, schema):
                self.calls.append(1)
                self.requests_made += 1
                _, items = parse_user(user)
                multi = "contextual group" in system and len(items) > 1
                rows = [make_row(label, t, ["X"] if multi else t.upper().replace(".", "").split()) for label, t in items]
                from isl_gloss.llm import LLMResponse
                return LLMResponse(text=results_json(rows), model="m")
        spec = {"passages": [{"id": "p", "text": "Speaker 1: Maria did not leave early today. She will not return soon either."}]}
        with tempfile.TemporaryDirectory() as d:
            cfg = TranslatorConfig(batch_size=1)  # direct mode sends one sentence per request; grouping sends two together
            client = DropsNegationWhenGrouped()
            s = run_compare(spec, cfg, lambda mode: Translator(cfg, client), Path(d))
            self.assertTrue(s["automatic_regressions"])
            self.assertIn("possible regression", s["automatic_verdict"])


class IncompleteComparisonTests(unittest.TestCase):
    def test_a_stopped_run_is_reported_as_incomplete_never_as_no_regression(self):
        from isl_gloss.llm import QuotaExhausted
        spec = {"passages": [{"id": "p", "text": "Speaker 1: Maria left. She was tired."}]}
        with tempfile.TemporaryDirectory() as d:
            cfg = TranslatorConfig()
            client = FakeClient([QuotaExhausted("daily")] * 4)
            s = run_compare(spec, cfg, lambda mode: Translator(cfg, client), Path(d))
            self.assertIn("INCOMPLETE", s["automatic_verdict"])
            self.assertNotIn("no regression", s["automatic_verdict"])
            self.assertIn("quota_exhausted", s["incomplete_reasons"]["direct"])
            self.assertEqual(s["totals"]["grouped"]["translated"], 0)


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.dir / "structured").mkdir()
        self.cleaned = self.dir / "structured" / "talk.cleaned.json"
        doc = cleaned("Speaker 1: Maria bought a laptop. She says it is fast. Speaker 2: Really? Speaker 1: Yes, it is.")
        self.cleaned.write_text(json.dumps(doc), encoding="utf-8")

    def run_cli(self, *args, env=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.dict(os.environ, env or {}, clear=False):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_group_command_needs_no_key_and_writes_an_inspection_file(self):
        code, out, _ = self.run_cli("group", self.cleaned)
        self.assertEqual(code, 0)
        self.assertIn("sentences ->", out)
        doc = json.loads((self.dir / "translations" / "talk.groups.json").read_text(encoding="utf-8"))
        self.assertEqual(len(doc["sentences"]), 4)
        self.assertEqual([g["boundary_reason"] for g in doc["groups"]][:2], ["turn_change", "turn_change"])
        self.assertEqual(doc["sentences"][0]["sentence_id"], "talk-s0001")
        self.assertIn("depends_on_previous", doc["sentences"][1])

    def test_group_limits_come_from_the_command_line(self):
        code, out, _ = self.run_cli("group", self.cleaned, "--max-group-sentences", "1", "--no-time-gap")
        doc = json.loads((self.dir / "translations" / "talk.groups.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["summary"]["sentences_per_group_max"], 1)
        self.assertEqual(doc["summary"]["config"]["group_max_gap_seconds"], None)

    def test_translate_groups_end_to_end_then_direct_mode_for_comparison(self):
        with MockGemini() as srv:
            cfg = self.dir / "cfg.json"
            cfg.write_text(json.dumps({"provider": "gemini", "base_url": srv.base_url, "requests_per_minute": 6000}), encoding="utf-8")
            env = {"GEMINI_API_KEY": KEY}
            code, out, err = self.run_cli("translate", self.cleaned, "--config", cfg, "--env-file", self.dir / "none.env", env=env)
            self.assertEqual(code, 0, out + err)
            doc = json.loads((self.dir / "translations" / "talk.gloss.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["mode"], "groups")
            self.assertTrue(doc["assembly_validation"]["passed"])
            self.assertEqual([r["sentence_id"] for r in doc["results"]], [f"talk-s000{i}" for i in range(1, 5)])
            self.assertTrue(all(r["group_id"] for r in doc["results"]))
            self.assertEqual(len(doc["groups"]), len(srv.requests))
            self.assertTrue((self.dir / "translations" / "talk.groups.json").exists())
            n = len(srv.requests)
            code, out, err = self.run_cli("translate", self.cleaned, "--mode", "direct", "--config", cfg,
                                          "--env-file", self.dir / "none.env", env=env)
            self.assertEqual(code, 0, out + err)
            self.assertTrue((self.dir / "translations" / "talk.direct.gloss.json").exists())
            self.assertEqual(len(srv.requests), n + 1)  # direct mode: one window for the 4 chunks

    def test_dry_run_shows_groups_and_the_bare_text_payload(self):
        code, out, _ = self.run_cli("translate", self.cleaned, "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("4 chunks -> 4 sentences -> 3 groups", out)
        self.assertIn("] Maria bought a laptop.", out)
        self.assertNotIn("Speaker", out)  # sentence ids are the only metadata sent
        self.assertFalse((self.dir / "translations").exists())

    def test_compare_dry_run(self):
        spec = self.dir / "p.json"
        spec.write_text(json.dumps({"passages": [{"id": "x", "text": "Speaker 1: One thing. Two things."}]}), encoding="utf-8")
        code, out, _ = self.run_cli("compare", "--passages", spec, "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("requests direct 1 vs grouped 1", out)


if __name__ == "__main__":
    unittest.main()
