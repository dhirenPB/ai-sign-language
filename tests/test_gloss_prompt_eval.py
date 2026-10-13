import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from isl_gloss.cli import main
from isl_gloss.config import TranslatorConfig
from isl_gloss.llm import QuotaExhausted
from isl_gloss.prompt_eval import FAILURE_TYPES, diagnose, names_in, run_prompt_comparison
from isl_gloss.store import ResultStore, cache_key
from isl_gloss.translator import Translator
from tests.gloss_helpers import FakeClient, chunk, make_row, reply_from


def row(text, tokens, needs=False, warnings=None, status="translated"):
    return {"llm_input_text": text, "source_text": text, "translation_status": status,
            "final_tokens": tokens, "candidate_tokens": tokens, "llm": {"needs_human_review": needs},
            "validation": {"warnings": warnings or []}}


def kinds(text, tokens, names=None, labels=None, **kw):
    return diagnose(row(text, tokens, **kw), names or names_in(text), labels or set())


class DiagnosticsObserveOnly(unittest.TestCase):
    def test_each_observed_failure_type_is_detected(self):
        self.assertIn("empty_gloss_not_flagged", kinds("Water flows.", []))
        self.assertIn("empty_gloss_flagged_for_review", kinds("Water flows.", [], needs=True))
        self.assertIn("name_not_fingerspelled", kinds("We met Maria Gonzalez today.", ["TODAY", "WE", "MARIA", "MEET"]))
        self.assertIn("name_compound_label", kinds("Professor Jonathan Young invited us.", ["PROFESSOR_JONATHAN_YOUNG", "INVITE"]))
        self.assertIn("fs_multiword_token", kinds("We met Jonathan Young.", ["WE", "FS:JONATHAN_YOUNG", "MEET"]))
        self.assertIn("name_missing", kinds("We met Maria today.", ["WE", "MEET", "TODAY"]))
        self.assertIn("condition_dropped", kinds("If it rains we stay home.", ["RAIN", "WE", "STAY", "HOME"]))
        self.assertIn("negation_dropped", kinds("I do not understand.", ["I", "UNDERSTAND"]))
        self.assertIn("cause_dropped", kinds("It dried because the rain stopped.", ["DRY", "RAIN", "STOP"]))
        self.assertIn("time_dropped", kinds("Yesterday we walked.", ["WE", "WALK"]))
        self.assertIn("number_dropped", kinds("We walked 5 kilometres.", ["WE", "WALK", "KILOMETRE"]))
        self.assertIn("notation_invalid_token", kinds("Water flows.", ["water", "FLOW"]))
        self.assertIn("notation_inconsistent_multiword", kinds("It dried up.", ["IT", "DRY", "UP"], labels={"DRY_UP"}))
        warns = [{"code": "source_text_mismatch", "severity": "review"}]
        self.assertIn("alignment_problem", kinds("Water flows.", ["WATER"], warnings=warns))

    def test_clean_glosses_show_no_failure(self):
        self.assertEqual(kinds("Professor Jonathan Young invited us.", ["PROFESSOR", "FS:JONATHAN", "FS:YOUNG", "INVITE", "WE"]), [])
        self.assertEqual(kinds("If it rains we stay home.", ["IF", "RAIN", "WE", "STAY", "HOME"]), [])
        self.assertEqual(kinds("I do not understand.", ["I", "UNDERSTAND", "NOT"]), [])
        self.assertEqual(kinds("Yesterday we walked 5 kilometres.", ["YESTERDAY", "WE", "WALK", "5", "KILOMETRE"]), [])

    def test_failed_or_skipped_sentences_are_not_diagnosed_as_translation_failures(self):
        self.assertEqual(kinds("Water flows.", None, status="failed"), [])
        self.assertEqual(kinds("Water flows.", [], status="skipped_no_translatable_text"), [])

    def test_diagnose_never_modifies_the_row_and_every_type_is_documented(self):
        r = row("If Professor Jonathan Young comes we stay.", ["PROFESSOR_JONATHAN_YOUNG", "COME", "STAY"])
        before = copy.deepcopy(r)
        got = diagnose(r, names_in(r["llm_input_text"]), set())
        self.assertEqual(r, before)
        self.assertTrue(got)
        self.assertTrue(set(got) <= set(FAILURE_TYPES))


class PromptComparisonRun(unittest.TestCase):
    SPEC = {"passages": [
        {"id": "p1", "covers": "conditions", "text": "If you practise every day, your mind will become calm. We do not rush."},
        {"id": "p2", "covers": "names", "text": "Professor Jonathan Young invited us to Bakersfield."}]}

    def run_cmp(self, out, clients, versions=("5.0", "6.0")):
        cfg = TranslatorConfig(reasoning_effort="medium")
        made = {}

        def make(cfg_v):
            made[cfg_v.prompt_version] = clients[cfg_v.prompt_version]
            return Translator(cfg_v, clients[cfg_v.prompt_version])
        return run_prompt_comparison(self.SPEC, cfg, make, Path(out), list(versions)), made

    def test_same_inputs_to_both_versions_outputs_kept_and_config_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            clients = {"5.0": FakeClient(), "6.0": FakeClient()}
            s, made = self.run_cmp(d, clients)
            ins = {v: [c["user"] for c in clients[v].calls] for v in clients}
            self.assertEqual(len(ins["5.0"]), len(ins["6.0"]))  # identical grouping: one request per group for both
            self.assertTrue(all("\n1. " in u or u.startswith("1. ") for u in ins["5.0"]))  # each in its own input format
            self.assertTrue(all("[p1-s0001]" in ins["6.0"][0] for _ in [0]))
            for v in ("5.0", "6.0"):
                self.assertTrue((Path(d) / f"results_prompt_{v}.json").exists())  # candidate outputs retained
                kept = json.loads((Path(d) / f"results_prompt_{v}.json").read_text(encoding="utf-8"))
                self.assertEqual(kept["config"]["reasoning_effort"], "medium")
                self.assertEqual(kept["config"]["prompt_version"], v)
                self.assertTrue(all("observed_failures" in r and "candidate_tokens" in r for r in kept["results"]))
                self.assertEqual(s["per_version"][v]["config"]["model"], "openai/gpt-oss-120b")
            sheet = (Path(d) / "comparison_sheet.csv").read_text(encoding="utf-8-sig")
            for col in ("gloss_5.0", "gloss_6.0", "observed_failures_5.0", "observed_failures_6.0", "better_prompt"):
                self.assertIn(col, sheet)
            self.assertTrue((Path(d) / "comparison.json").exists())

    def test_failure_types_are_recorded_per_version(self):
        bad = reply_from(lambda label, text: make_row(label, text, ["PROFESSOR_JONATHAN_YOUNG", "INVITE"]
                                                     if "Professor" in text else ["PRACTISE", "MIND", "CALM"]))
        with tempfile.TemporaryDirectory() as d:
            s, _ = self.run_cmp(d, {"5.0": FakeClient([bad, bad]), "6.0": FakeClient()})
            v5, v6 = s["per_version"]["5.0"], s["per_version"]["6.0"]
            self.assertIn("name_compound_label", v5["observed_failure_types"])
            self.assertIn("condition_dropped", v5["observed_failure_types"])
            self.assertGreater(v5["sentences_with_an_observed_failure"], 0)
            self.assertEqual(v6["sentences"], v5["sentences"])

    def test_no_claim_of_improvement_and_incomplete_runs_are_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            s, _ = self.run_cmp(d, {"5.0": FakeClient(), "6.0": FakeClient()})
            text = json.dumps(s).lower()
            self.assertNotIn("improved", text.replace("do not show", ""))
            self.assertIn("not a measure of translation accuracy", s["interpretation"])
            self.assertIn("reviewer", s["interpretation"])
        with tempfile.TemporaryDirectory() as d:
            s, _ = self.run_cmp(d, {"5.0": FakeClient(), "6.0": FakeClient([QuotaExhausted("daily")])})
            self.assertFalse(s["all_versions_complete"])
            self.assertFalse(s["per_version"]["6.0"]["complete"])
            self.assertIn("quota_exhausted", s["per_version"]["6.0"]["stop_reason"])

    def test_dry_run_needs_no_translator_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            s = run_prompt_comparison(self.SPEC, TranslatorConfig(), None, Path(d), ["5.0", "6.0"], dry_run=True)
            self.assertEqual(s["requests_per_version"], 2)
            self.assertFalse(list(Path(d).iterdir()))

    def test_cli_dry_run_over_the_shipped_inputs(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["compare-prompts", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("requests per prompt version: 11", out.getvalue())


class CacheKeyCoversGenerationSettings(unittest.TestCase):
    def test_a_result_cached_at_one_reasoning_effort_is_not_served_to_another(self):
        self.assertNotEqual(cache_key("m", "p", "t", "id", "effort=low"), cache_key("m", "p", "t", "id", "effort=medium"))
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "c.jsonl"
            Translator(TranslatorConfig(reasoning_effort="low"), FakeClient(), ResultStore(cache)).run([chunk("s1", "He is tall.")])
            same = FakeClient()
            Translator(TranslatorConfig(reasoning_effort="low"), same, ResultStore(cache)).run([chunk("s1", "He is tall.")])
            self.assertEqual(same.calls, [])
            other = FakeClient()
            Translator(TranslatorConfig(reasoning_effort="medium"), other, ResultStore(cache)).run([chunk("s1", "He is tall.")])
            self.assertEqual(len(other.calls), 1)  # different configuration: a fresh request, never a stale answer


if __name__ == "__main__":
    unittest.main()
