"""Layer 2 through the real CLI (in-process and as a real subprocess) and genuine missing-dependency failures.

The LLM is a local mock HTTP server (no real API call). Tests that need spaCy are skipped when it is not installed;
run them with the project's .venv python."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from isl_gloss.cli import main
from tests.gloss_helpers import MockGemini
from tests.test_layer2 import FIXTURE, HAVE_SPACY, l1_row, write_l1
from transcript_cleaner.config import CleaningConfig
from transcript_cleaner.pipeline import process_bytes

ROOT = Path(__file__).resolve().parent.parent
KEY = "test-key-123456"
RAW = b"Title: Test talk\n\nSpeaker 1: We will not go outside. Where is Sanket going? She bought 3 apples.\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CliBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.dir / "structured").mkdir()
        self.cleaned = self.dir / "structured" / "talk.cleaned.json"
        self.cleaned.write_text(json.dumps(process_bytes(RAW, "talk.txt", CleaningConfig()).doc), encoding="utf-8")

    def cfg_for(self, srv):
        p = self.dir / "cfg.json"
        p.write_text(json.dumps({"provider": "gemini", "base_url": srv.base_url, "requests_per_minute": 6000}), encoding="utf-8")
        return p

    def cli(self, *args, env=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.dict(os.environ, {"GEMINI_API_KEY": KEY, **(env or {})}, clear=False):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def translate(self, srv, out_dir, *extra):
        return self.cli("translate", self.cleaned, "--config", self.cfg_for(srv), "--env-file", self.dir / "none.env", "-o", out_dir, *extra)


class TranslateLayer2Tests(CliBase):
    def test_enabled_runs_both_paths_saves_results_and_leaves_layer1_unchanged(self):
        with MockGemini() as plain, MockGemini() as withl2:
            code0, out0, err0 = self.translate(plain, self.dir / "plain")
            code, out, err = self.translate(withl2, self.dir / "with", "--layer2")
            self.assertEqual((code0, code), (0, 0), out0 + err0 + out + err)
            self.assertEqual((len(plain.requests), len(withl2.requests)), (1, 1))  # Layer 2 adds NO LLM request
        gloss = self.dir / "with" / "talk.gloss.json"
        # Layer 1 output is the same with or without Layer 2
        a = json.loads((self.dir / "plain" / "talk.gloss.json").read_text(encoding="utf-8"))
        b = json.loads(gloss.read_text(encoding="utf-8"))
        self.assertEqual(a["results"], b["results"])
        self.assertEqual(a["counts"], b["counts"])
        l2 = self.dir / "with" / "talk.layer2"
        for name in ("spacy_raw_outputs.json", "candidates_side_by_side.json", "candidates_side_by_side.csv", "selection_decisions.csv",
                     "warnings.csv", "summary.json", "summary_report.md", "run_manifest.json"):
            self.assertTrue((l2 / name).is_file(), name)
        manifest = json.loads((l2 / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["layer1_input"]["sha256_before"], sha(gloss))  # Layer 2 read exactly this file...
        self.assertTrue(manifest["layer1_input"]["unchanged"])  # ...and did not change it
        self.assertEqual(manifest["llm_requests_made_by_layer2"], 0)
        self.assertFalse((self.dir / "plain" / "talk.layer2").exists())
        rows = json.loads((l2 / "candidates_side_by_side.json").read_text(encoding="utf-8"))["results"]
        self.assertEqual([r["sentence_id"] for r in rows], [r["sentence_id"] for r in b["results"]])
        self.assertTrue(all(r["llm_gloss"] for r in rows))  # the LLM candidate is present either way
        if HAVE_SPACY:
            self.assertTrue(all(r["spacy_status"] == "ok" for r in rows))
            by = {r["source_text"]: r for r in rows}
            self.assertEqual(by["We will not go outside."]["spacy_gloss"], FIXTURE["We will not go outside."])
            self.assertEqual(by["Where is Sanket going?"]["spacy_gloss"], FIXTURE["Where is Sanket going?"])
        else:
            self.assertTrue(all(r["spacy_status"] == "unavailable" and r["selected_candidate"] == "LLM" for r in rows))
        self.assertIn("layer 2:", out)

    def test_without_the_flag_layer2_does_not_run(self):
        with MockGemini() as srv:
            code, out, err = self.translate(srv, self.dir / "o")
        self.assertEqual(code, 0, out + err)
        self.assertNotIn("layer 2", out)
        self.assertFalse((self.dir / "o" / "talk.layer2").exists())

    def test_disabled_through_the_cli_config(self):
        cfg = self.dir / "l2.json"
        cfg.write_text(json.dumps({"enabled": False}), encoding="utf-8")
        with MockGemini() as srv:
            code, out, err = self.translate(srv, self.dir / "o", "--layer2", "--layer2-config", cfg)
        self.assertEqual(code, 0, out + err)
        self.assertIn("disabled", out)
        self.assertTrue((self.dir / "o" / "talk.gloss.json").is_file())  # Layer 1 still wrote its output
        self.assertFalse((self.dir / "o" / "talk.layer2").exists())

    def test_layer2_config_options_reach_the_pipeline(self):
        cfg = self.dir / "l2.json"
        cfg.write_text(json.dumps({"llm_field": "final_tokens", "policy": {"allow_check_based_preference": True}}), encoding="utf-8")
        with MockGemini() as srv:
            code, out, err = self.translate(srv, self.dir / "o", "--layer2", "--layer2-config", cfg)
        self.assertEqual(code, 0, out + err)
        m = json.loads((self.dir / "o" / "talk.layer2" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["candidate_a_field"], "final_tokens")
        self.assertTrue(m["selection_policy"]["allow_check_based_preference"])

    def test_dry_run_with_layer2_makes_no_request_and_no_files(self):
        with MockGemini() as srv:
            code, out, err = self.translate(srv, self.dir / "o", "--layer2", "--dry-run")
            self.assertEqual(len(srv.requests), 0)
        self.assertEqual(code, 0, out + err)
        self.assertFalse((self.dir / "o" / "talk.layer2").exists())

    def test_incomplete_layer1_run_is_labelled_in_layer2_output(self):
        daily = (429, {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
        with MockGemini([daily]) as srv:
            code, out, err = self.translate(srv, self.dir / "o", "--layer2")
        self.assertEqual(code, 1, out + err)  # Layer 1's own exit status is preserved
        self.assertIn("INCOMPLETE", out)
        l2 = self.dir / "o" / "talk.layer2"
        m = json.loads((l2 / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertIs(m["layer1_input"]["layer1_complete"], False)
        self.assertIn("Layer 1 run was incomplete", (l2 / "summary_report.md").read_text(encoding="utf-8"))

    def test_standalone_layer2_command_and_disabled_config(self):
        with MockGemini() as srv:
            self.translate(srv, self.dir / "o")
        gloss = self.dir / "o" / "talk.gloss.json"
        before = sha(gloss)
        code, out, err = self.cli("layer2", gloss, "-o", self.dir / "l2")
        self.assertEqual(code, 0, out + err)
        self.assertIn("has NOT been established", out)
        first = (self.dir / "l2" / "candidates_side_by_side.json").read_bytes()
        self.cli("layer2", gloss, "-o", self.dir / "l2")  # re-run: same files, no duplicates
        self.assertEqual(first, (self.dir / "l2" / "candidates_side_by_side.json").read_bytes())
        self.assertEqual(len(list((self.dir / "l2").iterdir())), 8)
        off = self.dir / "off.json"
        off.write_text('{"enabled": false}', encoding="utf-8")
        code, out, _ = self.cli("layer2", gloss, "-o", self.dir / "l2off", "--layer2-config", off)
        self.assertEqual(code, 0)
        self.assertIn("disabled", out)
        self.assertFalse((self.dir / "l2off").exists())
        self.assertEqual(sha(gloss), before)

    def test_layer2_refuses_to_write_into_the_layer1_folder(self):
        with MockGemini() as srv:
            self.translate(srv, self.dir / "o")
        code, _, err = self.cli("layer2", self.dir / "o" / "talk.gloss.json", "-o", self.dir / "o")
        self.assertEqual(code, 2)
        self.assertIn("different from the folder", err)


class RealSubprocessTests(CliBase):
    def test_real_process_translate_with_layer2(self):
        with MockGemini() as srv:
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "GEMINI_API_KEY": KEY}
            p = subprocess.run([sys.executable, "-m", "isl_gloss.cli", "translate", str(self.cleaned), "--config", str(self.cfg_for(srv)),
                                "--env-file", str(self.dir / "none.env"), "-o", str(self.dir / "o"), "--layer2"],
                               capture_output=True, text=True, cwd=ROOT, env=env, timeout=240)
            n = len(srv.requests)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(n, 1)
        self.assertIn("layer 2:", p.stdout)
        self.assertNotIn(KEY, p.stdout + p.stderr)  # secrets never reach the logs
        self.assertTrue((self.dir / "o" / "talk.layer2" / "candidates_side_by_side.csv").is_file())


BLOCK_PACKAGE = """
import importlib.abc, sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ('spacy', 'nltk'):
            raise ModuleNotFoundError("No module named '" + fullname + "'", name=fullname)
sys.meta_path.insert(0, _Block())
"""

# Hides ONLY the model: spaCy itself is real and its own spacy.load() runs and raises its own error. The model is made
# invisible both as an importable module and in package metadata, which is how spaCy decides a model is installed.
HIDE_MODEL = """
import importlib.abc, importlib.metadata as md, sys
_real = md.distribution
def _dist(name, *a, **k):
    if str(name).lower().replace('_', '-') == 'en-core-web-sm':
        raise md.PackageNotFoundError(name)
    return _real(name, *a, **k)
md.distribution = _dist
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] == 'en_core_web_sm':
            raise ModuleNotFoundError("No module named 'en_core_web_sm'", name='en_core_web_sm')
sys.meta_path.insert(0, _Block())
"""
RUN_CLI = "\nimport sys\nfrom isl_gloss.cli import main\nsys.exit(main(sys.argv[1:]))\n"


class MissingDependencyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        rows = [l1_row("m-s1", "We will not go outside.", ["WE", "NOT", "GO", "OUTSIDE"]),
                l1_row("m-s2", "Where is Sanket going?", ["WHERE", "FS:SANKET", "GO"])]
        self.l1 = write_l1(self.dir, rows)
        self.before = sha(self.l1)

    def run_cli(self, prelude, *args):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        return subprocess.run([sys.executable, "-c", prelude + RUN_CLI, *map(str, args)], capture_output=True, text=True, cwd=ROOT, env=env, timeout=240)

    def assert_llm_survives(self, kind_message_part):
        out = self.dir / "l2"
        rows = json.loads((out / "candidates_side_by_side.json").read_text(encoding="utf-8"))["results"]
        self.assertEqual(len(rows), 2)
        for r, expected in zip(rows, ("WE NOT GO OUTSIDE", "WHERE FS:SANKET GO")):
            self.assertEqual(r["llm_gloss"], expected)  # the valid LLM result is intact
            self.assertEqual((r["spacy_status"], r["spacy_gloss"], r["spacy_error"]["kind"]), ("unavailable", None, "dependency_missing"))
            self.assertIn(kind_message_part, r["spacy_error"]["message"])
            self.assertIn("python -m spacy download en_core_web_sm", r["spacy_error"]["setup"])  # actionable
            self.assertEqual((r["selection_status"], r["selected_candidate"]), ("llm_primary", "LLM"))
            self.assertIn("POLICY DEFAULT", r["selection_reason"])
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["spacy"]["dependency_unavailable"], 2)
        self.assertEqual(sha(self.l1), self.before)

    def test_missing_spacy_package_is_explicit_and_keeps_llm_results(self):
        p = self.run_cli(BLOCK_PACKAGE, "layer2", self.l1, "-o", self.dir / "l2")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)  # a missing optional layer is not a crash
        self.assertIn("layer2-check", p.stdout)
        self.assert_llm_survives("Python package not installed")
        c = self.run_cli(BLOCK_PACKAGE, "layer2-check")
        self.assertEqual(c.returncode, 1)
        self.assertIn("python -m pip install spacy nltk", c.stdout)

    @unittest.skipUnless(HAVE_SPACY, "needs spaCy installed so that only the model can be hidden")
    def test_missing_model_uses_the_real_spacy_load_error(self):
        p = self.run_cli(HIDE_MODEL, "layer2", self.l1, "-o", self.dir / "l2")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assert_llm_survives("en_core_web_sm")
        r = json.loads((self.dir / "l2" / "candidates_side_by_side.json").read_text(encoding="utf-8"))["results"][0]
        self.assertIn("spaCy model en_core_web_sm is not available", r["spacy_error"]["message"])
        self.assertIn("E050", r["spacy_error"]["message"])  # the genuine spaCy exception text, not a mock
        c = self.run_cli(HIDE_MODEL, "layer2-check")
        self.assertEqual(c.returncode, 1)
        self.assertIn("python -m spacy download en_core_web_sm", c.stdout)

    @unittest.skipUnless(HAVE_SPACY, "needs spaCy installed")
    def test_control_without_hiding_the_model_works(self):
        p = self.run_cli("", "layer2", self.l1, "-o", self.dir / "l2")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        rows = json.loads((self.dir / "l2" / "candidates_side_by_side.json").read_text(encoding="utf-8"))["results"]
        self.assertTrue(all(r["spacy_status"] == "ok" for r in rows))


if __name__ == "__main__":
    unittest.main()
