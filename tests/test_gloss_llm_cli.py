import io
import json
import os
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from isl_gloss.cli import main
from isl_gloss.config import TranslatorConfig
from isl_gloss.llm import (AuthError, BadRequest, BlockedResponse, GeminiClient, MissingApiKey, QuotaExhausted,
                           RateLimiter, TransientError, TruncatedResponse, parse_results_json)
from isl_gloss.prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT
from tests.gloss_helpers import MockGemini

KEY = "test-key-123456"


def ok_body(text):
    return 200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def client_for(server, **kw):
    sleeps = []
    cfg = TranslatorConfig(base_url=server.base_url, requests_per_minute=6000, backoff_base_seconds=1.0, **kw)
    return GeminiClient(cfg, api_key=KEY, sleep=sleeps.append), sleeps


class GeminiClientTests(unittest.TestCase):
    def test_request_shape_and_key_handling(self):
        with MockGemini() as srv:
            c, _ = client_for(srv, model="gemini-2.5-flash")
            resp = c.complete(SYSTEM_PROMPT, "1. Hi.", RESPONSE_SCHEMA)
            req = srv.requests[0]
            self.assertTrue(req["path"].endswith("/models/gemini-2.5-flash:generateContent"))
            self.assertNotIn(KEY, req["path"])  # never in the URL
            headers = {k.lower(): v for k, v in req["headers"].items()}
            self.assertEqual(headers["x-goog-api-key"], KEY)
            body = req["body"]
            self.assertEqual(body["systemInstruction"]["parts"][0]["text"], SYSTEM_PROMPT)
            gen = body["generationConfig"]
            self.assertEqual((gen["temperature"], gen["responseMimeType"]), (0.0, "application/json"))
            self.assertEqual(gen["responseSchema"], RESPONSE_SCHEMA)
            self.assertEqual(gen["thinkingConfig"], {"thinkingBudget": 0})
            self.assertEqual(json.loads(resp.text)["results"][0]["n"], 1)
            self.assertEqual(body["contents"][0]["parts"][0]["text"], "1. Hi.")  # bare text, nothing else
            self.assertEqual(resp.usage["totalTokenCount"], 15)

    def test_thinking_budget_only_for_25_series(self):
        with MockGemini() as srv:
            c, _ = client_for(srv, model="gemini-3-flash-preview")
            c.complete("s", "1. Hello.", RESPONSE_SCHEMA)
            self.assertNotIn("thinkingConfig", srv.requests[0]["body"]["generationConfig"])

    def test_per_minute_429_is_retried_with_backoff(self):
        limited = (429, {"error": {"code": 429, "message": "Quota exceeded for requests per minute", "status": "RESOURCE_EXHAUSTED",
                                   "details": [{"retryDelay": "7s"}]}})
        with MockGemini([limited, limited, ok_body('{"results": []}')]) as srv:
            c, sleeps = client_for(srv)
            self.assertEqual(c.complete("s", "u", RESPONSE_SCHEMA).text, '{"results": []}')
            self.assertEqual(len(srv.requests), 3)
            backoffs = [s for s in sleeps if s >= 0.5]  # ignore the tiny rate-limiter pauses
            self.assertEqual(len(backoffs), 2)
            self.assertGreaterEqual(backoffs[0], 7)  # honours the server's retryDelay

    def test_daily_quota_is_not_retried(self):
        daily = (429, {"error": {"code": 429, "message": "Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
        with MockGemini([daily]) as srv:
            c, _ = client_for(srv)
            with self.assertRaises(QuotaExhausted):
                c.complete("s", "u", RESPONSE_SCHEMA)
            self.assertEqual(len(srv.requests), 1)

    def test_unknown_model_is_a_distinct_run_stopping_error(self):
        from isl_gloss.llm import ModelUnavailable
        with MockGemini([(404, {"error": {"message": "no longer available to new users"}})]) as srv:
            c, _ = client_for(srv)
            with self.assertRaises(ModelUnavailable) as cm:
                c.complete("s", "u", RESPONSE_SCHEMA)
            self.assertIn(c.model, str(cm.exception))

    def test_non_retryable_errors(self):
        for status, exc in ((400, BadRequest), (404, BadRequest), (401, AuthError), (403, AuthError)):
            with MockGemini([(status, {"error": {"message": f"nope {status}"}})]) as srv:
                c, _ = client_for(srv)
                with self.assertRaises(exc):
                    c.complete("s", "u", RESPONSE_SCHEMA)
                self.assertEqual(len(srv.requests), 1, status)

    def test_server_errors_retry_then_give_up(self):
        attempts = TranslatorConfig().max_attempts
        with MockGemini([(503, {"error": {"message": "overloaded"}})] * attempts) as srv:
            c, sleeps = client_for(srv)
            with self.assertRaises(TransientError) as cm:
                c.complete("s", "u", RESPONSE_SCHEMA)
            self.assertEqual(len(srv.requests), attempts)
            self.assertEqual(len([s for s in sleeps if s >= 0.5]), attempts - 1)
            self.assertIn(f"Gave up after {attempts} attempts", str(cm.exception))

    def test_blocked_truncated_and_empty_responses(self):
        cases = [
            ((200, {"promptFeedback": {"blockReason": "SAFETY"}}), BlockedResponse),
            ((200, {"candidates": []}), BlockedResponse),
            ((200, {"candidates": [{"content": {"parts": [{"text": '{"results": ['}]}, "finishReason": "MAX_TOKENS"}]}), TruncatedResponse),
            ((200, {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}), BlockedResponse),
        ]
        for reply, exc in cases:
            with MockGemini([reply]) as srv:
                c, _ = client_for(srv)
                with self.assertRaises(exc):
                    c.complete("s", "u", RESPONSE_SCHEMA)

    def test_key_is_scrubbed_from_error_messages(self):
        with MockGemini([(400, {"error": {"message": f"API key {KEY} is invalid"}})]) as srv:
            c, _ = client_for(srv)
            with self.assertRaises(BadRequest) as cm:
                c.complete("s", "u", RESPONSE_SCHEMA)
            self.assertNotIn(KEY, str(cm.exception))

    def test_missing_key_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingApiKey) as cm:
                GeminiClient(TranslatorConfig.for_provider("gemini"))
            self.assertIn("GEMINI_API_KEY", str(cm.exception))

    def test_rate_limiter_spaces_requests(self):
        now, slept = [0.0], []

        def sleep(s):
            slept.append(s)
            now[0] += s
        rl = RateLimiter(6, clock=lambda: now[0], sleep=sleep)  # 6 rpm => 10 s apart
        rl.wait()
        now[0] += 3
        rl.wait()
        self.assertEqual(slept, [7.0])

    def test_parse_results_json(self):
        good = '{"results": [{"id": "a", "gloss": ["X"], "uncertain": false, "notes": ""}]}'
        self.assertEqual(parse_results_json(good)[0]["id"], "a")
        self.assertEqual(parse_results_json("```json\n" + good + "\n```")[0]["id"], "a")
        self.assertEqual(parse_results_json("Sure! Here you go: " + good)[0]["id"], "a")
        with self.assertRaises(ValueError):
            parse_results_json("no json at all")


class CliEndToEndTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        # real cleaner output as the translator's input
        from transcript_cleaner.config import CleaningConfig
        from transcript_cleaner.pipeline import process_bytes
        raw = (b"Title: Test talk\n\nSpeaker 1: Hmm, we study the Jane tradition. The result was 15, not 50. "
               b"Ek divas tu ane hu. I will meet you tomorrow.\n")
        res = process_bytes(raw, "talk.txt", CleaningConfig())
        (self.dir / "structured").mkdir()
        self.cleaned = self.dir / "structured" / "talk.cleaned.json"
        self.cleaned.write_text(json.dumps(res.doc), encoding="utf-8")

    def run_cli(self, *args, env=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.dict(os.environ, env or {}, clear=False):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_translate_end_to_end_against_mock_server_then_resume_from_cache(self):
        cfg = self.dir / "cfg.json"
        with MockGemini() as srv:
            cfg.write_text(json.dumps({"provider": "gemini", "base_url": srv.base_url, "requests_per_minute": 6000}), encoding="utf-8")
            code, out, err = self.run_cli("translate", self.cleaned, "--config", cfg, "--env-file", self.dir / "none.env",
                                          env={"GEMINI_API_KEY": KEY})
            self.assertEqual(code, 0, out + err)
            self.assertEqual(len(srv.requests), 1)
            sent = srv.requests[0]["body"]["contents"][0]["parts"][0]["text"]
            self.assertNotIn("Hmm", sent)
            self.assertNotIn("Jane", sent)
            self.assertNotIn("Ek divas", sent)
            self.assertNotIn("Speaker", sent)  # sentence ids are the only metadata the model gets
            self.assertNotIn("Speaker", sent)
            self.assertTrue(sent.startswith("[talk-s0001] "))

            doc = json.loads((self.dir / "translations" / "talk.gloss.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["review_status"], "unreviewed")
            self.assertTrue(doc["complete"])
            self.assertEqual(len(doc["results"]), 4)
            by_id = {r["sentence_id"]: r for r in doc["results"]}
            first = by_id["talk-s0001"]
            self.assertEqual(first["source_text"], "Hmm, we study the Jane tradition.")
            self.assertEqual({s["text"] for s in first["skipped_spans"]}, {"Hmm", "Jane"})
            self.assertEqual(first["final_gloss"], "WE STUDY TRADITION")
            self.assertEqual(by_id["talk-s0003"]["translation_status"], "skipped_no_translatable_text")
            # the echo mock returns every word upper-cased; the rule pass strips the article and keeps the negation
            self.assertEqual(by_id["talk-s0002"]["final_gloss"], "RESULT WAS 15 NOT 50")
            self.assertEqual(by_id["talk-s0002"]["validation"]["status"], "pass")
            self.assertEqual([a["rule_id"] for a in by_id["talk-s0002"]["applied_rules"]], ["R-ART-01"])
            self.assertTrue((self.dir / "translations" / "talk.review_sheet.csv").exists())
            self.assertTrue((self.dir / "translations" / "talk.gloss.txt").exists())

            code, out, _ = self.run_cli("translate", self.cleaned, "--config", cfg, "--env-file", self.dir / "none.env",
                                        env={"GEMINI_API_KEY": KEY})
            self.assertEqual(code, 0)
            self.assertEqual(len(srv.requests), 1)  # second run served entirely from the cache
            self.assertIn("cache_hits", out)

    def test_quota_exhaustion_exits_1_and_says_how_to_resume(self):
        daily = (429, {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
        with MockGemini([daily]) as srv:
            cfg = self.dir / "cfg.json"
            cfg.write_text(json.dumps({"provider": "gemini", "base_url": srv.base_url, "requests_per_minute": 6000}), encoding="utf-8")
            code, out, _ = self.run_cli("translate", self.cleaned, "--config", cfg, "--env-file", self.dir / "none.env",
                                        env={"GEMINI_API_KEY": KEY})
            self.assertEqual(code, 1)
            self.assertIn("INCOMPLETE", out)
            self.assertIn("resume", out)

    def test_missing_key_exits_2_but_dry_run_needs_no_key(self):
        env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GROQ_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            code, _, err = self.run_cli("translate", self.cleaned, "--env-file", self.dir / "none.env")
            self.assertEqual(code, 2)
            self.assertIn("GROQ_API_KEY", err)  # groq is the default provider
            code, _, err = self.run_cli("translate", self.cleaned, "--provider", "gemini", "--env-file", self.dir / "none.env")
            self.assertEqual(code, 2)
            self.assertIn("GEMINI_API_KEY", err)
            code, out, _ = self.run_cli("translate", self.cleaned, "--dry-run")
            self.assertEqual(code, 0)
            self.assertIn("would be sent", out)
            self.assertFalse((self.dir / "translations").exists())

    def test_env_file_is_read_but_does_not_override_environment(self):
        envf = self.dir / "k.env"
        envf.write_text('GEMINI_API_KEY="from-file"\n# comment\n', encoding="utf-8")
        from isl_gloss.llm import load_env_file
        with mock.patch.dict(os.environ, {}, clear=True):
            load_env_file(envf)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "from-file")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}, clear=True):
            load_env_file(envf)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "from-env")

    def test_evaluate_writes_reviewer_sheet(self):
        sentences = self.dir / "s.json"
        sentences.write_text(json.dumps({"sentences": [
            {"id": "e1", "category": "simple", "text": "I am happy."},
            {"id": "e2", "category": "negation", "text": "He is not here."}]}), encoding="utf-8")
        with MockGemini() as srv:
            cfg = self.dir / "cfg.json"
            cfg.write_text(json.dumps({"provider": "gemini", "base_url": srv.base_url, "requests_per_minute": 6000}), encoding="utf-8")
            out_dir = self.dir / "eval"
            code, out, err = self.run_cli("evaluate", "--sentences", sentences, "-o", out_dir, "--config", cfg,
                                          "--env-file", self.dir / "none.env", env={"GEMINI_API_KEY": KEY})
            self.assertEqual(code, 0, out + err)
            sheet = (out_dir / "eval.review_sheet.csv").read_text(encoding="utf-8-sig")
            self.assertIn("candidate_gloss", sheet)
            self.assertIn("meaning_preserved", sheet)
            self.assertIn("I AM HAPPY", sheet)


if __name__ == "__main__":
    unittest.main()
