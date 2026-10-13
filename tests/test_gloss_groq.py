import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from isl_gloss.cli import main
from isl_gloss.config import PROVIDER_DEFAULTS, TranslatorConfig
from isl_gloss.llm import (AuthError, BadRequest, BlockedResponse, GeminiClient, GroqClient, MissingApiKey,
                           ModelUnavailable, QuotaExhausted, RequestTooLarge, TokenPacer, TransientError,
                           TruncatedResponse, make_client, parse_wait_seconds, to_json_schema)
from isl_gloss.models import TRANSLATED
from isl_gloss.prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT
from isl_gloss.translator import Translator
from tests.gloss_helpers import FakeClient, MockGroq, chunk, echo_reply

KEY = "gsk_test_key_123456"


def ok(text, finish="stop", usage=None):
    return 200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                 "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def client_for(server, **kw):
    sleeps = []
    cfg = TranslatorConfig(base_url=server.base_url, requests_per_minute=6000, tokens_per_minute=None,
                           backoff_base_seconds=1.0, **kw)
    return GroqClient(cfg, api_key=KEY, sleep=sleeps.append), sleeps


class GroqRequestTests(unittest.TestCase):
    def test_defaults_are_groq_gpt_oss_120b(self):
        c = TranslatorConfig()
        self.assertEqual((c.provider, c.model, c.api_key_env, c.base_url),
                         ("groq", "openai/gpt-oss-120b", "GROQ_API_KEY", "https://api.groq.com/openai/v1"))

    def test_request_shape_auth_and_strict_schema(self):
        with MockGroq() as srv:
            c, _ = client_for(srv)
            resp = c.complete(SYSTEM_PROMPT, "[s1] Hello there.", RESPONSE_SCHEMA)
            req = srv.requests[0]
            self.assertEqual(req["path"], "/openai/v1/chat/completions")
            headers = {k.lower(): v for k, v in req["headers"].items()}
            self.assertEqual(headers["authorization"], f"Bearer {KEY}")
            self.assertTrue(headers["user-agent"].startswith("isl-gloss"))  # not the default Python-urllib agent
            self.assertNotIn(KEY, req["path"])
            self.assertNotIn(KEY, json.dumps(req["body"]))
            b = req["body"]
            self.assertEqual(b["model"], "openai/gpt-oss-120b")
            self.assertEqual([m["role"] for m in b["messages"]], ["system", "user"])
            self.assertEqual(b["messages"][0]["content"], SYSTEM_PROMPT)
            self.assertEqual(b["messages"][1]["content"], "[s1] Hello there.")  # bare text plus the id, nothing else
            self.assertEqual((b["temperature"], b["max_completion_tokens"]), (0.0, 8192))
            self.assertEqual((b["reasoning_effort"], b["include_reasoning"]), ("low", False))
            self.assertNotIn("stream", b)
            rf = b["response_format"]
            self.assertEqual((rf["type"], rf["json_schema"]["strict"]), ("json_schema", True))
            schema = rf["json_schema"]["schema"]
            self.assertEqual(schema["type"], "object")
            self.assertFalse(schema["additionalProperties"])
            item = schema["properties"]["results"]["items"]
            self.assertEqual((item["type"], item["additionalProperties"]), ("object", False))
            self.assertEqual(set(item["required"]), set(item["properties"]))  # strict mode: every property required
            self.assertEqual((item["properties"]["sentence_id"]["type"], item["properties"]["needs_human_review"]["type"]),
                             ("string", "boolean"))
            self.assertEqual(json.loads(resp.text)["results"][0]["sentence_id"], "s1")
            self.assertEqual(resp.usage["total_tokens"], 150)
            self.assertEqual(resp.model, "openai/gpt-oss-120b")

    def test_reasoning_fields_only_for_reasoning_models(self):
        with MockGroq() as srv:
            c, _ = client_for(srv, model="llama-3.3-70b-versatile")
            c.complete("s", "1. Hi.", RESPONSE_SCHEMA)
            b = srv.requests[0]["body"]
            self.assertNotIn("reasoning_effort", b)
            self.assertNotIn("include_reasoning", b)
        with MockGroq() as srv:
            c, _ = client_for(srv, reasoning_effort="high")
            c.complete("s", "1. Hi.", RESPONSE_SCHEMA)
            self.assertEqual(srv.requests[0]["body"]["reasoning_effort"], "high")

    def test_to_json_schema_converts_and_does_not_mutate_the_source(self):
        before = json.dumps(RESPONSE_SCHEMA, sort_keys=True)
        out = to_json_schema(RESPONSE_SCHEMA)
        self.assertEqual(json.dumps(RESPONSE_SCHEMA, sort_keys=True), before)
        self.assertEqual(out["properties"]["results"]["type"], "array")
        self.assertEqual(out["properties"]["results"]["items"]["properties"]["candidate_gloss"]["items"]["type"], "string")


class GroqErrorTests(unittest.TestCase):
    def test_per_minute_token_limit_is_retried_using_the_servers_wait_hint(self):
        tpm = (429, {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
                                          "Limit 8000, Used 7000, Requested 3000. Please try again in 7.5s.",
                               "type": "tokens"}})
        with MockGroq([tpm, ok('{"results": []}')]) as srv:
            c, sleeps = client_for(srv)
            self.assertEqual(c.complete("s", "u", RESPONSE_SCHEMA).text, '{"results": []}')
            self.assertEqual(len(srv.requests), 2)
            backoffs = [s for s in sleeps if s >= 0.5]
            self.assertEqual(len(backoffs), 1)
            self.assertGreaterEqual(backoffs[0], 7.5)

    def test_retry_after_header_is_honoured(self):
        limited = (429, {"error": {"message": "Rate limit reached on requests per minute (RPM)."}}, {"retry-after": "12"})
        with MockGroq([limited, ok('{"results": []}')]) as srv:
            c, sleeps = client_for(srv)
            c.complete("s", "u", RESPONSE_SCHEMA)
            self.assertGreaterEqual([s for s in sleeps if s >= 0.5][0], 12)

    def test_daily_limits_stop_the_run_without_retrying(self):
        for msg in ("Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 199900, "
                    "Requested 3000. Please try again in 6m52.8s.",
                    "Rate limit reached on requests per day (RPD): Limit 1000, Used 1000. Please try again in 14m3s."):
            with MockGroq([(429, {"error": {"message": msg}})]) as srv:
                c, _ = client_for(srv)
                with self.assertRaises(QuotaExhausted):
                    c.complete("s", "u", RESPONSE_SCHEMA)
                self.assertEqual(len(srv.requests), 1)

    def test_a_very_long_wait_is_treated_as_exhausted_quota(self):
        with MockGroq([(429, {"error": {"message": "Rate limit reached."}}, {"retry-after": "3600"})]) as srv:
            c, _ = client_for(srv)
            with self.assertRaises(QuotaExhausted):
                c.complete("s", "u", RESPONSE_SCHEMA)

    def test_status_code_mapping(self):
        cases = ((401, AuthError), (403, AuthError), (404, ModelUnavailable), (400, BadRequest), (413, RequestTooLarge))
        for status, exc in cases:
            with MockGroq([(status, {"error": {"message": f"nope {status}", "type": "invalid_request_error"}})]) as srv:
                c, _ = client_for(srv)
                with self.assertRaises(exc):
                    c.complete("s", "u", RESPONSE_SCHEMA)
                self.assertEqual(len(srv.requests), 1, status)
        self.assertTrue(issubclass(RequestTooLarge, TruncatedResponse))  # so the translator splits the group

    def test_server_errors_and_flex_capacity_are_retried_then_give_up(self):
        for status in (503, 498):
            attempts = TranslatorConfig().max_attempts
            with MockGroq([(status, {"error": {"message": "overloaded"}})] * attempts) as srv:
                c, sleeps = client_for(srv)
                with self.assertRaises(TransientError) as cm:
                    c.complete("s", "u", RESPONSE_SCHEMA)
                self.assertEqual(len(srv.requests), attempts)
                self.assertIn(f"Gave up after {attempts} attempts", str(cm.exception))

    def test_truncated_blocked_and_empty_replies(self):
        cases = [
            (ok('{"results": [', finish="length"), TruncatedResponse),
            ((200, {"choices": []}), BlockedResponse),
            (ok(""), BlockedResponse),
            (ok("", finish="content_filter"), BlockedResponse),
        ]
        for reply, exc in cases:
            with MockGroq([reply]) as srv:
                c, _ = client_for(srv)
                with self.assertRaises(exc):
                    c.complete("s", "u", RESPONSE_SCHEMA)

    def test_key_is_scrubbed_from_errors_and_missing_key_is_clear(self):
        with MockGroq([(400, {"error": {"message": f"Invalid API key {KEY}"}})]) as srv:
            c, _ = client_for(srv)
            with self.assertRaises(BadRequest) as cm:
                c.complete("s", "u", RESPONSE_SCHEMA)
            self.assertNotIn(KEY, str(cm.exception))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingApiKey) as cm:
                GroqClient(TranslatorConfig())
            self.assertIn("GROQ_API_KEY", str(cm.exception))
            self.assertIn("Groq", str(cm.exception))

    def test_request_too_large_makes_the_translator_split_the_group(self):
        client = FakeClient([RequestTooLarge("too big"), echo_reply, echo_reply])
        tr = Translator(TranslatorConfig(batch_size=4), client)
        out = tr.run([chunk(f"c{i}", f"Statement number {i} here.") for i in range(4)])
        self.assertTrue(all(r.status == TRANSLATED for r in out.results))
        self.assertEqual(len(client.calls), 3)  # one rejected, then two halves


class WaitParsingAndPacingTests(unittest.TestCase):
    def test_parse_wait_seconds(self):
        self.assertEqual(parse_wait_seconds("Limit reached. Please try again in 7.5s."), 7.5)
        self.assertAlmostEqual(parse_wait_seconds("Please try again in 6m52.8s."), 412.8)
        self.assertAlmostEqual(parse_wait_seconds("Please try again in 850ms."), 0.85)
        self.assertEqual(parse_wait_seconds("Please try again in 1h2m3s"), 3723)
        self.assertIsNone(parse_wait_seconds("no hint here"))

    def test_token_pacer_keeps_a_rolling_minute_under_budget(self):
        now, slept = [0.0], []

        def sleep(s):
            slept.append(s)
            now[0] += s
        pacer = TokenPacer(1000, clock=lambda: now[0], sleep=sleep)
        pacer.wait(800)  # nothing sent yet
        self.assertEqual(slept, [])
        pacer.record(800)
        now[0] = 10.0
        pacer.wait(100)  # 800 + 100 fits
        self.assertEqual(slept, [])
        pacer.wait(500)  # 800 + 500 > 1000: wait for the first batch to age out (at t=60)
        self.assertEqual(len(slept), 1)
        self.assertAlmostEqual(now[0], 60.05, places=2)
        pacer.record(500)
        pacer.wait(400)  # only the 500 is left in the window: 500 + 400 fits
        self.assertEqual(len(slept), 1)

    def test_pacer_never_blocks_forever_and_can_be_disabled(self):
        now, slept = [0.0], []
        pacer = TokenPacer(1000, clock=lambda: now[0], sleep=slept.append)
        pacer.wait(5000)  # larger than the whole budget: let it through, waiting would never help
        self.assertEqual(slept, [])
        off = TokenPacer(None, clock=lambda: now[0], sleep=slept.append)
        off.record(10 ** 9)
        off.wait(10 ** 9)
        self.assertEqual(slept, [])

    def test_client_paces_by_tokens_using_real_usage_numbers(self):
        now, slept = [0.0], []

        def sleep(s):
            slept.append(s)
            now[0] += s
        cfg = TranslatorConfig(requests_per_minute=6000, tokens_per_minute=1000, backoff_base_seconds=1.0)
        usage = {"prompt_tokens": 600, "completion_tokens": 300, "total_tokens": 900}
        with MockGroq([ok('{"results": []}', usage=usage), ok('{"results": []}', usage=usage)]) as srv:
            cfg.base_url = srv.base_url
            c = GroqClient(cfg, api_key=KEY, sleep=sleep, clock=lambda: now[0])
            c.complete("s", "u", RESPONSE_SCHEMA)
            self.assertEqual([s for s in slept if s > 1], [])
            c.complete("s", "u", RESPONSE_SCHEMA)  # 900 recorded + ~500 estimate > 1000: must wait about a minute
            self.assertTrue(any(55 <= s <= 61 for s in slept), slept)


class ProviderConfigTests(unittest.TestCase):
    def test_for_provider_and_from_dict_use_the_providers_own_defaults(self):
        g = TranslatorConfig.for_provider("gemini")
        self.assertEqual((g.provider, g.model, g.api_key_env, g.requests_per_minute, g.tokens_per_minute),
                         ("gemini", "gemini-3.6-flash", "GEMINI_API_KEY", 5.0, None))
        c = TranslatorConfig.from_dict({"provider": "gemini", "model": "gemini-3.5-flash-lite"})
        self.assertEqual((c.base_url, c.model), (PROVIDER_DEFAULTS["gemini"]["base_url"], "gemini-3.5-flash-lite"))
        self.assertEqual(TranslatorConfig.from_dict({"batch_size": 3}).provider, "groq")  # default provider
        with self.assertRaises(ValueError):
            TranslatorConfig.for_provider("openai")
        with self.assertRaises(ValueError):
            TranslatorConfig.from_dict({"provider": "nope"})
        with self.assertRaises(ValueError):
            TranslatorConfig(reasoning_effort="extreme").validate()
        with self.assertRaises(ValueError):
            TranslatorConfig(tokens_per_minute=10).validate()

    def test_make_client_returns_the_configured_provider_only(self):
        self.assertIsInstance(make_client(TranslatorConfig(), api_key="k"), GroqClient)
        self.assertIsInstance(make_client(TranslatorConfig.for_provider("gemini"), api_key="k"), GeminiClient)
        bad = TranslatorConfig()
        bad.provider = "other"
        with self.assertRaises(ValueError):
            make_client(bad, api_key="k")


class GroqCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        from tests.test_gloss_sentences_grouping import cleaned
        (self.dir / "structured").mkdir()
        self.cleaned = self.dir / "structured" / "talk.cleaned.json"
        self.cleaned.write_text(json.dumps(cleaned(
            "Speaker 1: Hmm, we study the Jane tradition. The result was 15, not 50. I will meet you tomorrow.")), encoding="utf-8")

    def run_cli(self, *args, env=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.dict(os.environ, env or {}, clear=False):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_translate_with_groq_by_default_end_to_end_and_resume_from_cache(self):
        with MockGroq() as srv:
            cfg = self.dir / "cfg.json"
            cfg.write_text(json.dumps({"base_url": srv.base_url, "requests_per_minute": 6000, "tokens_per_minute": None}), encoding="utf-8")
            env = {"GROQ_API_KEY": KEY}
            code, out, err = self.run_cli("translate", self.cleaned, "--config", cfg, "--env-file", self.dir / "none.env", env=env)
            self.assertEqual(code, 0, out + err)
            self.assertEqual(len(srv.requests), 1)
            body = srv.requests[0]["body"]
            sent = body["messages"][1]["content"]
            self.assertEqual(body["model"], "openai/gpt-oss-120b")
            self.assertTrue(sent.startswith("[talk-s0001] "))
            for leaked in ("Hmm", "Jane", "Speaker", "00:00"):
                self.assertNotIn(leaked, sent)
            doc = json.loads((self.dir / "translations" / "talk.gloss.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["metadata"]["model"], "groq/openai/gpt-oss-120b")
            self.assertTrue(doc["complete"])
            self.assertTrue(doc["assembly_validation"]["passed"])
            self.assertEqual(doc["results"][1]["final_gloss"], "RESULT WAS 15 NOT 50")
            code, out, _ = self.run_cli("translate", self.cleaned, "--config", cfg, "--env-file", self.dir / "none.env", env=env)
            self.assertEqual(code, 0)
            self.assertEqual(len(srv.requests), 1)  # served from the cache

    def test_flags_switch_provider_effort_and_pacing(self):
        code, out, _ = self.run_cli("translate", self.cleaned, "--dry-run", "--provider", "gemini")
        self.assertIn("model gemini-3.6-flash", out)
        code, out, _ = self.run_cli("translate", self.cleaned, "--dry-run")
        self.assertIn("model openai/gpt-oss-120b", out)
        from isl_gloss.cli import _config, build_parser
        args = build_parser().parse_args(["translate", str(self.cleaned), "--tpm", "5000", "--reasoning-effort", "medium",
                                          "--rpm", "10", "--model", "openai/gpt-oss-20b"])
        cfg = _config(args)
        self.assertEqual((cfg.tokens_per_minute, cfg.reasoning_effort, cfg.requests_per_minute, cfg.model),
                         (5000, "medium", 10, "openai/gpt-oss-20b"))

    def test_daily_quota_stops_with_resume_hint(self):
        with MockGroq([(429, {"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000. Please try again in 20m."}})]) as srv:
            cfg = self.dir / "cfg.json"
            cfg.write_text(json.dumps({"base_url": srv.base_url, "requests_per_minute": 6000}), encoding="utf-8")
            code, out, _ = self.run_cli("translate", self.cleaned, "--config", cfg, "--env-file", self.dir / "none.env",
                                        env={"GROQ_API_KEY": KEY})
            self.assertEqual(code, 1)
            self.assertIn("INCOMPLETE", out)
            self.assertIn("resume", out)


if __name__ == "__main__":
    unittest.main()
