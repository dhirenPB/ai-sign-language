"""Interactive pipeline tester: backend stages, the local server and its safety checks. The LLM is always a test double."""

import http.client
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from isl_gloss.cli import build_parser
from isl_gloss.llm import QuotaExhausted
from isl_gloss.webui import demo
from isl_gloss.webui.server import make_server
from tests.gloss_helpers import FakeClient
from tests.test_layer2 import HAVE_SPACY

TEXT = "Hmm, we will not go outside unless it stops raining. Why do we get angry so quickly? She bought 3 apples."


class BackendTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_analyze_returns_cleaning_numbered_sentences_and_groups(self):
        a = demo.analyze(TEXT)
        self.assertEqual([s["index"] for s in a["sentences"]], [1, 2, 3])
        self.assertEqual(a["sentences"][0]["text"], "Hmm, we will not go outside unless it stops raining.")
        self.assertEqual(a["sentences"][0]["skip_spans"][0]["text"], "Hmm")  # fillers stay visible but are marked skipped
        self.assertEqual(a["cleaning"]["chunks"][0]["flags"][0]["code"], "expression_cue")
        self.assertEqual((len(a["groups"]), a["groups"][0]["sentences"], a["estimated_llm_requests"]), (1, [1, 2, 3], 1))
        self.assertIn("end of the text", a["groups"][0]["boundary_text"])
        self.assertTrue(all(s["group_index"] == 1 for s in a["sentences"]))

    def test_input_limits_and_empty_input(self):
        for bad in ("", "   ", None):
            with self.assertRaises(demo.DemoError):
                demo.analyze(bad)
        with self.assertRaises(demo.DemoError) as cm:
            demo.analyze("x" * (demo.MAX_CHARS + 1))
        self.assertIn(str(demo.MAX_CHARS), str(cm.exception))
        with self.assertRaises(demo.DemoError):
            demo.analyze(" ".join(f"Sentence number {i} is here." for i in range(demo.MAX_SENTENCES + 5)))

    def test_bad_prompt_version_is_reported_not_crashed(self):
        with self.assertRaises(demo.DemoError):
            demo.analyze(TEXT, {"prompt_version": "99.0"})

    def test_translate_runs_both_layers_and_compares(self):
        client = FakeClient()
        t = demo.translate(TEXT, {"call_llm": True, "use_cache": False}, env_file=self.dir / "none.env", data_dir=self.dir, client=client)
        self.assertEqual(client.requests_made, 1)  # one group -> one request
        rows = t["llm"]["rows"]
        self.assertEqual([r["status"] for r in rows], ["translated"] * 3)
        self.assertEqual(rows[0]["candidate_tokens"][:2], ["WE", "WILL"])  # the double echoes words; "Hmm" was skipped before the model saw it
        self.assertNotIn("Hmm", t["llm"]["requests"][0]["user_payload"])
        self.assertIn("[input-s0001]", t["llm"]["requests"][0]["user_payload"])
        self.assertEqual(len(t["spacy"]["rows"]), 3)
        items = t["comparison"]["items"]
        self.assertEqual([i["index"] for i in items], [1, 2, 3])
        self.assertTrue(all(i["primary_candidate"] == "LLM" for i in items))
        self.assertEqual(t["comparison"]["summary"]["selection"]["evidence_based_selections"], 0)
        self.assertIn("UNREVIEWED", t["comparison"]["note"])
        if HAVE_SPACY:
            self.assertEqual(t["spacy"]["rows"][2]["spacy_status"], "ok")
        else:
            self.assertEqual(t["spacy"]["rows"][2]["spacy_status"], "unavailable")

    def test_llm_switched_off_shows_the_request_and_still_runs_spacy(self):
        client = FakeClient()
        t = demo.translate(TEXT, {"call_llm": False}, env_file=self.dir / "none.env", data_dir=self.dir, client=client)
        self.assertEqual(client.requests_made, 0)
        self.assertFalse(t["llm"]["enabled"])
        self.assertTrue(all(r["status"] == "not_attempted" for r in t["llm"]["rows"]))
        self.assertIn("[input-s0001]", t["llm"]["requests"][0]["user_payload"])  # what WOULD be sent
        self.assertEqual(len(t["spacy"]["rows"]), 3)

    def test_missing_api_key_is_reported_and_spacy_still_runs(self):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            t = demo.translate(TEXT, {"call_llm": True}, env_file=self.dir / "none.env", data_dir=self.dir)
        self.assertEqual(t["llm"]["error"]["kind"], "missing_api_key")
        self.assertEqual(len(t["spacy"]["rows"]), 3)
        self.assertEqual(len(t["comparison"]["items"]), 3)

    def test_llm_failure_is_shown_and_does_not_hide_spacy(self):
        client = FakeClient([QuotaExhausted("daily quota used up")])
        t = demo.translate(TEXT, {"call_llm": True, "use_cache": False}, env_file=self.dir / "none.env", data_dir=self.dir, client=client)
        self.assertIn("quota", (t["llm"]["stop_reason"] or "") + json.dumps(t["llm"]["error"]).lower())
        self.assertEqual(len(t["spacy"]["rows"]), 3)

    def test_identical_request_is_served_from_the_saved_answers(self):
        first = FakeClient()
        demo.translate(TEXT, {"call_llm": True, "use_cache": True}, env_file=self.dir / "none.env", data_dir=self.dir, client=first)
        second = FakeClient()
        t = demo.translate(TEXT, {"call_llm": True, "use_cache": True}, env_file=self.dir / "none.env", data_dir=self.dir, client=second)
        self.assertEqual((first.requests_made, second.requests_made), (1, 0))
        self.assertTrue(all(r["cache_hit"] for r in t["llm"]["rows"]))

    def test_scrub_removes_a_configured_key_value(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_FAKEFAKEFAKE12345"}):
            self.assertEqual(demo.scrub('{"x": "gsk_FAKEFAKEFAKE12345"}'), '{"x": "***"}')


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = make_server(0, env_file="__no_such_env__", data_dir=tempfile.mkdtemp())
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def req(self, method, path, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        data = raw if raw is not None else (None if body is None else json.dumps(body))
        c.request(method, path, body=data, headers=hdrs)
        r = c.getresponse()
        payload = r.read()
        c.close()
        return r.status, r.getheader("Content-Type"), payload

    def test_page_and_status(self):
        st, ctype, body = self.req("GET", "/")
        self.assertEqual((st, ctype), (200, "text/html; charset=utf-8"))
        self.assertIn(b"ISL Pipeline Tester", body)
        st, _, body = self.req("GET", "/api/status")
        info = json.loads(body)
        self.assertEqual(st, 200)
        self.assertEqual(info["limits"]["max_sentences"], demo.MAX_SENTENCES)
        self.assertNotIn("api_key", body.decode().lower().replace("llm_key_present", ""))  # only a yes/no, never a value

    def test_analyze_over_http(self):
        st, _, body = self.req("POST", "/api/analyze", {"text": TEXT})
        self.assertEqual(st, 200)
        self.assertEqual(len(json.loads(body)["sentences"]), 3)

    def test_user_errors_are_400_with_a_readable_message(self):
        st, _, body = self.req("POST", "/api/analyze", {"text": "  "})
        self.assertEqual(st, 400)
        self.assertIn("Type a sentence", json.loads(body)["error"])
        st, _, _ = self.req("POST", "/api/analyze", raw="{not json")
        self.assertEqual(st, 400)

    def test_translate_with_llm_off_over_http(self):
        st, _, body = self.req("POST", "/api/translate", {"text": TEXT, "call_llm": False})
        self.assertEqual(st, 200)
        t = json.loads(body)
        self.assertFalse(t["llm"]["enabled"])
        self.assertEqual(len(t["spacy"]["rows"]), 3)

    def test_foreign_host_and_foreign_origin_are_refused(self):
        self.assertEqual(self.req("GET", "/api/status", headers={"Host": "evil.example.com"})[0], 403)
        st, _, _ = self.req("POST", "/api/translate", {"text": TEXT}, headers={"Origin": "https://evil.example.com"})
        self.assertEqual(st, 403)
        st, _, _ = self.req("POST", "/api/analyze", {"text": TEXT}, headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(st, 200)  # the page's own origin is fine

    def test_post_must_be_json_and_small(self):
        self.assertEqual(self.req("POST", "/api/analyze", raw="text=hello", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.req("POST", "/api/analyze", raw=json.dumps({"text": "x" * 70000}))[0], 413)
        self.assertEqual(self.req("GET", "/nope")[0], 404)
        self.assertEqual(self.req("POST", "/api/other", {"text": "x"})[0], 404)

    def test_only_one_llm_run_at_a_time(self):
        gate, started = threading.Event(), threading.Event()

        def slow(*a, **k):
            started.set()
            gate.wait(10)
            return {"ok": True}

        results = []
        with mock.patch.object(demo, "translate", slow):
            t = threading.Thread(target=lambda: results.append(self.req("POST", "/api/translate", {"text": TEXT})[0]))
            t.start()
            self.assertTrue(started.wait(10))
            self.assertEqual(self.req("POST", "/api/translate", {"text": TEXT})[0], 429)
            gate.set()
            t.join(10)
        self.assertEqual(results, [200])

    def test_server_only_binds_to_loopback(self):
        with self.assertRaises(ValueError):
            make_server(0, host="0.0.0.0")

    def test_response_never_contains_the_api_key(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_FAKEFAKEFAKE12345"}):
            _, _, body = self.req("POST", "/api/translate", {"text": TEXT, "call_llm": False})
        self.assertNotIn(b"gsk_FAKEFAKEFAKE12345", body)


class CliTests(unittest.TestCase):
    def test_serve_command_is_registered(self):
        args = build_parser().parse_args(["serve", "--port", "9001"])
        self.assertEqual((args.cmd, args.port, args.env_file), ("serve", 9001, ".env"))


if __name__ == "__main__":
    unittest.main()
