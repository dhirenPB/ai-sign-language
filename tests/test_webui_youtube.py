"""The tester page's YouTube mode: background runner, status steps, the new routes, and a finished job feeding the rest of the pipeline.
YouTube, FFmpeg and the speech model are doubles; the cleaner, sentence splitter, grouper, translator (test double LLM) and Layer 2 are real."""

import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from isl_gloss.webui import demo
from isl_gloss.webui.server import make_server
from isl_gloss.webui.youtube_runner import YouTubeRunner
from isl_gloss.youtube.errors import IngestError, InvalidUrlError
from isl_gloss.youtube.transcribe import EngineResult
from tests.gloss_helpers import FakeClient
from tests.test_layer2 import HAVE_SPACY
from tests.youtube_helpers import URL, VIDEO_ID, FakeYDL, ScriptedEngine, components, seg

JOB = f"yt-{VIDEO_ID}"


def wait_done(runner, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        s = runner.status(job_id)
        if not s["running"]:
            return s
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.jobs = self.dir / "jobs"

    def runner(self, **kw):
        comps, self.ydl, self.ffrunner, self.engine = components(**kw)
        return YouTubeRunner(self.jobs, 30, components_factory=lambda cfg, log: comps)


class RunnerTests(Base):
    def test_steps_report_exactly_what_happened_in_order(self):
        r = self.runner()
        started = r.start(URL, "base.en")
        self.assertEqual((started["job_id"], started["started"]), (JOB, True))
        s = wait_done(r, JOB)
        self.assertEqual([(x["key"], x["state"], x["label"]) for x in s["steps"]],
                         [("video", "done", "Video detected successfully"), ("audio", "done", "Audio extracted successfully"),
                          ("transcript", "done", "Audio transcribed successfully")])
        self.assertTrue(s["ready"])
        self.assertIsNone(s["error"])
        v, a, t = (x["detail"] for x in s["steps"])
        self.assertEqual((v["title"], v["duration_seconds"]), ("Test talk", 12.0))
        self.assertEqual((a["sample_rate"], a["channels"]), (16000, 1))
        self.assertEqual((t["segments"], t["timestamp_level"], t["model"]), (3, "mixed", "base.en"))
        self.assertEqual(s["transcript"]["segments"][0]["text"], "We will not go outside unless it stops raining.")
        self.assertEqual((s["transcript"]["segments"][0]["start"], s["transcript"]["segments"][0]["end"]), (0.4, 3.2))
        self.assertFalse(any(x["reused"] for x in s["steps"]))

    def test_status_while_the_transcript_step_is_still_running(self):
        gate, entered = threading.Event(), threading.Event()

        class Slow(ScriptedEngine):
            def transcribe_window(self, wav_path, language):
                entered.set()
                gate.wait(20)
                return super().transcribe_window(wav_path, language)
        r = self.runner(engine=Slow([EngineResult([seg(1, 2, "Hello there friend.")])]))
        r.start(URL)
        self.assertTrue(entered.wait(20))
        s = r.status(JOB)
        self.assertTrue(s["running"])
        self.assertEqual([(x["key"], x["state"]) for x in s["steps"]], [("video", "done"), ("audio", "done"), ("transcript", "running")])
        self.assertFalse(s["ready"])  # the rest of the pipeline must not start before the transcript exists
        gate.set()
        self.assertTrue(wait_done(r, JOB)["ready"])

    def test_a_failure_marks_the_step_and_nothing_after_it_is_ready(self):
        for ydl_kwargs, failed, stage in ((dict(extract_error=Exception("ERROR: Private video")), "video", "metadata"),
                                          (dict(download_error=Exception("getaddrinfo failed")), "audio", "download")):
            r = self.runner(ydl=FakeYDL(**ydl_kwargs))
            self.jobs.mkdir(exist_ok=True)
            import shutil
            shutil.rmtree(self.jobs / JOB, ignore_errors=True)
            r.start(URL)
            s = wait_done(r, JOB)
            states = {x["key"]: x["state"] for x in s["steps"]}
            self.assertEqual(states[failed], "failed")
            self.assertFalse(s["ready"])
            self.assertEqual(s["error"]["stage"], stage)
            self.assertNotIn("transcript", s)
            later = [k for k in ("video", "audio", "transcript") if k > failed] if False else [k for k in states if list(states).index(k) > list(states).index(failed)]
            self.assertTrue(all(states[k] == "pending" for k in later), states)

    def test_missing_speech_engine_fails_the_transcript_step(self):
        from isl_gloss.youtube.errors import DependencyMissingError
        comps, *_ = components()
        comps.engine_factory = lambda cfg: (_ for _ in ()).throw(DependencyMissingError("faster-whisper is not installed", hint="pip install faster-whisper"))
        r = YouTubeRunner(self.jobs, 30, components_factory=lambda cfg, log: comps)
        r.start(URL)
        s = wait_done(r, JOB)
        self.assertEqual({x["key"]: x["state"] for x in s["steps"]}, {"video": "done", "audio": "done", "transcript": "failed"})
        self.assertEqual((s["error"]["stage"], s["error"]["kind"]), ("transcription", "dependency_missing"))
        self.assertIn("pip install faster-whisper", s["error"]["hint"])

    def test_second_run_shows_the_steps_as_reused(self):
        r = self.runner()
        r.start(URL)
        wait_done(r, JOB)
        time.sleep(1.1)  # stage timestamps have one-second resolution
        r.start(URL)
        s = wait_done(r, JOB)
        self.assertTrue(all(x["reused"] for x in s["steps"]), s["steps"])
        self.assertEqual([c for c in self.ydl.calls if c[0] == "download"].__len__(), 1)  # nothing was downloaded again

    def test_invalid_link_is_rejected_before_anything_starts(self):
        r = self.runner()
        for bad in ("", "not a url", "https://vimeo.com/1", "https://www.youtube.com/playlist?list=PL1"):
            with self.assertRaises(InvalidUrlError):
                r.start(bad)
        self.assertFalse(self.jobs.exists())

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(IngestError):
            self.runner().start(URL, "gigantic-ultra")

    def test_polling_as_fast_as_possible_never_breaks_the_running_job(self):
        """Regression: on Windows a poll that has job.json open used to make the job's own write fail with 'Access is denied'."""
        r = self.runner()
        stop, polls, errors = threading.Event(), [0], []

        def hammer():
            while not stop.is_set():
                try:
                    r.status(JOB)
                    polls[0] += 1
                except Exception as exc:  # the status call itself must never raise either
                    errors.append(exc)
        threads = [threading.Thread(target=hammer) for _ in range(3)]
        for t in threads:
            t.start()
        r.start(URL)
        s = wait_done(r, JOB)
        stop.set()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertGreater(polls[0], 20)
        self.assertTrue(s["ready"], s["error"])
        self.assertIsNone(s["error"])

    def test_job_file_write_survives_a_reader_holding_the_file_open(self):
        from isl_gloss.youtube.jobs import atomic_write_text
        target = self.dir / "job.json"
        target.write_text("old", encoding="utf-8")
        release = threading.Event()
        opened = threading.Event()

        def reader():
            with open(target, "rb"):
                opened.set()
                release.wait(5)
        t = threading.Thread(target=reader)
        t.start()
        opened.wait(5)
        threading.Timer(0.3, release.set).start()  # the reader lets go shortly after the write has started
        atomic_write_text(target, "new")
        t.join()
        self.assertEqual(target.read_text(encoding="utf-8"), "new")
        self.assertEqual([p.name for p in self.dir.glob("*.tmp")], [])

    def test_one_video_at_a_time(self):
        gate, entered = threading.Event(), threading.Event()

        class Slow(ScriptedEngine):
            def transcribe_window(self, wav_path, language):
                entered.set()
                gate.wait(20)
                return super().transcribe_window(wav_path, language)
        r = self.runner(engine=Slow([EngineResult([seg(1, 2, "Hello there friend.")])]))
        r.start(URL)
        entered.wait(20)
        self.assertFalse(r.start(URL)["started"])  # same video: not started twice
        with self.assertRaises(IngestError) as cm:
            r.start("https://youtu.be/AAAAAAAAAAA")
        self.assertIn("Another video", cm.exception.message)
        gate.set()
        wait_done(r, JOB)

    def test_never_exceeds_the_configured_video_length(self):
        r = YouTubeRunner(self.jobs, 0.1, components_factory=lambda cfg, log: components(cfg=cfg)[0])  # 6 seconds; the fake video is 12 s
        r.start(URL)
        s = wait_done(r, JOB)
        self.assertEqual((s["error"]["kind"], s["steps"][0]["state"]), ("video_too_long", "failed"))


class JobAsInputTests(Base):
    def finished_job(self, segments=None):
        eng = ScriptedEngine([EngineResult(segments, "en", 0.99)]) if segments else None
        r = self.runner(engine=eng)
        r.start(URL)
        wait_done(r, JOB)
        return r

    def test_the_rest_of_the_pipeline_continues_from_the_saved_transcript_with_video_times(self):
        self.finished_job()
        a = demo.analyze(None, {"job_id": JOB}, self.jobs)
        self.assertEqual(a["source"]["type"], "youtube")
        self.assertEqual([s["text"] for s in a["sentences"]], ["We will not go outside unless it stops raining.", "Why do we get angry so quickly?",
                                                                "If you practise every day, your mind will become calm."])
        self.assertEqual([(s["index"], s["time"]["start"], s["time"]["end"], s["time"]["basis"]) for s in a["sentences"]],
                         [(1, 0.4, 3.2, "word"), (2, 5.0, 8.1, "segment"), (3, 9.0, 11.5, "segment")])
        self.assertEqual((a["source"]["title"], a["source"]["video_id"], a["source"]["timestamp_level"]), ("Test talk", VIDEO_ID, "mixed"))
        self.assertNotIn("segments", a["source"])
        self.assertEqual(a["estimated_llm_requests"], 1)
        t = demo.translate(None, {"job_id": JOB, "call_llm": True, "use_cache": False}, env_file=self.dir / "none.env", data_dir=self.dir / "demo",
                           client=FakeClient(), jobs_dir=self.jobs)
        self.assertEqual([r["status"] for r in t["llm"]["rows"]], ["translated"] * 3)
        self.assertTrue(t["llm"]["requests"][0]["user_payload"].startswith("[yt-dqw4w9wgxcq-s0001]"))
        self.assertEqual(len(t["comparison"]["items"]), 3)
        self.assertEqual(t["spacy"]["rows"][0]["spacy_status"], "ok" if HAVE_SPACY else "unavailable")

    def test_only_the_first_n_sentences_go_on(self):
        segs = [seg(i * 2.4, i * 2.4 + 2.0, f"Sentence number {n} is here.") for i, n in enumerate(("one", "two", "three", "four", "five"))]  # all inside the 12 s of fake audio
        self.finished_job(segs)
        a = demo.analyze(None, {"job_id": JOB, "max_sentences": 2}, self.jobs)
        self.assertEqual((len(a["sentences"]), a["source"]["total_sentences"], a["source"]["limit"]), (2, 5, 2))
        self.assertEqual(len(a["cleaning"]["chunks"]), 2)  # the cleaned view follows the sentences that go on
        t = demo.translate(None, {"job_id": JOB, "call_llm": False, "max_sentences": 2}, env_file=self.dir / "none.env", data_dir=self.dir / "demo", jobs_dir=self.jobs)
        self.assertEqual(len(t["llm"]["rows"]), 2)
        self.assertEqual(demo.analyze(None, {"job_id": JOB, "max_sentences": 999}, self.jobs)["source"]["limit"], demo.MAX_SENTENCES)
        self.assertEqual(len(demo.analyze(None, {"job_id": JOB}, self.jobs)["sentences"]), 5)  # default 10 covers all five

    def test_a_job_without_a_valid_transcript_cannot_continue(self):
        r = self.runner(ydl=FakeYDL(extract_error=Exception("Video unavailable")))
        r.start(URL)
        wait_done(r, JOB)
        for fn in (lambda: demo.analyze(None, {"job_id": JOB}, self.jobs), lambda: demo.translate(None, {"job_id": JOB}, jobs_dir=self.jobs)):
            with self.assertRaises(demo.DemoError) as cm:
                fn()
            self.assertIn("no valid transcript", str(cm.exception))

    def test_job_ids_cannot_point_outside_the_jobs_folder(self):
        for bad in ("../secret", "yt-..", "yt-" + "a" * 12, "yt-short", "/etc/passwd", "yt-dQw4w9WgXc/../x", None, 5):
            with self.assertRaises(demo.DemoError):
                demo.analyze(None, {"job_id": bad}, self.jobs)

    def test_a_corrupted_saved_transcript_is_refused(self):
        self.finished_job()
        p = self.jobs / JOB / "transcript" / "transcript.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc["transcription"]["segments"][0]["end"] = 9999
        p.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(demo.DemoError):
            demo.analyze(None, {"job_id": JOB}, self.jobs)

    def test_typed_text_still_works_unchanged(self):
        a = demo.analyze("We go home. They stay.", {}, self.jobs)
        self.assertEqual((a["source"]["type"], len(a["sentences"]), a["sentences"][0]["time"]), ("text", 2, None))


class PageDefaultsTests(Base):
    """The page has no option controls: it always runs prompt 8.0, medium reasoning and the small.en speech model."""

    def test_fixed_defaults(self):
        cfg = demo._config({})
        self.assertEqual((cfg.prompt_version, cfg.reasoning_effort, cfg.provider, cfg.model, cfg.temperature),
                         ("8.0", "medium", "groq", "openai/gpt-oss-120b", 0.0))
        self.assertEqual(self.runner().config(None).model, "small.en")
        self.assertEqual(demo.status(self.dir / "none.env")["default_prompt_version"], "8.0")

    def test_a_request_without_options_really_uses_prompt_8_and_reports_it(self):
        client = FakeClient()
        t = demo.translate("We go home. They stay.", {}, env_file=self.dir / "none.env", data_dir=self.dir / "demo", client=client, jobs_dir=self.jobs)
        self.assertEqual((t["llm"]["prompt_version"], t["llm"]["reasoning_effort"]), ("8.0", "medium"))
        self.assertIn("(version 8.0)", client.calls[0]["system"])  # the prompt that was actually sent
        self.assertTrue(t["llm"]["enabled"])
        self.assertEqual(t["llm"]["model"], "groq/openai/gpt-oss-120b")

    def test_the_project_wide_defaults_are_untouched(self):
        from isl_gloss.config import TranslatorConfig
        from isl_gloss.prompt import DEFAULT_PROMPT_VERSION
        from isl_gloss.youtube.config import IngestConfig
        self.assertEqual((TranslatorConfig().prompt_version, DEFAULT_PROMPT_VERSION, IngestConfig().model), ("7.0", "7.0", "small.en"))

    def test_the_api_can_still_override_for_other_callers(self):
        self.assertEqual(demo._config({"prompt_version": "7.0", "reasoning_effort": "low"}).prompt_version, "7.0")
        self.assertEqual(self.runner().config("base.en").model, "base.en")

    def test_the_page_only_offers_the_gloss_limit(self):
        from pathlib import Path as P
        html = (P(demo.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
        for gone in ('id="ytmodel"', 'id="ytforce"', 'id="prompt"', 'id="effort"', 'id="callllm"', 'id="usecache"', "Speech model", "Reasoning", "Reuse saved answers",
                     "Redo from scratch", "Call the LLM", '$("prompt")', '$("effort")', '$("callllm")', '$("usecache")', '$("ytmodel")', '$("ytforce")'):
            self.assertNotIn(gone, html, gone)
        self.assertIn('id="ytsent"', html)
        self.assertIn("Gloss the first", html)
        self.assertEqual(html.count("<select"), 0)  # no drop-downs left at all


class ServerRoutesTests(Base):
    def setUp(self):
        super().setUp()
        comps, self.ydl, self.ffrunner, self.engine = components()
        self.runner_ = YouTubeRunner(self.jobs, 30, components_factory=lambda cfg, log: comps)
        self.srv = make_server(0, env_file="__no_such_env__", data_dir=str(self.dir / "demo"), jobs_dir=str(self.jobs), runner=self.runner_)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (self.srv.shutdown(), self.srv.server_close()))

    def req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request(method, path, body=None if body is None else json.dumps(body), headers={"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, json.loads(data.decode()) if data else {}

    def test_start_poll_then_analyze_over_http(self):
        st, body = self.req("POST", "/api/youtube/start", {"url": URL, "model": "base.en"})
        self.assertEqual((st, body["job_id"]), (200, JOB))
        end = time.time() + 30
        while True:
            st, s = self.req("GET", f"/api/youtube/status?job_id={JOB}")
            self.assertEqual(st, 200)
            if not s["running"]:
                break
            self.assertLess(time.time(), end)
            time.sleep(0.05)
        self.assertTrue(s["ready"])
        self.assertEqual([x["label"] for x in s["steps"]], ["Video detected successfully", "Audio extracted successfully", "Audio transcribed successfully"])
        st, a = self.req("POST", "/api/analyze", {"job_id": JOB, "max_sentences": 2})
        self.assertEqual((st, len(a["sentences"]), a["sentences"][0]["time"]["start"]), (200, 2, 0.4))
        st, t = self.req("POST", "/api/translate", {"job_id": JOB, "call_llm": False, "max_sentences": 2})
        self.assertEqual((st, len(t["spacy"]["rows"])), (200, 2))

    def test_bad_link_is_a_400_with_a_hint(self):
        st, body = self.req("POST", "/api/youtube/start", {"url": "https://vimeo.com/123"})
        self.assertEqual(st, 400)
        self.assertIn("not a YouTube address", body["error"])
        self.assertTrue(body["hint"])
        self.assertEqual(self.req("POST", "/api/youtube/start", {"url": ""})[0], 400)

    def test_status_rejects_a_bad_job_id(self):
        for bad in ("..%2F..", "yt-short", "", "yt-dQw4w9WgXc%2F..%2Fx"):
            self.assertEqual(self.req("GET", f"/api/youtube/status?job_id={bad}")[0], 400, bad)

    def test_foreign_origin_cannot_start_a_download(self):
        st, _ = self.req("POST", "/api/youtube/start", {"url": URL}, headers={"Origin": "https://evil.example.com"})
        self.assertEqual(st, 403)
        self.assertFalse(self.jobs.exists())  # nothing was started

    def test_status_page_reports_youtube_readiness(self):
        st, info = self.req("GET", "/api/status")
        self.assertEqual(st, 200)
        self.assertIn("available", info["youtube"])
        self.assertEqual(info["default_video_sentences"], demo.DEFAULT_VIDEO_SENTENCES)

    def test_page_contains_the_youtube_tab_and_the_three_messages_come_from_the_server(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("GET", "/")
        html = c.getresponse().read().decode()
        c.close()
        for needle in ('id="tab-yt"', 'id="yturl"', "/api/youtube/start", "/api/youtube/status"):
            self.assertIn(needle, html)


if __name__ == "__main__":
    unittest.main()
