"""YouTube phase end to end: jobs, resume/cache, failure handling, the existing cleaner and gloss pipeline, timestamps, and the CLI.

YouTube, FFmpeg and the speech model are doubles. The cleaner and the `translate` command are the REAL existing implementation;
the LLM is a local mock server. No real video is downloaded."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from isl_gloss.cli import main
from isl_gloss.youtube import IngestConfig, run_youtube_job
from isl_gloss.youtube.align import align_sentences
from isl_gloss.youtube.errors import InvalidUrlError, NetworkError, NoSpeechError
from isl_gloss.youtube.jobs import COMPLETED, FAILED, STATES, Job, read_json
from isl_gloss.youtube.pipeline import build_timeline, render_job
from isl_gloss.youtube.transcribe import EngineResult
from tests.gloss_helpers import MockGemini
from tests.youtube_helpers import URL, VIDEO_ID, FakeRunner, FakeYDL, ScriptedEngine, components, seg

KEY = "test-key-123456"


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.cfg = IngestConfig(data_dir=str(self.dir / "jobs"))
        self.log = []

    def job_dir(self):
        return self.dir / "jobs" / f"yt-{VIDEO_ID}"

    def gloss_args(self, srv):
        cfg = self.dir / "llm.json"
        cfg.write_text(json.dumps({"provider": "gemini", "base_url": srv.base_url, "requests_per_minute": 6000}), encoding="utf-8")
        return ["--config", str(cfg), "--env-file", str(self.dir / "none.env")]

    def run_job(self, comps, srv=None, **kw):
        extra = self.gloss_args(srv) if srv else []
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": KEY}):
            return run_youtube_job(URL, self.cfg, comps, gloss_args=extra, log=self.log.append, **kw)


class FullPipelineTests(Base):
    def test_url_to_gloss_end_to_end_with_timestamps(self):
        comps, ydl, runner, engine = components()
        with MockGemini() as srv:
            out = self.run_job(comps, srv, gloss_limit=None)
            llm_requests = len(srv.requests)
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.job.state, COMPLETED)
        self.assertEqual(llm_requests, 1)  # the LLM is reached once, by the existing translate command, after a good transcript
        d = self.job_dir()
        for rel in ("metadata/video.json", "audio/audio.wav", "audio/audio.json", "transcript/transcript.json", "transcript/transcript.txt",
                    "transcript/transcript.vtt", "cleaning/structured/yt-dqw4w9wgxcq.cleaned.json", "gloss/yt-dqw4w9wgxcq.gloss.json", "timeline/timeline.json", "job.log"):
            self.assertTrue((d / rel).is_file(), rel)
        self.assertFalse((d / "audio" / "_download").exists())  # temporary download removed, required outputs kept
        # canonical transcript
        t = read_json(d / "transcript" / "transcript.json")
        self.assertEqual(t["transcription"]["timestamp_level"], "mixed")  # first segment has word times, the others do not: said so, not hidden
        self.assertEqual((t["transcription"]["segments"][0]["start"], t["transcription"]["segments"][0]["end"]), (0.4, 3.2))
        self.assertEqual(t["video"]["title"], "Test talk")
        self.assertEqual(t["audio"]["timeline_preserved"], True)
        # the existing cleaner received the timestamps and the segment ids
        cleaned = read_json(d / "cleaning" / "structured" / "yt-dqw4w9wgxcq.cleaned.json")
        self.assertTrue(cleaned["timestamps_available"])
        self.assertEqual([(c["start_time"], c["end_time"], c["cue_ids"]) for c in cleaned["chunks"]],
                         [("00:00:00.400", "00:00:03.200", ["seg-0001"]), ("00:00:05.000", "00:00:08.100", ["seg-0002"]),
                          ("00:00:09.000", "00:00:11.500", ["seg-0003"])])  # one chunk per cue here: each keeps its own real interval
        # the existing gloss pipeline ran on it and kept the timestamps
        gloss = read_json(d / "gloss" / "yt-dqw4w9wgxcq.gloss.json")
        self.assertEqual([r["source_text"] for r in gloss["results"]],
                         ["We will not go outside unless it stops raining.", "Why do we get angry so quickly?", "If you practise every day, your mind will become calm."])
        self.assertEqual([(r["start_timestamp"], r["end_timestamp"]) for r in gloss["results"]],
                         [("00:00:00.400", "00:00:03.200"), ("00:00:05.000", "00:00:08.100"), ("00:00:09.000", "00:00:11.500")])  # the existing stage kept them
        self.assertEqual(gloss["review_status"], "unreviewed")
        # the timeline maps each sentence back to original-video time, per sentence, read from real words/segments
        tl = read_json(d / "timeline" / "timeline.json")
        rows = tl["sentences"]
        self.assertEqual([(r["start"], r["end"], r["timing_basis"], r["alignment"]) for r in rows],
                         [(0.4, 3.2, "word", "exact"), (5.0, 8.1, "segment", "exact"), (9.0, 11.5, "segment", "exact")])
        self.assertEqual([r["segment_ids"] for r in rows], [["seg-0001"], ["seg-0002"], ["seg-0003"]])
        self.assertTrue(all(r["final_gloss"] for r in rows))
        self.assertEqual((tl["groups"][0]["start"], tl["groups"][0]["end"]), (0.4, 11.5))
        # job record
        job = read_json(d / "job.json")
        self.assertEqual([h["state"] for h in job["history"]], ["DOWNLOADING", "AUDIO_EXTRACTED", "TRANSCRIBING", "TRANSCRIBED", "PROCESSING_EXISTING_PIPELINE", "COMPLETED"])
        self.assertTrue(set(h["state"] for h in job["history"]) <= set(STATES))
        self.assertIsNone(job["error"])
        # secrets never reach the log
        self.assertNotIn(KEY, (d / "job.log").read_text(encoding="utf-8"))

    def test_gloss_limit_default_protects_the_quota(self):
        comps, *_ = components()
        with MockGemini() as srv:
            out = self.run_job(comps, srv, gloss_limit=2)
        gloss = read_json(self.job_dir() / "gloss" / "yt-dqw4w9wgxcq.gloss.json")
        self.assertEqual((out.ok, len(gloss["results"])), (True, 2))

    def test_through_transcript_and_clean_never_touch_the_llm(self):
        for through, gloss_dir in (("transcript", False), ("clean", False)):
            comps, *_ = components()
            with MockGemini() as srv:
                out = self.run_job(comps, srv, through=through)
                self.assertEqual(len(srv.requests), 0)
            self.assertEqual((out.ok, (self.job_dir() / "gloss").exists()), (True, gloss_dir))
            self.assertEqual((self.job_dir() / "cleaning").exists(), through == "clean")
            for sub in ("gloss", "cleaning"):
                import shutil
                shutil.rmtree(self.job_dir() / sub, ignore_errors=True)

    def test_no_llm_call_during_download_or_transcription(self):
        comps, *_ = components()
        with MockGemini() as srv:
            self.run_job(comps, srv, through="transcript")
            self.assertEqual(srv.requests, [])


class FailureTests(Base):
    def test_failed_transcription_never_reaches_cleaning_or_gloss(self):
        for engine in (ScriptedEngine([EngineResult([])]), ScriptedEngine(error=NoSpeechError("no speech"))):
            comps, *_ = components(engine=engine)
            with MockGemini() as srv:
                out = self.run_job(comps, srv, force=["all"])
                self.assertEqual(len(srv.requests), 0)  # gloss generation is never invoked with an empty transcript
            self.assertFalse(out.ok)
            self.assertEqual((out.job.state, out.error["stage"], out.error["kind"]), (FAILED, "transcription", "no_speech_detected"))
            self.assertFalse((self.job_dir() / "cleaning").exists() or (self.job_dir() / "gloss").exists())
            self.assertFalse((self.job_dir() / "transcript" / "transcript.json").exists())

    def test_engine_unavailable_is_a_failed_transcription_stage_not_a_silent_fallback(self):
        from isl_gloss.youtube.errors import DependencyMissingError
        calls = []

        def factory(cfg):
            calls.append(cfg.engine)
            raise DependencyMissingError("The faster-whisper package is not installed.", hint="pip install faster-whisper")
        comps, *_ = components()
        comps.engine_factory = factory
        out = self.run_job(comps)
        self.assertEqual((out.ok, out.error["kind"], out.error["stage"], calls), (False, "dependency_missing", "transcription", ["faster-whisper"]))
        self.assertIn("pip install faster-whisper", out.error["hint"])
        self.assertEqual(out.job.state, FAILED)

    def test_download_failure_is_recorded_and_a_rerun_resumes(self):
        ydl = FakeYDL(download_error=Exception("getaddrinfo failed"))
        comps, *_ = components(ydl=ydl)
        out = self.run_job(comps, through="transcript")
        self.assertEqual((out.ok, out.error["kind"], out.error["stage"]), (False, "network_error", "download"))
        saved = read_json(self.job_dir() / "job.json")
        self.assertEqual((saved["state"], saved["error"]["stage"]), (FAILED, "download"))
        self.assertTrue((self.job_dir() / "metadata" / "video.json").is_file())  # the finished stage was kept
        ydl.download_error = None  # the network comes back
        out2 = self.run_job(comps, through="transcript")
        self.assertTrue(out2.ok, out2.error)
        self.assertEqual([c for c in ydl.calls if c[0] == "extract_info"].__len__(), 1)  # metadata was NOT fetched again
        self.assertIsNone(read_json(self.job_dir() / "job.json")["error"])

    def test_audio_conversion_failure_and_missing_audio_stream(self):
        for runner, kind in ((FakeRunner(fail_convert="Unknown encoder"), "audio_conversion_failed"), (FakeRunner(has_audio=False), "no_audio_stream")):
            comps, *_ = components(runner=runner)
            out = self.run_job(comps, force=["all"])
            self.assertEqual((out.ok, out.error["kind"], out.error["stage"]), (False, kind, "audio"))
            self.assertFalse((self.job_dir() / "audio" / "audio.wav").exists())

    def test_ffmpeg_missing_fails_before_downloading(self):
        comps, ydl, *_ = components(runner=FakeRunner(missing=True))
        out = self.run_job(comps)
        self.assertEqual((out.error["kind"], [c for c in ydl.calls if c[0] == "download"]), ("dependency_missing", []))

    def test_unavailable_video(self):
        comps, *_ = components(ydl=FakeYDL(extract_error=Exception("ERROR: Private video")))
        out = self.run_job(comps)
        self.assertEqual((out.error["kind"], out.error["stage"], out.error["details"]["reason"]), ("video_unavailable", "metadata", "private"))

    def test_invalid_url_creates_no_job(self):
        comps, *_ = components()
        with self.assertRaises(InvalidUrlError):
            run_youtube_job("https://example.com/video", self.cfg, comps, log=self.log.append)
        self.assertFalse((self.dir / "jobs").exists())

    def test_gloss_failure_keeps_the_transcript_and_says_why(self):
        comps, *_ = components()
        env = {k: v for k, v in os.environ.items() if k not in ("GROQ_API_KEY", "GEMINI_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            out = run_youtube_job(URL, self.cfg, comps, gloss_args=["--env-file", str(self.dir / "none.env")], log=self.log.append)
        self.assertEqual((out.ok, out.error["stage"], out.job.state), (False, "gloss", FAILED))
        self.assertIn("GROQ_API_KEY", (self.job_dir() / "job.log").read_text(encoding="utf-8"))
        self.assertTrue((self.job_dir() / "transcript" / "transcript.json").is_file())  # nothing finished is thrown away


class ResumeAndCacheTests(Base):
    def test_second_run_reuses_every_finished_stage(self):
        comps, ydl, runner, engine = components()
        self.run_job(comps, through="transcript")
        before = (len(ydl.calls), len(runner.calls), engine.calls)
        out = self.run_job(comps, through="transcript")
        self.assertTrue(out.ok)
        self.assertEqual((len(ydl.calls), len(runner.calls), engine.calls), before)  # no download, no conversion, no transcription
        log = "\n".join(self.log)
        self.assertIn("reusing saved video information", log)
        self.assertIn("reusing the extracted audio", log)
        self.assertIn("reusing the saved transcript", log)

    def test_changed_transcription_settings_redo_only_the_transcript(self):
        comps, ydl, runner, engine = components()
        self.run_job(comps, through="transcript")
        engine2 = ScriptedEngine([EngineResult([seg(1.0, 2.0, "Different settings, different transcript.")])])
        comps2, *_ = components(ydl=ydl, runner=runner, engine=engine2)
        self.cfg.model = "base.en"
        n_dl = len(ydl.calls)
        out = self.run_job(comps2, through="transcript")
        self.assertTrue(out.ok)
        self.assertEqual((len(ydl.calls), engine2.calls), (n_dl, 1))
        t = read_json(self.job_dir() / "transcript" / "transcript.json")
        self.assertEqual((t["transcription"]["settings"]["model"], t["transcription"]["segments"][0]["text"]), ("base.en", "Different settings, different transcript."))

    def test_force_redoes_a_stage(self):
        comps, ydl, runner, engine = components()
        self.run_job(comps, through="transcript")
        engine.results.append(EngineResult([seg(1.0, 2.0, "Forced again.")]))
        out = self.run_job(comps, through="transcript", force=["transcript"])
        self.assertEqual((out.ok, engine.calls), (True, 2))
        self.assertEqual(len([c for c in ydl.calls if c[0] == "download"]), 1)

    def test_corrupt_saved_transcript_is_not_trusted(self):
        comps, ydl, runner, engine = components()
        self.run_job(comps, through="transcript")
        p = self.job_dir() / "transcript" / "transcript.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc["transcription"]["segments"][0]["end"] = 9999
        p.write_text(json.dumps(doc), encoding="utf-8")
        engine.results.append(EngineResult([seg(1.0, 2.0, "Recomputed.")]))
        self.assertTrue(self.run_job(comps, through="transcript").ok)
        self.assertEqual(engine.calls, 2)

    def test_gloss_rerun_is_served_from_the_llm_cache(self):
        comps, *_ = components()
        with MockGemini() as srv:
            self.run_job(comps, srv, gloss_limit=None)
            self.run_job(comps, srv, gloss_limit=None)
            self.assertEqual(len(srv.requests), 1)  # the second run spent no LLM request


class AlignmentTests(unittest.TestCase):
    def segments(self):
        return [{"id": "seg-0001", "start": 0.0, "end": 4.0, "text": "Um we will go outside. Then it rained", "words": [
                    {"text": " Um", "start": 0.0, "end": 0.3}, {"text": " we", "start": 0.4, "end": 0.6}, {"text": " will", "start": 0.6, "end": 0.9},
                    {"text": " go", "start": 0.9, "end": 1.1}, {"text": " outside.", "start": 1.1, "end": 1.8}, {"text": " Then", "start": 2.0, "end": 2.3},
                    {"text": " it", "start": 2.3, "end": 2.5}, {"text": " rained", "start": 2.5, "end": 3.5}]},
                {"id": "seg-0002", "start": 5.0, "end": 8.0, "text": "Why do we get angry?", "words": None}]

    def test_sentences_split_differently_from_segments_still_get_real_times(self):
        al = align_sentences(self.segments(), ["We will go outside.", "Then it rained", "Why do we get angry?"])
        self.assertEqual([a.status for a in al], ["exact", "exact", "exact"])  # the filler "Um" before the sentence is simply not part of it
        self.assertEqual((al[0].start, al[0].end, al[0].timing_basis), (0.4, 1.8, "word"))  # words' own times, "Um" (0.0-0.3) excluded
        self.assertEqual((al[1].start, al[1].end), (2.0, 3.5))
        self.assertEqual((al[2].start, al[2].end, al[2].timing_basis, al[2].segment_ids), (5.0, 8.0, "segment", ["seg-0002"]))  # segment times; not spread over words

    def test_words_the_engine_splits_differently_from_the_cleaned_text_still_align(self):
        """Regression (found on a real 18-minute video): Whisper times 'co-founder', '1.3', 'U.S.' and '150,000' as separate pieces
        while the cleaned sentence keeps each as one word; the match used to break there and the sentence lost its own times."""
        def w(text, s, e):
            return {"text": text, "start": s, "end": e}
        segs = [{"id": "seg-0001", "start": 0.0, "end": 9.0, "text": "x", "words": [
            w(" Nandan", 0.0, 0.5), w(" was", 0.5, 0.7), w(" co-", 0.7, 1.0), w("founder", 1.0, 1.5), w(" of", 1.5, 1.6), w(" Infosys.", 1.6, 2.2),
            w(" It", 3.0, 3.2), w(" covers", 3.2, 3.6), w(" 1", 3.6, 3.8), w(".", 3.8, 3.85), w("3", 3.85, 4.0), w(" billion.", 4.0, 4.5),
            w(" In", 5.0, 5.1), w(" the", 5.1, 5.2), w(" U.", 5.2, 5.5), w("S.", 5.5, 5.8), w(" We", 6.0, 6.2), w(" have", 6.2, 6.4), w(" 150", 6.4, 6.9),
            w(",", 6.9, 6.95), w("000", 6.95, 7.4), w(" sites.", 7.4, 8.0)]}]
        sentences = ["Nandan was co-founder of Infosys.", "It covers 1.3 billion.", "In the U.S.", "We have 150,000 sites."]
        al = align_sentences(segs, sentences)
        self.assertEqual([a.status for a in al], ["exact"] * 4)
        self.assertEqual([(a.start, a.end) for a in al], [(0.0, 2.2), (3.0, 4.5), (5.0, 5.8), (6.0, 8.0)])  # each from its own words
        # the same with a non-breaking hyphen in the cleaned text, and with the engine keeping the hyphenated word whole
        segs2 = [{"id": "seg-0001", "start": 0.0, "end": 3.0, "text": "x", "words": [w(" Our", 0.0, 0.3), w(" co-founder", 0.3, 1.0), w(" left.", 1.0, 1.5)]}]
        self.assertEqual(align_sentences(segs2, ["Our co‑founder left."])[0].status, "exact")
        self.assertEqual(align_sentences(segs2, ["Our co founder left."])[0].status, "exact")

    def test_a_gap_inside_a_sentence_is_approximate(self):
        al = align_sentences(self.segments(), ["We go outside."])  # the cleaner dropped "will" from the middle
        self.assertEqual((al[0].status, al[0].skipped_transcript_words, al[0].start, al[0].end), ("approximate", 1, 0.4, 1.8))

    def test_unmatched_sentence_is_unaligned_and_does_not_disturb_the_next(self):
        al = align_sentences(self.segments(), ["Something that was never said.", "Then it rained"])
        self.assertEqual([a.status for a in al], ["unaligned", "exact"])
        self.assertIsNone(al[0].start)

    def test_no_timing_is_invented_for_reordered_text(self):
        al = align_sentences(self.segments(), ["outside go will we"])
        self.assertEqual(al[0].status, "unaligned")


class TimelineTests(unittest.TestCase):
    def test_unaligned_sentence_keeps_the_chunk_interval_and_says_so(self):
        transcript = {"job_id": "j", "transcription": {"segments": [{"id": "seg-0001", "start": 1.0, "end": 2.0, "text": "hello there", "words": None}]}}
        cleaned = {"chunks": [{"chunk_id": "c1", "cue_ids": ["seg-0001"]}]}
        gloss = {"results": [{"sentence_id": "s1", "group_id": "g1", "source_text": "Totally different words", "start_timestamp": "00:00:01.000",
                              "end_timestamp": "00:00:02.000", "translation_status": "translated", "candidate_gloss": "X", "final_gloss": "X",
                              "needs_human_review": False, "source_metadata": {"parent_chunk_id": "c1", "cue_ids": ["seg-0001"]}}]}
        tl = build_timeline(transcript, cleaned, gloss)
        row = tl["sentences"][0]
        self.assertEqual((row["start"], row["end"], row["timing_basis"], row["alignment"], row["segment_ids"]), (1.0, 2.0, "chunk_interval", "unaligned", ["seg-0001"]))
        self.assertEqual(tl["coverage"]["alignment"]["unaligned"], 1)


class DirectInputStillWorksTests(unittest.TestCase):
    def test_plain_text_transcript_goes_through_the_unchanged_cleaner_and_translator(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "structured").mkdir()
            from transcript_cleaner.config import CleaningConfig
            from transcript_cleaner.pipeline import process_bytes
            res = process_bytes(b"Speaker 1: We will not go outside. Why do we get angry?\n", "talk.txt", CleaningConfig())
            f = d / "structured" / "talk.cleaned.json"
            f.write_text(json.dumps(res.doc), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main(["translate", str(f), "--dry-run"])
            self.assertEqual(code, 0, out.getvalue() + err.getvalue())
            self.assertIn("1 request(s) would be sent", out.getvalue())


class CliTests(Base):
    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.dict(os.environ, {"GEMINI_API_KEY": KEY}):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_bad_url_gives_a_clear_error_and_exit_2(self):
        code, out, _ = self.cli("youtube", "https://example.com/x", "--data-dir", self.dir / "jobs")
        self.assertEqual(code, 2)
        self.assertIn("not a YouTube address", out)
        self.assertFalse((self.dir / "jobs").exists())

    def test_run_inspect_and_failure_exit_codes(self):
        comps, *_ = components()
        with mock.patch("isl_gloss.youtube.pipeline.default_components", return_value=comps):
            code, out, err = self.cli("youtube", URL, "--data-dir", self.dir / "jobs", "--through", "transcript")
        self.assertEqual(code, 0, out + err)
        self.assertIn("COMPLETED", out)
        code, out, _ = self.cli("youtube-show", URL, "--data-dir", self.dir / "jobs", "--what", "transcript")
        self.assertEqual(code, 0)
        self.assertIn("[00:00.400 -> 00:03.200] We will not go outside unless it stops raining.", out)
        self.assertIn("seconds from the start of the video", out)
        code, out, _ = self.cli("youtube-jobs", "--data-dir", self.dir / "jobs")
        self.assertIn(f"yt-{VIDEO_ID}", out)
        # failure -> exit 1 with the stage and the hint
        comps2, *_ = components(ydl=FakeYDL(extract_error=Exception("Video unavailable")))
        self.cfg.data_dir = str(self.dir / "jobs2")
        with mock.patch("isl_gloss.youtube.pipeline.default_components", return_value=comps2):
            code, out, _ = self.cli("youtube", URL, "--data-dir", self.dir / "jobs2")
        self.assertEqual(code, 1)
        self.assertIn("Error (metadata)", out)
        code, out, _ = self.cli("youtube-show", f"yt-{VIDEO_ID}", "--data-dir", self.dir / "jobs2", "--what", "status")
        self.assertIn("FAILED at 'metadata'", out)

    def test_gloss_view_shows_original_video_times(self):
        comps, *_ = components()
        with MockGemini() as srv, mock.patch("isl_gloss.youtube.pipeline.default_components", return_value=comps):
            cfg = self.dir / "llm.json"
            cfg.write_text(json.dumps({"provider": "gemini", "base_url": srv.base_url, "requests_per_minute": 6000}), encoding="utf-8")
            code, out, err = self.cli("youtube", URL, "--data-dir", self.dir / "jobs", "--llm-config", cfg, "--env-file", self.dir / "none.env", "--gloss-limit", 3)
        self.assertEqual(code, 0, out + err)
        shown = render_job(self.dir / "jobs" / f"yt-{VIDEO_ID}", "gloss")
        self.assertIn("[00:05.000 -> 00:08.100]", shown)
        self.assertIn("GLOSS:", shown)

    def test_check_command_reports_each_dependency(self):
        code, out, _ = self.cli("youtube-check")
        self.assertIn('"ffmpeg"', out)
        self.assertIn(code, (0, 1))


if __name__ == "__main__":
    unittest.main()
