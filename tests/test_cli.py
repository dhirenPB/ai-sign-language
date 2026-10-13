import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from transcript_cleaner.cli import main


def run_cli(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main([str(a) for a in args])
    return code, buf.getvalue()


class BatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_bad_files_are_reported_and_the_batch_continues(self):
        good_bytes = b"Title: T\r\n\r\nSpeaker 1: Hello there.\r\nSecond line. Next sentence.\r\n"
        (self.dir / "a-good.txt").write_bytes(good_bytes)
        (self.dir / "b-empty.txt").write_bytes(b"")
        (self.dir / "c-header-only.txt").write_text("Title: only\n", encoding="utf-8")
        (self.dir / "d-also-good.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi all.\n", encoding="utf-8")

        code, out = run_cli(self.dir)
        self.assertEqual(code, 1)  # failures present
        report = json.loads((self.dir / "reports" / "batch_report.json").read_text(encoding="utf-8"))
        status = {Path(e["file"]).name: e["status"] for e in report["files"]}
        self.assertEqual(status, {"a-good.txt": "complete", "b-empty.txt": "failed",
                                  "c-header-only.txt": "failed", "d-also-good.srt": "complete"})
        self.assertEqual(report["files_failed"], 2)
        self.assertIn("FAILED", out)

        # raw copy is byte-for-byte identical
        self.assertEqual((self.dir / "raw" / "a-good.raw.txt").read_bytes(), good_bytes)
        # every successful source has all four artifacts (+ review queue)
        for name in ("structured/a-good.cleaned.json", "cleaned/a-good.cleaned.txt",
                     "reports/a-good.cleaning_report.json", "review_queue/a-good.review.json"):
            self.assertTrue((self.dir / name).exists(), name)
        # failed sources get a failed report, but no cleaned output
        failed = json.loads((self.dir / "reports" / "b-empty.cleaning_report.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["cleaning_status"], "failed")
        self.assertFalse((self.dir / "structured" / "b-empty.cleaned.json").exists())

    def test_rerun_is_idempotent_and_raw_is_never_overwritten(self):
        src = self.dir / "talk.txt"
        src.write_text("Speaker 1: Original words here.", encoding="utf-8")
        self.assertEqual(run_cli(self.dir)[0], 0)
        self.assertEqual(run_cli(self.dir)[0], 0)  # unchanged input: fine

        src.write_text("Speaker 1: Edited words here.", encoding="utf-8")
        code, out = run_cli(self.dir)
        self.assertEqual(code, 1)
        self.assertIn("never overwritten", out)
        self.assertEqual((self.dir / "raw" / "talk.raw.txt").read_text(encoding="utf-8"),
                         "Speaker 1: Original words here.")

    def test_cleaned_text_file_holds_only_spoken_words(self):
        (self.dir / "talk.txt").write_text(
            "Source: https://www.youtube.com/watch?v=ABCDEFGHIJK\nTitle: My Talk\n\n"
            "Speaker 1: Hello there. How are you? Speaker 2: Fine, thanks.\n", encoding="utf-8")
        run_cli(self.dir)
        txt = (self.dir / "cleaned" / "talk.cleaned.txt").read_text(encoding="utf-8")
        self.assertEqual(txt, "Hello there.\nHow are you?\nFine, thanks.\n")
        # the metadata is not lost, it lives in the structured JSON
        doc = json.loads((self.dir / "structured" / "talk.cleaned.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["source_metadata"]["title"], "My Talk")
        self.assertEqual(doc["chunks"][2]["speaker_id"], "Speaker 2")

    def test_directory_scan_ignores_generated_subfolders(self):
        (self.dir / "talk.txt").write_text("Hello there.", encoding="utf-8")
        run_cli(self.dir)
        code, out = run_cli(self.dir)
        self.assertEqual(code, 0)
        self.assertIn("1/1 processed", out)

    def test_colliding_source_ids_fail_the_second_file(self):
        (self.dir / "my talk.txt").write_text("First file.", encoding="utf-8")
        (self.dir / "my-talk.txt").write_text("Second file.", encoding="utf-8")
        code, _ = run_cli(self.dir)
        report = json.loads((self.dir / "reports" / "batch_report.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(e["status"] for e in report["files"]), ["complete", "failed"])
        self.assertEqual(code, 1)

    def test_fillers_removed_by_default_and_keepable_from_cli(self):
        (self.dir / "talk.txt").write_text("Um, hello there.", encoding="utf-8")
        run_cli(self.dir)
        rep = json.loads((self.dir / "reports" / "talk.cleaning_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["edit_counts_by_operation"].get("remove_filler"), 1)
        self.assertIn("Hello there.", (self.dir / "cleaned" / "talk.cleaned.txt").read_text(encoding="utf-8"))

        out = self.dir / "kept"
        run_cli(self.dir / "talk.txt", "--keep-fillers", "-o", out)
        rep = json.loads((out / "reports" / "talk.cleaning_report.json").read_text(encoding="utf-8"))
        self.assertFalse(rep["config"]["remove_fillers"])
        self.assertIn("Um, hello there.", (out / "cleaned" / "talk.cleaned.txt").read_text(encoding="utf-8"))

    def test_missing_input_and_bad_config(self):
        self.assertEqual(run_cli(self.dir / "nope")[0], 2)
        bad = self.dir / "bad.json"
        bad.write_text('{"not_a_real_option": 1}', encoding="utf-8")
        (self.dir / "x.txt").write_text("Hi.", encoding="utf-8")
        self.assertEqual(run_cli(self.dir / "x.txt", "--config", bad)[0], 2)


if __name__ == "__main__":
    unittest.main()
