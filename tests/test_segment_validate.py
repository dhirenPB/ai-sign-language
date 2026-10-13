import copy
import time
import unittest

from transcript_cleaner.config import CleaningConfig
from transcript_cleaner.normalize import comparable_tokens
from transcript_cleaner.schema import validate_against_schema
from transcript_cleaner.validate import validate_chunks
from tests.helpers import flag_codes, run, texts


class SegmentationTests(unittest.TestCase):
    def test_abbreviations_initials_and_decimals(self):
        self.assertEqual(texts(run("Dr. Smith arrived. He sat down.")), ["Dr. Smith arrived.", "He sat down."])
        self.assertEqual(texts(run("Mr. J. Young spoke. Then left.")), ["Mr. J. Young spoke.", "Then left."])
        self.assertEqual(texts(run("It costs 3.5 dollars. Fine.")), ["It costs 3.5 dollars.", "Fine."])

    def test_question_and_exclamation_split(self):
        self.assertEqual(texts(run("Really? Yes! Okay.")), ["Really?", "Yes!", "Okay."])

    def test_ellipsis_followed_by_lowercase_is_not_a_boundary(self):
        self.assertEqual(texts(run("Wait... what? No.")), ["Wait... what?", "No."])

    def test_overlong_unpunctuated_run_is_split_without_losing_words(self):
        words = [f"word{i}" for i in range(150)]
        r = run(" ".join(words), max_chunk_words=40)
        self.assertGreater(len(r.chunks), 3)
        for c in r.chunks:
            self.assertLessEqual(len(c.text.split()), 40)
        self.assertEqual([w for c in r.chunks for w in c.text.split()], words)
        self.assertTrue(any("forced_split" in flag_codes(c) for c in r.chunks))
        self.assertEqual(r.status, "complete", r.validation)

    def test_overlong_split_prefers_clause_punctuation(self):
        clause = " ".join(f"a{i}" for i in range(25)) + ","
        r = run(" ".join([clause] * 4), max_chunk_words=40)
        self.assertTrue(all(c.text.endswith(",") or c is r.chunks[-1] for c in r.chunks))
        self.assertFalse(any("forced_split" in flag_codes(c) for c in r.chunks))

    def test_chunk_ids_sequential_and_ordered(self):
        r = run("One. Two. Three.")
        self.assertEqual([c.chunk_id for c in r.chunks], ["sample-0001", "sample-0002", "sample-0003"])
        offsets = [c.source_start_offset for c in r.chunks]
        self.assertEqual(offsets, sorted(offsets))


class LongTranscriptTests(unittest.TestCase):
    def test_long_txt_is_fast_ordered_and_lossless(self):
        n = 6000
        body = " ".join(f"Speaker {i % 2 + 1}: This is sentence number {i}, and it is not short." for i in range(n))
        t0 = time.time()
        r = run(body)
        self.assertLess(time.time() - t0, 20)
        self.assertEqual(len(r.chunks), n)
        self.assertEqual(r.status, "complete", r.validation["errors"][:2])
        self.assertEqual(sum(len(comparable_tokens(c.text)) for c in r.chunks),
                         sum(len(comparable_tokens(p.text)) for p in r.parsed.pieces))

    def test_long_srt_single_turn_is_not_quadratic(self):
        n = 4000
        cues = []
        for i in range(n):
            def ts(sec):
                return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d},000"
            cues.append(f"{i + 1}\n{ts(i)} --> {ts(i + 1)}\nCue {i} says something. And more")
        t0 = time.time()
        r = run("\n\n".join(cues) + "\n", "long.srt")
        self.assertLess(time.time() - t0, 20)
        self.assertNotEqual(r.status, "failed", r.validation["errors"][:2])
        starts = [c.start_time for c in r.chunks]
        self.assertEqual(starts, sorted(starts))


class ValidationTests(unittest.TestCase):
    def test_output_conforms_to_schema_and_is_deterministic(self):
        raw = "Title: T\n\nSpeaker 1: Hello there. [inaudible] again.\n"
        r1, r2 = run(raw), run(raw)
        self.assertEqual(validate_against_schema(r1.doc), [])
        self.assertEqual(r1.doc, r2.doc)

    def test_schema_violations_are_reported(self):
        r = run("Hello there.")
        broken = copy.deepcopy(r.doc)
        del broken["chunks"][0]["speaker_id"]
        broken["source_type"] = "podcast"
        errors = validate_against_schema(broken)
        self.assertTrue(any("speaker_id" in e for e in errors))
        self.assertTrue(any("source_type" in e for e in errors))

    def test_lost_words_are_detected(self):
        r = run("One two three. Four five six. Seven eight nine.")
        cfg = CleaningConfig()
        dropped = [r.chunks[0], r.chunks[2]]
        result = validate_chunks(r.parsed, dropped, r.edits, cfg)
        self.assertFalse(result["passed"])
        self.assertIn("word_accounting_mismatch", [e["code"] for e in result["errors"]])

    def test_duplicated_words_are_detected(self):
        r = run("One two three. Four five six.")
        result = validate_chunks(r.parsed, r.chunks + [r.chunks[0]], r.edits, CleaningConfig())
        self.assertIn("word_accounting_mismatch", [e["code"] for e in result["errors"]])

    def test_speaker_label_leaking_into_text_is_detected(self):
        r = run("Speaker 1: Hello there.")
        r.chunks[0].text = "Speaker 1: Hello there."
        result = validate_chunks(r.parsed, r.chunks, r.edits, CleaningConfig())
        self.assertIn("metadata_in_text", [e["code"] for e in result["errors"]])

    def test_genuine_speaker_repetition_is_kept_with_a_warning(self):
        r = run("Speaker 1: Please listen carefully. Please listen carefully.")
        self.assertEqual(len(r.chunks), 2)
        self.assertIn("duplicate_consecutive_chunk", [w["code"] for w in r.validation["warnings"]])
        self.assertEqual(r.status, "complete")

    def test_dropping_everything_fails_the_file(self):
        r = run("Subtitles by someone\n\nShort.", boilerplate_patterns=(r".*",))  # absurd rule drops everything
        self.assertEqual(r.status, "failed")


if __name__ == "__main__":
    unittest.main()
