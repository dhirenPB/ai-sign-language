import unittest

from transcript_cleaner.ingest import canonical_timestamp
from tests.helpers import run, texts


class TimestampTests(unittest.TestCase):
    def test_canonical_forms(self):
        self.assertEqual(canonical_timestamp("00:00:10,000"), "00:00:10.000")
        self.assertEqual(canonical_timestamp("00:10.500"), "00:00:10.500")
        self.assertEqual(canonical_timestamp("1:02:03.4"), "01:02:03.400")

    def test_invalid_timestamps_are_none_not_guessed(self):
        self.assertIsNone(canonical_timestamp("00:61:00,000"))
        self.assertIsNone(canonical_timestamp("00:00:61,000"))
        self.assertIsNone(canonical_timestamp("abc"))


class SrtTests(unittest.TestCase):
    SRT = (
        "17\n00:00:10,000 --> 00:00:12,000\nSpeaker 2: The result was 15, not\n50.\n\n"
        "18\n00:00:12,500 --> 00:00:14,000\nWe are building\n\n"
        "19\n00:00:14,000 --> 00:00:15,500\na translation system.\n"
    )

    def test_cue_numbers_and_timing_are_metadata(self):
        r = run(self.SRT, "a.srt")
        self.assertEqual(r.status, "complete", r.validation)
        first = r.chunks[0]
        self.assertEqual(first.text, "The result was 15, not 50.")
        self.assertEqual((first.start_time, first.end_time), ("00:00:10.000", "00:00:12.000"))
        self.assertEqual(first.speaker_id, "Speaker 2")
        self.assertEqual(first.cue_ids, ["17"])
        for c in r.chunks:
            self.assertNotIn("-->", c.text)
            self.assertNotIn("Speaker", c.text)

    def test_cues_merge_into_one_sentence_with_span_timing(self):
        r = run(self.SRT, "a.srt")
        merged = r.chunks[1]
        self.assertEqual(merged.text, "We are building a translation system.")
        self.assertEqual((merged.start_time, merged.end_time), ("00:00:12.500", "00:00:15.500"))
        self.assertEqual(merged.cue_indices, [2, 3])
        self.assertEqual(merged.cue_ids, ["18", "19"])
        self.assertIsNone(merged.source_start_offset)  # no reliable offsets for cue formats

    def test_invalid_timestamp_becomes_null_with_warning(self):
        r = run("1\n00:00:61,000 --> 00:00:62,000\nHello there.\n", "a.srt")
        self.assertIsNone(r.chunks[0].start_time)
        self.assertIn("invalid_timestamp", [w["code"] for w in r.doc["warnings"]])


class VttTests(unittest.TestCase):
    def test_header_note_settings_tags_entities_and_voice(self):
        vtt = (
            "WEBVTT\n\nNOTE a comment\n\n"
            "00:01.000 --> 00:03.000 align:start position:0%\n<v Alice>Tom &amp; Jerry are <c.color>here</c>.</v>\n"
        )
        r = run(vtt, "a.vtt")
        self.assertEqual(texts(r), ["Tom & Jerry are here."])
        self.assertEqual(r.chunks[0].speaker_id, "Alice")
        self.assertEqual(r.chunks[0].start_time, "00:00:01.000")

    def test_rolling_caption_duplicates_removed_and_logged(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\nhello everyone welcome\n\n"
            "00:00:02.000 --> 00:00:04.000\nhello everyone welcome\nto the show\n\n"
            "00:00:04.000 --> 00:00:04.010\nto the show\n\n"
            "00:00:04.010 --> 00:00:06.000\nto the show\ntoday we talk\n"
        )
        r = run(vtt, "a.vtt")
        self.assertEqual(texts(r), ["hello everyone welcome to the show today we talk"])
        ops = [e.operation for e in r.edits]
        self.assertGreaterEqual(ops.count("remove_rolling_caption_duplicate"), 2)

    def test_genuine_repeated_single_line_cue_is_kept(self):
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nNo.\n\n00:00:01.000 --> 00:00:02.000\nNo.\n"
        self.assertEqual(texts(run(vtt, "a.vtt")), ["No.", "No."])


class TxtParsingTests(unittest.TestCase):
    def test_header_is_metadata_and_labels_are_separated(self):
        raw = ("Source: https://www.youtube.com/watch?v=ABCDEFGHIJK\nTitle: My Talk\n\n"
               "Speaker 1: Hello there. Speaker 2: Hi.\n")
        r = run(raw)
        self.assertEqual(texts(r), ["Hello there.", "Hi."])
        self.assertEqual([c.speaker_id for c in r.chunks], ["Speaker 1", "Speaker 2"])
        self.assertEqual(r.doc["source_metadata"]["video_id"], "ABCDEFGHIJK")
        self.assertEqual(r.doc["source_metadata"]["title"], "My Talk")

    def test_no_speakers_or_timestamps_stay_null(self):
        r = run("Just some words. And more words.")
        self.assertFalse(r.doc["timestamps_available"])
        self.assertFalse(r.doc["speakers_available"])
        for c in r.chunks:
            self.assertIsNone(c.speaker_id)
            self.assertIsNone(c.start_time)

    def test_offsets_point_into_the_raw_text(self):
        raw = "Title: T\n\nSpeaker 1: First   sentence\nwrapped. Speaker 2: Second one here.\n"
        r = run(raw)
        for c in r.chunks:
            piece = r.parsed.raw_text[c.source_start_offset:c.source_end_offset]
            self.assertEqual(" ".join(piece.split()), c.text)

    def test_crlf_and_bom(self):
        raw = b"\xef\xbb\xbfTitle: T\r\n\r\nSpeaker 1: Line one\r\nstill one. Next.\r\n"
        r = run(raw)
        self.assertEqual(texts(r), ["Line one still one.", "Next."])
        for c in r.chunks:
            self.assertNotIn("\r", c.text)
            piece = r.parsed.raw_text[c.source_start_offset:c.source_end_offset]
            self.assertEqual(" ".join(piece.split()), c.text)

    def test_non_utf8_falls_back_with_warning(self):
        r = run(b"Caf\xe9 is open. Really.")
        self.assertIn("encoding_fallback", [w["code"] for w in r.doc["warnings"]])
        self.assertEqual(r.chunks[0].text, "Café is open.")

    def test_empty_and_metadata_only_inputs_fail(self):
        for payload in ("", "   \n\n", "Title: only a header\nSource: https://example.com\n"):
            r = run(payload)
            self.assertEqual(r.status, "failed", payload)
            self.assertIn("no_chunks", [e["code"] for e in r.validation["errors"]])


if __name__ == "__main__":
    unittest.main()
