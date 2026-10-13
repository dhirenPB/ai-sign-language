import time
import unittest

from isl_gloss.config import TranslatorConfig
from isl_gloss.grouping import estimate_tokens, group_sentences, group_stats
from isl_gloss.sentences import Sentence, build_sentences, dependency_reasons, split_run_on
from transcript_cleaner.config import CleaningConfig
from transcript_cleaner.pipeline import process_bytes


def cleaned(text, name="talk.txt"):
    """Real cleaner output for a raw transcript (the existing cleaning module, unchanged)."""
    return process_bytes(text.encode("utf-8"), name, CleaningConfig()).doc


def mk_doc(texts, turns=None, times=None, speakers=None, source_id="t"):
    chunks = []
    for i, text in enumerate(texts):
        start, end = (times[i] if times else (None, None))
        chunks.append({"chunk_id": f"{source_id}-{i + 1:04d}", "text": text,
                       "turn_index": turns[i] if turns else 0, "speaker_id": speakers[i] if speakers else None,
                       "start_time": start, "end_time": end, "uncertainty_flags": [], "skip_translation_spans": []})
    return {"source_id": source_id, "chunks": chunks}


def sentences_of(texts, **kw):
    return build_sentences(mk_doc(texts, **kw), TranslatorConfig())


class SentenceChunkingTests(unittest.TestCase):
    def test_normally_punctuated_transcript(self):
        s = build_sentences(cleaned("Speaker 1: The sun rose. Birds sang loudly. We walked home."), TranslatorConfig())
        self.assertEqual([x.source_text for x in s], ["The sun rose.", "Birds sang loudly.", "We walked home."])
        self.assertEqual([x.sentence_id for x in s], ["talk-s0001", "talk-s0002", "talk-s0003"])
        self.assertTrue(all(x.source_metadata["split_method"] is None for x in s))

    def test_abbreviations_are_not_sentence_ends(self):
        s = build_sentences(cleaned("Speaker 1: Dr. Smith arrived. He sat down."), TranslatorConfig())
        self.assertEqual([x.source_text for x in s], ["Dr. Smith arrived.", "He sat down."])

    def test_missing_punctuation_is_split_at_capital_openers_not_at_word_counts(self):
        text = ("Hey Prabhu If you had not grasped this land If you had not taught us the values of Ahimsa, "
                "Truth non-stealing, celibacy, non-possessionness.")
        s = build_sentences(cleaned("Speaker 1: " + text), TranslatorConfig())
        self.assertEqual([x.source_text for x in s], [
            "Hey Prabhu If you had not grasped this land",
            "If you had not taught us the values of Ahimsa, Truth non-stealing, celibacy, non-possessionness."])
        self.assertEqual(" ".join(x.source_text for x in s), text)  # sliced, never rewritten, nothing lost
        self.assertTrue(all(x.source_metadata["split_method"] == "runon_capital_opener" for x in s))
        self.assertEqual([x.source_metadata["parent_chunk_id"] for x in s], ["talk-0001", "talk-0001"])
        self.assertEqual([x.source_metadata["piece"] for x in s], [[1, 2], [2, 2]])

    def test_opener_followed_by_a_comma_still_counts_as_an_opener(self):
        text = ("Hello Hi students The California State University where Professor Jonathan Young has invited to take "
                "a session on philosophy and religious studies Today, first of all, I would like to appreciate your curiosity.")
        s = build_sentences(cleaned("Speaker 1: " + text), TranslatorConfig())
        self.assertEqual(len(s), 2)
        self.assertTrue(s[1].source_text.startswith("Today, first of all"))
        self.assertEqual(" ".join(x.source_text for x in s), text)

    def test_short_or_properly_capitalised_text_is_not_split(self):
        self.assertEqual(len(sentences_of(["Whenever we do any action It creates an image."])), 1)  # under the length floor
        long_but_clean = "We walked along the river to the old market and met a friend of Maria who sells fresh fruit"
        self.assertEqual(len(sentences_of([long_but_clean])), 1)  # no capitalised opener in the middle
        opener_first = "If a b c d e f g h i j k l m"
        self.assertEqual(split_run_on(opener_first, [], 14), [(0, len(opener_first))])  # opener at the start: no cut

    def test_run_on_splitting_can_be_switched_off_and_never_cuts_inside_a_skip_span(self):
        text = "alpha beta gamma delta epsilon zeta eta theta If iota kappa lambda mu nu"
        self.assertEqual(split_run_on(text, [], 0), [(0, len(text))])
        self.assertEqual(len(split_run_on(text, [], 14)), 2)
        start = text.index("If")
        protected = [{"start": start - 1, "end": start + 4, "text": text[start - 1:start + 4]}]
        self.assertEqual(split_run_on(text, protected, 14), [(0, len(text))])

    def test_skip_spans_follow_their_words_into_the_right_sentence(self):
        text = "alpha beta gamma delta If one two three four five six seven eight"
        doc = mk_doc([text])
        j = text.index("one")
        doc["chunks"][0]["skip_translation_spans"] = [{"start": j, "end": j + 3, "text": "one", "reason": "skip_term"}]
        s = build_sentences(doc, TranslatorConfig(runon_min_words=8))
        self.assertEqual(len(s), 2)
        self.assertEqual(s[0].skip_spans, [])
        self.assertEqual(s[1].skip_spans[0]["text"], "one")
        self.assertEqual(s[1].source_text[s[1].skip_spans[0]["start"]:s[1].skip_spans[0]["end"]], "one")

    def test_timestamps_and_speakers_are_carried_over(self):
        srt = ("1\n00:00:01,000 --> 00:00:03,000\nSpeaker 1: Hello there. How are\n\n"
               "2\n00:00:03,000 --> 00:00:05,000\nyou today?\n\n"
               "3\n00:00:20,000 --> 00:00:22,000\nSpeaker 2: Fine, thanks.\n")
        s = build_sentences(cleaned(srt, "talk.srt"), TranslatorConfig())
        self.assertEqual([x.source_text for x in s], ["Hello there.", "How are you today?", "Fine, thanks."])
        self.assertEqual([(x.start_timestamp, x.end_timestamp) for x in s], [
            ("00:00:01.000", "00:00:03.000"), ("00:00:01.000", "00:00:05.000"), ("00:00:20.000", "00:00:22.000")])
        self.assertEqual([x.speaker_id for x in s], ["Speaker 1", "Speaker 1", "Speaker 2"])
        self.assertEqual(s[1].source_metadata["cue_ids"], ["1", "2"])

    def test_missing_timestamps_and_speakers_stay_none(self):
        s = build_sentences(cleaned("Just words. More words."), TranslatorConfig())
        for x in s:
            self.assertIsNone(x.start_timestamp)
            self.assertIsNone(x.end_timestamp)
            self.assertIsNone(x.speaker_id)

    def test_ids_are_deterministic_unique_and_ordered(self):
        doc = cleaned("Speaker 1: One. Two. Three. Speaker 2: Four. Five.")
        a, b = build_sentences(doc, TranslatorConfig()), build_sentences(doc, TranslatorConfig())
        self.assertEqual([x.to_dict() for x in a], [x.to_dict() for x in b])
        ids = [x.sentence_id for x in a]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), len(ids))

    def test_dependency_signals_for_pronouns_fragments_and_incomplete_sentences(self):
        s = sentences_of(["Maria bought a laptop.", "She says it is fast.", "and it works well.", "Yes.",
                          "We stopped for a rest,", "then we left."])
        self.assertEqual(s[0].dependency_reasons, [])  # nothing before it
        self.assertEqual(s[1].dependency_reasons, ["pronoun_start"])
        self.assertEqual(s[2].dependency_reasons, ["lowercase_start", "continuation_word"])
        self.assertIn("very_short", s[3].dependency_reasons)
        self.assertTrue(s[4].incomplete)  # ends with a comma
        self.assertIn("previous_incomplete", s[5].dependency_reasons)
        self.assertEqual(sentences_of(["It rained.", "We stayed home."])[1].dependency_reasons, [])  # 'We' is not anaphoric

    def test_dependency_reasons_without_a_previous_sentence(self):
        self.assertEqual(dependency_reasons("She left.", None), [])
        prev = Sentence(sentence_id="x", source_text="Hello,", incomplete=True)
        self.assertEqual(dependency_reasons("Everyone was there.", prev), ["previous_incomplete"])


class GroupingTests(unittest.TestCase):
    def groups(self, texts, cfg=None, **kw):
        cfg = cfg or TranslatorConfig()
        s = build_sentences(mk_doc(texts, **kw), cfg)
        return s, group_sentences(s, cfg)

    def assert_exact_cover(self, s, groups):
        self.assertEqual([i for g in groups for i in g.sentence_ids], [x.sentence_id for x in s])

    def test_related_sentences_stay_together_in_order(self):
        texts = ["Maria bought a new laptop yesterday.", "She says it is very fast.", "It replaced the old one that kept crashing.",
                 "Her brother wants one too."]
        s, g = self.groups(texts)
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0].sentence_ids, [x.sentence_id for x in s])
        self.assertEqual(g[0].context_text, " ".join(texts))
        self.assertEqual([x.source_text for x in g[0].sentences], texts)

    def test_unrelated_passages_in_separate_turns_are_not_combined(self):
        doc = cleaned("Speaker 1: The monsoon arrives in June. Farmers prepare their fields early.\n\n"
                      "Speaker 1: Our new library opens next week. It has a quiet reading room.")
        s = build_sentences(doc, TranslatorConfig())
        g = group_sentences(s, TranslatorConfig())
        self.assertEqual(len(g), 2)
        self.assertEqual(g[0].boundary_reason, "turn_change")
        self.assertIn("monsoon", g[0].context_text)
        self.assertNotIn("library", g[0].context_text)
        self.assertNotIn("monsoon", g[1].context_text)

    def test_speaker_change_and_time_gap_are_hard_boundaries(self):
        s, g = self.groups(["One here.", "Two here."], speakers=["A", "B"], turns=[0, 0])
        self.assertEqual([x.boundary_reason for x in g], ["speaker_change", "end"])
        times = [("00:00:01.000", "00:00:02.000"), ("00:00:03.000", "00:00:04.000"), ("00:00:30.000", "00:00:31.000")]
        s, g = self.groups(["First thing.", "Second thing.", "Much later thing."], times=times)
        self.assertEqual([x.sentence_ids for x in g], [[s[0].sentence_id, s[1].sentence_id], [s[2].sentence_id]])
        self.assertEqual(g[0].boundary_reason, "time_gap")
        self.assertEqual((g[0].start_timestamp, g[0].end_timestamp), ("00:00:01.000", "00:00:04.000"))
        cfg = TranslatorConfig(group_max_gap_seconds=None)  # timing ignored on request
        self.assertEqual(len(group_sentences(build_sentences(mk_doc(["First thing.", "Second thing.", "Much later thing."], times=times), cfg), cfg)), 1)

    def test_sentence_count_limit(self):
        s, g = self.groups([f"Independent statement number {i}." for i in range(10)], cfg=TranslatorConfig(group_max_sentences=3))
        self.assertEqual([len(x.sentences) for x in g], [3, 3, 3, 1])
        self.assertTrue(all(x.boundary_reason == "max_sentences" for x in g[:-1]))
        self.assert_exact_cover(s, g)

    def test_character_and_token_budgets(self):
        texts = [f"Sentence {i} " + "word " * 10 + "ends here." for i in range(8)]  # ~70 chars each
        cfg = TranslatorConfig(group_max_chars=150, group_max_tokens=300)
        s, g = self.groups(texts, cfg=cfg)
        self.assertTrue(all(len(" ".join(x.source_text for x in grp.sentences)) <= 150 for grp in g))
        self.assertIn("max_chars", {x.boundary_reason for x in g})
        self.assert_exact_cover(s, g)
        cfg = TranslatorConfig(group_max_tokens=40)
        s, g = self.groups(texts, cfg=cfg)
        self.assertTrue(all(estimate_tokens(" ".join(x.source_text for x in grp.sentences)) <= 40 for grp in g))
        self.assertIn("max_tokens", {x.boundary_reason for x in g})
        self.assert_exact_cover(s, g)

    def test_one_long_sentence_is_never_cut_and_is_flagged_oversize(self):
        long_sentence = "This single sentence is very " + "long and winding " * 30 + "indeed."
        s, g = self.groups(["Short one.", long_sentence, "Another short one."], cfg=TranslatorConfig(group_max_chars=200))
        big = [x for x in g if [y.source_text for y in x.sentences] == [long_sentence]]
        self.assertEqual(len(big), 1)
        self.assertTrue(big[0].oversize)
        self.assertEqual(big[0].sentences[0].source_text, long_sentence)  # whole sentence intact
        self.assert_exact_cover(s, g)

    def test_limit_splits_at_the_latest_clean_boundary_keeping_dependent_sentences_together(self):
        texts = ["Alpha opens the day.", "Beta follows closely.", "Gamma changes the plan.", "It surprises everyone.",
                 "Delta ends the story."]
        s, g = self.groups(texts, cfg=TranslatorConfig(group_max_sentences=3))
        # 'It surprises everyone.' depends on 'Gamma changes the plan.', so the cut moves before Gamma
        self.assertEqual([[x.source_text for x in grp.sentences] for grp in g],
                         [texts[:2], texts[2:]])
        self.assertFalse(any(x.forced_boundary for x in g))
        self.assert_exact_cover(s, g)

    def test_forced_cut_inside_a_dependent_chain_is_recorded_and_gets_context(self):
        texts = ["The engine started.", "It ran for an hour.", "It then overheated.", "It was shut down.", "It cooled slowly.",
                 "It was restarted."]
        s, g = self.groups(texts, cfg=TranslatorConfig(group_max_sentences=3))
        self.assertTrue(g[0].forced_boundary)
        self.assertEqual(len(g[0].sentences), 3)
        self.assertEqual(g[1].context_before_ids, [s[2].sentence_id])  # last sentence before the cut, read-only
        self.assertEqual(g[0].context_after_ids, [s[3].sentence_id])  # successor depends on this group
        self.assertNotIn(s[2].sentence_id, g[1].sentence_ids)
        self.assert_exact_cover(s, g)

    def test_read_only_context_only_when_needed(self):
        s, g = self.groups(["The first statement is about rain.", "The second statement is about wind."], cfg=TranslatorConfig(group_max_sentences=1))
        self.assertEqual([x.context_before_ids for x in g], [[], []])
        self.assertEqual([x.context_after_ids for x in g], [[], []])
        s, g = self.groups(["We opened the shop,", "and sales grew fast."], turns=[0, 1])
        self.assertEqual(g[1].context_before_ids, [s[0].sentence_id])  # lowercase start depends on the previous turn's text
        self.assertEqual(g[0].context_after_ids, [s[1].sentence_id])  # the first group ends with a comma
        self.assertNotIn(s[1].sentence_id, g[0].sentence_ids)  # ... but context is never part of the group's sentences
        self.assertIn("and sales grew fast.", g[0].context_text)

    def test_config_context_counts_and_validation(self):
        cfg = TranslatorConfig(context_sentences_before=0, context_sentences_after=0)
        s, g = self.groups(["We opened the shop,", "and sales grew fast."], turns=[0, 1], cfg=cfg)
        self.assertTrue(all(not x.context_before_ids and not x.context_after_ids for x in g))
        for bad in ({"group_max_sentences": 0}, {"group_max_chars": 5}, {"group_max_tokens": 1}, {"runon_min_words": -1}):
            with self.assertRaises(ValueError):
                TranslatorConfig(**bad).validate()

    def test_group_ids_and_mapping_are_stable(self):
        texts = [f"Statement {i} stands alone." for i in range(7)]
        cfg = TranslatorConfig(group_max_sentences=3)
        s = build_sentences(mk_doc(texts), cfg)
        a, b = group_sentences(s, cfg), group_sentences(s, cfg)
        self.assertEqual([x.to_dict() for x in a], [x.to_dict() for x in b])
        self.assertEqual([x.group_id for x in a], ["t-g0001", "t-g0002", "t-g0003"])
        self.assertEqual(group_sentences([], cfg), [])

    def test_empty_and_single_sentence_inputs(self):
        s, g = self.groups(["Only one."])
        self.assertEqual((len(g), g[0].boundary_reason), (1, "end"))

    def test_long_transcript_is_grouped_fast_in_order_without_loss(self):
        texts = [f"This is statement number {i} in a very long talk." if i % 5 else f"It continues with item {i}." for i in range(3000)]
        cfg = TranslatorConfig()
        s = build_sentences(mk_doc(texts), cfg)
        t0 = time.time()
        g = group_sentences(s, cfg)
        self.assertLess(time.time() - t0, 5)
        self.assert_exact_cover(s, g)
        self.assertTrue(all(len(x.sentences) <= cfg.group_max_sentences for x in g))
        stats = group_stats(g)
        self.assertEqual(stats["sentences"], 3000)


if __name__ == "__main__":
    unittest.main()
