import unittest

from tests.helpers import flag_codes, run, texts


class MeaningPreservationTests(unittest.TestCase):
    def test_numbers_and_negation_untouched(self):
        r = run("Speaker 2: The result was 15, not 50. It is not ready.")
        self.assertEqual(texts(r), ["The result was 15, not 50.", "It is not ready."])
        self.assertEqual(r.edits and [e for e in r.edits if e.stage == "clean"], [])

    def test_fillers_can_be_kept(self):
        r = run("uh, the model is not ready yet", remove_fillers=False)
        self.assertEqual(texts(r), ["uh, the model is not ready yet"])

    def test_fillers_removed_by_default_logged_and_negation_kept(self):
        r = run("uh, the model is not ready yet")
        self.assertEqual(texts(r), ["the model is not ready yet"])
        r = run("Um, the result was not 15, uh, but 50. Uh, ok.")
        self.assertEqual(texts(r), ["The result was not 15, but 50.", "Ok."])
        removed = [e for e in r.edits if e.operation == "remove_filler"]
        self.assertEqual(len(removed), 3)
        self.assertEqual(r.status, "complete", r.validation)  # word accounting still reconciles

    def test_repetition_flagged_not_deleted_by_default(self):
        r = run("We we need to test it.")
        self.assertEqual(texts(r), ["We we need to test it."])
        self.assertIn("repeated_word", flag_codes(r.chunks[0]))

    def test_repeat_collapse_opt_in_requires_review(self):
        r = run("We we need to test it.", collapse_repeated_words=True)
        self.assertEqual(texts(r), ["We need to test it."])
        self.assertTrue(any(e.operation == "collapse_repeated_word" and e.review_required for e in r.edits))
        self.assertEqual(r.status, "needs_review")

    def test_legitimate_double_words_not_flagged(self):
        r = run("He said that that was fine.")
        self.assertNotIn("repeated_word", flag_codes(r.chunks[0]))

    def test_repeated_phrase_flagged(self):
        r = run("I know I know what you mean.")
        self.assertEqual(texts(r), ["I know I know what you mean."])
        self.assertIn("repeated_phrase", flag_codes(r.chunks[0]))


class UncertaintyTests(unittest.TestCase):
    def test_inaudible_preserved_and_flagged_for_review(self):
        r = run("[inaudible] and then we deployed the model.")
        self.assertEqual(texts(r), ["[inaudible] and then we deployed the model."])
        flag = r.chunks[0].uncertainty_flags[0]
        self.assertEqual((flag.code, flag.text, flag.review_required), ("unclear_audio", "[inaudible]", True))
        self.assertEqual(r.status, "needs_review")
        self.assertTrue(r.doc["review_required"])
        self.assertEqual(len(r.review_items), 1)

    def test_unclear_audio_variants(self):
        for marker in ("[Unintelligible]", "(inaudible)", "[crosstalk]", "[inaudible 00:12]"):
            r = run(f"We said {marker} yesterday.")
            self.assertIn("unclear_audio", flag_codes(r.chunks[0]), marker)

    def test_sound_tags_kept_unless_configured(self):
        self.assertEqual(texts(run("[Music] Welcome back.")), ["[Music] Welcome back."])
        r = run("[Music] Welcome back.", drop_sound_tags=True)
        self.assertEqual(texts(r), ["Welcome back."])
        self.assertEqual(r.status, "complete", r.validation)

    def test_unclear_marker_never_dropped_even_with_drop_sound_tags(self):
        r = run("[Music] We said [inaudible] twice.", drop_sound_tags=True)
        self.assertEqual(texts(r), ["We said [inaudible] twice."])

    def test_romanised_gujarati_kept_and_marked_skip(self):
        r = run("Jya tu tya hu. He went home.")
        self.assertEqual(texts(r), ["Jya tu tya hu.", "He went home."])
        spans = r.chunks[0].to_dict()["skip_translation_spans"]
        self.assertEqual([(s["text"], s["reason"]) for s in spans], [("Jya tu tya hu", "skip_non_english")])
        self.assertEqual(r.chunks[1].to_dict()["skip_translation_spans"], [])
        self.assertEqual(r.review_items, [])  # decided, so not queued for review

    def test_only_the_gujarati_run_is_skipped_inside_a_mixed_sentence(self):
        r = run("Ek divas tu ane hu sarkha hashu Hey my Lord, one day you and I will be equal.")
        spans = r.chunks[0].to_dict()["skip_translation_spans"]
        self.assertEqual([s["text"] for s in spans], ["Ek divas tu ane hu sarkha hashu"])
        self.assertEqual(r.chunks[0].text[spans[0]["start"]:spans[0]["end"]], spans[0]["text"])

    def test_single_stray_marker_word_in_english_is_not_skipped(self):
        r = run("He said ek once and left.")
        self.assertEqual(r.chunks[0].to_dict()["skip_translation_spans"], [])

    def test_known_misrecognised_terms_kept_and_marked_skip(self):
        r = run("We study the Jane tradition today.")
        self.assertEqual(texts(r), ["We study the Jane tradition today."])
        spans = r.chunks[0].to_dict()["skip_translation_spans"]
        self.assertEqual([(s["text"], s["reason"]) for s in spans], [("Jane", "skip_term")])
        self.assertEqual(r.status, "complete")

    def test_expression_cues_kept_and_marked_skip(self):
        r = run("Hmm, that is true. Ooh, nice.")
        self.assertEqual(texts(r), ["Hmm, that is true.", "Ooh, nice."])
        got = [(s["text"], s["reason"]) for c in r.chunks for s in c.to_dict()["skip_translation_spans"]]
        self.assertEqual(got, [("Hmm", "expression_cue"), ("Ooh", "expression_cue")])
        self.assertEqual(r.review_items, [])

    def test_asr_full_stops_are_real_sentence_breaks(self):
        r = run("I'm happy to have you with us. or this evening depending where you are.")
        self.assertEqual(texts(r), ["I'm happy to have you with us.", "or this evening depending where you are."])
        self.assertEqual(flag_codes(r.chunks[1]), [])

    def test_boilerplate_chunk_removed_and_logged(self):
        r = run("Hello everyone. Subtitles by the Amara.org community")
        self.assertEqual(texts(r), ["Hello everyone."])
        self.assertEqual([e.operation for e in r.edits if e.stage == "clean"], ["remove_boilerplate"])
        self.assertEqual(r.status, "complete", r.validation)


class UnicodeTests(unittest.TestCase):
    def test_nbsp_zero_width_and_nfc(self):
        r = run("Café au​ lait. Café again.")
        self.assertEqual(texts(r), ["Café au lait.", "Café again."])
        counts = r.normalization_counts
        self.assertEqual(counts["unicode_spaces_replaced"], 1)
        self.assertEqual(counts["invisible_chars_removed"], 1)
        self.assertEqual(counts["unicode_nfc_normalized"], 1)
        self.assertEqual(r.status, "complete", r.validation)

    def test_mojibake_repaired_and_logged(self):
        r = run("Itâ€™s fine.")
        self.assertEqual(texts(r), ["It’s fine."])
        self.assertEqual([e.operation for e in r.edits if e.stage == "normalize"], ["repair_mojibake"])

    def test_unrepairable_mojibake_is_flagged(self):
        r = run("Bad Ã© and ก text.")  # mixed: cannot round-trip through cp1252
        self.assertIn("possible_mojibake", flag_codes(r.chunks[0]))

    def test_gujarati_script_preserved_including_zwj(self):
        s = "નમસ્તે ક્‍ષ everyone."
        r = run(s)
        self.assertEqual(texts(r), [s])

    def test_replacement_character_flagged(self):
        r = run("Broken � bytes here.")
        self.assertIn("replacement_character", flag_codes(r.chunks[0]))
        self.assertEqual(r.status, "needs_review")

    def test_hard_wrapped_lines_not_concatenated(self):
        r = run("This sentence is\nwrapped across\nthree lines.")
        self.assertEqual(texts(r), ["This sentence is wrapped across three lines."])


if __name__ == "__main__":
    unittest.main()
