import tempfile
import unittest
from pathlib import Path

from isl_gloss.config import TranslatorConfig
from isl_gloss.prompt import (DEFAULT_PROMPT_VERSION, PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT, build_system_prompt,
                              get_prompt)
from isl_gloss.store import ResultStore, cache_key
from isl_gloss.translator import Translator
from tests.gloss_helpers import FakeClient, chunk


class PromptVersionsTests(unittest.TestCase):
    def test_current_is_7_0_and_earlier_prompts_are_kept_for_comparison(self):
        self.assertEqual((DEFAULT_PROMPT_VERSION, PROMPT_VERSION), ("7.0", "7.0"))
        self.assertEqual(TranslatorConfig().prompt_version, "7.0")  # the pipeline really uses it by default
        self.assertEqual(Translator(TranslatorConfig(), None).spec.version, "7.0")
        self.assertEqual(sorted(PROMPTS), ["5.0", "6.0", "7.0", "8.0"])
        self.assertTrue(get_prompt("7.0").id_mode and get_prompt("6.0").id_mode)
        self.assertFalse(get_prompt("5.0").id_mode)
        self.assertEqual({v: get_prompt(v).key_field for v in PROMPTS}, {"5.0": "n", "6.0": "sentence_id", "7.0": "sentence_id", "8.0": "sentence_id"})
        self.assertEqual(get_prompt("7.0").schema, get_prompt("6.0").schema)  # same reply schema, so the two compare cleanly
        self.assertEqual(SYSTEM_PROMPT, get_prompt().build_system(None))
        with self.assertRaises(ValueError):
            get_prompt("9.9")
        with self.assertRaises(ValueError):
            TranslatorConfig(prompt_version="9.9").validate()

    def test_5_0_text_is_unchanged_so_the_comparison_is_against_the_real_previous_prompt(self):
        p = get_prompt("5.0").build_system(None)
        self.assertIn("ISL GRAMMAR CONVENTIONS (apply in this order)", p)
        self.assertIn("SUBJECT, OBJECT, VERB", p)
        self.assertEqual(len(p), 6697)  # exact size of the v5.0 text

    def test_6_0_text_is_unchanged(self):
        p = get_prompt("6.0").build_system(None)
        self.assertEqual(len(p), 7938)
        self.assertIn("# FINAL SELF-CHECK", p)


class PromptV7SuppliedByTheProjectOwner(unittest.TestCase):
    """Prompt 7.0 is the project's own system prompt. Check its ten sections are all present and intact."""

    P = SYSTEM_PROMPT

    def test_all_ten_sections_in_order(self):
        positions = [self.P.index(h) for h in (
            "## 1. Role and objective", "## 2. Input structure", "## 3. Sentence meaning and context",
            "## 4. Proper names and fingerspelling", "## 5. Preserve meaning-bearing information",
            "## 6. Gloss notation and English leakage", "## 7. Handle uncertainty without over-flagging",
            "## 8. Final self-check", "## 9. Required output format", "## 10. Quality objective")]
        self.assertEqual(positions, sorted(positions))

    def test_1_role_and_no_assumed_grammar_correspondence(self):
        for needle in ("This is a sentence-level translation task, not a word-by-word substitution task.",
                       "return a separate gloss result for every target sentence",
                       "Do not invent linguistic rules or assume that English grammar directly corresponds to ISL grammar."):
            self.assertIn(needle, self.P)
        for fixed_order in ("SUBJECT, OBJECT, VERB", "TIME FIRST", "apply in this order"):
            self.assertNotIn(fixed_order, self.P)

    def test_2_3_context_is_for_meaning_not_for_combining(self):
        for needle in ("Context is provided to resolve meaning, not to combine sentences into a single output.",
                       "Do not add information from neighboring sentences that is not part of the target sentence's meaning.",
                       "Preserve sentence boundaries and the original `sentence_id`.",
                       "translate the supported meaning without inventing missing information"):
            self.assertIn(needle, self.P)

    def test_3_pronoun_handling_forbids_noun_substitution_and_over_flagging(self):
        for needle in ('Do not automatically replace pronouns such as "he", "she", "they", "it", or "this" with the nouns they may refer to.',
                       "Resolve a pronoun's referent internally",
                       "flag the ambiguity for review",
                       "Do not flag a sentence merely because it contains a pronoun"):
            self.assertIn(needle, self.P)

    def test_4_proper_names(self):
        for needle in ("Do not automatically convert a complete proper name into one ordinary compound gloss label.",
                       "`FS:NAME`", "Keep titles, occupations, and roles separate from names", "Do not invent a lexical sign or name sign",
                       "Do not represent the entire phrase as `PROFESSOR_JONATHAN_YOUNG`", '"California State University"',
                       "A capitalized word at the beginning of a sentence is not automatically a proper name.",
                       '"June" or "Jainism"', "project-level notation, not a claim"):
            self.assertIn(needle, self.P)

    def test_5_meaning_bearing_information(self):
        for needle in ("Conditional statements and their consequences.", "Quantities, numbers, and comparisons.",
                       "Cause-and-effect relationships.", "Do not silently lose the condition",
                       "do not blindly insert the literal token `IF` into every output",
                       "Do not convert digits into words or discard them without a linguistic reason"):
            self.assertIn(needle, self.P)

    def test_6_notation_and_english_leakage(self):
        for needle in ("Do not mechanically retain English words such as `THAT`, `AND`, and `IT` as gloss labels.",
                       "Do not automatically delete these words either",
                       "Do not alternate between different representations of the same concept",
                       "`DRY_UP` and `WAY_OF_LIVING`", "Do not invent compound labels merely to preserve English word order."):
            self.assertIn(needle, self.P)

    def test_7_uncertainty_without_over_flagging(self):
        for needle in ("Distinguish genuine uncertainty from ordinary linguistic complexity.",
                       "Do not flag a sentence solely because:", "The translation differs from English word order.",
                       "It contains a word with multiple dictionary meanings when the context clearly resolves the intended meaning.",
                       "attempt a candidate translation even if some minor uncertainty remains",
                       "Never return an unexplained empty gloss."):
            self.assertIn(needle, self.P)

    def test_8_self_check_has_ten_questions_and_no_guarantee(self):
        check = self.P.split("## 8. Final self-check")[1].split("## 9.")[0]
        self.assertEqual(sum(f"\n{i}. " in check for i in range(1, 11)), 10)
        self.assertIn("Is a review flag genuinely necessary?", check)
        self.assertIn("It does not guarantee linguistic correctness.", check)

    def test_9_10_output_contract_and_quality_objective(self):
        for needle in ("Return only valid JSON matching the application's expected schema.", "`sentence_id`", "`source_text`",
                       "`candidate_gloss`", "`uncertainty_notes`", "`needs_human_review`", "Do not merge multiple sentence outputs.",
                       "Do not invent missing sentence identifiers or source text.",
                       "Do not include hidden reasoning or a chain-of-thought",
                       "Do not optimize for matching English word order, minimizing the number of gloss tokens, or appearing confident.",
                       "must not be represented as human-validated ISL until it has been reviewed by someone proficient in ISL"):
            self.assertIn(needle, self.P)

    def test_only_three_marked_contract_additions_were_made(self):
        self.assertEqual(self.P.lower().count("contract addition"), 3)
        self.assertIn("In this application (contract addition)", self.P)  # how input arrives here
        self.assertIn("one FS: token per word, for example FS:JONATHAN FS:YOUNG", self.P)  # multi-word names
        self.assertIn("`uncertain_parts` (contract addition required by the application schema", self.P)  # strict reply schema

    def test_verified_vocabulary_is_only_present_when_supplied(self):
        self.assertNotIn("Verified vocabulary reference", self.P)
        self.assertIn("If none appears, no verified vocabulary and no approved examples are available", self.P)
        p = build_system_prompt(frozenset({"HOME", "GO", "YOU"}))
        self.assertIn("## Verified vocabulary reference", p)
        self.assertIn('["GO", "HOME", "YOU"]', p)
        self.assertLess(p.index("## Verified vocabulary reference"), p.index("## 7. Handle uncertainty"))
        self.assertGreater(p.index("## Verified vocabulary reference"), p.index("## 6. Gloss notation"))
        self.assertEqual(build_system_prompt(frozenset({"B", "A"})), build_system_prompt(frozenset({"A", "B"})))


class PromptV6StillAvailable(unittest.TestCase):
    P = get_prompt("6.0").build_system(None)

    def test_no_fixed_word_order_recipe(self):
        self.assertNotIn("apply in this order", self.P)
        self.assertIn("do not mechanically reorder every sentence to subject-object-verb", self.P)

    def test_vocabulary_option(self):
        p = build_system_prompt(frozenset({"HOME", "GO"}), version="6.0")
        self.assertIn("# VOCABULARY (verified project vocabulary)", p)
        self.assertIn("# VOCABULARY", build_system_prompt(frozenset({"A"}), version="5.0"))


class PromptIsPartOfTheCacheKey(unittest.TestCase):
    def test_changed_prompt_or_vocabulary_never_reuses_old_answers(self):
        self.assertNotEqual(cache_key("m", "prompt A", "text", "id1"), cache_key("m", "prompt B", "text", "id1"))
        self.assertNotEqual(cache_key("m", "p", "text", "id1"), cache_key("m2", "p", "text", "id1"))
        self.assertNotEqual(cache_key("m", "p", "text", "id1"), cache_key("m", "p", "text", "id2"))
        self.assertNotEqual(cache_key("m", "p", "context A\n[id1] text", "id1"), cache_key("m", "p", "context B\n[id1] text", "id1"))
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "c.jsonl"
            Translator(TranslatorConfig(), FakeClient(), ResultStore(cache)).run([chunk("c1", "He is tall.")])
            same = FakeClient()
            Translator(TranslatorConfig(), same, ResultStore(cache)).run([chunk("c1", "He is tall.")])
            self.assertEqual(same.calls, [])  # identical prompt: served from cache
            with_vocab = FakeClient()
            Translator(TranslatorConfig(), with_vocab, ResultStore(cache), vocabulary=frozenset({"HE", "TALL"})).run(
                [chunk("c1", "He is tall.")])
            self.assertEqual(len(with_vocab.calls), 1)  # vocabulary changes the prompt: fresh request
            self.assertIn("## Verified vocabulary reference", with_vocab.calls[0]["system"])
            for older in ("6.0", "5.0"):
                other = FakeClient()
                Translator(TranslatorConfig(prompt_version=older), other, ResultStore(cache)).run([chunk("c1", "He is tall.")])
                self.assertEqual(len(other.calls), 1)  # a different prompt version never reuses another's answers


if __name__ == "__main__":
    unittest.main()
