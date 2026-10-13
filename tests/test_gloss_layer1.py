"""Layer 1 (LLM) tests for prompt 6.0.

What can and cannot be tested offline: a unit test cannot show that a real model OBEYS an instruction. These tests check
(a) that every requirement is stated in the prompt, (b) that the structured response contract is enforced, (c) that the
pipeline passes a model's output through faithfully, flags uncertainty honestly, and never "fixes" a reply behind the
model's back (no Layer 2 logic). Whether the model follows the prompt is measured separately with `compare-prompts` on
live outputs, and judged by an ISL-fluent reviewer.
"""

import unittest

from isl_gloss.config import TranslatorConfig
from isl_gloss.models import FAILED, TRANSLATED, Candidate
from isl_gloss.prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT, get_prompt
from isl_gloss.rules import RULE_SET_VERSION, RuleEngine
from isl_gloss.translator import Translator
from tests.gloss_helpers import FakeClient, chunk, echo_reply, make_row, parse_user, reply_from, results_json

P = get_prompt("6.0").build_system(None)  # these needles describe prompt 6.0; 7.0 is covered in test_gloss_prompt.py


def run(chunks, client, **cfg):
    t = Translator(TranslatorConfig(**cfg), client)
    return t, t.run(chunks)


def one(text, tokens, **row_kw):
    """Run one sentence through the pipeline with a scripted model reply; return (result, result_dict)."""
    client = FakeClient([reply_from(lambda label, t: make_row(label, t, tokens, **row_kw))])
    t, out = run([chunk("s1", text)], client)
    return out.results[0], out.results[0].to_dict(t.metadata())


def codes(result):
    return {w["code"] for w in result.validation["warnings"]}


# ----------------------------------------------------------------------------------------------------------------------
class Prompt60StatesEveryLayer1Requirement(unittest.TestCase):
    def test_1_proper_names_use_the_fs_name_convention(self):
        for needle in ("PROPER NAMES AND FINGERSPELLING", "project notation FS:NAME", "one token per word of the name",
                       "FS:JONATHAN FS:YOUNG", "not PROFESSOR_JONATHAN_YOUNG", "not CALIFORNIA_STATE_UNIVERSITY",
                       "Titles and roles", "PROFESSOR FS:JONATHAN FS:YOUNG", "Never invent a lexical sign or a name sign",
                       "verified sign label or an approved name sign", "FS:NAME is a project-level notation"):
            self.assertIn(needle, P)

    def test_2_conditions_and_their_consequences_are_preserved(self):
        for needle in ('("if", "unless", "whenever", "in case")', "keep the condition (IF, UNLESS, WHENEVER",
                       "the relationship between the condition and its consequence", "Do not silently omit it",
                       '"If it rains, we stay home."'):
            self.assertIn(needle, P)

    def test_3_negation_questions_time_quantities_cause_are_preserved(self):
        for needle in ("negation; conditions; questions; time references; quantities, numbers and comparisons; "
                       "cause-and-effect relationships", "BECAUSE or an equivalent", "question words and question meaning",
                       "every number and every name", "not necessarily every English word as a separate token"):
            self.assertIn(needle, P)

    def test_4_pronouns_and_referents_use_the_group_context(self):
        for needle in ("pronouns and their referents", 'who "he", "she", "it", "they", "this" or "that" refers to',
                       "ONE connected passage", "If a pronoun or reference cannot be resolved", "Do not guess",
                       "do not replace a pronoun with the name it refers to"):
            self.assertIn(needle, P)

    def test_5_every_sentence_gets_its_own_result(self):
        for needle in ("Return a SEPARATE result for every target sentence", "Never combine sentences",
                       "Never copy information from a neighbouring sentence", "sentence_id (exactly as given)",
                       "source_text (the sentence text exactly as given)"):
            self.assertIn(needle, P)

    def test_6_one_gloss_notation(self):
        for needle in ("one convention, used consistently", "UPPERCASE: letters A-Z, digits and underscores only",
                       "ONE token joined with underscores", "never alternate between DRY_UP and DRY UP",
                       "No explanations, commentary or reasoning inside the gloss", "Do not retain ordinary English words mechanically"):
            self.assertIn(needle, P)

    def test_7_never_empty_and_uncertainty_reported_outside_the_gloss(self):
        for needle in ("Do not return an empty candidate_gloss for an understandable sentence", "best-supported candidate",
                       "genuinely uninterpretable", "needs_human_review true and a concise reason in uncertainty_notes",
                       "never inside the gloss", "Do not invent a sign for them"):
            self.assertIn(needle, P)

    def test_8_final_self_check_for_omissions_additions_and_meaning_changes(self):
        self.assertIn("FINAL SELF-CHECK", P)
        check = P.split("FINAL SELF-CHECK")[1].split("# OUTPUT")[0]
        for needle in ("silently", "never write the checklist in your reply", "identify potential omissions, additions and meaning changes",
                       "comparing each candidate gloss with its source sentence", "does not guarantee that the gloss is correct ISL"):
            self.assertIn(needle, check)
        self.assertEqual(sum(f"\n{i}. " in check for i in range(1, 7)), 6)

    def test_9_no_oversimplified_grammar_rule(self):
        self.assertNotIn("SUBJECT, OBJECT, VERB", P)
        self.assertNotIn("apply in this order", P)
        self.assertIn("do not mechanically reorder every sentence to subject-object-verb", P)

    def test_the_prompt_is_honest_about_what_it_can_do(self):
        self.assertIn("CANDIDATE", P)
        self.assertIn("not linguistic gold", P)
        self.assertIn("a gloss records only part of it", P)


class ResponseContract(unittest.TestCase):
    ITEM = RESPONSE_SCHEMA["properties"]["results"]["items"]

    def test_each_target_sentence_returns_the_required_fields(self):
        self.assertEqual(set(self.ITEM["properties"]),
                         {"sentence_id", "source_text", "candidate_gloss", "uncertain_parts", "uncertainty_notes", "needs_human_review"})
        self.assertEqual(set(self.ITEM["required"]), set(self.ITEM["properties"]))
        self.assertEqual(self.ITEM["properties"]["candidate_gloss"], {"type": "ARRAY", "items": {"type": "STRING"}})
        self.assertEqual(self.ITEM["properties"]["needs_human_review"]["type"], "BOOLEAN")

    def test_the_sentence_id_field_is_pinned_to_the_ids_in_the_request(self):
        schema = get_prompt("6.0").schema_for(["a-s0001", "a-s0002"])
        self.assertEqual(schema["properties"]["results"]["items"]["properties"]["sentence_id"]["enum"], ["a-s0001", "a-s0002"])
        self.assertNotIn("enum", self.ITEM["properties"]["sentence_id"])  # the shared schema is not mutated

    def test_the_result_dict_carries_the_requested_fields_plus_the_existing_ones(self):
        r, d = one("We study hard.", ["WE", "STUDY", "HARD"])
        for key in ("sentence_id", "source_text", "candidate_gloss", "uncertainty_notes", "needs_human_review"):
            self.assertIn(key, d)
        for key in ("final_gloss", "applied_rules", "validation", "metadata", "group_id", "speaker_id", "review_status"):
            self.assertIn(key, d)  # nothing that already existed was dropped
        self.assertEqual((d["sentence_id"], d["source_text"], d["candidate_gloss"]), ("s1", "We study hard.", "WE STUDY HARD"))
        self.assertIsInstance(d["candidate_gloss"], str)  # no explanation can hide in it: it is the joined tokens only


class ProperNamesAndMeaningPassThroughUnchanged(unittest.TestCase):
    """The pipeline hands the model's candidate through. It must neither "fix" nor silently rewrite it."""

    def test_1_a_reply_that_follows_the_fs_name_convention_is_kept_exactly(self):
        r, d = one("Professor Jonathan Young invited us.", ["PROFESSOR", "FS:JONATHAN", "FS:YOUNG", "INVITE", "WE"])
        self.assertEqual(d["final_tokens"], ["PROFESSOR", "FS:JONATHAN", "FS:YOUNG", "INVITE", "WE"])
        self.assertEqual(d["applied_rules"], [])
        self.assertEqual(d["candidate_tokens"], d["final_tokens"])

    def test_2_a_conditional_reply_keeps_the_condition_and_its_consequence(self):
        r, d = one("If you pour water again, it flows the same way.", ["IF", "YOU", "POUR", "WATER", "AGAIN", "IT", "FLOW", "SAME"])
        self.assertEqual(d["final_tokens"][0], "IF")
        self.assertEqual(d["final_tokens"], d["candidate_tokens"])
        self.assertEqual(r.validation["status"], "pass")

    def test_3_negation_and_question_replies_are_kept(self):
        r, d = one("I do not understand.", ["I", "UNDERSTAND", "NOT"])
        self.assertEqual(d["final_tokens"], ["I", "UNDERSTAND", "NOT"])
        r, d = one("Where did you go yesterday?", ["YESTERDAY", "YOU", "GO", "WHERE"])
        self.assertEqual(d["final_tokens"], ["YESTERDAY", "YOU", "GO", "WHERE"])

    def test_no_layer_2_logic_was_added_a_violating_reply_is_not_corrected(self):
        # The model breaks the name convention AND drops the condition. Layer 1 reports what the model said; it does not
        # rewrite it. Detecting or correcting these is documented as future Layer 2 work.
        r, d = one("If Professor Jonathan Young comes, we stay.", ["PROFESSOR_JONATHAN_YOUNG", "COME", "WE", "STAY"])
        self.assertEqual(d["final_tokens"], ["PROFESSOR_JONATHAN_YOUNG", "COME", "WE", "STAY"])
        self.assertNotIn("IF", d["final_tokens"])  # nothing was inserted
        self.assertEqual(d["applied_rules"], [])
        layer2_codes = {"name_not_fingerspelled", "condition_may_be_lost", "cause_may_be_lost", "time_reference_may_be_lost"}
        self.assertFalse(codes(r) & layer2_codes)
        self.assertEqual(RuleEngine().active_ids, ["R-FMT-01", "R-ART-01", "R-SKIP-01"])  # the same three rules as before
        self.assertEqual(RULE_SET_VERSION, "0.1.0")


class ContextGroupsAndSeparateOutputs(unittest.TestCase):
    def test_4_a_context_dependent_pronoun_group_is_sent_as_one_request_with_separate_results(self):
        texts = ["Maria bought a laptop.", "She says it is fast.", "It replaced the old one."]
        client = FakeClient()
        chunks = [chunk(f"s{i}", t) for i, t in enumerate(texts, 1)]
        t, out = run(chunks, client)
        self.assertEqual(len(client.calls), 1)  # the whole passage in one request, so "She" and "it" can be resolved
        ctx, items = parse_user(client.calls[0]["user"])
        self.assertEqual(items, [("s1", texts[0]), ("s2", texts[1]), ("s3", texts[2])])
        self.assertEqual(len(out.results), 3)  # one result per sentence, never merged
        self.assertEqual([r.to_dict({})["sentence_id"] for r in out.results], ["s1", "s2", "s3"])

    def test_5_each_result_keeps_its_own_gloss_nothing_leaks_between_neighbours(self):
        gloss = {"s1": ["MARIA", "BUY", "LAPTOP"], "s2": ["SHE", "SAY", "FAST"], "s3": ["REPLACE", "OLD"]}
        client = FakeClient([reply_from(lambda label, text: gloss[label])])
        t, out = run([chunk("s1", "Maria bought a laptop."), chunk("s2", "She says it is fast."), chunk("s3", "It replaced the old one.")], client)
        self.assertEqual([r.candidate.tokens for r in out.results], [gloss["s1"], gloss["s2"], gloss["s3"]])
        self.assertEqual([r.final_tokens for r in out.results], [gloss["s1"], gloss["s2"], gloss["s3"]])

    def test_4_an_unresolved_reference_is_flagged_for_review_and_not_resolved_by_guessing(self):
        r, d = one("He apologised, but nobody believed him.", ["HE", "APOLOGISE", "BUT", "NOBODY", "BELIEVE", "HE"],
                   needs=True, parts=["he", "him"], notes="Referent of he/him not stated in the passage.")
        self.assertTrue(d["needs_human_review"])
        self.assertEqual(d["uncertain_parts"], ["he", "him"])
        self.assertIn("Referent", d["uncertainty_notes"])
        self.assertNotIn("JOHN", d["final_tokens"])  # pronouns stay pronouns
        self.assertEqual(r.validation["status"], "review_required")
        self.assertIn("llm_marked_uncertain", codes(r))


class ConsistentNotation(unittest.TestCase):
    def test_6_explanations_inside_a_gloss_token_are_flagged_by_the_existing_check(self):
        r, d = one("Water flows downhill.", ["water flows because of gravity", "FLOW"])
        self.assertIn("prose_in_gloss", codes(r))
        self.assertEqual(r.validation["status"], "review_required")

    def test_6_the_candidate_is_always_a_list_of_string_tokens(self):
        for bad in (123, [1, 2], {"a": 1}, None):
            row = make_row("s1", "Water flows.", ["X"])
            row["candidate_gloss"] = bad
            out = Translator(TranslatorConfig(), FakeClient([results_json([row])] * 10)).run([chunk("s1", "Water flows.")])
            self.assertEqual(out.results[0].status, FAILED, bad)
            self.assertIsNone(out.results[0].candidate)


class EmptyAndMalformedResponses(unittest.TestCase):
    def test_7_an_empty_gloss_for_an_understandable_sentence_is_a_contract_failure(self):
        r, d = one("The water flows in the same direction.", [], needs=False)
        self.assertEqual(r.status, TRANSLATED)
        self.assertEqual(r.validation["status"], "failed")
        self.assertIn("empty_gloss", codes(r))
        self.assertTrue(d["needs_human_review"])

    def test_7_an_uncertain_flag_is_not_an_excuse_for_an_empty_gloss_either_unless_uninterpretable(self):
        r, d = one("Water flows.", [], needs=True, notes="Hard.")
        # needs_human_review with an empty gloss IS the explicit "uninterpretable" result, so it is review, not failure
        self.assertEqual(r.validation["status"], "review_required")

    def test_7_a_genuinely_uninterpretable_sentence_gives_an_explicit_review_result_with_a_reason(self):
        r, d = one("Om Ooh Hello Done frio.", [], needs=True, notes="Garbled ASR text; no interpretable meaning.")
        self.assertEqual(r.validation["status"], "review_required")  # honest uncertainty, not a technical failure
        self.assertNotEqual(r.validation["status"], "failed")
        self.assertIn("llm_review_required", codes(r))
        self.assertEqual(d["uncertainty_notes"], "Garbled ASR text; no interpretable meaning.")
        self.assertEqual(d["final_gloss"], "")
        self.assertTrue(d["needs_human_review"])
        self.assertEqual(d["candidate"] if "candidate" in d else d["llm"]["status"], "review_required")

    def test_7_malformed_replies_are_reported_as_technical_failures_never_as_a_translation(self):
        for text in ("", "not json at all", '{"results": "nope"}', '{"results": [1, 2]}', "```"):
            client = FakeClient([text] * 10)
            out = Translator(TranslatorConfig(), client).run([chunk("s1", "Water flows.")])
            r = out.results[0]
            self.assertEqual(r.status, FAILED, text)
            self.assertEqual((r.final_tokens, r.candidate), ([], None))
            self.assertIn(r.error["kind"], ("malformed_response", "missing_result"))
            self.assertTrue(r.needs_human_review)

    def test_7_a_reply_missing_a_sentence_is_repaired_once_then_failed_not_invented(self):
        client = FakeClient([results_json([]), results_json([])])
        out = Translator(TranslatorConfig(), client).run([chunk("s1", "Water flows.")])
        self.assertEqual(out.results[0].status, FAILED)
        self.assertEqual(out.results[0].error["kind"], "missing_result")


class SentenceIdsAndSourceText(unittest.TestCase):
    def test_8_ids_and_original_source_text_are_preserved_exactly_and_in_order(self):
        texts = ["First thing here.", "Second thing here.", "Third thing here."]
        client = FakeClient()
        t, out = run([chunk(f"doc-s{i:04d}", x) for i, x in enumerate(texts, 1)], client)
        d = [r.to_dict(t.metadata()) for r in out.results]
        self.assertEqual([x["sentence_id"] for x in d], ["doc-s0001", "doc-s0002", "doc-s0003"])
        self.assertEqual([x["source_text"] for x in d], texts)
        self.assertTrue(out.assembly["passed"])

    def test_8_source_text_in_the_result_is_the_original_even_when_words_were_cut_before_the_model(self):
        text = "Hmm, we study the Jane tradition."
        j = text.index("Jane")
        c = chunk("s1", text, skip_spans=[{"start": 0, "end": 3, "text": "Hmm", "reason": "expression_cue"},
                                          {"start": j, "end": j + 4, "text": "Jane", "reason": "skip_term"}])
        client = FakeClient()
        t = Translator(TranslatorConfig(), client)
        d = t.run([c]).results[0].to_dict(t.metadata())
        self.assertEqual(d["source_text"], text)  # the original sentence
        self.assertEqual(d["llm_input_text"], "we study the tradition.")  # what the model was actually given
        self.assertEqual(parse_user(client.calls[0]["user"])[1], [("s1", "we study the tradition.")])

    def test_8_the_models_copy_of_the_source_text_is_checked_against_the_input(self):
        r, d = one("We study hard today.", ["WE", "STUDY"])
        self.assertEqual(d["llm"]["echoed_source_text"], "We study hard today.")
        self.assertNotIn("source_text_mismatch", codes(r))
        # whitespace differences alone are tolerated
        client = FakeClient([reply_from(lambda l, t: make_row(l, "We  study hard\ttoday.", ["WE", "STUDY"]))])
        self.assertNotIn("source_text_mismatch", codes(run([chunk("s1", "We study hard today.")], client)[1].results[0]))

    def test_8_a_different_or_missing_copy_means_the_gloss_may_belong_to_another_sentence(self):
        client = FakeClient([reply_from(lambda l, t: make_row(l, "A completely different sentence.", ["X"]))])
        r = run([chunk("s1", "We study hard today.")], client)[1].results[0]
        self.assertIn("source_text_mismatch", codes(r))
        self.assertEqual(r.validation["status"], "review_required")
        self.assertTrue(r.needs_human_review)
        row = make_row("s1", "We study hard today.", ["WE"])
        del row["source_text"]
        r = run([chunk("s1", "We study hard today.")], FakeClient([results_json([row])]))[1].results[0]
        self.assertIn("source_text_not_echoed", codes(r))

    def test_8_a_mistyped_sentence_id_cannot_attach_a_gloss_to_the_wrong_sentence(self):
        client = FakeClient([reply_from(lambda l, t: make_row(l.upper() + "x", t, ["X"]))] * 10)
        out = run([chunk("s1", "Water flows.")], client)[1]
        self.assertEqual((out.results[0].status, out.results[0].candidate), (FAILED, None))


class AmbiguousOrUnsupportedInputNeedsReview(unittest.TestCase):
    def test_9_uncertainty_is_reported_through_fields_never_inside_the_gloss(self):
        r, d = one("The old man the boats.", ["OLD", "PEOPLE", "BOAT", "OPERATE"], needs=True, parts=["man (verb?)"],
                   notes="Garden-path sentence; 'man' is a verb here.")
        self.assertTrue(d["needs_human_review"])
        self.assertEqual(d["uncertain_parts"], ["man (verb?)"])
        self.assertNotIn("?", d["candidate_gloss"])
        self.assertNotIn("Garden", d["candidate_gloss"])
        self.assertEqual(r.validation["status"], "review_required")

    def test_9_a_confident_clean_translation_does_not_ask_for_review(self):
        r, d = one("We study hard.", ["WE", "STUDY", "HARD"])
        self.assertFalse(d["needs_human_review"])
        self.assertEqual(r.validation["status"], "pass")

    def test_9_needs_human_review_also_covers_validation_findings_the_model_did_not_flag(self):
        r, d = one("I did not see 50 rupees.", ["I", "SEE", "RUPEES"], needs=False)  # lost negation + number, model confident
        self.assertEqual(r.candidate.needs_human_review, False)  # the model's own flag
        self.assertTrue(d["needs_human_review"])  # the aggregate: look at this one

    def test_9_uncertain_or_unsupported_input_is_never_silently_dropped(self):
        r, d = one("Frio done hello.", [], needs=True, notes="Unsupported.")
        self.assertEqual(r.status, TRANSLATED)  # the sentence is in the output with its reason, not omitted
        self.assertEqual(d["sentence_id"], "s1")

    def test_candidate_status_is_derived_consistently(self):
        self.assertEqual(Candidate(tokens=["A"]).status, "ok")
        self.assertEqual(Candidate(tokens=["A"], needs_human_review=True).status, "uncertain")
        self.assertEqual(Candidate(tokens=[], needs_human_review=True).status, "review_required")
        self.assertTrue(Candidate(tokens=["A"], uncertain=True).needs_human_review)  # older callers
        self.assertTrue(Candidate(tokens=[], status="review_required").needs_human_review)

    def test_replies_in_the_previous_5_0_format_still_parse(self):
        rows = [{"n": 1, "gloss": ["WATER"], "status": "uncertain", "uncertain_parts": ["water"], "notes": "n"}]
        client = FakeClient([results_json(rows)])
        t = Translator(TranslatorConfig(prompt_version="5.0"), client)
        r = t.run([chunk("s1", "Water flows.")]).results[0]
        self.assertEqual((r.candidate.status, r.candidate.needs_human_review, r.candidate.notes), ("uncertain", True, "n"))


class ConfigurationIsRecordedAndUnchanged(unittest.TestCase):
    def test_provider_and_model_are_unchanged_and_reasoning_effort_is_configurable(self):
        cfg = TranslatorConfig()
        self.assertEqual((cfg.provider, cfg.model, cfg.reasoning_effort), ("groq", "openai/gpt-oss-120b", "low"))
        for effort in ("low", "medium", "high"):
            self.assertEqual(TranslatorConfig(reasoning_effort=effort).reasoning_effort, effort)
        with self.assertRaises(ValueError):
            TranslatorConfig(reasoning_effort="max").validate()

    def test_the_llm_configuration_is_written_into_every_result(self):
        t = Translator(TranslatorConfig(reasoning_effort="medium", prompt_version="5.0"), FakeClient())
        d = t.run([chunk("s1", "We study hard.")]).results[0].to_dict(t.metadata())
        self.assertEqual(d["metadata"]["llm_config"], {
            "provider": "groq", "model": "openai/gpt-oss-120b", "reasoning_effort": "medium", "temperature": 0.0,
            "max_output_tokens": 8192, "prompt_version": "5.0", "thinking_budget": None})


if __name__ == "__main__":
    unittest.main()
