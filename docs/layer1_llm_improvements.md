# Layer 1 (LLM) improvements: prompt 6.0

Date: 13 Oct 2026. Scope: the LLM translation layer only. Cleaning, sentence chunking, context grouping and the provider
integration (Groq, `openai/gpt-oss-120b`) are unchanged. **No Layer 2 logic was added** (see section 8).

## 1. Files

New: `src/isl_gloss/prompt_eval.py` (evaluation tool, observation only), `evaluation/layer1_inputs.json` (11 representative
passages, 41 sentences), `docs/layer1_llm_improvements.md`, `docs/layer2_future_requirements.md`,
`tests/test_gloss_layer1.py` (40 tests), `tests/test_gloss_prompt_eval.py` (10 tests).

Modified: `src/isl_gloss/prompt.py` (prompt registry: 5.0 frozen, 6.0 new default), `translator.py` (requests keyed by sentence id,
echo check, generation settings in the cache key, `llm_config` metadata), `models.py` (`Candidate.needs_human_review`,
`echoed_source_text`; result fields), `config.py` (`prompt_version`), `store.py` (cache key, new fields), `export.py` (review
sheet columns), `cli.py` (`--prompt-version`, `compare-prompts`), `validate.py` (interprets the model's review flag: an empty
gloss with `needs_human_review` is "review required", without it a contract failure), tests updated for the new format.

Reverted (added last turn, out of scope here): rule `R-FS-01`, validator checks `name_not_fingerspelled`,
`condition_may_be_lost`, `cause_may_be_lost`, `time_reference_may_be_lost`. `rules.py` is back to rule-set `0.1.0`.

## 2. The prompt (version 6.0, system prompt)

Location: `src/isl_gloss/prompt.py` (`_V6_HEAD`, `_V6_VOCAB`, `_V6_TAIL`; `get_prompt("6.0")`). Full text, generated from the code:

```text
You are an expert Indian Sign Language (ISL) glossing assistant. You convert English sentences from transcripts of spoken talks into candidate ISL gloss. Your output is a CANDIDATE that a fluent ISL signer will review, so stay faithful to the meaning of the source and do not embellish. These instructions are mandatory for every request.

# WHAT TO PRODUCE
- Translate the MEANING and grammatical intent of each target sentence into candidate ISL gloss. Do not map English words to gloss labels one by one, and do not assume that an English sentence can be converted by a fixed word-order recipe (for example, do not mechanically reorder every sentence to subject-object-verb). ISL has its own grammar, and a gloss records only part of it: facial expression, head movement and use of space are not written in a gloss. Use sign-language-appropriate structure where you are confident; where you are not, stay close to the meaning-bearing order of the source and say so in uncertainty_notes.
- Use the whole group of sentences you are given to interpret each target sentence, but return a separate candidate gloss for every target sentence.
- Do not invent signs or glosses for words or interpretations you cannot support. Flag ambiguity and uncertainty; never guess silently.

# GLOSS NOTATION (one convention, used consistently throughout your reply)
- A candidate gloss is an ordered array of tokens. Tokens are UPPERCASE: letters A-Z, digits and underscores only. A sign that needs several English words is ONE token joined with underscores (THANK_YOU, HOW_MANY, GOOD_MORNING, WAY_OF_LIVING). Always write such a label the same way: never alternate between DRY_UP and DRY UP, or between two spellings of the same concept.
- Never output lowercase, punctuation or empty strings. No explanations, commentary or reasoning inside the gloss.
- Do not retain ordinary English words mechanically. THAT, AND, IT (as an empty subject), THERE, OF, TO, the articles, the verb "be" and similar words become a label only when their meaning is needed in this sentence according to the context and to ISL guidance. Equally, do not delete a word merely because such words are often unsigned: decide by meaning.
- Keep personal pronouns that refer to a person or thing distinct (I, YOU, HE, SHE, THEY, WE); do not merge them, and do not replace a pronoun with the name it refers to.

# PROPER NAMES AND FINGERSPELLING
- Identify proper names: people, places, institutions and other named entities. Represent a name that should be fingerspelled with the project notation FS:NAME, one token per word of the name: "Jonathan Young" becomes FS:JONATHAN FS:YOUNG. Do not write a name as an ordinary compound sign label (not PROFESSOR_JONATHAN_YOUNG, not CALIFORNIA_STATE_UNIVERSITY, not FS:JONATHAN_YOUNG).
- Titles and roles (Professor, Doctor, Lord, King, Teacher) stay separate from the name where appropriate: PROFESSOR FS:JONATHAN FS:YOUNG.
- Never invent a lexical sign or a name sign for a name you do not know. If a verified sign label or an approved name sign is given in the VOCABULARY section of this message, follow that instead of fingerspelling; if there is no such section, treat no name as having a verified sign.
- FS:NAME is a project-level notation. A downstream system will convert it into a fingerspelling sequence.

# PRESERVE MEANING-BEARING INFORMATION
Keep what the sentence means, not necessarily every English word as a separate token: negation; conditions; questions; time references; quantities, numbers and comparisons; cause-and-effect relationships; and pronouns and their referents where they matter for the meaning.
- If a sentence has a condition ("if", "unless", "whenever", "in case"), the gloss must keep the condition (IF, UNLESS, WHENEVER, ...) and the relationship between the condition and its consequence. Do not silently omit it.
- Keep cause-and-effect words (BECAUSE or an equivalent), question words and question meaning, negation, every number and every name.
- A word is not unimportant just because it is a function word in English.
Examples (they illustrate notation and preserved meaning only; the token order in them is NOT a word-order rule, and they are not linguistic gold):
"Professor Jonathan Young invited us." -> ["PROFESSOR","FS:JONATHAN","FS:YOUNG","INVITE","WE"]
"If it rains, we stay home." -> ["IF","RAIN","WE","STAY","HOME"]
"I do not understand." -> ["I","UNDERSTAND","NOT"]

# INPUT
The user message is plain text. Each target sentence is on its own line in the form `[sentence_id] sentence text`. The target sentences are consecutive sentences from one talk and together form ONE connected passage. The message may also contain read-only context blocks headed "Earlier speech" (before the target sentences) and "Following speech" (after them). Context blocks have no ids and are never to be translated or output.

# HOW TO USE THE PASSAGE
- Use the whole passage, and the context blocks, only to understand each target sentence: who "he", "she", "it", "they", "this" or "that" refers to, what an incomplete sentence is continuing, how a sentence relates to its neighbours.
- Return a SEPARATE result for every target sentence, with a candidate gloss that contains only that sentence's own content. Never combine sentences. Never copy information from a neighbouring sentence or from a context block into a gloss.
- If a pronoun or reference cannot be resolved from the passage, say so in uncertainty_notes and set needs_human_review to true. Do not guess.

# NEVER EMPTY, ALWAYS HONEST ABOUT UNCERTAINTY
- Do not return an empty candidate_gloss for an understandable sentence just because it is hard. Give the best-supported candidate.
- A fragment (for example one that continues the previous sentence) is glossed from the words it contains, using the passage to interpret it, and marked for review.
- If part of a sentence cannot be translated confidently, gloss the rest, list the doubtful words or tokens in uncertain_parts, explain briefly in uncertainty_notes, and set needs_human_review to true. Do not invent a sign for them.
- Only if the whole sentence is genuinely uninterpretable, return an empty candidate_gloss, needs_human_review true and a concise reason in uncertainty_notes.
- Uncertainty about the translation is NOT a failure and must be reported through needs_human_review, uncertain_parts and uncertainty_notes, never through malformed or missing output, and never inside the gloss.

# FINAL SELF-CHECK (do this silently before replying; never write the checklist in your reply)
Before returning your result, identify potential omissions, additions and meaning changes by comparing each candidate gloss with its source sentence:
1. Is there exactly one result for every target sentence, with sentence_id and source_text copied exactly?
2. Is every proper name written as FS:NAME, one token per word, with titles separate?
3. Are the conditions, negation, questions, time references, quantities, comparisons, cause-and-effect relationships and meaningful pronouns of the source still represented?
4. Do all labels follow the one notation above, with no mechanically retained English words and no explanatory prose in the gloss?
5. Was anything added, omitted or changed in meaning, or copied in from a neighbouring sentence?
6. Is every doubtful translation flagged through needs_human_review, uncertain_parts and uncertainty_notes?
This self-check is an instruction to improve consistency. It does not guarantee that the gloss is correct ISL.

# OUTPUT
Reply with JSON only, matching the schema: a result for each target sentence with sentence_id (exactly as given), source_text (the sentence text exactly as given), candidate_gloss (array of tokens), uncertain_parts (array of strings, empty if none), uncertainty_notes (at most 25 words, empty if none) and needs_human_review (true or false). Do not include the checklist or any commentary outside uncertainty_notes.
```

The user message contains only the target sentences, one per line as `[sentence_id] text`, plus optional read-only
"Earlier speech" / "Following speech" context blocks without ids. No speaker labels, timestamps or titles.

## 3. Response schema (before and after)

| | Prompt 5.0 (before) | Prompt 6.0 (now) |
|---|---|---|
| Key of each result | `n` (1..N within the request) | `sentence_id` (the real id; the schema restricts it, per request, to exactly the ids sent, so a strict decoder cannot return a mistyped id) |
| Source text | not returned | `source_text` (copy of the sentence; compared with the input, mismatch gives warning `source_text_mismatch`, absence `source_text_not_echoed`) |
| Gloss | `gloss` | `candidate_gloss` (array of tokens; no commentary allowed in it) |
| Uncertainty | `status` ok/uncertain/review_required, `notes` | `needs_human_review` (boolean), `uncertainty_notes` (separate from the gloss) |
| Kept | `uncertain_parts` | `uncertain_parts` |

Rules of the contract: one identifiable result per sentence; an empty gloss is accepted only with `needs_human_review: true` and a
reason (otherwise validation fails the sentence as a contract violation); a missing result is re-requested once, then recorded as
failed, never invented. Replies in the 5.0 format and cached 5.0 data still parse. Per-sentence result files gained
`needs_human_review` (model flag OR any validation finding), `uncertain_parts`, `llm_status` and
`metadata.llm_config` (provider, model, reasoning effort, temperature, max output tokens, prompt version). `chunk_id`/`sentence_id`
and the other existing fields are unchanged. The cache key now includes the generation settings, so a result cached at low reasoning
effort is never served to a medium run.

## 4. Tests (259 pass: `python -m unittest discover -s tests -t .`)

`tests/test_gloss_layer1.py` covers the nine requested categories: (1) names / `FS:NAME`, (2) conditionals, (3) negation and questions,
(4) context-dependent pronouns, (5) multi-sentence groups with separate outputs, (6) consistent notation, (7) empty and malformed
replies, (8) sentence-id and source-text preservation, (9) ambiguous or unsupported input needing human review; plus the response
contract, configuration recording and a boundary test proving no Layer 2 correction happens.
**What these tests can and cannot show:** they verify that each requirement is stated in the prompt, that the contract is enforced, and
that the pipeline passes the model's output through faithfully without "fixing" it. A unit test cannot show that a real model obeys an
instruction; that is what the live comparison (section 5) observes. `test_gloss_prompt_eval.py` tests the comparison tool.

## 5. Comparison: prompt 5.0 vs 6.0 on the same inputs (live)

Setup: Groq `openai/gpt-oss-120b`, reasoning effort **medium** for both, temperature 0, same cleaning, same sentences and groups
(11 passages, 41 sentences, one request per group), run 13 Oct 2026. All candidate outputs are kept in
`data/translations/prompt_comparison/` (`results_prompt_5.0.json`, `results_prompt_6.0.json`, `comparison.json`,
`comparison_sheet.csv` for the ISL reviewer). Both versions translated 41/41 sentences with no failures; all 41 `source_text`
copies matched in 6.0. One sample, one run per prompt. **This is observation, not evidence of improved accuracy.**

| Observed by the automatic detectors | 5.0 | 6.0 |
|---|---|---|
| Sentences flagged by the model for review | 5 | 11 |
| Sentences with `needs_human_review` (model flag or validation) | 8 | 12 |
| Sentences with at least one detector hit | 11 | 7 |
| Name not in `FS:` form (detector) | 8 | 5 |
| Condition dropped | 0 | 0 |
| Negation, cause, time dropped | 0 | 0 |
| Number "dropped" (see caveat) | 2 | 0 |
| Empty gloss | 0 | 1 (flagged for review, with reason) |
| Inconsistent multi-word notation | 1 | 1 |
| English function-word tokens (THAT, AND, IT, TO, WILL, WOULD, ...) | 13 | 8 |

Caveats about these numbers: the detectors are crude. The name detector also fires on "June", "Jainism", "Ahimsa" and "Truth", so most
hits in both columns are false alarms or debatable. "Number dropped" counted `FIVE` and `TWO` (digits written as words) as a drop, which
is arguable. The function-word list is short; 6.0 still retains prepositions such as TO, AT, FROM, FOR. A lower count is not evidence
of a better translation.

**Observed differences, by failure type** (outputs are in the sheet):

* *Names:* 6.0 fingerspelled "California State University" and "Peter" (5.0 wrote CALIFORNIA STATE UNIVERSITY as separate ordinary
  labels in one sentence and `PETER` plain). Both fingerspelled people and places, and both wrote `JAINISM` as a plain label.
* *Conditions (IF, UNLESS, WHENEVER) and BECAUSE:* kept by both. The earlier dropped-`IF` failure (seen with the older prompt at low effort)
  did not recur for either prompt in this run, so **this sample gives no evidence of a change** on conditionals.
* *Empty gloss:* 5.0 never returned one. For the garbled "Om Ooh Hello Done frio" 5.0 produced a gloss (`OM HELLO DONE FS:FRIO`);
  6.0 returned an empty gloss with `needs_human_review` and the reason "Uninterpretable garbled sentence", which is the designed behaviour.
  The ASR fragment "an impression in your mind." got a gloss in both.
* *Uncertainty:* 6.0 flagged the ambiguous "he"/"him" in the John/Peter passage and "duck"; 5.0 flagged only "duck". 6.0 also flagged
  11 of 41 sentences, some for word-sense doubts ("marked", "splits", "left back"): possible over-flagging that adds reviewer workload.
* *Pronouns:* **6.0 deviated from its own instruction.** It replaced pronouns with their referents ("SHE SAY LAPTOP VERY FAST",
  "LAPTOP REPLACE OLD LAPTOP", "her brother" became `FS:MARIA BROTHER`), although the prompt says not to. 5.0 kept SHE/IT. Whether
  resolving the referent is linguistically better is a question for the reviewer; it is not what the prompt asked for.
* *Word order:* with the fixed-order rules removed, 6.0 output follows English order more closely (for example `WE NOT GO OUTSIDE
  UNLESS RAIN STOP`, `YESTERDAY` placed last in one sentence). Whether that is more or less like real ISL was not measured. It is a
  risk of removing the ordering guidance that only a signer can assess.
* *Notation:* neither version was fully consistent (for example `DRY_UP` / `DRY UP` / `DRY`, `FLOW_OVER` / `FLOW OVER`).
* *Alignment:* no sentence-id, duplicate, or source-text problems in 6.0 (41/41 echoes matched).
* *Cost:* one extra HTTP attempt for 6.0 (12 requests vs 11), most likely a retry; no result was affected. The 6.0 prompt is longer
  (about 1,990 tokens; 5.0 was about 1,670), which counts against the free token limits.

## 6. Reasoning effort

`reasoning_effort` stays configurable (`--reasoning-effort`, config file) and is recorded in every result. The earlier observation that
medium beat low came from 10 sentences and one run; it is an observation, not a finding. The comparison above used medium for both prompts.

## 7. Known limitations

* No ISL-fluent review has happened. Nothing here shows the glosses are correct ISL, and `FS:NAME` is a project notation, not a universal convention.
* One small sample, one run per prompt, temperature 0 but still a stochastic model; results can differ between runs and effort levels.
* The 6.0 prompt asks for no pronoun substitution, yet the model substituted referents; instructions are not guarantees.
* Notation consistency, mechanically retained English words and pronoun handling are not checked by the pipeline (that would be Layer 2).
* Removing the fixed word-order guidance may push output toward English order; unmeasured.
* Over-flagging in 6.0 may raise review workload; a reviewer should decide whether the flag threshold is useful.
* `Jainism`, religions, months and domain terms have no agreed policy for `FS:`; the prompt leaves it to the model.
* The comparison detectors are crude (section 5 caveats). The comparison tool is evaluation tooling, not part of the pipeline.

## 8. Confirmation: no Layer 2 implementation was added

No new rule, rule engine, rule-based correction (names, conditions, negation), post-rule validation or transformation logging was added.
The pipeline's rule pass is the pre-existing one (`R-FMT-01`, `R-ART-01`, `R-SKIP-01`, three rules off by default, version `0.1.0`) and was
not modified. A test (`test_no_layer_2_logic_was_added_a_violating_reply_is_not_corrected`) shows a reply that breaks the name
convention and drops the condition passes through unchanged and no new warning codes exist. The comparison tool's detectors
only observe model output; they never change a gloss and do not run during translation. Future Layer 2 requirements are in
[layer2_future_requirements.md](layer2_future_requirements.md).

## 9. Prompt 7.0 (the project's own system prompt) and a three-way comparison

On 13 Oct 2026 the project owner supplied a complete system prompt. It is now **version 7.0 and the default**; 5.0 and 6.0 remain
selectable (`--prompt-version`). The text is the supplied prompt verbatim plus three marked "contract additions": how input arrives in
this application, the `uncertain_parts` field the strict reply schema needs, and one `FS:` token per word of a multi-word name
(`FS:JONATHAN FS:YOUNG`; the supplied text did not say how to write multi-word names). The reply schema is the same as 6.0. The prompt
is about 2,730 tokens (6.0: about 1,990, 5.0: about 1,670; so 7.0 is about 38% longer than 6.0), which counts against free-tier token limits.
Note: the supplied prompt repeatedly says to follow "the supplied ISL guidance" and "verified vocabulary". None has been supplied, so the
model improvises on conditionals, pronoun representation and notation.

Same live setup as section 5 (11 passages, 41 sentences, medium reasoning effort, Groq `openai/gpt-oss-120b`, identical grouping). The 5.0 and 6.0 rows
come from the cached outputs of the first run (same settings), so they are the same answers as before. All outputs are kept in
`data/translations/prompt_comparison/` (`results_prompt_5.0|6.0|7.0.json`, `comparison.json`, `comparison_sheet.csv`;
the earlier two-way summary is saved as `earlier_5.0_vs_6.0_*`). One run per prompt; observation only.

| Observed | 5.0 | 6.0 | 7.0 |
|---|---|---|---|
| Sentences flagged by the model for review | 5 | 11 | **4** |
| Empty glosses | 0 | 1 | **4** |
| Sentences with at least one detector hit | 12 | 9 | 14 |
| Name not in `FS:` form / compound name label (detector) | 8 / 0 | 5 / 0 | 6 / **1** |
| Condition marker dropped | 0 | 0 | **1** |
| Person pronoun replaced by a noun (detector) | 1 | 2 | 1 |
| English function-word tokens | 13 | 8 | 6 |
| Gloss tokens (41 sentences) | 253 | 252 | 218 |

Detector caveats from section 5 still apply (the name detector fires on "June", "Jainism", "Ahimsa"; counts are not accuracy).

**Observed with 7.0, relative to 6.0:**

* *Over-flagging reduced as intended:* 4 sentences flagged instead of 11, and the flags are about real problems.
* *But three understandable sentences came back with an EMPTY gloss and a review flag:* "John told Peter that he would be late.",
  "He apologised, but nobody believed him." and "I saw her duck." 6.0 glossed all three and flagged them. The prompt says to attempt a
  candidate when the sentence is understandable, so this goes against its own instruction. (The fourth empty gloss, the garbled
  "Om Ooh Hello Done frio", is the intended case.) Empty plus flag is reported honestly as review-required, not as a failure, but the
  sentence has no candidate.
* *Names:* "California State University" came out as one compound label `CALIFORNIA_STATE_UNIVERSITY` in the long real sentence (the
  exact example the prompt warns about; 6.0 fingerspelled it in that sentence), and "Prabhu" was written plain. In the short names passage 7.0 fingerspelled it correctly.
  `JAINISM` and `JUNE` are plain labels in all versions (the prompt tells the model to classify by meaning, so that may be correct).
* *Conditions:* `WHENEVER` was dropped in "...whenever water splits from a glass..." (the only condition loss in any run). IF, UNLESS and WHENEVER were kept elsewhere.
* *Dropped content:* the first-person pronoun and "truly" disappeared in "I truly believe that Jainism is Not a religion" (`BELIEVE JAINISM NOT
  RELIGION`) and "I" in "...I would like to appreciate...". Shorter output (218 tokens vs 252) is not evidence of a better translation.
* *Pronouns:* better for "her" (kept `HER`), still replaced "it" with nouns (`SHE SAY LAPTOP VERY FAST`, `BUT JAINISM PHILOSOPHY...`).
* *Notation:* mixed. `DRY_UP` was used consistently in one passage and `DRY` elsewhere; 7.0 invented `IMPRESS` and `STUDENT-PL`, wrote `MORE THAN`
  as two tokens, and replaced "Janmotsav" with the paraphrase `BIRTH_ANNIVERSARY`.
* *Word order:* English-like in many sentences, as with 6.0; whether that is good ISL is unmeasured.

**Conclusion allowed by this evidence:** 7.0 flags less, and it still keeps pronouns like "her". But it also gave up on three understandable sentences,
dropped some meaning-bearing content, and was less consistent on the institution name. There is no basis here to call 7.0 better or worse overall:
that needs an ISL-fluent reviewer on `comparison_sheet.csv`. Options that would address the empty-gloss behaviour (for the prompt owner to
decide, not applied): state explicitly that an ambiguous pronoun is never a reason for an empty gloss (keep the pronoun, gloss, flag); re-run the comparison at other reasoning efforts and a second time to see
how stable these differences are.
