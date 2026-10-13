# Layer 2 evaluation: LLM candidate vs spaCy candidate

**Status: unreviewed.** Candidates are machine-generated and UNREVIEWED. Technical checks and warning counts say nothing about linguistic accuracy. No ISL-fluent reviewer has judged either candidate, so it is not known whether combining the two approaches improves the gloss.

## Inputs
* Layer 1 file: `data\translations\prompt_comparison\results_prompt_7.0.json` (prompt 7.0; unchanged by this run: True)
* spaCy 3.8.16, en_core_web_sm 3.8.0, nltk 3.10.3; reference commit `f5d9484`
* LLM requests made by Layer 2: 0; Candidate A field: `candidate_tokens`
* Selection policy: {"policy_version": "2.0.0", "primary_candidate": "LLM", "spacy_fallback_when_llm_fails": true, "fallback_on_llm_abstention": false, "treat_identical_as_agreement": true, "allow_check_based_preference": false, "preference_codes": ["number_not_found"]}

## Counts
* Sentences: 41
* spaCy: 41 produced a gloss, 0 empty, 0 failed, 0 dependency unavailable
* LLM (Layer 1): 37 produced a gloss, 4 empty, 0 failed, 0 nothing to translate
* Technically unusable: LLM 4, spaCy 0
* Both usable: 37; of those, 37 differ and 0 are identical after normalising case/FS:/separators

## Selection
The LLM is the primary candidate; spaCy is an independent alternative that is always generated and kept.
* Carried forward: 37 of 41. LLM kept as primary by policy default: 37 (a default, not a finding that it beats spaCy); the two agree: 0
* spaCy selected: 0 (technical fallback after an LLM failure: 0)
* LLM declined on purpose, spaCy offered but NOT substituted: 3
* LLM unusable and spaCy rejected by the fallback check: 1; neither usable: 0
* Selections resting on human-reviewed evidence: 0
* Flagged for human review: 6 of 41 (the LLM's own flags, warnings, or an LLM failure). Every row is still in the reviewer sheet, because nothing is reviewed yet.
* Status counts: {'llm_abstained_alternative_available': 3, 'llm_failed_alternative_rejected': 1, 'llm_primary': 37}; rules: {'R-SEL-06': 37, 'R-SEL-07': 3, 'R-SEL-08': 1}; basis: {'none': 4, 'policy_default_primary': 37}

## Heuristic warnings (not errors)
* LLM: {'possible_condition_not_represented': 1, 'possible_name_not_found': 1}
* spaCy: {'possible_name_not_found': 1}
* Raised for both: {}

## What this does and does not show
* It shows how often each independent translator produced a technically usable candidate and how often they agree.
* It does **not** show that either candidate is correct, nor that combining them improves quality: no ISL-fluent reviewer has judged them.
* Fill in `candidates_side_by_side.csv` (reviewer columns) to establish that.
