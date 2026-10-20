"""Signable-gloss step: synonym fallback, name spelling, repeat merging, vocabulary availability and the table validator."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from isl_gloss.signable import consistently_capitalised, coverage, normalize_tokens, token_form
from isl_gloss.signable_cli import main as signable_main, process_file
from islrtc_index.vocab import Vocabulary, load_synonym_table, parse_yap_tokens, validate_synonyms


def asset(aid, label_norm, status="source_label", aliases=()):
    return {"asset_id": aid, "label_norm": label_norm, "status": status, "aliases": [{"text": a, "verified": True} for a in aliases]}


YAP = ["HELLO", "GOOD", "WE", "TELL", "PHONE", "MOBILE", "THANK_YOU"]
ASSETS = [asset("a1", "traditions"), asset("a2", "a lot"), asset("b1", "bank"), asset("b2", "bank"), asset("c1", "cup", "needs_review"), asset("d1", "tree", aliases=["woods"])]
VOCAB = Vocabulary(YAP, ASSETS, {})
SYN = {
    "hi": {"to": ["hello"], "confidence": "exact"},
    "us": {"to": ["we"], "confidence": "exact"},
    "cellphone": {"to": ["nonexistent", "mobile"], "confidence": "exact"},
    "tradition": {"to": ["traditions"], "confidence": "exact"},
    "say": {"to": ["tell"], "confidence": "close"},
    "many": {"to": ["a lot"], "confidence": "close"},
    "good": {"to": ["hello"], "confidence": "exact"},  # has its own sign: must never be used
}


class VocabularyTests(unittest.TestCase):
    def test_sources(self):
        self.assertEqual(VOCAB.source("hello"), "yap")
        self.assertEqual(VOCAB.source("THANK you"), "yap")
        self.assertEqual(VOCAB.source("Traditions"), "dictionary")
        self.assertEqual(VOCAB.source("woods"), "dictionary")  # a source-provided alias
        self.assertIsNone(VOCAB.source("hi"))

    def test_several_signs_need_a_fixed_default(self):
        self.assertIsNone(VOCAB.source("bank"))
        self.assertEqual(Vocabulary(YAP, ASSETS, {"bank": "b2"}).source("bank"), "dictionary")
        self.assertIsNone(Vocabulary(YAP, ASSETS, {"bank": "nonexistent"}).source("bank"))

    def test_a_needs_review_single_entry_is_not_usable(self):
        self.assertIsNone(VOCAB.source("cup"))

    def test_parse_yap_tokens_reads_the_real_file_when_present(self):
        ts = Path(__file__).resolve().parents[2] / "yap-render" / "lib" / "islVocabulary.ts"
        if not ts.is_file():
            self.skipTest("yap-render not next to this repo")
        tokens = parse_yap_tokens(ts)
        self.assertIn("hello", tokens)
        self.assertIn("thank you", tokens)
        self.assertGreater(len(tokens), 100)


class TableValidationTests(unittest.TestCase):
    def check(self, entries):
        return validate_synonyms({"entries": entries}, VOCAB)

    def test_valid_entries_are_normalised(self):
        ok, problems = self.check([{"from": "HI", "to": ["HELLO"], "confidence": "exact", "note": "greeting"}])
        self.assertEqual(ok["hi"], {"to": ["hello"], "confidence": "exact", "note": "greeting"})
        self.assertEqual(problems, [])

    def test_bad_entries_are_dropped_and_reported(self):
        ok, problems = self.check([
            {"from": "HI", "to": ["HELLO"], "confidence": "exact"},
            {"from": "HI", "to": ["WE"], "confidence": "exact"},          # duplicate source
            {"from": "GOOD", "to": ["HELLO"], "confidence": "exact"},     # already has its own sign
            {"from": "YO", "to": ["YO"], "confidence": "exact"},          # maps to itself
            {"from": "SUP", "to": ["NOPE"], "confidence": "exact"},       # target has no sign
            {"from": "HOWDY", "to": ["HI"], "confidence": "exact"},       # chain: HI is itself a source
            {"from": "OKAY", "to": ["HELLO"], "confidence": "maybe"},     # bad confidence
            {"from": "", "to": ["HELLO"], "confidence": "exact"},
        ])
        self.assertEqual(list(ok), ["hi"])
        joined = "\n".join(problems)
        for needle in ("duplicate source word", "already has a sign", "maps to itself", "no sign in the Yap avatar vocabulary", "itself a synonym source", "confidence must be", "empty 'from'"):
            self.assertIn(needle, joined)

    def test_unavailable_targets_are_removed_but_a_good_one_keeps_the_entry(self):
        ok, problems = self.check([{"from": "CELLPHONE", "to": ["NOPE", "MOBILE"], "confidence": "exact"}])
        self.assertEqual(ok["cellphone"]["to"], ["mobile"])
        self.assertEqual(len(problems), 1)

    def test_the_curated_table_is_fully_valid_against_the_real_vocabulary(self):
        root = Path(__file__).resolve().parents[1]
        ts, manifest = root.parent / "yap-render" / "lib" / "islVocabulary.ts", root / "data" / "islrtc" / "manifest.json"
        if not ts.is_file() or not manifest.is_file():
            self.skipTest("needs yap-render and the built manifest")
        vocab = Vocabulary.from_files(ts, manifest, root / "data" / "islrtc" / "approved_mappings.json")
        table = load_synonym_table(root / "data" / "vocab" / "synonyms.json")
        valid, problems = validate_synonyms(table, vocab)
        self.assertEqual(problems, [])
        self.assertEqual(len(valid), len(table["entries"]))


class NormalizeTests(unittest.TestCase):
    def run_norm(self, tokens, level="exact", capitalised=None):
        return normalize_tokens(tokens, VOCAB, SYN, level, capitalised)

    def test_unsigned_word_gets_its_exact_synonym(self):
        n = self.run_norm(["HI", "STUDENT"])
        self.assertEqual(n.tokens, ["HELLO", "STUDENT"])
        self.assertEqual([(c.kind, c.before, c.after, c.confidence) for c in n.changes], [("synonym", "HI", "HELLO", "exact")])

    def test_a_words_own_sign_always_wins(self):
        self.assertEqual(self.run_norm(["GOOD"]).tokens, ["GOOD"])
        self.assertEqual(self.run_norm(["HELLO"]).tokens, ["HELLO"])

    def test_close_synonyms_only_at_the_close_level(self):
        self.assertEqual(self.run_norm(["SAY"]).tokens, ["SAY"])
        self.assertEqual(self.run_norm(["SAY"], "close").tokens, ["TELL"])
        self.assertEqual(self.run_norm(["MANY"], "close").tokens, ["A_LOT"])  # multiword targets use the gloss token form
        self.assertEqual(self.run_norm(["HI", "SAY"], "off").tokens, ["HI", "SAY"])

    def test_targets_are_tried_in_order_and_dictionary_targets_work(self):
        self.assertEqual(self.run_norm(["CELLPHONE"]).tokens, ["MOBILE"])
        self.assertEqual(self.run_norm(["TRADITION"]).tokens, ["TRADITIONS"])

    def test_fs_tokens_and_unknown_words_are_left_alone(self):
        self.assertEqual(self.run_norm(["FS:HI", "ZZZ"]).tokens, ["FS:HI", "ZZZ"])

    def test_a_repeat_created_by_a_substitution_is_merged_but_a_real_repeat_is_not(self):
        n = self.run_norm(["HELLO", "HI", "STUDENT"])
        self.assertEqual(n.tokens, ["HELLO", "STUDENT"])
        self.assertEqual([c.kind for c in n.changes], ["synonym", "merge"])
        self.assertEqual(self.run_norm(["HELLO", "HELLO"]).tokens, ["HELLO", "HELLO"])
        self.assertEqual(self.run_norm(["HI", "HI"]).tokens, ["HELLO", "HELLO"])  # the speaker said it twice

    def test_invalid_level_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_norm(["HI"], "loose")

    def test_spelling_rule_needs_a_consistently_capitalised_word_without_a_sign(self):
        cap = {"jainism"}
        n = self.run_norm(["JAINISM", "HELLO"], capitalised=cap)
        self.assertEqual(n.tokens, ["FS:JAINISM", "HELLO"])
        self.assertEqual(n.changes[0].kind, "spell")
        self.assertEqual(self.run_norm(["JAINISM"]).tokens, ["JAINISM"])  # rule is off without the evidence
        self.assertEqual(self.run_norm(["RELIGIOUS_STUDIES"], capitalised={"religious_studies"}).tokens, ["RELIGIOUS_STUDIES"])  # never for compounds
        self.assertEqual(self.run_norm(["HELLO"], capitalised={"hello"}).tokens, ["HELLO"])  # never for a word that has a sign

    def test_a_synonym_beats_spelling(self):
        self.assertEqual(self.run_norm(["HI"], capitalised={"hi"}).tokens, ["HELLO"])


class CapitalisationTests(unittest.TestCase):
    def test_consistently_capitalised_mid_sentence_words(self):
        texts = ["We learn about Jainism today.", "Many people follow Jainism in India."]
        self.assertIn("jainism", consistently_capitalised(texts))

    def test_lowercase_anywhere_disqualifies(self):
        self.assertNotIn("jainism", consistently_capitalised(["We study Jainism.", "the jainism way of life"]))

    def test_sentence_initial_capitals_are_not_evidence(self):
        self.assertNotIn("today", consistently_capitalised(["Today is good.", "Today we begin."]))

    def test_ordinary_english_words_are_never_taken_for_names(self):
        texts = ["Om Ooh Hello Done, frio", "Though it rains we stay. We came Though late."]
        found = consistently_capitalised(texts)
        self.assertNotIn("done", found)
        self.assertNotIn("though", found)
        self.assertIn("bhagwan", consistently_capitalised(["We bow to Bhagwan Mahavir today.", "Praise Bhagwan always."]))

    def test_short_words_are_ignored(self):
        self.assertIn("asia", consistently_capitalised(["We visit Asia often."]))  # 4 letters is the minimum length
        self.assertNotIn("eye", consistently_capitalised(["We see Eye of it."]))  # 3 letters: too short to trust


class CoverageTests(unittest.TestCase):
    def test_counts(self):
        c = coverage(["HELLO", "TRADITIONS", "FS:AMIT", "ZZZ", "HI"], VOCAB)
        self.assertEqual(c, {"avatar": 1, "dictionary": 1, "spelled": 1, "no_sign": 2, "total": 5})

    def test_token_form(self):
        self.assertEqual(token_form("a lot"), "A_LOT")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        t = self.tmp
        (t / "manifest.json").write_text(json.dumps({"assets": ASSETS}), encoding="utf-8")
        (t / "mappings.json").write_text(json.dumps({"select": {}}), encoding="utf-8")
        (t / "vocab.ts").write_text("export const SUPPORTED_TOKENS: readonly string[] = [\n  'HELLO', 'GOOD', 'WE', 'TELL',\n]\n", encoding="utf-8")
        (t / "syn.json").write_text(json.dumps({"entries": [{"from": "HI", "to": ["HELLO"], "confidence": "exact"},
                                                              {"from": "SAY", "to": ["TELL"], "confidence": "close"}]}), encoding="utf-8")
        gloss = {"source_id": "x", "results": [
            {"chunk_id": "x-1", "source_text": "Hello Hi students, we learn about Jainism and Jainism again.", "final_tokens": ["HELLO", "HI", "STUDENT", "WE", "LEARN", "JAINISM", "JAINISM"]},
            {"chunk_id": "x-2", "source_text": "failed", "final_tokens": None, "translation_status": "failed"}]}
        (t / "x.gloss.json").write_text(json.dumps(gloss), encoding="utf-8")

    def args(self, *extra):
        t = self.tmp
        return ["--gloss", str(t / "x.gloss.json"), "--out", str(t / "out"), "--manifest", str(t / "manifest.json"), "--mappings", str(t / "mappings.json"),
                "--synonyms", str(t / "syn.json"), "--yap-vocab", str(t / "vocab.ts"), *extra]

    def test_end_to_end_writes_a_report_and_never_edits_the_source(self):
        before = hashlib.sha256((self.tmp / "x.gloss.json").read_bytes()).hexdigest()
        self.assertEqual(signable_main(self.args()), 0)
        self.assertEqual(hashlib.sha256((self.tmp / "x.gloss.json").read_bytes()).hexdigest(), before)
        report = json.loads((self.tmp / "out" / "x.signable.json").read_text(encoding="utf-8"))
        first = report["results"][0]
        self.assertEqual(first["signable_tokens"], ["HELLO", "STUDENT", "WE", "LEARN", "FS:JAINISM", "FS:JAINISM"])  # merge, and the speaker's repeat kept
        self.assertEqual(first["final_tokens"][1], "HI")  # the original is preserved beside the new tokens
        self.assertEqual({c["kind"] for c in first["changes"]}, {"synonym", "merge", "spell"})
        self.assertEqual(report["results"][1]["signable_tokens"], [])  # a failed sentence stays empty, no crash
        self.assertEqual(report["review_status"], "unreviewed")
        self.assertLess(report["coverage_after"]["no_sign"], report["coverage_before"]["no_sign"])

    def test_off_level_and_no_spell_names(self):
        self.assertEqual(signable_main(self.args("--level", "off", "--no-spell-names")), 0)
        report = json.loads((self.tmp / "out" / "x.signable.json").read_text(encoding="utf-8"))
        self.assertEqual(report["results"][0]["signable_tokens"], report["results"][0]["final_tokens"])

    def test_missing_inputs_are_reported(self):
        a = self.args()
        a[a.index("--manifest") + 1] = str(self.tmp / "absent.json")
        self.assertEqual(signable_main(a), 2)

    def test_real_gloss_improves_when_the_data_is_present(self):
        root = Path(__file__).resolve().parents[1]
        files = [root / "data/translations/01-csu-bakersfield-english-session.gloss.json", root.parent / "yap-render/lib/islVocabulary.ts", root / "data/islrtc/manifest.json"]
        if not all(f.is_file() for f in files):
            self.skipTest("real data not present")
        vocab = Vocabulary.from_files(files[1], files[2], root / "data/islrtc/approved_mappings.json")
        syn, _ = validate_synonyms(load_synonym_table(root / "data/vocab/synonyms.json"), vocab)
        report = process_file(files[0], vocab, syn, "exact", True)
        self.assertLess(report["coverage_after"]["no_sign"], report["coverage_before"]["no_sign"])


if __name__ == "__main__":
    unittest.main()
