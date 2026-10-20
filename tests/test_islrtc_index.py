"""ISLRTC dictionary indexer: file-name parsing, status rules, manifest/SQLite output, review application, probing."""

import csv
import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from islrtc_index.index import apply_reviews, assign_status, asset_id_for, run_index, scan, to_manifest
from islrtc_index.parse import normalize_label, parse_stem

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

# Shared with yap-render tests/normalize.test.ts: both implementations must give exactly these results.
NORMALIZE_CASES = [
    ("  Religion! ", "religion"),
    ("GOOD_MORNING", "good morning"),
    ("FS:JONATHAN_YOUNG", "jonathan young"),
    ("Mother-in-law", "mother in law"),
    ("Don't", "dont"),
    ("A/B  Testing", "a b testing"),
    ("Ache, Pain", "ache pain"),
    ("120 Volt", "120 volt"),
    ("Café", "café"),
]


class NormalizeTests(unittest.TestCase):
    def test_shared_cases(self):
        for raw, expected in NORMALIZE_CASES:
            self.assertEqual(normalize_label(raw), expected, raw)


class ParseTests(unittest.TestCase):
    def test_plain_term(self):
        p = parse_stem("A_Line_Dress", "A")
        self.assertEqual((p.label, p.label_norm, p.variant, p.qualifier, p.entry_type, p.flags), ("A Line Dress", "a line dress", None, None, "term", []))

    def test_alternate_sign_variants(self):
        self.assertEqual(parse_stem("Aamir_Khan_(Sign_2)", "A").variant, "sign_2")
        self.assertEqual(parse_stem("Aamir_Khan_(Sign_2)", "A").label_norm, "aamir khan")
        self.assertEqual(parse_stem("X_(Sign)", "X").variant, "sign")
        self.assertEqual(parse_stem("Colour_Palette_(2)", "C").variant, "2")

    def test_explanations_are_concept_entries(self):
        for stem in ("Able_(Explanation)", "Abundant_(Sign_and_Explanation)", "Ancestor_(Explaination)", "Foo_(Explanation_2)", "Bar_(Example)"):
            self.assertEqual(parse_stem(stem, stem[0]).entry_type, "explanation", stem)

    def test_abbreviation_in_brackets_is_a_source_alias(self):
        p = parse_stem("Aam_Aadmi_Party_(AAP)_(Sign_2)", "A")
        self.assertEqual(p.label_norm, "aam aadmi party")
        self.assertEqual([(a.text, a.source, a.verified) for a in p.aliases], [("AAP", "filename_abbreviation", True)])
        self.assertEqual(p.variant, "sign_2")

    def test_sense_qualifier_is_flagged_not_dropped(self):
        p = parse_stem("Bank_(Financial_Term)", "B")
        self.assertEqual((p.label_norm, p.qualifier), ("bank", "Financial Term"))
        self.assertIn("sense_qualifier", p.flags)

    def test_numbers_folder_pairs_digits_and_word(self):
        p = parse_stem("2000_Two_Thousand", "Numbers")
        self.assertEqual(p.label_norm, "two thousand")
        self.assertEqual([a.text for a in p.aliases], ["2000"])
        q = parse_stem("120 Volt", "Numbers")
        self.assertEqual(q.label_norm, "120 volt")  # no "digits_word" pair: left as the dataset wrote it

    def test_synonym_lists_become_source_aliases_but_long_phrases_do_not(self):
        p = parse_stem("Abstain_or_Abstention", "A")
        self.assertEqual(sorted(a.text for a in p.aliases), ["Abstain", "Abstention"])
        self.assertIn("synonym_list", p.flags)
        long = parse_stem("Organisational_Structure_Secretary_Additional_Secretary_Joint_Secretary,Division_Deputy_Secretary", "O")
        self.assertEqual((long.entry_type, [a.text for a in long.aliases]), ("phrase", []))
        self.assertIn("long_label", long.flags)

    def test_language_tag_and_trailing_number(self):
        p = parse_stem("AB Testing- English", "A")
        self.assertEqual((p.label_norm, p.language_tag), ("ab testing", "English"))
        self.assertEqual(parse_stem("Hair Pins - English_1", "H").variant, "1")
        self.assertIn("trailing_number", parse_stem("Lean_1", "L").flags)

    def test_weak_label_and_folder_mismatch(self):
        self.assertIn("weak_label", parse_stem("_", "A").flags)
        self.assertIn("folder_initial_mismatch", parse_stem("Gun_Range", "R").flags)

    def test_spacing_glitches_are_collapsed(self):
        self.assertEqual(parse_stem("United _Nations_Convention_on _the_Rights", "U").label, "United Nations Convention on the Rights")


def make_dataset(root: Path, names, real_video=False):
    for folder, name in names:
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"{name}.mp4"
        if real_video:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1", "-c:v", "libx265",
                            "-pix_fmt", "yuv420p", str(target)], check=True)
        else:
            target.write_bytes(b"not a real video")


NAMES = [("A", "Able_(Explanation)"), ("A", "Aback"), ("B", "Bank"), ("B", "Bank_(Sign_2)"), ("B", "Bank_(Financial_Term)"), ("T", "Tree"),
         ("Numbers", "5_Five")]


class IndexTests(unittest.TestCase):
    def read_review(self):
        with open(self.out / "review_queue.csv", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "ds"
        self.out = self.tmp / "out"
        make_dataset(self.data, NAMES)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scan_is_stable_and_relative(self):
        entries = scan(self.data)
        self.assertEqual(len(entries), len(NAMES))
        self.assertTrue(all(not Path(e.rel_path).is_absolute() and "\\" not in e.rel_path for e in entries))
        self.assertEqual({e.asset_id for e in entries}, {e.asset_id for e in scan(self.data)})
        self.assertEqual(asset_id_for("A/Aback.mp4"), asset_id_for("a/aback.mp4"))

    def test_status_rules(self):
        entries = scan(self.data)
        assign_status(entries)
        status = {e.filename: e.status for e in entries}
        self.assertEqual(status["Able_(Explanation).mp4"], "concept")
        self.assertEqual(status["Aback.mp4"], "source_label")
        self.assertEqual(status["Bank_(Financial_Term).mp4"], "needs_review")
        self.assertEqual(status["Tree.mp4"], "source_label")

    def test_duplicates_are_flagged_not_dropped(self):
        make_dataset(self.data, [("C", "Tree")])  # same label as T/Tree, different folder
        entries = scan(self.data)
        assign_status(entries)
        dup = [e for e in entries if e.status == "duplicate"]
        self.assertEqual(len(dup), 1)
        self.assertIsNotNone(dup[0].duplicate_of)
        self.assertEqual(len(entries), len(NAMES) + 1)

    def test_run_writes_all_outputs_without_touching_media(self):
        before = {p: p.stat().st_mtime_ns for p in self.data.rglob("*.mp4")}
        report = run_index(self.data, self.out, probe=False)
        for name in ("islrtc_index.sqlite", "manifest.json", "review_queue.csv", "audit.json"):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertEqual(report["total_video_files"], len(NAMES))
        self.assertEqual(report["by_status"]["concept"], 1)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in self.data.rglob("*.mp4")})
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertTrue(all(a["rel_path"].count("/") == 1 for a in manifest["assets"]))
        db = sqlite3.connect(self.out / "islrtc_index.sqlite")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM assets").fetchone()[0], len(NAMES))
        self.assertEqual(db.execute("SELECT aliases.text FROM aliases JOIN assets USING(asset_id) WHERE assets.filename='5_Five.mp4'").fetchone()[0], "5")
        db.close()

    def test_multiple_signs_for_one_headword_reach_the_review_queue(self):
        run_index(self.data, self.out, probe=False)
        rows = self.read_review()
        banks = [r for r in rows if r["label_norm"] == "bank"]
        self.assertEqual(len(banks), 3)
        self.assertTrue(all("multiple_signs:3" in r["issues"] for r in banks))
        self.assertFalse(any(r["label_norm"] == "tree" for r in rows))  # a clean single sign needs no review
        self.assertTrue(all(r["decision"] == "" for r in rows))  # never pre-filled

    def test_reviews_apply_decisions_and_defaults(self):
        run_index(self.data, self.out, probe=False)
        rows = self.read_review()
        plain = next(r for r in rows if r["rel_path"] == "B/Bank.mp4")
        fin = next(r for r in rows if r["rel_path"] == "B/Bank_(Financial_Term).mp4")
        for r in rows:
            r["decision"] = {plain["asset_id"]: "default", fin["asset_id"]: "reject"}.get(r["asset_id"], "")
        plain["aliases_add"] = "lender;money house"
        reviews = self.tmp / "reviews.csv"
        with open(reviews, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        report = run_index(self.data, self.out, probe=False, reviews=reviews)
        self.assertEqual(report["reviews_applied"], 2)
        manifest = {a["asset_id"]: a for a in json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))["assets"]}
        self.assertEqual((manifest[plain["asset_id"]]["status"], manifest[plain["asset_id"]]["reviewed"]), ("verified", True))
        self.assertEqual(manifest[fin["asset_id"]]["status"], "rejected")
        self.assertEqual([a["text"] for a in manifest[plain["asset_id"]]["aliases"]], ["lender", "money house"])
        self.assertEqual(json.loads((self.out / "approved_mappings.json").read_text())["select"], {"bank": plain["asset_id"]})

    def test_review_with_unknown_ids_or_decisions_is_reported_not_applied(self):
        entries = scan(self.data)
        assign_status(entries)
        bad = self.tmp / "bad.csv"
        bad.write_text("asset_id,decision\nnope,verified\n%s,maybe\n" % entries[0].asset_id, encoding="utf-8")
        result = apply_reviews(entries, bad)
        self.assertEqual(result.applied, 0)
        self.assertEqual(len(result.unknown_ids), 2)

    def test_missing_data_dir_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            run_index(self.tmp / "absent", self.out, probe=False)

    def test_manifest_round_trips_for_the_resolver(self):
        entries = scan(self.data)
        assign_status(entries)
        m = to_manifest(entries)
        self.assertEqual({"asset_id", "label", "label_norm", "aliases", "variant", "rel_path", "container", "codec", "duration_s", "status"},
                         set(m["assets"][0]) & {"asset_id", "label", "label_norm", "aliases", "variant", "rel_path", "container", "codec", "duration_s", "status"})


class DefaultSignTests(unittest.TestCase):
    """One deterministic default sign per ambiguous key, so a word always shows the same sign."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data, self.out = self.tmp / "ds", self.tmp / "out"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, names):
        make_dataset(self.data, names)
        report = run_index(self.data, self.out, probe=False)
        maps = json.loads((self.out / "approved_mappings.json").read_text(encoding="utf-8"))
        manifest = {a["asset_id"]: a for a in json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))["assets"]}
        return report, maps, manifest

    def chosen(self, maps, manifest, key):
        return manifest[maps["select"][key]]["rel_path"]

    def test_the_unsuffixed_base_file_is_the_default(self):
        _, maps, man = self.build([("G", "Good"), ("G", "Good_(Sign_2)"), ("G", "Good_(Sign_3)")])
        self.assertEqual(self.chosen(maps, man, "good"), "G/Good.mp4")
        self.assertEqual(len(maps["auto_default"]["good"]), 3)
        self.assertEqual(maps["auto_default"]["good"][0], maps["select"]["good"])

    def test_without_a_base_the_lowest_numbered_alternate_wins(self):
        _, maps, man = self.build([("A", "Aam_(Sign_3)"), ("A", "Aam_(Sign_2)"), ("A", "Aam_(Sign_10)")])
        self.assertEqual(self.chosen(maps, man, "aam"), "A/Aam_(Sign_2).mp4")

    def test_a_clean_entry_beats_a_sense_qualifier_and_a_needs_review_entry(self):
        _, maps, man = self.build([("B", "Bank_(Financial_Term)"), ("B", "Bank_(Sign_2)"), ("B", "Bank")])
        self.assertEqual(self.chosen(maps, man, "bank"), "B/Bank.mp4")
        _, maps2, man2 = self.build([("C", "Cell_(Biology)"), ("C", "Cell_(Sign_2)")])
        self.assertEqual(self.chosen(maps2, man2, "cell"), "C/Cell_(Sign_2).mp4")  # clean alternate beats the qualified sense

    def test_qualifier_only_groups_still_get_one_deterministic_default(self):
        _, maps, man = self.build([("A", "Angle_(Math)"), ("A", "Angle_(Academic)")])
        self.assertEqual(self.chosen(maps, man, "angle"), "A/Angle_(Academic).mp4")  # alphabetical by qualifier, then path

    def test_a_word_that_is_an_alias_of_another_entry_prefers_the_entry_that_is_the_word(self):
        _, maps, man = self.build([("A", "Act_or_Statute"), ("A", "Act")])
        self.assertEqual(self.chosen(maps, man, "act"), "A/Act.mp4")

    def test_single_sign_words_get_no_default_entry(self):
        _, maps, _ = self.build([("T", "Tree"), ("G", "Good"), ("G", "Good_(Sign_2)")])
        self.assertNotIn("tree", maps["select"])

    def test_defaults_are_identical_across_reindexing_and_reported(self):
        names = [("G", "Good"), ("G", "Good_(Sign_2)"), ("N", "Name"), ("N", "Name_(Sign_2)"), ("N", "Name_(Sign_3)")]
        report, first, _ = self.build(names)
        second = run_index(self.data, self.out, probe=False)
        self.assertEqual(first, json.loads((self.out / "approved_mappings.json").read_text(encoding="utf-8")))
        self.assertEqual((report["auto_defaults"], second["auto_defaults"]), (2, 2))

    def test_reviewer_decision_overrides_the_automatic_default(self):
        _, maps, man = self.build([("G", "Good"), ("G", "Good_(Sign_2)")])
        alt = next(i for i, a in man.items() if a["rel_path"] == "G/Good_(Sign_2).mp4")
        with open(self.out / "review_queue.csv", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        marked = [r["rel_path"] for r in rows if r["auto_default"] == "default"]
        self.assertEqual(marked, ["G/Good.mp4"])
        for r in rows:
            r["decision"] = "default" if r["asset_id"] == alt else ""
        reviews = self.tmp / "r.csv"
        with open(reviews, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        report = run_index(self.data, self.out, probe=False, reviews=reviews)
        maps2 = json.loads((self.out / "approved_mappings.json").read_text(encoding="utf-8"))
        self.assertEqual(maps2["select"]["good"], alt)
        self.assertNotIn("good", maps2["auto_default"])  # no longer automatic
        self.assertEqual((report["auto_defaults"], report["reviewer_defaults"]), (0, 1))

    def test_rejected_and_concept_entries_are_never_the_default(self):
        _, _, _ = self.build([("G", "Good"), ("G", "Good_(Sign_2)"), ("G", "Good_(Explanation)")])
        maps = json.loads((self.out / "approved_mappings.json").read_text(encoding="utf-8"))
        manifest = {a["asset_id"]: a for a in json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))["assets"]}
        for ids in maps["auto_default"].values():
            self.assertTrue(all(manifest[i]["status"] in ("source_label", "verified", "needs_review") for i in ids))


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")
class ProbeTests(unittest.TestCase):
    def test_probe_reads_codec_and_flags_unreadable_files_and_caches(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            data, out = tmp / "ds", tmp / "out"
            make_dataset(data, [("A", "Good")], real_video=True)
            make_dataset(data, [("A", "Broken")])  # bytes that are not a video
            report = run_index(data, out, probe=True, workers=2)
            self.assertEqual(report["codecs"].get("hevc"), 1)
            self.assertEqual(report["by_status"].get("unreadable"), 1)
            db = sqlite3.connect(out / "islrtc_index.sqlite")
            row = db.execute("SELECT codec,width,height,duration_s FROM assets WHERE filename='Good.mp4'").fetchone()
            db.close()
            self.assertEqual((row[0], row[1], row[2]), ("hevc", 64, 64))
            self.assertAlmostEqual(row[3], 1.0, delta=0.3)
            again = run_index(data, out, probe=True, workers=2)
            self.assertEqual(again["probe"]["probed"], 0)
            self.assertEqual(again["probe"]["reused_from_cache"], 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
