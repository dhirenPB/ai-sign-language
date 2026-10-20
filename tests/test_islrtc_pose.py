"""Pose-extraction PoC: quality metrics, asset selection and the landmark document contract (no MediaPipe needed)."""

import json
import tempfile
import unittest
from pathlib import Path

from islrtc_pose.cli import select_assets
from islrtc_pose.quality import assess


def hand(x, y):
    return [[x, y, 0.0] for _ in range(21)]


def pose(vis=0.9):
    return [[0.5, 0.5, 0.0, vis] for _ in range(33)]


def frame(i, left=None, right=None, with_pose=True, vis=0.9):
    return {"i": i, "t": i * 0.04, "pose": pose(vis) if with_pose else None, "left_hand": left, "right_hand": right, "face": None}


def doc(frames):
    return {"frames": frames}


class QualityTests(unittest.TestCase):
    def test_empty_document(self):
        q = assess(doc([]))
        self.assertFalse(q["usable_for_retargeting"])
        self.assertTrue(q["issues"])

    def test_clean_two_handed_clip_is_usable(self):
        frames = [frame(i, hand(0.2 + i * 0.01, 0.5), hand(0.8 - i * 0.01, 0.5)) for i in range(30)]
        q = assess(doc(frames))
        self.assertEqual(q["detection_rate"]["both_hands"], 1.0)
        self.assertEqual(q["moving_hands_measured"], "both")
        self.assertEqual(q["issues"], [])
        self.assertTrue(q["usable_for_retargeting"])

    def test_short_dropout_is_reported_but_can_be_interpolated(self):
        frames = [frame(i, None, hand(0.8 - i * 0.01, 0.5) if i not in (10, 11, 12) else None) for i in range(30)]
        q = assess(doc(frames))
        self.assertEqual(q["hand_dropout_runs"]["right"], [3])
        self.assertTrue(any("right hand lost mid-sign" in s and "longest 3" in s for s in q["issues"]))
        self.assertEqual(q["active_window"]["right"], {"start": 0, "end": 29, "frames": 30, "coverage": 0.9})
        self.assertTrue(q["usable_for_retargeting"])  # 27/30 in window, gap of 3 is bridgeable

    def test_long_dropout_blocks_usability(self):
        frames = [frame(i, None, hand(0.8 - i * 0.01, 0.5) if i not in range(10, 15) else None) for i in range(30)]
        q = assess(doc(frames))
        self.assertEqual(q["hand_dropout_runs"]["right"], [5])
        self.assertFalse(q["usable_for_retargeting"])

    def test_hand_at_rest_before_and_after_the_sign_does_not_count_against_the_clip(self):
        frames = [frame(i, None, hand(0.2 + i * 0.02, 0.5) if 20 <= i < 50 else None) for i in range(70)]
        q = assess(doc(frames))
        self.assertLess(q["detection_rate"]["right_hand"], 0.5)
        self.assertEqual(q["active_window"]["right"]["coverage"], 1.0)
        self.assertTrue(q["usable_for_retargeting"])

    def test_leading_and_trailing_absence_is_not_a_dropout(self):
        frames = [frame(i, None, hand(0.5 + i * 0.01, 0.5) if 5 <= i < 25 else None) for i in range(30)]
        self.assertEqual(assess(doc(frames))["hand_dropout_runs"]["right"], [])

    def test_no_hand_at_all(self):
        q = assess(doc([frame(i) for i in range(10)]))
        self.assertIn("no hand detected in any frame", q["issues"])
        self.assertFalse(q["usable_for_retargeting"])
        self.assertEqual(q["moving_hands_measured"], "none")

    def test_missing_pose_and_low_wrist_visibility(self):
        frames = [frame(i, None, hand(0.5, 0.5), with_pose=(i % 2 == 0), vis=0.2) for i in range(20)]
        q = assess(doc(frames))
        self.assertTrue(any("pose missing" in s for s in q["issues"]))
        self.assertTrue(any("wrist low visibility" in s for s in q["issues"]))

    def test_single_moving_hand_is_identified(self):
        frames = [frame(i, None, hand(0.2 + i * 0.02, 0.5)) for i in range(30)]
        self.assertEqual(assess(doc(frames))["moving_hands_measured"], "right")

    def test_jittery_tracking_is_flagged(self):
        frames = [frame(i, None, hand(0.5 + (0.1 if i % 2 else -0.1), 0.5)) for i in range(30)]
        self.assertTrue(any("jittery" in s for s in assess(doc(frames))["issues"]))


class SelectionTests(unittest.TestCase):
    ASSETS = [
        {"asset_id": "a1", "label_norm": "today", "status": "source_label"},
        {"asset_id": "b1", "label_norm": "bank", "status": "source_label"},
        {"asset_id": "b2", "label_norm": "bank", "status": "source_label"},
        {"asset_id": "c1", "label_norm": "cup", "status": "needs_review"},
        {"asset_id": "d1", "label_norm": "tree", "status": "concept"},
    ]

    def test_only_unambiguous_clean_labels_are_selected(self):
        import contextlib
        import io

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            chosen = select_assets({"assets": self.ASSETS}, ["Today", "Bank", "cup", "tree", "missing"], ["a1", "zzz"])
        self.assertEqual([a["asset_id"] for a in chosen], ["a1", "a1"])  # the id and the label both resolve to a1; nothing guessed
        text = err.getvalue()
        for needle in ("'Bank'", "'cup'", "'tree'", "'missing'", "zzz"):
            self.assertIn(needle, text)


class DocumentContractTests(unittest.TestCase):
    def test_sample_document_round_trips_through_json(self):
        d = {"format": "islrtc-landmarks", "format_version": "1.0", "frames": [frame(0, hand(0.1, 0.2), None)]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text(json.dumps(d), encoding="utf-8")
            back = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(back["frames"][0]["right_hand"], None)
        self.assertEqual(len(back["frames"][0]["left_hand"]), 21)


if __name__ == "__main__":
    unittest.main()
