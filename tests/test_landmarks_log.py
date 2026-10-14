"""The full-skeleton record (logging.mode: research).

It is a research record: no decision reads it. These tests pin that it
writes what MediaPipe delivered, that research records frames, events and
skeleton, and that deployment leaves nothing on disk and changes no decision
(§3.6: the deployed detector keeps nothing).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from pef_fall_detector.audit_log import LANDMARK_FIELDS, landmark_row  # noqa: E402
from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.pose_frontend import PoseFrame  # noqa: E402
from tests.support import TEST_CFG, FakeSource, make_pose, short_fall  # noqa: E402


class TestRow(unittest.TestCase):

    def test_schema_is_33_image_and_33_world_landmarks(self) -> None:
        self.assertEqual(len(LANDMARK_FIELDS), 3 + 33 * 4 + 33 * 3)
        self.assertEqual(len(set(LANDMARK_FIELDS)), len(LANDMARK_FIELDS))

    def test_a_detected_frame_writes_what_mediapipe_gave(self) -> None:
        pf = make_pose(7, 0.25)
        pf.visibility = np.linspace(0.0, 1.0, 33)
        pf.world_landmarks = np.arange(99, dtype=float).reshape(33, 3) / 100.0
        row = landmark_row(pf)
        self.assertEqual(set(row) - set(LANDMARK_FIELDS), set())
        self.assertEqual(row["frame_index"], 7)
        self.assertEqual(row["lm23_y"], f"{pf.landmarks[23][1]:.2f}")
        self.assertEqual(row["lm11_x"], f"{pf.landmarks[11][0]:.2f}")
        self.assertEqual(row["lm32_v"], "1.000")
        self.assertEqual(row["w05_z"], f"{pf.world_landmarks[5][2]:.4f}")
        self.assertEqual(len(row), len(LANDMARK_FIELDS))

    def test_without_world_landmarks_only_the_image_ones(self) -> None:
        pf = make_pose(1, 0.0)
        pf.world_landmarks = np.empty((0, 3))
        row = landmark_row(pf)
        self.assertIn("lm00_x", row)
        self.assertNotIn("w00_x", row)

    def test_an_undetected_frame_is_just_its_index(self) -> None:
        pf = PoseFrame(frame_index=3, timestamp=0.1, detected=False, frame_size=(64, 48))
        self.assertEqual(landmark_row(pf), {"frame_index": 3, "timestamp_s": "0.1000",
                                            "detected": 0})


class TestHeadless(unittest.TestCase):
    """What main.py records in each logging.mode (0.2.0.dev3)."""

    def _run(self, mode: str) -> Path:
        tmp = tempfile.mkdtemp()
        frames = short_fall()
        built = []

        class _Scripted(FramePipeline):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                built.append(self)

            def process(self, frame_bgr, frame_index, timestamp):
                return self.analyze(frames[frame_index])

        cfg = copy.deepcopy(TEST_CFG)
        cfg["logging"]["output_dir"] = tmp
        cfg["logging"]["mode"] = mode
        saved = (main.load_config, main.VideoFileSource, main.FramePipeline)
        main.load_config = lambda _p: Config(cfg)
        main.VideoFileSource = lambda _p: FakeSource(len(frames))
        main.FramePipeline = _Scripted
        try:
            args = argparse.Namespace(video="A00-S1-NotRecovered.mp4", config=None,
                                      expected_seconds=None, headless=True)
            with contextlib.redirect_stdout(io.StringIO()):
                main.run_headless(args)
        finally:
            main.load_config, main.VideoFileSource, main.FramePipeline = saved
        self.n_frames = len(frames)
        self.events = [(e.timestamp, e.verdict, e.severity, e.reason)
                       for e in built[-1].machine.events]
        return Path(tmp)

    def test_research_writes_one_landmark_row_per_frame(self) -> None:
        d = self._run("research")
        lm = list(d.glob("*-landmarks-*.csv"))
        self.assertEqual(len(lm), 1)
        with lm[0].open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), self.n_frames)
        self.assertEqual(rows[0]["lm23_x"], "640.00")
        self.assertTrue(lm[0].with_suffix(".meta.json").exists())

    def test_research_writes_frames_events_and_landmarks(self) -> None:
        names = sorted(p.name for p in self._run("research").glob("*.csv"))
        self.assertEqual([n.split("-2")[0] for n in names],
                         ["A00-S1-NotRecovered", "A00-S1-NotRecovered-events",
                          "A00-S1-NotRecovered-landmarks"])

    def test_deployment_writes_nothing(self) -> None:
        # §3.6: the deployed detector persists nothing.
        self.assertEqual(list(self._run("deployment").iterdir()), [])

    def test_the_mode_changes_what_is_written_never_what_is_decided(self) -> None:
        self._run("research")
        research = self.events
        self._run("deployment")
        self.assertTrue(research)
        self.assertEqual(self.events, research)

    def test_the_shipped_config_is_deployment(self) -> None:
        # Decided in B2 (0.2.0.dev2): config.yaml runs as the deployed device
        # does; dataset runs switch to research.
        from pef_fall_detector.config import load_config, research_mode
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertEqual(cfg.logging.mode, "deployment")
        self.assertFalse(research_mode(cfg))


if __name__ == "__main__":
    unittest.main()
