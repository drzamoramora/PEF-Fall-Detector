"""Registro experimental del esqueleto completo (logging.save_landmarks).

Es un registro de investigación: ninguna decisión lo lee. Estas pruebas fijan
tres cosas: que escribe lo que MediaPipe entregó, que no cambia ni un byte del
registro de cuadros, y que apagado no deja rastro (§3.6: el dispositivo
desplegado no guarda landmarks).
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
from tests.test_main_headless import _FakeSource, _short_fall  # noqa: E402
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402


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

    def _run(self, save: bool | None) -> Path:
        tmp = tempfile.mkdtemp()
        frames = _short_fall()

        class _Scripted(FramePipeline):
            def process(self, frame_bgr, frame_index, timestamp):
                return self.analyze(frames[frame_index])

        cfg = copy.deepcopy(TEST_CFG)
        cfg["logging"]["output_dir"] = tmp
        if save is not None:
            cfg["logging"]["save_landmarks"] = save
        saved = (main.load_config, main.VideoFileSource, main.FramePipeline)
        main.load_config = lambda _p: Config(cfg)
        main.VideoFileSource = lambda _p: _FakeSource(len(frames))
        main.FramePipeline = _Scripted
        try:
            args = argparse.Namespace(video="A00-S1-NotRecovered.mp4", config=None,
                                      expected_seconds=None, headless=True)
            with contextlib.redirect_stdout(io.StringIO()):
                main.run_headless(args)
        finally:
            main.load_config, main.VideoFileSource, main.FramePipeline = saved
        self.n_frames = len(frames)
        return Path(tmp)

    @staticmethod
    def _frames_csv(d: Path) -> str:
        f = [p for p in d.glob("*.csv") if "-events-" not in p.name
             and "-landmarks-" not in p.name]
        assert len(f) == 1, f
        return f[0].read_text()

    def test_on_writes_one_row_per_frame(self) -> None:
        d = self._run(True)
        lm = list(d.glob("*-landmarks-*.csv"))
        self.assertEqual(len(lm), 1)
        with lm[0].open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), self.n_frames)
        self.assertEqual(rows[0]["lm23_x"], "640.00")
        self.assertTrue(lm[0].with_suffix(".meta.json").exists())

    def test_it_changes_nothing_in_the_frame_record(self) -> None:
        on, off = self._run(True), self._run(False)
        self.assertEqual(self._frames_csv(on), self._frames_csv(off))
        ev_on = next(on.glob("*-events-*.csv")).read_text()
        ev_off = next(off.glob("*-events-*.csv")).read_text()
        self.assertEqual(ev_on, ev_off)

    def test_off_or_absent_leaves_no_file(self) -> None:
        for save in (False, None):
            self.assertEqual(list(self._run(save).glob("*-landmarks-*")), [])

    def test_the_shipped_config_turns_it_on(self) -> None:
        from pef_fall_detector.config import load_config
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertTrue(cfg.logging.save_landmarks)


if __name__ == "__main__":
    unittest.main()
