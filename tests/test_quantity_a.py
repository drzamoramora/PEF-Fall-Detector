"""Quantity A (phase 8.1): 3D head height against gravity.

Measured and recorded; with its switches off (as in TEST_CFG) no stage
reads it. These tests pin the geometry (with a tilted camera, the real
case), the causal calibration on the subject's own standing, and that no
decision changes.
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.audit_log import PHASE2_FIELDS  # noqa: E402
from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.quantities import GravityCalibrator, knee_angle_3d  # noqa: E402
from tests.support import (  # noqa: E402
    POSE_KNEELING,
    POSE_LYING,
    POSE_SITTING_FLOOR,
    POSE_STANDING,
    TEST_CFG,
    attach_world,
    make_pose,
    toppling_pose,
    world_body,
    world_pose,
)

DT = 1.0 / 30.0


def _calibrated(pitch=30.0, seconds=1.0) -> GravityCalibrator:
    g = GravityCalibrator(min_standing_s=0.5)
    for i in range(int(round(seconds / DT))):
        g.observe(i * DT, world_body(POSE_STANDING, pitch), standing=True)
    return g


class TestKneeAngle(unittest.TestCase):

    def test_straight_legs_are_180(self) -> None:
        self.assertAlmostEqual(knee_angle_3d(world_body(POSE_STANDING)), 180.0, places=4)

    def test_kneeling_is_a_right_angle(self) -> None:
        self.assertAlmostEqual(knee_angle_3d(world_body(POSE_KNEELING)), 90.0, places=4)

    def test_unusable_input_is_nan(self) -> None:
        self.assertTrue(math.isnan(knee_angle_3d(np.empty((0, 3)))))
        bad = world_body(POSE_STANDING)
        bad[25, 0] = np.nan
        self.assertTrue(math.isnan(knee_angle_3d(bad)))


class TestCalibration(unittest.TestCase):

    def test_nan_until_half_a_second_of_standing(self) -> None:
        g = GravityCalibrator(min_standing_s=0.5)
        for i in range(15):                                  # 14 steps = 0.47 s
            g.observe(i * DT, world_body(POSE_STANDING), standing=True)
        self.assertFalse(g.calibrated)
        self.assertTrue(math.isnan(g.head_ratio(world_body(POSE_STANDING))))
        g.observe(15 * DT, world_body(POSE_STANDING), standing=True)   # 0.50 s
        g.observe(16 * DT, world_body(POSE_STANDING), standing=True)
        self.assertTrue(g.calibrated)

    def test_up_is_recovered_from_a_pitched_camera(self) -> None:
        g = _calibrated(pitch=30.0)
        a = math.radians(30.0)
        expected = np.array([0.0, -math.cos(a), -math.sin(a)])  # (0,-1,0) seen by the camera
        self.assertLess(math.degrees(math.acos(float(np.dot(g.up, expected)))), 0.5)
        self.assertAlmostEqual(g.stature, 1.6, places=3)

    def test_postures(self) -> None:
        g = _calibrated(pitch=30.0)
        self.assertAlmostEqual(g.head_ratio(world_body(POSE_STANDING)), 1.0, places=3)
        self.assertAlmostEqual(g.head_ratio(world_body(POSE_LYING)), 0.05 / 1.6, places=3)
        self.assertAlmostEqual(g.head_ratio(world_body(POSE_KNEELING)), 1.15 / 1.6, places=3)
        self.assertAlmostEqual(g.head_ratio(world_body(POSE_SITTING_FLOOR)), 0.75 / 1.6, places=3)
        self.assertAlmostEqual(g.hip_ratio(world_body(POSE_STANDING)), 1.0, places=3)
        self.assertAlmostEqual(g.hip_ratio(world_body(POSE_SITTING_FLOOR)), 0.05 / 0.9, places=3)

    def test_the_same_posture_reads_the_same_under_any_camera_pitch(self) -> None:
        for pitch in (0.0, 15.0, 45.0):
            g = _calibrated(pitch=pitch)
            self.assertAlmostEqual(g.head_ratio(world_body(POSE_KNEELING, pitch)), 1.15 / 1.6,
                                   places=3)

    def test_up_is_a_unit_vector_even_from_mixed_samples(self) -> None:
        g = GravityCalibrator()
        for i in range(40):                                   # camera shaking 20°..40°
            g.observe(i * DT, world_body(POSE_STANDING, 20.0 if i % 2 else 40.0), standing=True)
        self.assertAlmostEqual(float(np.linalg.norm(g.up)), 1.0, places=9)

    def test_new_standing_frames_update_the_calibration(self) -> None:
        g = _calibrated(seconds=0.6)
        self.assertAlmostEqual(g.stature, 1.6, places=3)
        taller = dict(POSE_STANDING)
        taller[0] = (0, -0.9, 0)     # head 1.8 m above ankles
        for i in range(18, 120):
            g.observe(i * DT, world_body(taller), standing=True)
        self.assertAlmostEqual(g.stature, 1.8, places=3)

    def test_non_standing_frames_never_calibrate(self) -> None:
        g = GravityCalibrator()
        for i in range(90):
            g.observe(i * DT, world_body(POSE_LYING), standing=False)
        self.assertFalse(g.calibrated)
        self.assertIsNone(g.up)

    def test_a_gap_does_not_count_as_standing(self) -> None:
        g = GravityCalibrator(min_standing_s=0.5)
        for t in (0.0, 10.0, 20.0, 30.0, 40.0):               # five frames, 40 s apart
            g.observe(t, world_body(POSE_STANDING), standing=True)
        self.assertAlmostEqual(g.standing_s, 0.4)             # 4 gaps, 0.1 s each
        self.assertFalse(g.calibrated)

    def test_memory_is_bounded_and_reset_forgets(self) -> None:
        g = GravityCalibrator(max_samples=10)
        for i in range(100):
            g.observe(i * DT, world_body(POSE_STANDING), standing=True)
        self.assertEqual(len(g._ups), 10)
        g.reset()
        self.assertFalse(g.calibrated)
        self.assertEqual(g.standing_s, 0.0)

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            GravityCalibrator(min_standing_s=0.0)
        with self.assertRaises(ValueError):
            GravityCalibrator(max_samples=2)


def _clip(with_world: bool):
    """Stand 1.5 s (upright in the image, straight legs in 3D), fall, lie 4 s."""
    frames = [make_pose(i, i * DT) for i in range(45)]
    frames += [toppling_pose(i, i * DT, angle_deg=(i - 45) * 4.5, hip_y=500.0 + (i - 45) * 12.0)
               for i in range(45, 65)]
    frames += [toppling_pose(i, i * DT, angle_deg=90.0, hip_y=740.0) for i in range(65, 185)]
    if with_world:
        for i, pf in enumerate(frames):
            attach_world(pf, world_body(POSE_STANDING if i < 45 else POSE_LYING))
    else:
        for pf in frames:
            pf.world_landmarks = np.empty((0, 3))
    return frames


class TestPipeline(unittest.TestCase):

    def _run(self, with_world: bool):
        pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)), labelling=True)
        results = [pipe.analyze(pf) for pf in _clip(with_world)]
        pipe.finalise(results[-1].pose.timestamp)
        return pipe, results

    def test_a_is_recorded_standing_then_lying(self) -> None:
        _, res = self._run(True)
        a = [r.quantities["A_head"] for r in res]
        self.assertTrue(math.isnan(a[0]), "A must be NaN before calibrating")
        self.assertAlmostEqual(a[40], 1.0, places=3)          # calibrated, standing
        self.assertAlmostEqual(a[-1], 0.05 / 1.6, places=3)   # on the floor

    def test_no_decision_changes(self) -> None:
        with_w, res_w = self._run(True)
        without, res_0 = self._run(False)
        self.assertEqual([r.stage for r in res_w], [r.stage for r in res_0])
        ev_w = [(e.timestamp, e.verdict, e.severity, e.reason) for e in with_w.machine.events]
        ev_0 = [(e.timestamp, e.verdict, e.severity, e.reason) for e in without.machine.events]
        self.assertEqual(ev_w, ev_0)
        self.assertTrue(ev_w, "the scenario must produce an event")

    def test_upright_trunk_with_bent_knees_does_not_calibrate(self) -> None:
        # Sitting upright on a chair: T is 0 in the image, but the knees are
        # bent in 3D — that is not "standing", and must not set the height.
        pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)))
        sitting = world_pose(ankles=((-0.1, 0.45, 0.45), (0.1, 0.45, 0.45)),
                        knees=((-0.1, 0.0, 0.45), (0.1, 0.0, 0.45)),
                        shoulders_y=-0.5, nose=(0, -0.7, 0))
        for i in range(60):
            pipe.analyze(attach_world(make_pose(i, i * DT), world_body(sitting)))
        self.assertFalse(pipe._gravity.calibrated)

    def test_unreliable_frames_are_not_read(self) -> None:
        pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)))
        for i in range(60):
            pf = attach_world(make_pose(i, i * DT), world_body(POSE_STANDING))
            pf.core_visibility = 0.1
            r = pipe.analyze(pf)
            self.assertTrue(math.isnan(r.quantities["A_head"]))
        self.assertFalse(pipe._gravity.calibrated)

    def test_without_world_landmarks_a_is_nan(self) -> None:
        _, res = self._run(False)
        self.assertTrue(all(math.isnan(r.quantities["A_head"]) for r in res))

    def test_record_and_config(self) -> None:
        self.assertIn("A_head_EXP", PHASE2_FIELDS)
        self.assertIn("A_hip_EXP", PHASE2_FIELDS)
        from pef_fall_detector.config import load_config
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        qa = cfg.as_dict()["quantity_a"]
        self.assertEqual((qa["standing_T_deg"], qa["standing_knee_deg"], qa["min_standing_s"]),
                         (20.0, 150.0, 0.5))
        pipe = FramePipeline(cfg)
        self.assertEqual(pipe._gravity.min_standing_s, 0.5)

    def test_the_section_is_required(self) -> None:
        # Before 0.2.0.dev2 a config without the section fell back to these
        # values; now the section is required, and TEST_CFG declares them.
        from pef_fall_detector.config import ConfigError
        cfg = copy.deepcopy(TEST_CFG)
        pipe = FramePipeline(Config(cfg))
        self.assertEqual((pipe._a_standing_t, pipe._a_standing_knee), (20.0, 150.0))
        cfg.pop("quantity_a")
        with self.assertRaisesRegex(ConfigError, r"missing section 'quantity_a'"):
            Config(cfg)


class TestLabCurve(unittest.TestCase):

    def test_pef_lab_draws_a(self) -> None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest("PySide6 is not installed")
        from pef_fall_detector.config import load_config
        from pef_fall_detector.gui.lab_window import LabWindow
        _app = QApplication.instance() or QApplication([])  # noqa: F841
        w = LabWindow(load_config(Path(__file__).resolve().parents[1] / "config.yaml"))
        try:
            self.assertIn("A_head", w.plots.series)
            self.assertIn("A_hip", w.plots.series)
            self.assertIn(w.plots.a_plot, w.plots.curve_plots)
            # Regression 25/09: the window grew with every plot (1341 px minimum
            # height) and on the Mac the bottom buttons ended up off the screen. Now
            # the curves sit in a scroll area and the window has to fit a laptop.
            from PySide6.QtWidgets import QScrollArea
            self.assertLessEqual(w.minimumSizeHint().height(), 760)
            self.assertLessEqual(w.minimumSizeHint().width(), 1280)
            for i in range(w.plots.tabs.count()):
                self.assertIsInstance(w.plots.tabs.widget(i), QScrollArea)
            self.assertEqual(w.plots.tabs.count(), 2)
            # Open video and add folder are always in sight, at the top.
            w.resize(1280, 760)
            w.show()
            QApplication.processEvents()
            for btn in (w.open_video_btn, w.queue_add_btn):
                self.assertTrue(btn.isVisible())
                top_left = btn.mapTo(w, btn.rect().topLeft())
                self.assertLess(top_left.y() + btn.height(), 120)
        finally:
            w.close()

    def test_the_a_plot_lines_are_the_config_values(self) -> None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            import pyqtgraph as pg
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest("PySide6 is not installed")
        from pef_fall_detector.gui.lab_window import LabWindow
        _app = QApplication.instance() or QApplication([])  # noqa: F841
        cfg = copy.deepcopy(TEST_CFG)
        cfg["quantity_a"].update({"band_floor": 0.2, "floor_reached": 0.4, "band_standing": 0.9})
        w = LabWindow(Config(cfg))
        try:
            lines = sorted(round(item.value(), 6) for item in w.plots.a_plot.getPlotItem().items
                           if isinstance(item, pg.InfiniteLine) and item.angle == 0)
            self.assertEqual(lines, [0.2, 0.4, 0.9])
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
