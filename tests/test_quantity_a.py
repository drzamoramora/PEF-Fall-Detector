"""Cantidad A (fase 8.1): altura 3D de la cabeza contra la gravedad.

Solo se mide y se registra: ninguna etapa la lee. Estas pruebas fijan la
geometría (con una cámara inclinada, que es el caso real), la calibración
causal con el propio sujeto de pie, y que ninguna decisión cambie.
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
from tests.test_main_headless import _toppling_pose  # noqa: E402
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402

DT = 1.0 / 30.0


def _body(points: dict[int, tuple[float, float, float]], pitch_deg: float = 30.0) -> np.ndarray:
    """33×3 world landmarks from a gravity-frame body, seen by a pitched camera.

    Gravity frame: y DOWN (MediaPipe's convention), hips at the origin. The
    camera looks down by ``pitch_deg``, so its axes are rotated about x: the
    calibration must recover "up" from the subject, not assume (0, -1, 0).
    """
    w = np.zeros((33, 3))
    for i, p in points.items():
        w[i] = p
    a = math.radians(pitch_deg)
    rot = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    return w @ rot.T


def _pose(ankles, knees, shoulders_y, nose, hips=((-0.1, 0, 0), (0.1, 0, 0)), shoulders_z=0.0):
    return {0: nose, 11: (-0.18, shoulders_y, shoulders_z), 12: (0.18, shoulders_y, shoulders_z),
            23: hips[0], 24: hips[1], 25: knees[0], 26: knees[1], 27: ankles[0], 28: ankles[1]}


STANDING = _pose(ankles=((-0.1, 0.9, 0), (0.1, 0.9, 0)), knees=((-0.1, 0.45, 0), (0.1, 0.45, 0)),
                 shoulders_y=-0.5, nose=(0, -0.7, 0))                   # head 1.6 m above ankles
LYING = _pose(ankles=((-0.1, 0, 0.9), (0.1, 0, 0.9)), knees=((-0.1, 0, 0.45), (0.1, 0, 0.45)),
              shoulders_y=0.0, shoulders_z=-0.5, nose=(0, -0.05, -0.7))  # head 0.05 m above
KNEELING = _pose(ankles=((-0.1, 0.45, 0.45), (0.1, 0.45, 0.45)),
                 knees=((-0.1, 0.45, 0), (0.1, 0.45, 0)),
                 shoulders_y=-0.5, nose=(0, -0.7, 0))                   # 1.15 m → 0.72
SITTING_FLOOR = _pose(ankles=((-0.1, 0.05, 0.9), (0.1, 0.05, 0.9)),
                      knees=((-0.1, 0.0, 0.45), (0.1, 0.0, 0.45)),
                      shoulders_y=-0.5, nose=(0, -0.7, 0))              # 0.75 m → 0.47


def _calibrated(pitch=30.0, seconds=1.0) -> GravityCalibrator:
    g = GravityCalibrator(min_standing_s=0.5)
    for i in range(int(round(seconds / DT))):
        g.observe(i * DT, _body(STANDING, pitch), standing=True)
    return g


class TestKneeAngle(unittest.TestCase):

    def test_straight_legs_are_180(self) -> None:
        self.assertAlmostEqual(knee_angle_3d(_body(STANDING)), 180.0, places=4)

    def test_kneeling_is_a_right_angle(self) -> None:
        self.assertAlmostEqual(knee_angle_3d(_body(KNEELING)), 90.0, places=4)

    def test_unusable_input_is_nan(self) -> None:
        self.assertTrue(math.isnan(knee_angle_3d(np.empty((0, 3)))))
        bad = _body(STANDING); bad[25, 0] = np.nan
        self.assertTrue(math.isnan(knee_angle_3d(bad)))


class TestCalibration(unittest.TestCase):

    def test_nan_until_half_a_second_of_standing(self) -> None:
        g = GravityCalibrator(min_standing_s=0.5)
        for i in range(15):                                  # 14 steps = 0.47 s
            g.observe(i * DT, _body(STANDING), standing=True)
        self.assertFalse(g.calibrated)
        self.assertTrue(math.isnan(g.head_ratio(_body(STANDING))))
        g.observe(15 * DT, _body(STANDING), standing=True)   # 0.50 s
        g.observe(16 * DT, _body(STANDING), standing=True)
        self.assertTrue(g.calibrated)

    def test_up_is_recovered_from_a_pitched_camera(self) -> None:
        g = _calibrated(pitch=30.0)
        a = math.radians(30.0)
        expected = np.array([0.0, -math.cos(a), -math.sin(a)])  # (0,-1,0) seen by the camera
        self.assertLess(math.degrees(math.acos(float(np.dot(g.up, expected)))), 0.5)
        self.assertAlmostEqual(g.stature, 1.6, places=3)

    def test_postures(self) -> None:
        g = _calibrated(pitch=30.0)
        self.assertAlmostEqual(g.head_ratio(_body(STANDING)), 1.0, places=3)
        self.assertAlmostEqual(g.head_ratio(_body(LYING)), 0.05 / 1.6, places=3)
        self.assertAlmostEqual(g.head_ratio(_body(KNEELING)), 1.15 / 1.6, places=3)
        self.assertAlmostEqual(g.head_ratio(_body(SITTING_FLOOR)), 0.75 / 1.6, places=3)
        self.assertAlmostEqual(g.hip_ratio(_body(STANDING)), 1.0, places=3)
        self.assertAlmostEqual(g.hip_ratio(_body(SITTING_FLOOR)), 0.05 / 0.9, places=3)

    def test_the_same_posture_reads_the_same_under_any_camera_pitch(self) -> None:
        for pitch in (0.0, 15.0, 45.0):
            g = _calibrated(pitch=pitch)
            self.assertAlmostEqual(g.head_ratio(_body(KNEELING, pitch)), 1.15 / 1.6, places=3)

    def test_up_is_a_unit_vector_even_from_mixed_samples(self) -> None:
        g = GravityCalibrator()
        for i in range(40):                                   # camera shaking 20°..40°
            g.observe(i * DT, _body(STANDING, 20.0 if i % 2 else 40.0), standing=True)
        self.assertAlmostEqual(float(np.linalg.norm(g.up)), 1.0, places=9)

    def test_new_standing_frames_update_the_calibration(self) -> None:
        g = _calibrated(seconds=0.6)
        self.assertAlmostEqual(g.stature, 1.6, places=3)
        taller = dict(STANDING); taller[0] = (0, -0.9, 0)     # head 1.8 m above ankles
        for i in range(18, 120):
            g.observe(i * DT, _body(taller), standing=True)
        self.assertAlmostEqual(g.stature, 1.8, places=3)

    def test_non_standing_frames_never_calibrate(self) -> None:
        g = GravityCalibrator()
        for i in range(90):
            g.observe(i * DT, _body(LYING), standing=False)
        self.assertFalse(g.calibrated)
        self.assertIsNone(g.up)

    def test_a_gap_does_not_count_as_standing(self) -> None:
        g = GravityCalibrator(min_standing_s=0.5)
        for t in (0.0, 10.0, 20.0, 30.0, 40.0):               # five frames, 40 s apart
            g.observe(t, _body(STANDING), standing=True)
        self.assertAlmostEqual(g.standing_s, 0.4)             # 4 gaps, 0.1 s each
        self.assertFalse(g.calibrated)

    def test_memory_is_bounded_and_reset_forgets(self) -> None:
        g = GravityCalibrator(max_samples=10)
        for i in range(100):
            g.observe(i * DT, _body(STANDING), standing=True)
        self.assertEqual(len(g._ups), 10)
        g.reset()
        self.assertFalse(g.calibrated)
        self.assertEqual(g.standing_s, 0.0)

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            GravityCalibrator(min_standing_s=0.0)
        with self.assertRaises(ValueError):
            GravityCalibrator(max_samples=2)


def _with_world(pf, world):
    pf.world_landmarks = world
    return pf


def _clip(with_world: bool):
    """Stand 1.5 s (upright in the image, straight legs in 3D), fall, lie 4 s."""
    frames = [make_pose(i, i * DT) for i in range(45)]
    frames += [_toppling_pose(i, i * DT, angle_deg=(i - 45) * 4.5, hip_y=500.0 + (i - 45) * 12.0)
               for i in range(45, 65)]
    frames += [_toppling_pose(i, i * DT, angle_deg=90.0, hip_y=740.0) for i in range(65, 185)]
    if with_world:
        for i, pf in enumerate(frames):
            _with_world(pf, _body(STANDING if i < 45 else LYING))
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
        self.assertTrue(math.isnan(a[0]), "A antes de calibrar debe ser NaN")
        self.assertAlmostEqual(a[40], 1.0, places=3)          # calibrated, standing
        self.assertAlmostEqual(a[-1], 0.05 / 1.6, places=3)   # on the floor

    def test_no_decision_changes(self) -> None:
        with_w, res_w = self._run(True)
        without, res_0 = self._run(False)
        self.assertEqual([r.stage for r in res_w], [r.stage for r in res_0])
        ev_w = [(e.timestamp, e.verdict, e.severity, e.reason) for e in with_w.machine.events]
        ev_0 = [(e.timestamp, e.verdict, e.severity, e.reason) for e in without.machine.events]
        self.assertEqual(ev_w, ev_0)
        self.assertTrue(ev_w, "el escenario debe producir un evento")

    def test_upright_trunk_with_bent_knees_does_not_calibrate(self) -> None:
        # Sitting upright on a chair: T is 0 in the image, but the knees are
        # bent in 3D — that is not "standing", and must not set the height.
        pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)))
        sitting = _pose(ankles=((-0.1, 0.45, 0.45), (0.1, 0.45, 0.45)),
                        knees=((-0.1, 0.0, 0.45), (0.1, 0.0, 0.45)),
                        shoulders_y=-0.5, nose=(0, -0.7, 0))
        for i in range(60):
            pipe.analyze(_with_world(make_pose(i, i * DT), _body(sitting)))
        self.assertFalse(pipe._gravity.calibrated)

    def test_unreliable_frames_are_not_read(self) -> None:
        pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)))
        for i in range(60):
            pf = _with_world(make_pose(i, i * DT), _body(STANDING))
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

    def test_a_config_without_the_section_still_works(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        cfg.pop("quantity_a", None)
        pipe = FramePipeline(Config(cfg))
        self.assertEqual((pipe._a_standing_t, pipe._a_standing_knee), (20.0, 150.0))


class TestLabCurve(unittest.TestCase):

    def test_pef_lab_draws_a(self) -> None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest("PySide6 no esta instalado")
        from pef_fall_detector.config import load_config
        from pef_fall_detector.gui.lab_window import LabWindow
        _app = QApplication.instance() or QApplication([])  # noqa: F841
        w = LabWindow(load_config(Path(__file__).resolve().parents[1] / "config.yaml"))
        try:
            self.assertIn("A_head", w._series)
            self.assertIn("A_hip", w._series)
            self.assertIn(w.a_plot, w._curve_plots)
            # Regresion 25/09: la ventana crecia con cada grafica (alto
            # minimo 1341 px) y en el Mac los botones de abajo quedaban fuera
            # de la pantalla. Ahora las curvas van en un area con
            # desplazamiento y la ventana tiene que caber en una laptop.
            from PySide6.QtWidgets import QScrollArea
            self.assertLessEqual(w.minimumSizeHint().height(), 760)
            self.assertLessEqual(w.minimumSizeHint().width(), 1280)
            for i in range(w.curve_tabs.count()):
                self.assertIsInstance(w.curve_tabs.widget(i), QScrollArea)
            self.assertEqual(w.curve_tabs.count(), 2)
            # Abrir video y agregar carpeta estan siempre a la vista, arriba.
            w.resize(1280, 760)
            w.show()
            QApplication.processEvents()
            for btn in (w.open_video_btn, w.queue_add_btn):
                self.assertTrue(btn.isVisible())
                top_left = btn.mapTo(w, btn.rect().topLeft())
                self.assertLess(top_left.y() + btn.height(), 120)
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
