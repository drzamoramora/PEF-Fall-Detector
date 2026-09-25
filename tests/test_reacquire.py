"""Fase 4: re-adquisición del rastreador de MediaPipe (pose_frontend).

El caso real es A16-S1: 5.4 s con el núcleo a visibilidad 0.00-0.05 mientras
el rastreador seguía "enganchado" a una región vieja y el sujeto caía a la
vista. Aquí se prueba la regla sin MediaPipe: un grafo falso cuenta los
reinicios.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.pose_frontend import PoseFrontend  # noqa: E402

DT = 1.0 / 30.0


class _Graph:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def _frontend(wait: float, threshold: float = 0.5) -> PoseFrontend:
    fe = PoseFrontend.__new__(PoseFrontend)      # sin cargar MediaPipe
    fe._pose = _Graph()
    fe._visibility_threshold = threshold
    fe._reacquire_after_s = wait
    fe._low_since = None
    fe._reseeded = False
    fe.reacquisitions = []
    return fe


def _feed(fe: PoseFrontend, visibilities, t0: float = 0.0) -> None:
    for i, v in enumerate(visibilities):
        fe._watch_tracker(t0 + i * DT, v)


class TestReacquire(unittest.TestCase):

    def test_a_sustained_low_visibility_lock_reseeds_the_tracker(self) -> None:
        fe = _frontend(0.5)
        _feed(fe, [0.02] * 20)                    # 0.63 s pegado
        self.assertEqual(fe._pose.resets, 1)
        self.assertAlmostEqual(fe.reacquisitions[0], 15 * DT, places=6)

    def test_shorter_than_the_wait_does_nothing(self) -> None:
        fe = _frontend(0.5)
        _feed(fe, [0.02] * 15)                    # 0.47 s
        self.assertEqual(fe._pose.resets, 0)

    def test_one_good_frame_restarts_the_count(self) -> None:
        fe = _frontend(0.5)
        _feed(fe, [0.02] * 14 + [0.9] + [0.02] * 14)
        self.assertEqual(fe._pose.resets, 0)

    def test_at_the_threshold_is_not_low(self) -> None:
        fe = _frontend(0.5)
        _feed(fe, [0.5] * 60)
        self.assertEqual(fe._pose.resets, 0)

    def test_a_long_lock_is_retried_every_wait(self) -> None:
        fe = _frontend(0.5)
        _feed(fe, [0.02] * 64)                    # ~2.1 s
        self.assertEqual(fe._pose.resets, 4)

    def test_zero_disables_it(self) -> None:
        fe = _frontend(0.0)
        _feed(fe, [0.0] * 300)
        self.assertEqual(fe._pose.resets, 0)
        self.assertEqual(fe.reacquisitions, [])


class TestLostDetectionRestartsTheCount(unittest.TestCase):

    def test_a_frame_without_a_person_is_not_a_locked_tracker(self) -> None:
        # Sin persona, MediaPipe ya vuelve a correr el detector por su cuenta;
        # ese cuadro no puede sumar al tiempo "pegado".
        import types
        import numpy as np
        fe = _frontend(0.5)
        fe._pose.process = lambda rgb: types.SimpleNamespace(pose_landmarks=None)
        _feed(fe, [0.02] * 14)
        fe.process(np.zeros((4, 4, 3), np.uint8), 14, 14 * DT)
        _feed(fe, [0.02] * 14, t0=15 * DT)
        self.assertEqual(fe._pose.resets, 0)


    def test_process_watches_every_detected_frame(self) -> None:
        import types
        import numpy as np
        lm = [types.SimpleNamespace(x=0.5, y=0.1 + 0.02 * k, z=0.0, visibility=0.02)
              for k in range(33)]
        fe = _frontend(0.5)
        fe._pose.process = lambda rgb: types.SimpleNamespace(
            pose_landmarks=types.SimpleNamespace(landmark=lm),
            pose_world_landmarks=None)
        frame = np.zeros((4, 4, 3), np.uint8)
        flags = [fe.process(frame, i, i * DT).reacquired for i in range(20)]
        self.assertEqual(fe._pose.resets, 1)
        # Reset after frame 15: frame 16 is the fresh detection, and only it.
        self.assertEqual([i for i, f in enumerate(flags) if f], [16])


class TestConfigWiring(unittest.TestCase):

    def test_config_has_it_on(self) -> None:
        from pef_fall_detector.config import load_config
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertEqual(float(cfg.pose.reacquire_after_s), 0.5)

    def test_the_pipeline_passes_it_to_the_frontend(self) -> None:
        import copy
        from unittest import mock
        from pef_fall_detector.config import Config
        from pef_fall_detector.pipeline import FramePipeline
        from tests.test_pipeline import TEST_CFG
        cfg = copy.deepcopy(TEST_CFG)
        cfg["pose"]["reacquire_after_s"] = 0.7
        pipe = FramePipeline(Config(cfg))
        with mock.patch("pef_fall_detector.pipeline.PoseFrontend") as fake:
            fake.return_value.process.side_effect = RuntimeError("basta")
            with self.assertRaises(RuntimeError):
                pipe.process(None, 0, 0.0)
        kw = fake.call_args.kwargs
        self.assertEqual(kw["reacquire_after_s"], 0.7)
        self.assertEqual(kw["visibility_threshold"],
                         float(cfg["pose"]["visibility_threshold"]))

    def test_old_configs_without_the_key_keep_it_off(self) -> None:
        import copy
        from unittest import mock
        from pef_fall_detector.config import Config
        from pef_fall_detector.pipeline import FramePipeline
        from tests.test_pipeline import TEST_CFG
        cfg = copy.deepcopy(TEST_CFG)
        cfg["pose"].pop("reacquire_after_s", None)
        pipe = FramePipeline(Config(cfg))
        with mock.patch("pef_fall_detector.pipeline.PoseFrontend") as fake:
            fake.return_value.process.side_effect = RuntimeError("basta")
            with self.assertRaises(RuntimeError):
                pipe.process(None, 0, 0.0)
        self.assertEqual(fake.call_args.kwargs["reacquire_after_s"], 0.0)


class TestDiscontinuity(unittest.TestCase):
    """A reacquired frame breaks continuity like a lost one.

    NOTE: a plain detection loss deliberately does NOT clear the trigger's
    peak window. Measured (fase 4): clearing it on every loss lost A16-S2,
    A17-S4, A14-S3 and A09-S4 — real falls whose detection flickers during
    the fall itself — for one gain. The reacquisition case differs: there
    the frames before the reset were a stale lock, not the same body.
    """

    def _pipe(self):
        import copy
        from pef_fall_detector.config import Config
        from pef_fall_detector.pipeline import FramePipeline
        from tests.test_pipeline import TEST_CFG
        cfg = copy.deepcopy(TEST_CFG)
        cfg["stage1"]["trigger_formulation"] = "score"
        cfg["stage1"]["trigger_peak_window_s"] = 0.8
        return FramePipeline(Config(cfg))

    def _stale_peak_then(self, gap_frame):
        """Inverted skeleton (T 170, not falling), a break, then a slow sink.

        Across the break, the peak window would pair the stale T with the new
        V; nothing about the new skeleton is a fall.
        """
        from tests.test_main_headless import _toppling_pose
        from tests.test_pipeline import make_pose
        pipe = self._pipe()
        # A17-S3: the clip opens on a still, inverted skeleton (V 0: no fire).
        for i in range(15):
            pipe.analyze(_toppling_pose(i, i * DT, angle_deg=170.0, hip_y=500.0))
        pipe.analyze(gap_frame(15))
        stages = []
        for i in range(16, 60):                    # 1 px/frame down: V < 0, T 0
            r = pipe.analyze(make_pose(i, i * DT, hip_y=500.0 + (i - 16),
                                       shoulder_y=300.0 + (i - 16)))
            stages.append(r.stage)
        return pipe, stages

    def test_a_reacquired_frame_is_a_discontinuity(self) -> None:
        from tests.test_pipeline import make_pose

        def reacq(i):
            pf = make_pose(i, i * DT, hip_y=499.0, shoulder_y=299.0)
            pf.reacquired = True
            return pf
        pipe, stages = self._stale_peak_then(reacq)
        self.assertEqual(set(stages), {"MONITORING"}, "disparo con un T viejo")


if __name__ == "__main__":
    unittest.main()
