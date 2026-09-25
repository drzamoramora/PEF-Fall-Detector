"""Fase 6a: levantarse en curso al final de la observación (§3.5).

El §3.5 baja la severidad ante "torso and hip landmark recovery to upright".
Hasta la fase 6 solo el torso (T) decidía. A06-S14: T termina en 120-150 grados
("en el suelo") mientras la cadera sube 0.70-0.81 torsos en los últimos 2 s:
se está levantando. Estas pruebas fijan la medida y la regla.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.state_machine import (  # noqa: E402
    CONFIRMED_FALL,
    MODERATE,
    SEVERE,
    Stage3Evaluator,
)

DT = 1.0 / 30.0
NAN = float("nan")
TORSO = 100.0


def _ev(rise_torsos: float = 0.0, threshold: float = 0.45, t_final: float = 130.0,
        seconds: float = 3.0, jitter_frame: bool = False) -> Stage3Evaluator:
    """Observe `seconds` on the floor; the hip rises linearly over the last 2 s."""
    ev = Stage3Evaluator(defer_to_end=True, getup_rise_torsos=threshold,
                         getup_window_s=2.0)
    ev.start(0.0)
    n = int(round(seconds / DT))
    start_rise = n - int(round(2.0 / DT))
    for i in range(n):
        ts = i * DT
        frac = max(0.0, (i - start_rise) / (n - 1 - start_rise))
        y = 500.0 - rise_torsos * TORSO * frac
        if jitter_frame and i == n - 1:
            y = 100.0                                 # one wild frame at the end
        ev.observe(ts, t_final, 0.1, NAN, NAN)
        ev.observe_hip(ts, y, TORSO)
    ev.finalise()
    return ev


class TestRise(unittest.TestCase):

    def test_a_rising_hip_measures_its_rise(self) -> None:
        # Medians of the first and last half-second: a bit under the full rise.
        r = _ev(rise_torsos=0.8).getup_rise()
        self.assertGreater(r, 0.55)
        self.assertLess(r, 0.8)

    def test_a_still_hip_measures_zero(self) -> None:
        self.assertAlmostEqual(_ev(rise_torsos=0.0).getup_rise(), 0.0, places=6)

    def test_a_sinking_hip_is_negative(self) -> None:
        self.assertLess(_ev(rise_torsos=-0.8).getup_rise(), 0.0)

    def test_one_wild_frame_cannot_fake_a_rise(self) -> None:
        self.assertAlmostEqual(_ev(rise_torsos=0.0, jitter_frame=True).getup_rise(),
                               0.0, places=6)

    def test_too_short_a_window_is_not_measured(self) -> None:
        ev = Stage3Evaluator(defer_to_end=True, getup_rise_torsos=0.45)
        ev.start(0.0)
        for i in range(30):                           # 1 s < 0.75 x 2 s
            ev.observe_hip(i * DT, 500.0 - 5 * i, TORSO)
        self.assertTrue(ev.getup_rise() != ev.getup_rise())   # NaN

    def test_a_thinly_sampled_start_is_not_measured(self) -> None:
        # Two frames, then detection back 0.6 s later: the first half-second
        # has 2 samples, too few for its median to mean anything.
        ev = Stage3Evaluator(defer_to_end=True, getup_rise_torsos=0.45)
        ev.start(0.0)
        ev.observe_hip(0.0, 400.0, TORSO)
        ev.observe_hip(DT, 400.0, TORSO)
        for i in range(18, 60):
            ev.observe_hip(i * DT, 500.0, TORSO)
        self.assertTrue(ev.getup_rise() != ev.getup_rise())   # NaN

    def test_only_the_last_window_is_kept(self) -> None:
        ev = _ev(rise_torsos=0.8, seconds=6.0)
        self.assertGreaterEqual(ev._hip[0][0], ev._hip[-1][0] - 2.0 - 1e-9)

    def test_nothing_is_kept_before_start_or_with_bad_input(self) -> None:
        ev = Stage3Evaluator(defer_to_end=True, getup_rise_torsos=0.45)
        ev.observe_hip(0.0, 500.0, TORSO)             # not started
        ev.start(0.0)
        ev.observe_hip(0.1, NAN, TORSO)
        ev.observe_hip(0.2, 500.0, 0.0)
        ev.observe_hip(0.3, 500.0, NAN)
        self.assertEqual(ev._hip, [])

    def test_start_forgets_the_previous_event(self) -> None:
        ev = _ev(rise_torsos=0.8)
        ev.start(10.0)
        self.assertEqual(ev._hip, [])


class TestVerdict(unittest.TestCase):

    def test_down_but_getting_up_is_moderate_with_its_reason(self) -> None:
        ev = _ev(rise_torsos=0.8)
        self.assertEqual(ev.verdict(), (CONFIRMED_FALL, MODERATE))
        self.assertIn("la cadera subio", ev.reason)
        self.assertIn("se esta levantando", ev.reason)

    def test_down_and_still_stays_severe(self) -> None:
        ev = _ev(rise_torsos=0.1)
        self.assertEqual(ev.verdict(), (CONFIRMED_FALL, SEVERE))
        self.assertIn("termino en el suelo", ev.reason)

    def test_threshold_is_inclusive_and_read_from_the_rise(self) -> None:
        r = _ev(rise_torsos=0.8).getup_rise()
        self.assertEqual(_ev(rise_torsos=0.8, threshold=r).verdict()[1], MODERATE)
        self.assertEqual(_ev(rise_torsos=0.8, threshold=r + 1e-6).verdict()[1], SEVERE)

    def test_zero_disables_it(self) -> None:
        self.assertEqual(_ev(rise_torsos=0.8, threshold=0.0).verdict(),
                         (CONFIRMED_FALL, SEVERE))

    def test_it_only_touches_the_down_case(self) -> None:
        # Mid band (sat up): already moderate by T, and the reason is T's.
        ev = _ev(rise_torsos=0.8, t_final=45.0)
        self.assertEqual(ev.verdict()[1], MODERATE)
        self.assertIn("sentado o arrodillado", ev.reason)

    def test_unmeasured_rise_leaves_it_severe(self) -> None:
        ev = Stage3Evaluator(defer_to_end=True, getup_rise_torsos=0.45)
        ev.start(0.0)
        for i in range(90):
            ev.observe(i * DT, 130.0, 0.1, NAN, NAN)  # no hip fed at all
        ev.finalise()
        self.assertEqual(ev.verdict(), (CONFIRMED_FALL, SEVERE))


class TestWiring(unittest.TestCase):

    def test_config_turns_it_on(self) -> None:
        from pef_fall_detector.config import load_config
        from pef_fall_detector.pipeline import FramePipeline
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertEqual(float(cfg.stage3.getup_rise_torsos), 0.45)
        st3 = FramePipeline(cfg, labelling=True).machine.stage3
        self.assertEqual((st3.getup_rise_torsos, st3.getup_window_s), (0.45, 2.0))

    def test_old_configs_keep_it_off(self) -> None:
        from pef_fall_detector.config import Config
        from pef_fall_detector.pipeline import FramePipeline
        from tests.test_pipeline import TEST_CFG
        cfg = copy.deepcopy(TEST_CFG)
        cfg["stage3"].pop("getup_rise_torsos", None)
        self.assertEqual(FramePipeline(Config(cfg)).machine.stage3.getup_rise_torsos, 0.0)

    def _clip(self, rise_px_per_frame: float):
        """Fall, lie with T 90; the hip climbs over the last 2 s; clip ends."""
        from pef_fall_detector.config import Config
        from pef_fall_detector.pipeline import FramePipeline
        from tests.test_main_headless import _toppling_pose
        from tests.test_pipeline import TEST_CFG
        cfg = copy.deepcopy(TEST_CFG)
        cfg["stage3"]["getup_rise_torsos"] = 0.45
        pipe = FramePipeline(Config(cfg), labelling=True)
        frames = [_toppling_pose(i, i * DT, angle_deg=i * 4.5, hip_y=500.0 + i * 12.0)
                  for i in range(20)]
        frames += [_toppling_pose(i, i * DT, angle_deg=90.0, hip_y=740.0)
                   for i in range(20, 120)]
        frames += [_toppling_pose(i, i * DT, angle_deg=90.0,
                                  hip_y=740.0 - rise_px_per_frame * (i - 119))
                   for i in range(120, 180)]
        for pf in frames:
            pipe.analyze(pf)
        pipe.finalise(frames[-1].timestamp)
        return pipe.machine.events[-1]

    def test_the_pipeline_feeds_the_hip_while_observing(self) -> None:
        # 200 px trunk; 4 px/frame over 60 frames = 240 px = 1.2 torsos.
        ev = self._clip(4.0)
        self.assertEqual(ev.severity, MODERATE)
        self.assertIn("se esta levantando", ev.reason)

    def test_the_pipeline_lying_still_stays_severe(self) -> None:
        self.assertEqual(self._clip(0.0).severity, SEVERE)


if __name__ == "__main__":
    unittest.main()
