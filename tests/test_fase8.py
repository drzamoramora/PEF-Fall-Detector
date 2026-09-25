"""Fases 8.2 y 8.3: la Etapa 3 lee la cantidad A (altura 3D de la cabeza).

8.2: con A calibrada, la postura final sale de A (<= 0.26 en el suelo,
>= 0.82 de pie, en medio sentado o arrodillado). 8.3: si la cabeza nunca bajo
de 0.31 alrededor del disparo y la cadera termino elevada, no hubo caida
(stage3_not_down, NoFall, sin alarma); se abstiene si A no estaba calibrada al
disparar. Las dos solo en modo etiquetado, igual que las fases 3 y 6a.
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.alerts import AlertDispatcher  # noqa: E402
from pef_fall_detector.classification import (  # noqa: E402
    NO_FALL,
    UNDETERMINED,
    class_for_event,
    classify_clip,
)
from pef_fall_detector.config import Config, load_config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.state_machine import (  # noqa: E402
    CONFIRMED_FALL,
    MILD,
    MODERATE,
    NOT_DOWN,
    NULLIFIED,
    SEVERE,
    Stage3Evaluator,
)
from tests.test_main_headless import _toppling_pose  # noqa: E402
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402
from tests.test_quantity_a import (  # noqa: E402
    KNEELING,
    LYING,
    SITTING_FLOOR,
    STANDING,
    _body,
    _pose,
    _with_world,
)

DT = 1.0 / 30.0
NAN = float("nan")


def _ev(a_head, a_hip=0.5, *, t_final=90.0, calibrated=True, u1=True, u2=True,
        history=(), still=0.1, seconds=5.0, trigger_t=0.0, arm=True) -> Stage3Evaluator:
    """Observe `seconds` from the trigger with A given per timestamp by `a_head`."""
    ev = Stage3Evaluator(defer_to_end=True, persistent_still_s=0.5, persistent_fraction=0.5,
                         a_band_floor=0.26 if u1 else 0.0, a_band_standing=0.82,
                         a_reach_floor=0.31 if u2 else 0.0, a_hip_raised=0.22)
    ev.start(trigger_t + 0.3)
    if arm:
        ev.arm_a(trigger_t, calibrated, history)
    n = int(round(seconds / DT))
    for i in range(n):
        ts = trigger_t + 0.3 + i * DT
        head = a_head(ts) if callable(a_head) else a_head
        hip = a_hip(ts) if callable(a_hip) else a_hip
        ev.observe(ts, t_final, still, NAN, NAN)
        ev.observe_a(ts, head, hip)
    ev.finalise()
    return ev


class TestReachedFloor(unittest.TestCase):
    """8.3: "¿llego al suelo?"."""

    def test_head_high_and_hip_raised_is_not_a_fall(self) -> None:
        ev = _ev(0.5, 0.45)
        self.assertEqual(ev.verdict(), (NOT_DOWN, ""))
        self.assertIn("no llego al suelo", ev.reason)
        self.assertIn("A minima 0.50 > 0.31", ev.reason)

    def test_head_that_reached_the_floor_is_judged_as_before(self) -> None:
        ev = _ev(lambda t: 0.2 if t < 2.0 else 0.6, 0.45)
        self.assertNotEqual(ev.verdict()[0], NOT_DOWN)

    def test_the_cut_is_strict_and_at_031(self) -> None:
        self.assertNotEqual(_ev(0.31, 0.45).verdict()[0], NOT_DOWN)
        self.assertEqual(_ev(0.32, 0.45).verdict()[0], NOT_DOWN)

    def test_hip_on_the_floor_protects_the_slumped_fall(self) -> None:
        # A17-S2: cabeza 0.44 contra la pared, cadera 0.07 en el piso.
        ev = _ev(0.44, 0.07)
        self.assertEqual(ev.verdict()[0], CONFIRMED_FALL)

    def test_abstains_without_calibration_at_the_trigger(self) -> None:
        # A16-S3 en el Mac: calibra 2.6 s despues de caer, ya de pie.
        ev = _ev(0.5, 0.45, calibrated=False)
        self.assertNotEqual(ev.verdict()[0], NOT_DOWN)

    def test_only_the_window_counts(self) -> None:
        # Floor readings after trigger + 4 s do not say it reached the floor.
        late = _ev(lambda t: 0.1 if t > 4.5 else 0.6, 0.45, seconds=6.0)
        self.assertEqual(late.verdict()[0], NOT_DOWN)
        self.assertAlmostEqual(late.a_min_window, 0.6)

    def test_the_second_before_the_trigger_counts(self) -> None:
        # Stage 3 starts after Stage 2; the trigger's own second comes from history.
        hist = [(-0.9 + i * DT, 0.2, 0.45) for i in range(10)]
        ev = _ev(0.6, 0.45, history=hist)
        self.assertNotEqual(ev.verdict()[0], NOT_DOWN)
        self.assertAlmostEqual(ev.a_min_window, 0.2)

    def test_history_older_than_the_window_is_ignored(self) -> None:
        hist = [(-3.0 + i * DT, 0.1, 0.45) for i in range(10)]
        ev = _ev(0.6, 0.45, history=hist)
        self.assertEqual(ev.verdict()[0], NOT_DOWN)

    def test_switched_off(self) -> None:
        self.assertNotEqual(_ev(0.5, 0.45, u2=False).verdict()[0], NOT_DOWN)

    def test_nothing_before_arming_and_reset_forgets(self) -> None:
        ev = _ev(0.5, 0.45, arm=False)
        # Unarmed means no A at all: T decides (90 deg -> severe), not A (0.5).
        self.assertEqual(ev.verdict(), (CONFIRMED_FALL, SEVERE))
        self.assertTrue(math.isnan(ev.a_min_window))
        idle = Stage3Evaluator(a_reach_floor=0.31)
        idle.arm_a(0.0, True)
        self.assertFalse(idle.a_armed, "sin start() no hay episodio que armar")
        ev2 = _ev(0.5, 0.45)
        ev2.reset()
        self.assertFalse(ev2.a_armed)
        self.assertTrue(math.isnan(ev2.a_min_window))

    def test_nan_readings_are_not_evidence(self) -> None:
        self.assertNotEqual(_ev(NAN, 0.45).verdict()[0], NOT_DOWN)
        self.assertNotEqual(_ev(0.5, NAN).verdict()[0], NOT_DOWN)

    def test_counts_as_nofall_and_raises_no_alarm(self) -> None:
        self.assertEqual(class_for_event(NOT_DOWN, ""), NO_FALL)
        self.assertEqual(class_for_event(NOT_DOWN, "", unresolved_as=UNDETERMINED), NO_FALL)
        self.assertFalse(AlertDispatcher().should_dispatch(NOT_DOWN))


class TestSeverityFromA(unittest.TestCase):
    """8.2: la postura final sale de A cuando esta calibrada."""

    def test_three_bands(self) -> None:
        self.assertEqual(_ev(0.10, 0.1, u2=False, t_final=20.0).verdict(), (CONFIRMED_FALL, SEVERE))
        self.assertEqual(_ev(0.95, 0.9, u2=False, t_final=90.0).verdict(), (NULLIFIED, MILD))
        self.assertEqual(_ev(0.50, 0.1, u2=False, t_final=90.0).verdict(), (CONFIRMED_FALL, MODERATE))

    def test_band_edges(self) -> None:
        self.assertEqual(_ev(0.26, 0.1, u2=False).verdict()[1], SEVERE)
        self.assertEqual(_ev(0.82, 0.9, u2=False).verdict()[1], MILD)

    def test_reason_names_a_and_t(self) -> None:
        r = _ev(0.10, 0.1, u2=False, t_final=28.0).reason
        self.assertIn("A final 0.10", r)
        self.assertIn("T final 28 deg", r)

    def test_persistence_still_applies_in_the_middle(self) -> None:
        ev = _ev(0.50, 0.1, u2=False, still=2.0)
        self.assertEqual(ev.verdict(), (CONFIRMED_FALL, SEVERE))
        self.assertIn("la inmovilidad persistio", ev.reason)

    def test_without_a_t_decides_as_before(self) -> None:
        self.assertEqual(_ev(NAN, NAN, u2=False, t_final=90.0).verdict(), (CONFIRMED_FALL, SEVERE))
        self.assertEqual(_ev(0.95, 0.9, u1=False, u2=False, t_final=90.0).verdict(),
                         (CONFIRMED_FALL, SEVERE))

    def test_getting_up_keeps_precedence(self) -> None:
        ev = Stage3Evaluator(defer_to_end=True, getup_rise_torsos=0.45, getup_window_s=2.0,
                             a_band_floor=0.26)
        ev.start(0.0)
        ev.arm_a(0.0, True)
        n = int(round(3.0 / DT))
        for i in range(n):
            ts = i * DT
            frac = max(0.0, (i - (n - 60)) / 59)
            ev.observe(ts, 130.0, 0.1, NAN, NAN)
            ev.observe_hip(ts, 500.0 - 80.0 * frac, 100.0)
            ev.observe_a(ts, 0.1, 0.1)
        ev.finalise()
        self.assertEqual(ev.verdict(), (CONFIRMED_FALL, MODERATE))
        self.assertIn("se esta levantando", ev.reason)

    def test_bad_bands(self) -> None:
        with self.assertRaises(ValueError):
            Stage3Evaluator(a_band_floor=0.5, a_band_standing=0.4)
        with self.assertRaises(ValueError):
            Stage3Evaluator(a_reach_after_s=0.0)


# ---------------------------------------------------------------------------
# Through the pipeline: stand 1.5 s, topple in 2D, then a 3D posture.

SITTING_CHAIR = _pose(ankles=((-0.1, 0.45, 0.45), (0.1, 0.45, 0.45)),
                      knees=((-0.1, 0.0, 0.45), (0.1, 0.0, 0.45)),
                      shoulders_y=-0.5, nose=(0, -0.7, 0))   # bent knees: never "standing"


def _clip(after, before=STANDING):
    frames = [make_pose(i, i * DT) for i in range(45)]
    frames += [_toppling_pose(i, i * DT, angle_deg=(i - 45) * 4.5, hip_y=500.0 + (i - 45) * 12.0)
               for i in range(45, 65)]
    frames += [_toppling_pose(i, i * DT, angle_deg=90.0, hip_y=740.0) for i in range(65, 185)]
    for i, pf in enumerate(frames):
        _with_world(pf, _body(before if i < 45 else after))
    return frames


def _cfg(u1=True, u2=True):
    cfg = copy.deepcopy(TEST_CFG)
    cfg["quantity_a"] = {"severity_from_a": u1, "reach_check": u2, "band_floor": 0.26,
                         "band_standing": 0.82, "floor_reached": 0.31, "hip_raised": 0.22,
                         "reach_before_s": 1.0, "reach_after_s": 4.0}
    return Config(cfg)


def _run(frames, cfg):
    pipe = FramePipeline(cfg, labelling=True)
    res = [pipe.analyze(pf) for pf in frames]
    pipe.finalise(res[-1].pose.timestamp)
    return pipe


class TestPipeline(unittest.TestCase):

    def test_kneeling_after_the_trigger_never_reached_the_floor(self) -> None:
        pipe = _run(_clip(KNEELING), _cfg())
        ev = pipe.machine.events[-1]
        self.assertEqual(ev.verdict, NOT_DOWN)
        self.assertEqual(classify_clip(pipe.machine.events)[0], NO_FALL)
        self.assertFalse(pipe.alerts.should_dispatch(ev.verdict))

    def test_lying_is_severe_from_a(self) -> None:
        pipe = _run(_clip(LYING), _cfg())
        ev = pipe.machine.events[-1]
        self.assertEqual((ev.verdict, ev.severity), (CONFIRMED_FALL, SEVERE))
        self.assertIn("A final 0.03", ev.reason)
        self.assertTrue(pipe.alerts.should_dispatch(ev.verdict))

    def test_sitting_on_the_floor_is_moderate_not_rejected(self) -> None:
        pipe = _run(_clip(SITTING_FLOOR), _cfg())
        ev = pipe.machine.events[-1]
        self.assertEqual((ev.verdict, ev.severity), (CONFIRMED_FALL, MODERATE))
        self.assertIn("sentado o arrodillado", ev.reason)

    def test_the_second_before_stage3_comes_from_history(self) -> None:
        # Lying only during the topple, kneeling afterwards: the floor reading
        # exists only before Stage 3 starts, so only the history carries it.
        frames = _clip(KNEELING)
        for i in range(45, 65):
            _with_world(frames[i], _body(LYING))
        ev = _run(frames, _cfg()).machine.events[-1]
        self.assertNotEqual(ev.verdict, NOT_DOWN)

    def test_late_calibration_abstains(self) -> None:
        # A16-S3 en el Mac: sin calibrar al caer, calibra ya de pie despues.
        frames = _clip(LYING, before=SITTING_CHAIR)
        for i in range(105, 185):
            frames[i] = _with_world(make_pose(i, i * DT), _body(STANDING))
        pipe = _run(frames, _cfg())
        self.assertTrue(pipe._gravity.calibrated, "el escenario debe calibrar tarde")
        self.assertNotEqual(pipe.machine.events[-1].verdict, NOT_DOWN)

    def test_frames_after_stage3_starts_are_read(self) -> None:
        # Down during the fall, kneeling at the end: the end posture is only
        # in the frames Stage 3 observes itself.
        frames = _clip(KNEELING)
        for i in range(45, 100):
            _with_world(frames[i], _body(LYING))
        ev = _run(frames, _cfg()).machine.events[-1]
        self.assertEqual((ev.verdict, ev.severity), (CONFIRMED_FALL, MODERATE))

    def test_history_is_bounded(self) -> None:
        pipe = _run(_clip(LYING), _cfg())
        self.assertLessEqual(len(pipe._a_history), int(5.0 / DT) + 2)

    def test_uncalibrated_at_the_trigger_abstains(self) -> None:
        # Never standing in 3D before the fall: no calibration, T decides.
        pipe = _run(_clip(KNEELING, before=SITTING_CHAIR), _cfg())
        ev = pipe.machine.events[-1]
        self.assertNotEqual(ev.verdict, NOT_DOWN)
        self.assertNotIn("A final", ev.reason)

    def test_switches_off_reproduce_81(self) -> None:
        base = _run(_clip(KNEELING), Config(copy.deepcopy(TEST_CFG)))
        off = _run(_clip(KNEELING), _cfg(False, False))
        key = [(e.timestamp, e.verdict, e.severity, e.reason) for e in base.machine.events]
        self.assertEqual(key, [(e.timestamp, e.verdict, e.severity, e.reason)
                               for e in off.machine.events])
        self.assertNotEqual(base.machine.events[-1].verdict, NOT_DOWN)

    def test_project_config(self) -> None:
        qa = load_config(Path(__file__).resolve().parents[1] / "config.yaml").as_dict()["quantity_a"]
        self.assertTrue(qa["severity_from_a"])
        self.assertTrue(qa["reach_check"])
        self.assertEqual((qa["band_floor"], qa["band_standing"], qa["floor_reached"],
                          qa["hip_raised"], qa["reach_before_s"], qa["reach_after_s"]),
                         (0.26, 0.82, 0.31, 0.22, 1.0, 4.0))


if __name__ == "__main__":
    unittest.main()
