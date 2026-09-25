"""Fase 2 (variante B) y fase 3 del plan de caídas (23/09).

Fase 2: si se pierde la detección durante la Etapa 3, el evento se cierra con
lo último que se vio del sujeto cuando se lo vio EN EL SUELO; si se lo vio de
pie, se abandona como antes. Caso real: A08-S3.

Fase 3: un final ni tumbado ni de pie (sentado, arrodillado, recostado) es
``severe`` en vez de ``moderate`` cuando la inmovilidad persistió (§3.5).
Caso real: A17-S2 y A17-S4, recostados contra la pared.
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.state_machine import (  # noqa: E402
    CONFIRMED_FALL,
    MILD,
    MODERATE,
    NULLIFIED,
    SEVERE,
    FallStateMachine,
    Stage,
    Stage2Evaluator,
    Stage3Evaluator,
    TriggerEvent,
)
from tests.test_main_headless import _toppling_pose  # noqa: E402
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402
from tests.test_state_machine import trigger  # noqa: E402

DT = 1.0 / 30.0
NAN = float("nan")


# --------------------------------------------------------------------------- #
# Fase 3 — inmovilidad persistente en finalise()
# --------------------------------------------------------------------------- #
class TestPersistentImmobility(unittest.TestCase):

    def _ending(self, t_deg: float, ext: float, still_share: float,
                fraction: float = 0.5, n: int = 90):
        """n frames of one ending; the first still_share of them are still."""
        ev = Stage3Evaluator(defer_to_end=True, persistent_fraction=fraction)
        ev.start(0.0)
        n_still = round(n * still_share)
        for i in range(n):
            i_still = 1.0 if i < n_still else 0.1
            ev.observe(i * DT, t_deg, i_still, ext, NAN)
        ev.finalise()
        return ev.verdict()

    def test_reclined_against_a_wall_and_still_is_severe(self) -> None:
        # A17-S4's ending: T 25.3 (a back against a wall keeps the trunk
        # vertical), legs folded (ratio 0.69), 57.8 % of frames still.
        self.assertEqual(self._ending(25.3, 0.69, 0.578), (CONFIRMED_FALL, SEVERE))

    def test_sat_up_and_moving_stays_moderate(self) -> None:
        # The most still PartiallyRecovered ending measured: 31.5 % (A15-S2).
        self.assertEqual(self._ending(23.1, 0.69, 0.315), (CONFIRMED_FALL, MODERATE))

    def test_mid_band_still_is_severe_and_moving_is_moderate(self) -> None:
        self.assertEqual(self._ending(40.0, 0.69, 0.80), (CONFIRMED_FALL, SEVERE))
        self.assertEqual(self._ending(40.0, 0.69, 0.20), (CONFIRMED_FALL, MODERATE))

    def test_the_cut_is_inclusive(self) -> None:
        self.assertEqual(self._ending(25.0, 0.69, 0.50), (CONFIRMED_FALL, SEVERE))

    def test_standing_with_legs_extended_stays_recovered_however_still(self) -> None:
        # Standing up IS the get-up evidence §3.5 asks for; stillness while
        # standing is not a failure to recover.
        self.assertEqual(self._ending(5.0, 1.3, 1.0), (NULLIFIED, MILD))

    def test_lying_is_severe_regardless(self) -> None:
        self.assertEqual(self._ending(85.0, 0.3, 0.0), (CONFIRMED_FALL, SEVERE))

    def test_disabled_restores_plain_moderate(self) -> None:
        self.assertEqual(self._ending(25.3, 0.69, 1.0, fraction=0.0),
                         (CONFIRMED_FALL, MODERATE))

    def test_counters_reset_between_events(self) -> None:
        ev = Stage3Evaluator(defer_to_end=True, persistent_fraction=0.5)
        ev.start(0.0)
        for i in range(90):
            ev.observe(i * DT, 25.0, 1.0, 0.69, NAN)
        ev.reset()
        ev.start(10.0)
        for i in range(90):
            ev.observe(10.0 + i * DT, 25.0, 0.1, 0.69, NAN)
        ev.finalise()
        self.assertEqual(ev.verdict(), (CONFIRMED_FALL, MODERATE))


# --------------------------------------------------------------------------- #
# Fase 2 — pérdida de detección durante la Etapa 3
# --------------------------------------------------------------------------- #
def _lost(pf):
    return replace(pf, detected=False)


def _fall_then(after):
    """Topple (0.67 s), then the frames `after` builds from index 20."""
    frames = [_toppling_pose(i, i * DT, angle_deg=i * 4.5, hip_y=500.0 + i * 12.0)
              for i in range(20)]
    return frames + after(20)


class TestLossOfDetectionDuringStage3(unittest.TestCase):

    def _run(self, frames, labelling: bool = True):
        pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)), labelling=labelling)
        alerts = []
        pipe.alerts.add_sink(alerts.append)
        closed_on = []
        for pf in frames:
            r = pipe.analyze(pf)
            if r.resolved_event is not None:
                closed_on.append((pf.timestamp, r.resolved_event, pf.detected))
        return pipe, alerts, closed_on

    @staticmethod
    def _down_then_lost(start):
        down = [_toppling_pose(i, i * DT, angle_deg=90.0, hip_y=740.0)
                for i in range(start, start + 30)]          # 1 s on the floor
        lost = [_lost(make_pose(i, i * DT)) for i in range(start + 30, start + 90)]
        return down + lost                                   # then 2 s undetected

    @staticmethod
    def _up_then_lost(start):
        down = [_toppling_pose(i, i * DT, angle_deg=90.0, hip_y=740.0)
                for i in range(start, start + 12)]
        up = [make_pose(i, i * DT, hip_y=500.0, shoulder_y=300.0, ankle_y=760.0,
                        feet_x=640.0) for i in range(start + 12, start + 30)]
        lost = [_lost(make_pose(i, i * DT)) for i in range(start + 30, start + 90)]
        return down + up + lost

    def test_seen_on_the_floor_then_lost_is_closed_as_severe(self) -> None:
        pipe, alerts, closed = self._run(_fall_then(self._down_then_lost))
        self.assertEqual(len(pipe.machine.events), 1, "la Etapa 1 no disparo")
        self.assertEqual(len(closed), 1, "el evento no se cerro al perderlo")
        t, ev, detected = closed[0]
        self.assertFalse(detected, "debe cerrarse durante la perdida")
        self.assertEqual((ev.verdict, ev.severity), ("stage3_confirmed", "severe"))
        self.assertEqual(len(alerts), 1, "un evento cerrado asi debe alertar")
        self.assertEqual(pipe.machine.stage.value, "MONITORING")

    def test_it_closes_only_after_the_gap_threshold(self) -> None:
        _, _, closed = self._run(_fall_then(self._down_then_lost))
        loss_start = 50 * DT
        self.assertGreater(closed[0][0] - loss_start, TEST_CFG["stage1"]["history_max_gap_s"])

    def test_the_same_holds_on_the_live_path(self) -> None:
        pipe, alerts, closed = self._run(_fall_then(self._down_then_lost), labelling=False)
        self.assertEqual((closed[0][1].verdict, closed[0][1].severity),
                         ("stage3_confirmed", "severe"))

    def test_seen_upright_then_lost_is_abandoned_as_before(self) -> None:
        pipe, alerts, closed = self._run(_fall_then(self._up_then_lost))
        self.assertEqual(len(pipe.machine.events), 1, "la Etapa 1 no disparo")
        self.assertEqual(closed, [])
        self.assertEqual(alerts, [])
        self.assertFalse(pipe.machine.events[0].verdict.startswith("stage3"))
        self.assertEqual(pipe.machine.stage.value, "MONITORING")


class TestOnlyFromStage3(unittest.TestCase):

    def test_an_event_still_in_stage2_is_never_closed_on_a_stale_tail(self) -> None:
        """reset() keeps Stage 3's trailing window; only start() clears it.

        So while a NEW event sits in CONFIRMING, that window still holds the
        PREVIOUS event's frames. If the previous subject ended on the floor,
        closing now would judge this event on another event's evidence. The
        OBSERVING guard is what prevents it.
        """
        m = FallStateMachine(trigger(), Stage2Evaluator(), Stage3Evaluator())
        m.stage3.start(0.0)
        for i in range(30):                                  # previous event:
            m.stage3.observe(i * DT, 160.0, 0.0, NAN, NAN)   # ended lying down
        m.stage3.reset()
        self.assertTrue(m.stage3.last_seen_down(), "precondicion: cola vieja en el suelo")
        m._pending = TriggerEvent(frame_index=400, timestamp=13.3, t_deg=50.0,
                                  v_tps=-2.0, formulation="score")
        m.stage = Stage.CONFIRMING
        self.assertIsNone(m.close_if_last_seen_down(14.0))
        self.assertEqual(m._pending.verdict, "stage1_only")


class TestLastSeenDown(unittest.TestCase):

    def _ev(self, angles):
        ev = Stage3Evaluator(defer_to_end=True)
        ev.start(0.0)
        for i, t in enumerate(angles):
            ev.observe(i * DT, t, 0.0, NAN, NAN)
        return ev

    def test_median_of_the_trailing_observed_window(self) -> None:
        self.assertTrue(self._ev([5.0] * 30 + [160.0] * 30).last_seen_down())
        self.assertFalse(self._ev([160.0] * 30 + [3.0] * 30).last_seen_down())

    def test_nothing_observed_is_not_down(self) -> None:
        self.assertFalse(self._ev([NAN] * 30).last_seen_down())
        self.assertFalse(Stage3Evaluator().last_seen_down())


if __name__ == "__main__":
    unittest.main()
