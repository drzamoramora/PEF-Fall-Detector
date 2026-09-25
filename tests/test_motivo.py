"""El motivo de cada veredicto: la regla que decidió, con los números que leyó.

§3.5 promete que *"a downstream reviewer can reconstruct the decision"*. Sin
el motivo, eso exige leer el código para saber qué rama cerró el evento. Estas
pruebas fijan que cada rama deje su motivo, y que diga lo que de verdad decidió.
"""

from __future__ import annotations

import copy
import csv
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.audit_log import EVENT_FIELDS  # noqa: E402
from pef_fall_detector.state_machine import (  # noqa: E402
    CONFIRMED,
    INCONCLUSIVE,
    REJECTED,
    FallStateMachine,
    Stage,
    Stage2Evaluator,
    Stage3Evaluator,
    TriggerEvent,
)
from tests.test_main_headless import TestHeadlessLabelsLikePefLab  # noqa: E402
from tests.test_state_machine import trigger  # noqa: E402

DT = 1.0 / 30.0
NAN = float("nan")


def _ending(t_deg, ext, still_share=0.0, fraction=0.5, n=90):
    ev = Stage3Evaluator(defer_to_end=True, persistent_fraction=fraction)
    ev.start(0.0)
    n_still = round(n * still_share)
    for i in range(n):
        ev.observe(i * DT, t_deg, 1.0 if i < n_still else 0.1, ext, NAN)
    ev.finalise()
    return ev


class TestFinaliseReasons(unittest.TestCase):

    def test_lying(self) -> None:
        r = _ending(85.0, 0.3).reason
        self.assertIn("T final 85", r)
        self.assertIn("suelo", r)

    def test_mid_band_moving_says_moderate_and_the_share(self) -> None:
        r = _ending(40.0, 0.69, still_share=0.2).reason
        self.assertIn("sentado o arrodillado", r)
        self.assertIn("20% < 50%", r)

    def test_mid_band_persistent_says_why_it_is_severe(self) -> None:
        r = _ending(40.0, 0.69, still_share=0.8).reason
        self.assertIn("80% >= 50%", r)
        self.assertIn("persistio", r)

    def test_upright_with_legs_extended(self) -> None:
        self.assertIn("se levanto", _ending(5.0, 1.3).reason)

    def test_upright_without_verifiable_legs(self) -> None:
        r = _ending(3.0, NAN, still_share=0.63).reason
        self.assertIn("sin piernas verificables", r)
        self.assertIn("63% >= 50%", r)

    def test_persistence_clause_absent_when_disabled(self) -> None:
        self.assertNotIn("quieto", _ending(40.0, 0.69, 0.9, fraction=0.0).reason)

    def test_nothing_measured(self) -> None:
        self.assertIn("sin esqueleto", _ending(NAN, NAN).reason)


class TestLiveReasons(unittest.TestCase):

    def _play(self, frames, **kw):
        ev = Stage3Evaluator(threshold_w_s=2.0, **kw)
        ev.start(0.0)
        for i, (t, still, ext) in enumerate(frames):
            ev.observe(i * DT, t, still, ext, NAN)
        return ev

    def test_immobility_lying(self) -> None:
        ev = self._play([(85.0, min(3.0, i * DT), 0.3) for i in range(120)])
        self.assertIn("W 2.0", ev.reason)
        self.assertIn("tumbado", ev.reason)
        self.assertIn("T 85 deg >= 60", ev.reason)

    def test_immobility_not_lying(self) -> None:
        ev = self._play([(40.0, min(3.0, i * DT), 0.3) for i in range(120)])
        self.assertIn("no tumbado", ev.reason)

    def test_recovery(self) -> None:
        ev = self._play([(85.0, 0.1, 0.3)] * 10 + [(5.0, 0.0, 1.3)] * 60)
        self.assertIn("se levanto", ev.reason)

    def test_window_ran_out(self) -> None:
        ev = Stage3Evaluator(window_s=1.0)
        ev.start(0.0)
        for i in range(40):
            ev.observe(i * DT, 85.0, 0.1, 0.3, NAN)
        self.assertIn("ventana", ev.reason)


class TestMachineReasons(unittest.TestCase):

    def _machine(self):
        return FallStateMachine(trigger(), Stage2Evaluator(min_samples=3),
                                Stage3Evaluator())

    def test_stage2_texts(self) -> None:
        m = self._machine()
        m.stage2.start(0.0)
        for p in (0.4, 0.4, -0.2, -0.3):          # 2 of 4 outside
            m.stage2.observe(p)
        self.assertIn("50% >= 50%", m._stage2_text(CONFIRMED, False))
        self.assertIn("descenso controlado", m._stage2_text(REJECTED, True))
        m.stage2.reset()
        m.stage2.start(0.0)
        for p in (0.4, -0.2, -0.3, -0.1):         # 1 of 4 outside
            m.stage2.observe(p)
        self.assertIn("25% < 50% (4 muestras)", m._stage2_text(REJECTED, False))
        m.stage2.reset()
        m.stage2.start(0.0)
        m.stage2.observe(0.4)
        self.assertIn("pies no medibles (1 muestras", m._stage2_text(INCONCLUSIVE, False))

    def test_abandoned_event_says_so_and_keeps_what_it_had(self) -> None:
        m = self._machine()
        m._pending = TriggerEvent(frame_index=1, timestamp=1.0, t_deg=50.0,
                                  v_tps=-2.0, formulation="score",
                                  reason="Etapa 2: COM fuera del apoyo 90% >= 50% (11 muestras)")
        m.stage = Stage.OBSERVING
        ev = m._pending
        m.reset()
        self.assertTrue(ev.reason.startswith("abandonado"))
        self.assertIn("COM fuera del apoyo", ev.reason)

    def test_closed_on_loss_is_prefixed(self) -> None:
        m = self._machine()
        m._pending = TriggerEvent(frame_index=1, timestamp=1.0, t_deg=50.0,
                                  v_tps=-2.0, formulation="score")
        m.stage = Stage.OBSERVING
        m.stage3.start(1.0)
        for i in range(30):
            m.stage3.observe(1.0 + i * DT, 160.0, 0.1, NAN, NAN)
        closed = m.close_if_last_seen_down(3.0)
        self.assertTrue(closed.reason.startswith("deteccion perdida en el suelo"))
        self.assertIn("T final 160", closed.reason)


class TestLivePathCarriesTheReason(unittest.TestCase):
    """Greedy (live) resolution must put Stage 3's reason on the event too."""

    def test_immobility_confirmation_reason_reaches_the_event(self) -> None:
        from pef_fall_detector.config import Config
        from pef_fall_detector.pipeline import FramePipeline
        from tests.test_main_headless import _toppling_pose
        from tests.test_pipeline import TEST_CFG
        pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)), labelling=False)
        frames = [_toppling_pose(i, i * DT, angle_deg=i * 4.5, hip_y=500.0 + i * 12.0)
                  for i in range(20)]
        frames += [_toppling_pose(i, i * DT, angle_deg=90.0, hip_y=740.0)
                   for i in range(20, 260)]        # 8 s still: past W = 5 s
        alerts = []
        pipe.alerts.add_sink(alerts.append)
        for pf in frames:
            pipe.analyze(pf)
        ev = pipe.machine.events[0]
        self.assertEqual(ev.verdict, "stage3_confirmed")
        self.assertIn("quieto", ev.reason)
        self.assertIn("W 5.0", ev.reason)
        # And the alert carries it: the rule, not only the numbers.
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].reason, ev.reason)
        self.assertIn(f"why: {ev.reason}", alerts[0].message())


class TestReasonReachesTheRecord(TestHeadlessLabelsLikePefLab):
    """The events CSV written by main.py carries the reason."""

    def test_events_csv_has_the_reason(self) -> None:
        self.assertIn("motivo", EVENT_FIELDS)
        rows = self._run()
        self.assertIn("T final 90", rows[0]["motivo"])
        self.assertIn("suelo", rows[0]["motivo"])


if __name__ == "__main__":
    unittest.main()
