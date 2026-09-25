"""Fase 4: sin enfriamiento tras un rechazo de la Etapa 2.

A17-S3: un disparo falso a los 2.5 s, rechazado por la Etapa 2, dejaba al
sistema 3 s en COOLDOWN; la caída real (4.0-5.0 s) cayó dentro y no se vio.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.state_machine import (  # noqa: E402
    CONFIRMED,
    REJECTED,
    FallStateMachine,
    Stage,
    Stage2Evaluator,
    Stage3Evaluator,
    TriggerEvent,
)
from tests.test_state_machine import trigger  # noqa: E402


def _machine(after_rejection: bool) -> FallStateMachine:
    return FallStateMachine(trigger(), Stage2Evaluator(min_samples=3),
                            Stage3Evaluator(), cooldown_s=3.0,
                            cooldown_after_rejection=after_rejection)


def _resolve(m: FallStateMachine, verdict: str) -> None:
    m._pending = TriggerEvent(frame_index=1, timestamp=1.0, t_deg=50.0,
                              v_tps=-2.0, formulation="score")
    m._pending.verdict = verdict
    m.stage = Stage.CONFIRMING
    m._resolve(2.0)


class TestCooldownAfterRejection(unittest.TestCase):

    def test_off_a_rejection_goes_straight_back_to_monitoring(self) -> None:
        m = _machine(False)
        _resolve(m, REJECTED)
        self.assertIs(m.stage, Stage.MONITORING)
        self.assertIsNone(m._resolved_at)

    def test_off_a_real_event_still_cools_down(self) -> None:
        m = _machine(False)
        _resolve(m, CONFIRMED)
        self.assertIs(m.stage, Stage.COOLDOWN)
        self.assertEqual(m._resolved_at, 2.0)

    def test_on_is_the_previous_behaviour(self) -> None:
        m = _machine(True)
        _resolve(m, REJECTED)
        self.assertIs(m.stage, Stage.COOLDOWN)

    def test_default_keeps_the_previous_behaviour(self) -> None:
        m = FallStateMachine(trigger(), Stage2Evaluator(min_samples=3))
        self.assertTrue(m.cooldown_after_rejection)

    def test_the_rejected_event_is_still_recorded(self) -> None:
        m = _machine(False)
        _resolve(m, REJECTED)
        self.assertEqual(m.just_resolved.verdict, REJECTED)


class TestWiring(unittest.TestCase):

    def test_config_turns_it_off(self) -> None:
        from pef_fall_detector.config import load_config
        from pef_fall_detector.pipeline import FramePipeline
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertFalse(cfg.stage3.cooldown_after_rejection)
        self.assertFalse(FramePipeline(cfg).machine.cooldown_after_rejection)

    def test_old_configs_keep_the_cooldown(self) -> None:
        import copy
        from pef_fall_detector.config import Config
        from pef_fall_detector.pipeline import FramePipeline
        from tests.test_pipeline import TEST_CFG
        cfg = copy.deepcopy(TEST_CFG)
        cfg["stage3"].pop("cooldown_after_rejection", None)
        self.assertTrue(FramePipeline(Config(cfg)).machine.cooldown_after_rejection)


if __name__ == "__main__":
    unittest.main()
