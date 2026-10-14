"""Phase 4: no cooldown after a Stage-2 rejection.

A17-S3: a false trigger at 2.5 s, rejected by Stage 2, left the system 3 s in
COOLDOWN; the real fall (4.0-5.0 s) landed inside it and was not seen.
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
from tests.support import TEST_CFG, trigger  # noqa: E402


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

    def test_true_keeps_the_cooldown_and_the_key_is_required(self) -> None:
        # Before 0.2.0.dev2 a config without the key kept the cooldown; now
        # the key is required, and true is how the old behaviour is asked for.
        import copy

        from pef_fall_detector.config import Config, ConfigError
        from pef_fall_detector.pipeline import FramePipeline
        cfg = copy.deepcopy(TEST_CFG)
        self.assertIs(cfg["stage3"]["cooldown_after_rejection"], True)
        self.assertTrue(FramePipeline(Config(cfg)).machine.cooldown_after_rejection)
        cfg["stage3"].pop("cooldown_after_rejection")
        with self.assertRaisesRegex(ConfigError, r"missing key 'stage3\.cooldown_after_rejection'"):
            Config(cfg)


if __name__ == "__main__":
    unittest.main()
