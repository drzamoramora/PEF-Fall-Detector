"""Phase 8.4: Quantity A as an additional Stage-1 condition.

Stage1Trigger fires when the head reaches ``threshold_a_low`` (0.3 of standing
height) with a reading at/above ``threshold_a_high`` (0.7) in the last
``a_window_s`` (1.5 s), while the body is still descending fast on that same
frame (V below ``threshold_v_tps``), held ``hold_s``, with its own latch. It
is OR'd with the formulation and the H fallback, never mixed into them, and
the event records which condition raised it (``trigger_source``).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from pef_fall_detector.audit_log import EVENT_FIELDS, AuditLogger  # noqa: E402
from pef_fall_detector.config import Config, load_config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.state_machine import (  # noqa: E402
    A_FALLBACK,
    H_FALLBACK,
    SCORE,
    FallStateMachine,
    Stage,
    Stage1Trigger,
    Stage2Evaluator,
    Stage3Evaluator,
    TriggerEvent,
)
from tests.test_main_headless import _FakeSource  # noqa: E402
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402
from tests.test_quantity_a import STANDING, _body, _pose, _with_world  # noqa: E402

DT = 1.0 / 30.0
NAN = float("nan")


def _trig(**kw) -> Stage1Trigger:
    """The project's trigger (score 2.6 over a 0.8 s peak window) plus the A drop."""
    base = dict(formulation=SCORE, threshold_score=2.6, peak_window_s=0.8, hold_s=0.1,
                threshold_a_high=0.7, threshold_a_low=0.3, a_window_s=1.5)
    base.update(kw)
    return Stage1Trigger(**base)


def _series(segments):
    """Frames from (seconds, a_start, a_end, v) segments, A linear inside each."""
    out, t = [], 0.0
    for seconds, a0, a1, v in segments:
        n = int(round(seconds / DT))
        for i in range(n):
            a = a0 if (isinstance(a0, float) and math.isnan(a0)) else a0 + (a1 - a0) * i / max(n - 1, 1)
            out.append((t, a, v))
            t += DT
    return out


def _fire_times(trg, frames, t_deg=5.0):
    """Timestamps (and source) of every firing. T = 5 deg keeps the score low."""
    fired = []
    for t, a, v in frames:
        if trg.update(t, t_deg, v, NAN, a):
            fired.append((round(t, 3), trg.last_source))
    return fired


# A3-S2 in miniature: standing, the head drops to the floor in 0.5 s while
# the body is still falling fast, then stays down.
FALL = [(1.0, 1.0, 1.0, 0.0), (0.5, 1.0, 0.2, -2.5), (1.0, 0.2, 0.2, -2.5)]


class TestADropCondition(unittest.TestCase):

    def test_a_fast_drop_fires_once_as_A(self) -> None:
        fired = _fire_times(_trig(), _series(FALL))
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][1], A_FALLBACK)
        # A reaches 0.3 at ~1.44 s; held 0.1 s.
        self.assertGreater(fired[0][0], 1.5)
        self.assertLess(fired[0][0], 1.6)

    def test_off_by_default_and_keeps_no_history(self) -> None:
        trg = Stage1Trigger(formulation=SCORE, threshold_score=2.6, peak_window_s=0.8)
        self.assertEqual(_fire_times(trg, _series(FALL)), [])
        self.assertEqual(len(trg._a_history), 0)

    def test_arriving_slowly_does_not_fire(self) -> None:
        # Lying down on purpose (B11-S2): the head gets there, the body brakes.
        slow = [(1.0, 1.0, 1.0, 0.0), (0.5, 1.0, 0.2, -0.5), (1.0, 0.2, 0.2, -0.5)]
        self.assertEqual(_fire_times(_trig(), _series(slow)), [])

    def test_a_fast_descent_that_ended_before_the_head_arrived_does_not_fire(self) -> None:
        # B12-S2: fast down to kneeling, then a slow lie-down. V is read on
        # the frame the head arrives, not over the peak window.
        frames = _series([(1.0, 1.0, 1.0, 0.0), (0.3, 1.0, 0.5, -2.5), (0.5, 0.45, 0.2, 0.5),
                          (1.0, 0.2, 0.2, 0.3)])
        self.assertEqual(_fire_times(_trig(), frames), [])

    def test_the_high_reading_must_be_recent(self) -> None:
        # Sat down (0.5) 2 s ago, then dropped fast to the floor: no reading
        # >= 0.7 in the last 1.5 s, so this is not a drop from standing.
        frames = _series([(1.0, 1.0, 1.0, 0.0), (0.3, 1.0, 0.5, -0.5), (2.0, 0.5, 0.5, 0.0),
                          (0.2, 0.5, 0.2, -2.5), (1.0, 0.2, 0.2, -2.5)])
        self.assertEqual(_fire_times(_trig(), frames), [])
        self.assertEqual(len(_fire_times(_trig(a_window_s=3.0), frames)), 1)

    def test_the_window_is_inclusive(self) -> None:
        # One high reading exactly 1.5 s before the low ones, NaN in between.
        frames = [(0.0, 0.9, 0.0)] + [(i * DT, NAN, -2.5) for i in range(1, 45)]
        frames += [(45 * DT + i * DT, 0.2, -2.5) for i in range(0, 4)]
        trg = _trig(hold_s=0.0)
        self.assertEqual(len(_fire_times(trg, frames)), 1)
        late = [(0.0, 0.9, 0.0)] + [(i * DT, NAN, -2.5) for i in range(1, 46)]
        late += [(46 * DT + i * DT, 0.2, -2.5) for i in range(0, 4)]
        self.assertEqual(_fire_times(_trig(hold_s=0.0), late), [])

    def test_the_window_edge_survives_float_rounding(self) -> None:
        # 1.6 - 1.5 is 0.10000000000000009 in floating point: the reading
        # taken at 0.1 is exactly 1.5 s old and must still count.
        trg = _trig(hold_s=0.0)
        trg.update(0.1, 5.0, 0.0, NAN, 0.9)
        self.assertTrue(trg.update(1.6, 5.0, -2.5, NAN, 0.2))

    def test_uncalibrated_A_never_fires(self) -> None:
        nan_all = [(i * DT, NAN, -2.5) for i in range(90)]
        trg = _trig()
        self.assertEqual(_fire_times(trg, nan_all), [])
        self.assertEqual(len(trg._a_history), 0, "only measured readings are kept")
        # Calibration that arrives with the subject already down: no reading
        # from standing, so no drop.
        late_cal = nan_all + [(3.0 + i * DT, 0.2, -2.5) for i in range(30)]
        self.assertEqual(_fire_times(_trig(), late_cal), [])

    def test_nan_v_is_not_evidence(self) -> None:
        frames = [(t, a, NAN if a <= 0.3 else v) for t, a, v in _series(FALL)]
        self.assertEqual(_fire_times(_trig(), frames), [])

    def test_edges_inclusive_for_A_strict_for_V(self) -> None:
        exact = _series([(1.0, 0.7, 0.7, 0.0), (0.2, 0.3, 0.3, -2.5)])
        self.assertEqual(len(_fire_times(_trig(), exact)), 1)
        just_above = _series([(1.0, 0.7, 0.7, 0.0), (0.2, 0.301, 0.301, -2.5)])
        self.assertEqual(_fire_times(_trig(), just_above), [])
        just_below_high = _series([(1.0, 0.699, 0.699, 0.0), (0.2, 0.3, 0.3, -2.5)])
        self.assertEqual(_fire_times(_trig(), just_below_high), [])
        v_at_threshold = _series([(1.0, 0.9, 0.9, 0.0), (0.2, 0.2, 0.2, -1.5)])
        self.assertEqual(_fire_times(_trig(), v_at_threshold), [])
        v_below = _series([(1.0, 0.9, 0.9, 0.0), (0.2, 0.2, 0.2, -1.51)])
        self.assertEqual(len(_fire_times(_trig(), v_below)), 1)

    def test_hold_time(self) -> None:
        # Two frames (0.067 s) is a flicker, not a drop (B07-S4: 1 frame).
        blip = _series([(1.0, 0.9, 0.9, 0.0), (2 * DT, 0.2, 0.2, -3.0), (1.0, 0.9, 0.9, 0.0)])
        self.assertEqual(_fire_times(_trig(), blip), [])
        four = _series([(1.0, 0.9, 0.9, 0.0), (4 * DT, 0.2, 0.2, -3.0), (1.0, 0.9, 0.9, 0.0)])
        self.assertEqual(len(_fire_times(_trig(), four)), 1)

    def test_latch_rearms_only_after_the_condition_clears(self) -> None:
        frames = _series(FALL + [(1.0, 1.0, 1.0, 0.0), (0.5, 1.0, 0.2, -2.5), (0.5, 0.2, 0.2, -2.5)])
        self.assertEqual([s for _t, s in _fire_times(_trig(), frames)], [A_FALLBACK, A_FALLBACK])

    def test_reset_forgets_the_readings(self) -> None:
        trg = _trig()
        _fire_times(trg, _series([(1.0, 1.0, 1.0, 0.0)]))
        trg.reset()
        self.assertEqual(len(trg._a_history), 0)
        self.assertEqual(_fire_times(trg, [(1.0 + i * DT, 0.2, -2.5) for i in range(10)]), [])

    def test_reset_clears_the_hold_and_the_latch(self) -> None:
        trg = _trig()
        _fire_times(trg, _series([(1.0, 1.0, 1.0, 0.0)]))
        self.assertFalse(trg.update(1.0, 5.0, -2.5, NAN, 0.2))   # condition starts, not held
        self.assertIsNotNone(trg._a_condition_since)
        trg.reset()
        self.assertIsNone(trg._a_condition_since)
        fired = _trig()
        for t, a, v in _series(FALL):
            if fired.update(t, 5.0, v, NAN, a):
                break
        self.assertEqual((fired._a_fired, fired.last_source), (True, A_FALLBACK))
        fired.reset()
        self.assertEqual((fired._a_fired, fired.last_source), (False, ""))

    def test_the_formulation_wins_the_source_on_the_same_frame(self) -> None:
        trg = _trig(peak_window_s=0.0, hold_s=0.0)
        trg.update(0.0, 5.0, 0.0, NAN, 1.0)
        self.assertTrue(trg.update(DT, 90.0, -3.5, NAN, 0.2))
        self.assertEqual(trg.last_source, SCORE)
        self.assertTrue(trg._a_fired, "the A latch is consumed too")

    def test_the_H_fallback_is_named(self) -> None:
        trg = _trig(threshold_h_ratio=0.35, peak_window_s=0.0, hold_s=0.0)
        self.assertTrue(trg.update(0.0, 5.0, -2.0, 0.2, NAN))
        self.assertEqual(trg.last_source, H_FALLBACK)
        self.assertFalse(trg.update(DT, 5.0, 0.0, 1.0, NAN))
        self.assertEqual(trg.last_source, "")

    def test_bad_parameters(self) -> None:
        with self.assertRaises(ValueError):
            _trig(threshold_a_low=0.7)
        with self.assertRaises(ValueError):
            _trig(a_window_s=0.0)
        Stage1Trigger(threshold_a_high=0.0, threshold_a_low=0.9, a_window_s=0.0)  # off: not checked


# ---------------------------------------------------------------------------
# The state machine: source on the event, A seen in every stage.

def _machine(stage2=None, stage3=None, cooldown_s=3.0, reject_cooldown=True):
    return FallStateMachine(_trig(), stage2=stage2, stage3=stage3, cooldown_s=cooldown_s,
                            cooldown_after_rejection=reject_cooldown)


def _run_machine(m, frames, t_deg=5.0, p_offset=NAN, start=0):
    events = []
    for i, (t, a, v) in enumerate(frames):
        ev = m.update(start + i, t, t_deg, v, p_offset, NAN, NAN, NAN, a)
        if ev is not None:
            events.append(ev)
    return events


class TestMachine(unittest.TestCase):

    def test_the_event_says_A_raised_it(self) -> None:
        m = _machine(stage2=Stage2Evaluator())
        events = _run_machine(m, _series(FALL))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].trigger_source, A_FALLBACK)
        self.assertEqual(events[0].formulation, SCORE)

    def test_a_score_event_says_score(self) -> None:
        m = _machine()
        frames = [(i * DT, 1.0, -3.5 if i > 30 else 0.0) for i in range(60)]
        events = _run_machine(m, frames, t_deg=90.0)
        self.assertEqual([e.trigger_source for e in events], [SCORE])

    def test_readings_during_cooldown_count(self) -> None:
        # A first event (no Stage 2: resolved at once, 0.5 s cooldown). A is
        # measured only during the cooldown (standing); the drop comes right
        # after it, so only readings seen while cooling down can count.
        m = _machine(cooldown_s=0.5)
        first = _run_machine(m, [(i * DT, NAN, -3.5) for i in range(4)], t_deg=90.0)
        self.assertEqual(len(first), 1)
        cooling = [(i * DT, 1.0, 0.0) for i in range(4, 18)]
        _run_machine(m, cooling, start=4)
        self.assertEqual(m.stage, Stage.COOLDOWN)
        drop = [(i * DT, 0.2, -2.5) for i in range(18, 28)]
        second = _run_machine(m, drop, start=18)
        self.assertEqual([e.trigger_source for e in second], [A_FALLBACK])

    def test_readings_during_stage2_count(self) -> None:
        # Stage 2 rejects (COM inside) and, without a rejection cooldown, the
        # machine is back to MONITORING at once. A is measured only while
        # Stage 2 judges; those readings still belong to the window.
        m = _machine(stage2=Stage2Evaluator(window_s=0.33, min_samples=3), reject_cooldown=False)
        first = _run_machine(m, [(i * DT, NAN, -3.5) for i in range(5)], t_deg=90.0, p_offset=-0.5)
        self.assertEqual(len(first), 1)
        judging = [(i * DT, 1.0, 0.0) for i in range(5, 13)]
        _run_machine(m, judging, p_offset=-0.5, start=5)
        self.assertEqual(m.stage, Stage.CONFIRMING)
        drop = [(i * DT, 0.2, -2.5) for i in range(13, 23)]
        second = _run_machine(m, drop, p_offset=-0.5, start=13)
        self.assertEqual(first[0].verdict, "stage2_rejected")
        self.assertEqual([e.trigger_source for e in second], [A_FALLBACK])

    def test_a_drop_while_observing_is_consumed_not_replayed(self) -> None:
        # The drop happens while Stage 3 observes an earlier event; once the
        # machine is free again the same drop must not raise a second event.
        st3 = Stage3Evaluator(window_s=0.5)             # live: resolves at the window end
        m = _machine(stage2=Stage2Evaluator(window_s=0.1, min_samples=1), stage3=st3,
                     cooldown_s=0.0)
        first = _run_machine(m, [(i * DT, 1.0, -3.5) for i in range(6)], t_deg=90.0,
                             p_offset=0.5)
        self.assertEqual(len(first), 1)
        frames = [(i * DT, 1.0, 0.0) for i in range(6, 9)]
        frames += [(i * DT, 1.0 - 0.8 * (i - 8) / 6, -2.5) for i in range(9, 15)]
        frames += [(i * DT, 0.2, -2.5) for i in range(15, 45)]
        seen = []
        for i, (t, a, v) in enumerate(frames, start=6):
            ev = m.update(i, t, 5.0, v, 0.5, NAN, NAN, NAN, a)
            seen.append(m.stage)
            self.assertIsNone(ev, f"second event at frame {i}")
        self.assertIn(Stage.OBSERVING, seen[:12], "the drop must happen while observing")
        self.assertEqual(seen[-1], Stage.MONITORING, "the scenario must free the machine")

    def test_without_a_the_machine_behaves_as_in_83(self) -> None:
        def run(trg, pass_a):
            m = FallStateMachine(trg, stage2=Stage2Evaluator())
            out = []
            for i, (t, a, v) in enumerate(_series(FALL)):
                args = (i, t, 90.0 if 40 < i < 50 else 5.0, v, 0.5, NAN, NAN, NAN)
                ev = m.update(*args, a) if pass_a else m.update(*args)
                if ev is not None:
                    out.append((ev.frame_index, ev.trigger_source))
            return out
        off = run(Stage1Trigger(formulation=SCORE, threshold_score=2.6, peak_window_s=0.8), True)
        self.assertEqual(off, run(_trig(), False))
        self.assertEqual([s for _f, s in off], [SCORE])


# ---------------------------------------------------------------------------
# Through the pipeline: a vertical collapse the T/V formulation cannot see.

SLUMPED = _pose(ankles=((-0.1, 0.05, 0.9), (0.1, 0.05, 0.9)),
                knees=((-0.1, 0.0, 0.45), (0.1, 0.0, 0.45)),
                shoulders_y=-0.25, nose=(0, -0.35, 0))       # 0.40 m above the ankles -> 0.25


def _collapse(descent_frames: int):
    """Stand 1.5 s; the head drops to 0.25 in 0.3 s (3D); the trunk stays
    vertical in the image (T = 0) while the hips go down 190 px over
    ``descent_frames``; then still."""
    w_stand, w_down = _body(STANDING), _body(SLUMPED)
    frames = []
    for i in range(160):
        k = min(max(i - 45, 0), descent_frames)
        hip = 500.0 + 190.0 * k / descent_frames
        pf = make_pose(i, i * DT, hip_y=hip, shoulder_y=hip - 150.0, ankle_y=900.0)
        s = min(max((i - 45) / 8.0, 0.0), 1.0)
        frames.append(_with_world(pf, (1.0 - s) * w_stand + s * w_down))
    return frames


def _cfg(on=True):
    cfg = copy.deepcopy(TEST_CFG)
    cfg["quantity_a"] = {"trigger_from_a": on, "trigger_high": 0.7, "trigger_low": 0.3,
                         "trigger_window_s": 1.5}
    return Config(cfg)


def _pipeline_run(frames, cfg):
    pipe = FramePipeline(cfg, labelling=True)
    res = [pipe.analyze(pf) for pf in frames]
    pipe.finalise(res[-1].pose.timestamp)
    return pipe, res


class TestPipeline(unittest.TestCase):

    def test_a_vertical_collapse_raises_an_A_event(self) -> None:
        pipe, res = _pipeline_run(_collapse(16), _cfg())
        self.assertEqual([e.trigger_source for e in pipe.machine.events], [A_FALLBACK])
        ev = pipe.machine.events[0]
        self.assertLess(ev.v_tps, -1.5)
        self.assertLess(abs(ev.t_deg), 5.0, "the trunk stayed vertical")
        a_at = res[ev.frame_index].quantities["A_head"]
        self.assertLessEqual(a_at, 0.3)

    def test_switched_off_the_collapse_is_not_seen(self) -> None:
        pipe, _ = _pipeline_run(_collapse(16), _cfg(on=False))
        self.assertEqual(pipe.machine.events, [])
        self.assertEqual(pipe.machine.trigger.threshold_a_high, 0.0)

    def test_without_the_section_it_is_off(self) -> None:
        pipe, _ = _pipeline_run(_collapse(16), Config(copy.deepcopy(TEST_CFG)))
        self.assertEqual(pipe.machine.events, [])

    def test_a_slow_arrival_raises_nothing(self) -> None:
        pipe, _ = _pipeline_run(_collapse(110), _cfg())
        self.assertEqual(pipe.machine.events, [])

    def test_config_values_reach_the_trigger(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        cfg["quantity_a"] = {"trigger_from_a": True, "trigger_high": 0.8, "trigger_low": 0.25,
                             "trigger_window_s": 2.0}
        trg = FramePipeline(Config(cfg)).machine.trigger
        self.assertEqual((trg.threshold_a_high, trg.threshold_a_low, trg.a_window_s),
                         (0.8, 0.25, 2.0))

    def test_project_config(self) -> None:
        qa = load_config(Path(__file__).resolve().parents[1] / "config.yaml").as_dict()["quantity_a"]
        self.assertTrue(qa["trigger_from_a"])
        self.assertEqual((qa["trigger_high"], qa["trigger_low"], qa["trigger_window_s"]),
                         (0.7, 0.3, 1.5))


# ---------------------------------------------------------------------------
# The records: both writers carry the source.

class TestRecords(unittest.TestCase):

    def test_the_field_exists(self) -> None:
        self.assertIn("trigger_source", EVENT_FIELDS)

    def test_the_headless_runner_writes_it(self) -> None:
        frames = _collapse(16)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(TEST_CFG)
            cfg["logging"]["output_dir"] = tmp
            cfg["quantity_a"] = {"trigger_from_a": True}

            class _Scripted(FramePipeline):
                def process(self, frame_bgr, frame_index, timestamp):
                    return self.analyze(frames[frame_index])

            saved = (main.load_config, main.VideoFileSource, main.FramePipeline)
            main.load_config = lambda _path: Config(cfg)
            main.VideoFileSource = lambda _path: _FakeSource(len(frames))
            main.FramePipeline = _Scripted
            try:
                args = argparse.Namespace(video="A00-S1-Recovered.mp4", config=None,
                                          expected_seconds=None, headless=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    main.run_headless(args)
            finally:
                main.load_config, main.VideoFileSource, main.FramePipeline = saved
            path = sorted(Path(tmp).glob("*-events-*.csv"))[0]
            with path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        self.assertEqual([r["trigger_source"] for r in rows], [A_FALLBACK])

    def test_the_lab_window_writes_it(self) -> None:
        try:
            from pef_fall_detector.gui.lab_window import LabWindow
        except ImportError as exc:                          # pragma: no cover
            self.skipTest(f"GUI not available: {exc}")
        ev = TriggerEvent(frame_index=3, timestamp=0.1, t_deg=2.0, v_tps=-2.0,
                          formulation=SCORE, trigger_source=A_FALLBACK)
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip-events", fields=EVENT_FIELDS)
            fake = SimpleNamespace(event_logger=logger,
                                   pipeline=SimpleNamespace(machine=SimpleNamespace(events=[ev])))
            LabWindow._log_event(fake, ev)
            logger.close()
            with next(Path(tmp).glob("clip-events*.csv")).open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        self.assertEqual(rows[0]["trigger_source"], A_FALLBACK)


if __name__ == "__main__":
    unittest.main()
