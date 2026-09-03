"""Tests for Stage 1 and the confirmation state machine (§3.5).

Synthetic (T, V) tracks only. This is the first module in the project that
*decides* rather than measures, so the properties pinned here are about the
decision itself: that it fires when it should, that it does not fire on a
single noisy frame, that one fall yields one event, and that the three
formulations of §3.5's "instantaneous combination" behave the way the
measurements said they do.

The formulation tests are the load-bearing ones. §3.5 as written admits
several readings, and the difference between them was 9 of 13 real falls
against 12 of 13. Encoding each reading's behaviour as a test is what keeps
that difference from being re-litigated from memory.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from pef_fall_detector.state_machine import (
    CONFIRMED,
    CONFIRMED_FALL,
    MILD,
    MODERATE,
    NULLIFIED,
    SEVERE,
    UNRESOLVED,
    INCONCLUSIVE,
    REJECTED,
    SCORE,
    SEQUENTIAL,
    SIMULTANEOUS,
    V_ONLY,
    FallStateMachine,
    Stage,
    Stage1Trigger,
    Stage2Evaluator,
    Stage3Evaluator,
)

T_TH, V_TH = 45.0, -1.5
DT = 1.0 / 30.0
HOLD = 0.1                       # 3 frames at 30 fps


def trigger(formulation: str = SEQUENTIAL, **kw) -> Stage1Trigger:
    kw.setdefault("threshold_t_deg", T_TH)
    kw.setdefault("threshold_v_tps", V_TH)
    kw.setdefault("hold_s", HOLD)
    return Stage1Trigger(formulation=formulation, **kw)


def feed(trg: Stage1Trigger, samples, t0: float = 0.0) -> list[float]:
    """Feed (T, V) pairs at 30 fps; return the times at which it fired."""
    fired = []
    for i, (t_deg, v_tps) in enumerate(samples):
        ts = t0 + i * DT
        if trg.update(ts, t_deg, v_tps):
            fired.append(ts)
    return fired


class TestHysteresis(unittest.TestCase):
    """A decision must not rest on one frame."""

    def test_a_single_frame_spike_does_not_fire(self) -> None:
        # One frame of landmark noise crossing both thresholds. This is the
        # false alarm hysteresis exists to prevent.
        samples = [(0.0, 0.0)] * 5 + [(90.0, -5.0)] + [(0.0, 0.0)] * 5
        self.assertEqual(feed(trigger(SIMULTANEOUS), samples), [])

    def test_a_sustained_condition_fires(self) -> None:
        samples = [(0.0, 0.0)] * 3 + [(90.0, -5.0)] * 10
        self.assertEqual(len(feed(trigger(SIMULTANEOUS), samples)), 1)

    def test_the_hold_is_time_not_frames(self) -> None:
        # The same physical event at two capture rates must fire in both.
        # A frame count would fire at 30 fps and stay silent at 10.
        for fps in (10.0, 30.0, 60.0):
            trg = trigger(SIMULTANEOUS)
            fired = []
            for i in range(int(1.0 * fps)):
                if trg.update(i / fps, 90.0, -5.0):
                    fired.append(i / fps)
            self.assertTrue(fired, f"no disparo a {fps} fps")
            # And it fires at essentially the same moment in wall-clock time.
            self.assertLessEqual(fired[0], HOLD + 1.0 / fps + 1e-9)

    def test_zero_hold_fires_immediately(self) -> None:
        # Hysteresis off is a legitimate ablation setting; it must not crash
        # or silently keep a minimum.
        trg = trigger(SIMULTANEOUS, hold_s=0.0)
        self.assertTrue(trg.update(0.0, 90.0, -5.0))


class TestFormulations(unittest.TestCase):
    """The three readings of §3.5's "instantaneous combination"."""

    #: A fall whose T and V peaks are OFFSET, as real falls measured on this
    #: project's data are (0.03 s to 1.8 s apart). V drops first, the trunk
    #: rotates past threshold afterwards.
    OFFSET_FALL = (
        [(10.0, 0.0)] * 5           # standing
        + [(20.0, -5.0)] * 5        # dropping fast, trunk still fairly upright
        + [(30.0, -0.5)] * 5        # velocity gone
        + [(80.0, -0.2)] * 10       # now the trunk is down
    )

    def test_simultaneous_misses_an_offset_fall(self) -> None:
        # The failure mode measured on real data: both quantities clearly
        # witness the fall, never at the same time.
        self.assertEqual(feed(trigger(SIMULTANEOUS), self.OFFSET_FALL), [])

    def test_sequential_catches_an_offset_fall(self) -> None:
        self.assertEqual(len(feed(trigger(SEQUENTIAL), self.OFFSET_FALL)), 1)

    def test_v_only_catches_an_offset_fall(self) -> None:
        self.assertEqual(len(feed(trigger(V_ONLY), self.OFFSET_FALL)), 1)

    def test_sequential_needs_the_trunk_to_confirm(self) -> None:
        # A fast drop that never leans: sequential must NOT fire, because it
        # still requires both quantities. This is what separates it from
        # v_only and keeps it faithful to §3.5's "V and T".
        drop_only = [(10.0, 0.0)] * 5 + [(10.0, -5.0)] * 5 + [(10.0, 0.0)] * 20
        self.assertEqual(feed(trigger(SEQUENTIAL), drop_only), [])
        self.assertEqual(len(feed(trigger(V_ONLY), drop_only)), 1)

    def test_sequential_gives_up_after_the_confirm_window(self) -> None:
        # The trunk arrives far too late to belong to the same event.
        late = (
            [(10.0, -5.0)] * 5           # V event
            + [(10.0, 0.0)] * 90         # 3 s of nothing
            + [(80.0, 0.0)] * 10         # trunk down, unrelated
        )
        self.assertEqual(feed(trigger(SEQUENTIAL, confirm_window_s=1.0), late), [])

    def test_walking_fires_nothing_in_any_formulation(self) -> None:
        # Small oscillating velocity, upright trunk.
        walk = [(12.0, -0.4), (14.0, 0.3)] * 40
        for f in (SIMULTANEOUS, SEQUENTIAL, V_ONLY):
            self.assertEqual(feed(trigger(f), walk), [], f"{f} disparo caminando")


class TestScoreFormulation(unittest.TestCase):
    """The reading of §3.5 that follows its sentence literally.

    §3.5: "an instantaneous combination of the two quantities exceeds a
    configurable trigger threshold while V remains negative (downward)".
    All four clauses are testable, and each is tested here, because this
    formulation exists to close a divergence rather than to add a feature —
    if it drifts from the sentence it stops being worth having.
    """

    def score(self, s: float = 2.2):
        return trigger(SCORE, threshold_score=s)

    def test_both_quantities_exactly_at_threshold_scores_two(self) -> None:
        # The anchor that makes the number readable: each term is its
        # quantity over its own threshold.
        trg = self.score(2.0)
        self.assertEqual(len(feed(trg, [(T_TH + 0.01, V_TH - 0.01)] * 10)), 1)

    def test_one_quantity_alone_at_threshold_does_not_reach_two(self) -> None:
        trg = self.score(2.0)
        self.assertEqual(feed(trg, [(T_TH + 0.01, -0.1)] * 10), [])

    def test_it_catches_the_offset_fall_the_conjunction_misses(self) -> None:
        # The whole point: when V is deep the trunk need not have finished
        # rotating for the pair to be, together, past the threshold.
        self.assertEqual(len(feed(self.score(), TestFormulations.OFFSET_FALL)), 1)
        self.assertEqual(feed(trigger(SIMULTANEOUS), TestFormulations.OFFSET_FALL), [])

    def test_it_is_instantaneous_not_a_sequence(self) -> None:
        # T and V large but never on the same frame: a sequential rule fires,
        # an instantaneous one must not. This is what keeps the formulation
        # faithful to the word §3.5 actually uses.
        alternating = [(90.0, 0.0), (0.0, -5.0)] * 30
        self.assertEqual(feed(self.score(), alternating), [])

    def test_upward_motion_never_fires_however_large(self) -> None:
        # §3.5: "while V remains negative (downward)". A subject standing up
        # fast with the trunk down must not score, no matter the magnitude.
        self.assertEqual(feed(self.score(), [(120.0, +9.0)] * 20), [])

    def test_a_barely_upward_drift_with_the_trunk_down_never_fires(self) -> None:
        """The case where §3.5's downward clause actually earns its keep.

        A large positive V is self-suppressing: its term goes strongly
        negative and drags the score under any threshold. The dangerous case
        is the opposite — a subject already on the floor, trunk inverted
        (T = 180 contributes 4.0 on its own) and drifting upward by a hair,
        so the velocity term subtracts almost nothing. Without the explicit
        sign check the score sails past the threshold on the trunk angle
        alone, and the system alarms on someone lying still.
        """
        self.assertEqual(feed(self.score(), [(180.0, +0.01)] * 20), [])
        # And the same posture drifting DOWNWARD does fire, which is the
        # asymmetry the clause is there to create.
        self.assertEqual(len(feed(self.score(), [(180.0, -0.01)] * 20)), 1)

    def test_walking_does_not_fire(self) -> None:
        walk = [(12.0, -0.4), (14.0, 0.3)] * 40
        self.assertEqual(feed(self.score(), walk), [])

    def test_the_threshold_is_configurable(self) -> None:
        modest = [(30.0, -1.0)] * 20          # score = 0.67 + 0.67 = 1.33
        self.assertEqual(feed(self.score(2.0), modest), [])
        self.assertEqual(len(feed(self.score(1.3), modest)), 1)

    def test_nan_in_either_quantity_never_fires(self) -> None:
        nan = float("nan")
        self.assertEqual(feed(self.score(), [(nan, -5.0)] * 20), [])
        self.assertEqual(feed(self.score(), [(90.0, nan)] * 20), [])


class TestInputValidation(unittest.TestCase):

    def test_unknown_formulation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            trigger("whatever_seems_right")

    def test_a_non_negative_v_threshold_is_rejected(self) -> None:
        # §3.5 requires the trigger to fire "while V remains negative". A
        # positive threshold would make the condition true while standing
        # still, firing forever.
        with self.assertRaises(ValueError):
            trigger(V_ONLY, threshold_v_tps=0.0)

    def test_nan_quantities_never_fire(self) -> None:
        # NaN means "not computable", not "zero". An unmeasured frame must
        # not be allowed to vote.
        nan = float("nan")
        for f in (SIMULTANEOUS, SEQUENTIAL, V_ONLY):
            samples = [(nan, nan)] * 30
            self.assertEqual(feed(trigger(f), samples), [], f"{f} disparo con NaN")


class TestStateMachine(unittest.TestCase):
    """Event lifecycle: one fall, one event."""

    def _machine(self, formulation: str = SIMULTANEOUS,
                 cooldown_s: float = 3.0) -> FallStateMachine:
        return FallStateMachine(trigger(formulation), cooldown_s=cooldown_s)

    def _run(self, machine: FallStateMachine, samples) -> list:
        events = []
        for i, (t_deg, v_tps) in enumerate(samples):
            ev = machine.update(i, i * DT, t_deg, v_tps)
            if ev is not None:
                events.append(ev)
        return events

    def test_one_sustained_fall_raises_exactly_one_event(self) -> None:
        # Without a cooldown the trigger would re-arm and fire repeatedly
        # while the subject lies there, turning one fall into an alarm burst.
        events = self._run(self._machine(), [(90.0, -5.0)] * 120)
        self.assertEqual(len(events), 1)

    def test_the_event_carries_the_evidence_that_caused_it(self) -> None:
        events = self._run(self._machine(), [(90.0, -5.0)] * 20)
        ev = events[0]
        self.assertAlmostEqual(ev.t_deg, 90.0)
        self.assertAlmostEqual(ev.v_tps, -5.0)
        self.assertEqual(ev.formulation, SIMULTANEOUS)
        self.assertEqual(ev.frame_index, int(round(ev.timestamp / DT)))

    def test_events_are_marked_stage1_only_until_the_other_stages_exist(self) -> None:
        # A record that said "confirmed" while Stages 2 and 3 are unbuilt
        # would misrepresent what the system actually checked.
        events = self._run(self._machine(), [(90.0, -5.0)] * 20)
        self.assertEqual(events[0].verdict, "stage1_only")

    def test_cooldown_expires_and_a_second_fall_is_caught(self) -> None:
        # The refractory period must suppress duplicates, not deafen the
        # system: a genuine second event after it has to be seen.
        machine = self._machine(cooldown_s=1.0)
        samples = [(90.0, -5.0)] * 10 + [(0.0, 0.0)] * 60 + [(90.0, -5.0)] * 10
        self.assertEqual(len(self._run(machine, samples)), 2)

    def test_cooldown_merges_a_flickering_condition_into_one_event(self) -> None:
        """What the cooldown is actually for, now that the latch exists.

        The latch already collapses a *steadily* true condition into one
        firing. What it cannot collapse is a condition that oscillates —
        T and V crossing back and forth around threshold during a single
        tumble, which is what real landmark noise on a real fall looks like.
        Each dip below re-arms the latch, so without a refractory period one
        fall arrives as several events.
        """
        # Three separate qualifying bursts inside one second.
        burst = [(90.0, -5.0)] * 5 + [(0.0, 0.0)] * 5
        samples = burst * 3
        self.assertEqual(len(self._run(self._machine(cooldown_s=3.0), samples)), 1)
        # With no refractory period the same input yields one per burst.
        self.assertEqual(len(self._run(self._machine(cooldown_s=0.0), samples)), 3)

    def test_reset_drops_pending_state_but_keeps_history(self) -> None:
        machine = self._machine()
        self._run(machine, [(90.0, -5.0)] * 20)
        machine.reset()
        self.assertIs(machine.stage, Stage.MONITORING)
        self.assertEqual(len(machine.events), 1)

    def test_reset_disarms_a_pending_sequential_trigger(self) -> None:
        # A V event seen before a break in observation must not be confirmed
        # by a trunk angle measured after it — that trunk may belong to
        # another body, or to another point in the timeline after a seek.
        machine = self._machine(SEQUENTIAL)
        for i in range(6):
            machine.update(i, i * DT, 10.0, -5.0)       # arm on V
        machine.reset()
        ev = machine.update(6, 6 * DT, 80.0, 0.0)       # trunk arrives after
        self.assertIsNone(ev)


class TestStage2Evaluator(unittest.TestCase):
    """The geometric check on P, in isolation."""

    def _ev(self, **kw) -> Stage2Evaluator:
        kw.setdefault("window_s", 0.3)
        kw.setdefault("outside_fraction", 0.5)
        kw.setdefault("min_samples", 3)
        return Stage2Evaluator(**kw)

    def _run(self, offsets, ev=None) -> str:
        ev = ev or self._ev()
        ev.start(0.0)
        for i, p in enumerate(offsets):
            ev.observe(p)
        return ev.verdict()

    def test_com_outside_throughout_confirms(self) -> None:
        # A topple: the COM is beyond the feet for the whole window.
        self.assertEqual(self._run([0.4] * 10), CONFIRMED)

    def test_com_inside_throughout_rejects(self) -> None:
        # The controlled sit-down §3.5 names by hand: "COM stays within the
        # support polygon". This is the false positive Stage 2 exists to kill.
        self.assertEqual(self._run([-0.3] * 10), REJECTED)

    def test_a_minority_of_outside_frames_rejects(self) -> None:
        # Two frames of landmark noise must not confirm a fall.
        self.assertEqual(self._run([-0.3] * 8 + [0.2] * 2), REJECTED)

    def test_the_vote_is_a_fraction_of_measurable_frames(self) -> None:
        # Half the window unmeasurable, and every measurable frame outside.
        # A count-based rule ("5 of 10") would reject; a fraction confirms,
        # which is the point of expressing it that way.
        nan = float("nan")
        self.assertEqual(self._run([0.4, nan] * 5), CONFIRMED)

    def test_occluded_feet_are_inconclusive_not_rejected(self) -> None:
        # Feet out of frame for the whole window. "I could not look" must not
        # be recorded as "no fall".
        self.assertEqual(self._run([float("nan")] * 10), INCONCLUSIVE)

    def test_too_few_samples_is_inconclusive(self) -> None:
        nan = float("nan")
        self.assertEqual(self._run([0.4, nan, nan, nan, nan], ), INCONCLUSIVE)

    def test_nan_frames_do_not_count_as_inside(self) -> None:
        # The subtle version of the same rule: if NaN were read as 0.0 (not
        # outside), an occluded fall would come back REJECTED.
        nan = float("nan")
        self.assertEqual(self._run([0.4] * 4 + [nan] * 6), CONFIRMED)

    def test_the_window_is_time_based(self) -> None:
        ev = self._ev(window_s=0.3)
        ev.start(10.0)
        self.assertFalse(ev.ready(10.2))
        self.assertTrue(ev.ready(10.3))

    def test_invalid_configuration_is_rejected(self) -> None:
        for kw in ({"window_s": 0.0}, {"outside_fraction": 0.0},
                   {"outside_fraction": 1.5}):
            with self.assertRaises(ValueError):
                self._ev(**kw)


class TestStage2InTheMachine(unittest.TestCase):
    """The funnel with both stages wired together."""

    def _machine(self, **kw) -> FallStateMachine:
        return FallStateMachine(
            trigger(SIMULTANEOUS),
            Stage2Evaluator(window_s=0.3, outside_fraction=0.5, min_samples=3),
            cooldown_s=kw.get("cooldown_s", 3.0),
        )

    def _run(self, samples) -> list:
        machine = self._machine()
        raised = []
        for i, (t_deg, v_tps, p) in enumerate(samples):
            ev = machine.update(i, i * DT, t_deg, v_tps, p)
            if ev is not None:
                raised.append(ev)
        return raised

    def test_a_fall_with_the_com_outside_is_confirmed(self) -> None:
        events = self._run([(90.0, -5.0, 0.4)] * 40)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].verdict, CONFIRMED)
        self.assertAlmostEqual(events[0].p_outside_fraction, 1.0)
        self.assertGreater(events[0].p_samples, 0)

    def test_a_sit_down_that_fires_stage_1_is_rejected_by_stage_2(self) -> None:
        # This is the whole architectural claim of §3.5 in one test: Stage 1
        # is allowed to be wrong, and Stage 2 catches it.
        events = self._run([(90.0, -5.0, -0.4)] * 40)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].verdict, REJECTED)

    def test_the_verdict_is_not_decided_on_the_firing_frame(self) -> None:
        # At the trigger the body is still moving; the geometry has not
        # happened yet. The event must exist immediately and be judged later.
        machine = self._machine()
        ev = None
        for i in range(4):
            ev = machine.update(i, i * DT, 90.0, -5.0, 0.4) or ev
        self.assertIsNotNone(ev)
        self.assertEqual(ev.verdict, "stage1_only")
        for i in range(4, 20):
            machine.update(i, i * DT, 90.0, -5.0, 0.4)
        self.assertEqual(ev.verdict, CONFIRMED)

    def test_without_stage_2_the_event_stands_on_stage_1_alone(self) -> None:
        # The §3.7 ablation "remove Stage 2" must be a configuration, not a
        # code change — and must be visible in the verdict.
        machine = FallStateMachine(trigger(SIMULTANEOUS), None, cooldown_s=3.0)
        ev = None
        for i in range(20):
            ev = machine.update(i, i * DT, 90.0, -5.0, -0.9) or ev
        self.assertIsNotNone(ev)
        self.assertEqual(ev.verdict, "stage1_only")

    def test_a_discontinuity_abandons_the_pending_verdict(self) -> None:
        # Losing the subject mid-window means Stage 2 never finished looking.
        # Judging on a truncated window would invent a verdict.
        machine = self._machine()
        ev = None
        # Long enough to fire (hold 0.1 s = frame 3 at 30 fps), short enough
        # that the 0.3 s Stage-2 window is still open.
        for i in range(6):
            ev = machine.update(i, i * DT, 90.0, -5.0, 0.4) or ev
        self.assertIsNotNone(ev, "no disparo antes del reset")
        self.assertIs(machine.stage, Stage.CONFIRMING)
        machine.reset()
        self.assertEqual(ev.verdict, "stage1_only")
        self.assertIs(machine.stage, Stage.MONITORING)

    def test_no_second_event_starts_while_one_is_being_judged(self) -> None:
        events = self._run([(90.0, -5.0, 0.4)] * 60)
        self.assertEqual(len(events), 1)


class TestStage3Evaluator(unittest.TestCase):
    """Immobility, recovery and the §3.3 severity tags, in isolation.

    Severity here is defined by RECOVERY, not by impact — §3.3 is explicit —
    so the tests are written as three recovery stories: got up, sat up, never
    moved.
    """

    def _ev(self, **kw) -> Stage3Evaluator:
        kw.setdefault("window_s", 10.0)
        kw.setdefault("threshold_w_s", 3.0)
        kw.setdefault("upright_t_deg", 30.0)
        kw.setdefault("recovery_hold_s", 1.0)
        kw.setdefault("standing_extension", 1.1)
        return Stage3Evaluator(**kw)

    def _play(self, frames, ev=None):
        """frames = [(t_deg, i_still_s, extension_ratio)] at 30 fps."""
        ev = ev or self._ev()
        ev.start(0.0)
        for i, (t_deg, still, ext) in enumerate(frames):
            ev.observe(i * DT, t_deg, still, ext)
        return ev.verdict()

    def test_a_subject_who_gets_up_is_mild_and_nullified(self) -> None:
        # Down briefly, then upright with the legs extended and held.
        frames = ([(85.0, 0.5, 0.4)] * 15          # on the floor
                  + [(10.0, 0.0, 1.3)] * 60)       # standing, held 2 s
        self.assertEqual(self._play(frames), (NULLIFIED, MILD))

    def test_a_subject_who_only_sits_up_is_moderate(self) -> None:
        # Trunk upright, legs still folded: §3.3's "partially recovered".
        # This is the case the trunk angle alone cannot distinguish from
        # standing, which is why the leg extension is read at all.
        frames = ([(85.0, 0.5, 0.4)] * 15
                  + [(10.0, 0.0, 0.5)] * 60)       # upright but not standing
        self.assertEqual(self._play(frames), (CONFIRMED_FALL, MODERATE))

    def test_a_subject_who_stays_down_is_severe(self) -> None:
        # Immobility reaches W with the trunk never returning upright.
        frames = [(85.0, min(4.0, i * DT), 0.3) for i in range(150)]
        self.assertEqual(self._play(frames), (CONFIRMED_FALL, SEVERE))

    def test_a_brief_wobble_upright_is_not_a_recovery(self) -> None:
        # Half a second upright, then back down: not a get-up. Without the
        # hold requirement this would nullify a real fall.
        frames = ([(85.0, 0.0, 0.3)] * 10
                  + [(10.0, 0.0, 1.3)] * 10        # 0.33 s upright
                  + [(85.0, 4.0, 0.3)] * 130)
        self.assertEqual(self._play(frames), (CONFIRMED_FALL, SEVERE))

    def test_recovery_resolves_before_the_window_expires(self) -> None:
        # Latency matters: an alarm must not wait out 30 s of observation
        # when the subject stood up after two.
        ev = self._ev(window_s=30.0)
        ev.start(0.0)
        for i in range(90):                        # 3 s
            ev.observe(i * DT, 10.0, 0.0, 1.3)
        self.assertTrue(ev.ready(3.0))

    def test_moving_on_the_floor_the_whole_window_is_unresolved(self) -> None:
        # Never still enough to confirm, never upright enough to clear.
        # Reporting a severity here would be inventing one.
        frames = [(85.0, 0.1, 0.3)] * 300
        verdict, _ = self._play(frames)
        self.assertEqual(verdict, UNRESOLVED)

    def test_without_leg_extension_a_recovery_is_never_called_mild(self) -> None:
        # The conservative fallback: claiming a full recovery that cannot be
        # verified is the error that matters, so it reports moderate.
        frames = [(85.0, 0.5, 0.4)] * 15 + [(10.0, 0.0, 1.3)] * 60
        ev = self._ev(use_leg_extension=False)
        self.assertEqual(self._play(frames, ev), (NULLIFIED, MODERATE))

    def test_standing_once_counts_even_if_the_subject_sits_afterwards(self) -> None:
        # The latch, stated as a story: the subject gets to their feet, then
        # lowers themselves onto a chair with the trunk still upright. They
        # recovered — that happened, and a later sit does not un-happen it.
        # Re-reading the extension every frame instead of latching would
        # downgrade this to a partial recovery.
        frames = ([(85.0, 0.5, 0.4)] * 10
                  + [(10.0, 0.0, 1.3)] * 5         # briefly standing
                  + [(10.0, 0.0, 0.5)] * 60)       # then seated, trunk upright
        self.assertEqual(self._play(frames), (NULLIFIED, MILD))

    def test_the_legs_may_extend_after_the_trunk_straightens(self) -> None:
        # A real get-up: the trunk comes up first, the knees finish after.
        # Requiring both in the same frame would miss it.
        frames = ([(85.0, 0.5, 0.4)] * 10
                  + [(10.0, 0.0, 0.5)] * 5         # upright, knees still bent
                  + [(10.0, 0.0, 1.3)] * 60)       # now standing
        self.assertEqual(self._play(frames), (NULLIFIED, MILD))


class TestFullFunnel(unittest.TestCase):
    """All three stages of §3.5 chained, on synthetic quantity tracks."""

    def _machine(self) -> FallStateMachine:
        return FallStateMachine(
            trigger(SIMULTANEOUS),
            Stage2Evaluator(window_s=0.3, outside_fraction=0.5, min_samples=3),
            Stage3Evaluator(window_s=10.0, threshold_w_s=2.0, upright_t_deg=30.0,
                            recovery_hold_s=1.0, standing_extension=1.1),
            cooldown_s=3.0,
        )

    def _play(self, frames):
        """frames = [(T, V, P, I, ext)]."""
        machine = self._machine()
        raised = []
        for i, f in enumerate(frames):
            ev = machine.update(i, i * DT, *f)
            if ev is not None:
                raised.append(ev)
        return raised

    def test_a_fall_that_stays_down_is_confirmed_severe(self) -> None:
        frames = ([(85.0, -5.0, 0.4, 0.0, 0.3)] * 15
                  + [(85.0, 0.0, 0.4, min(3.0, i * DT), 0.3) for i in range(150)])
        events = self._play(frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].verdict, CONFIRMED_FALL)
        self.assertEqual(events[0].severity, SEVERE)

    def test_a_controlled_sit_down_never_reaches_stage_3(self) -> None:
        # Stage 1 fires, Stage 2 sees the COM stay inside, and the funnel
        # ends there. Crucially it must NOT be rescued by lying still
        # afterwards — a person who sat down does lie still.
        frames = ([(85.0, -5.0, -0.4, 0.0, 0.5)] * 15
                  + [(85.0, 0.0, -0.4, min(5.0, i * DT), 0.5) for i in range(150)])
        events = self._play(frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].verdict, REJECTED)
        self.assertEqual(events[0].severity, "")

    def test_a_fall_the_subject_recovers_from_is_nullified(self) -> None:
        frames = ([(85.0, -5.0, 0.4, 0.0, 0.3)] * 15
                  + [(10.0, 0.0, -0.5, 0.0, 1.3)] * 90)
        events = self._play(frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].verdict, NULLIFIED)
        self.assertEqual(events[0].severity, MILD)

    def test_an_inconclusive_geometry_still_reaches_stage_3(self) -> None:
        # Feet unmeasurable. The event must not die at Stage 2 — the whole
        # point of INCONCLUSIVE — and Stage 3 confirms it on immobility.
        nan = float("nan")
        frames = ([(85.0, -5.0, nan, 0.0, 0.3)] * 15
                  + [(85.0, 0.0, nan, min(3.0, i * DT), 0.3) for i in range(150)])
        events = self._play(frames)
        self.assertEqual(events[0].verdict, CONFIRMED_FALL)
        self.assertEqual(events[0].severity, SEVERE)


if __name__ == "__main__":
    unittest.main()
