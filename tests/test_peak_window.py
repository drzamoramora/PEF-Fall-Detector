"""Stage 1's peak window.

WHY THIS FILE EXISTS. §3.5 asks for the event to move to Stage 2 when *"an
instantaneous combination of the two quantities exceeds a configurable
trigger threshold"*. Requiring the same frame is an **additional**
condition: §3.4 describes the temporal signature of each quantity
separately — T reaching the fall range *"within a short time window"*, V
strongly negative *"within a short window"* — and does not say they
coincide.

Measured on PEF-FallDB they never coincide: in the twelve falls the trigger
missed, the T peak and the V peak are 0.6 s to 3.1 s apart. And it is
mechanical, not accidental — T is a position and V a velocity of the same
body.

These tests pin three things, in order of importance:

* that with ``peak_window_s = 0`` the trigger behaves **exactly** as before
  the parameter existed, because the §3.7 ablation depends on it being a
  configuration change and not a code change;
* that the window recovers an out-of-step fall;
* that the window does **not** invent evidence: it does not cross a
  discontinuity, does not read a NaN as zero, and forgets what is old.

Run with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import math
import unittest

from pef_fall_detector.state_machine import (
    SCORE,
    SIMULTANEOUS,
    PeakWindow,
    Stage1Trigger,
)

NAN = float("nan")


def drive(trigger, samples, fps=30.0, start=0.0):
    """Feed (T, V) frame by frame; return the times at which it fired."""
    fired = []
    t = start
    for t_deg, v_tps in samples:
        if trigger.update(t, t_deg, v_tps):
            fired.append(round(t, 4))
        t += 1.0 / fps
    return fired


def trigger(**kw):
    base = dict(formulation=SCORE, threshold_t_deg=45.0, threshold_v_tps=-1.5,
                hold_s=0.0, threshold_score=2.2)
    base.update(kw)
    return Stage1Trigger(**base)


class TestTheWindowItself(unittest.TestCase):

    def test_returns_the_maximum_of_T_and_the_minimum_of_V(self) -> None:
        w = PeakWindow(1.0)
        w.update(0.0, 10.0, -0.5)
        w.update(0.1, 90.0, -0.2)
        t, v = w.update(0.2, 20.0, -3.0)
        self.assertAlmostEqual(t, 90.0)
        self.assertAlmostEqual(v, -3.0)

    def test_forgets_what_left_the_window(self) -> None:
        # A long clip cannot pile up evidence from ten seconds ago: the fall
        # that matters is the one happening now.
        w = PeakWindow(0.5)
        w.update(0.0, 170.0, -5.0)
        t, v = w.update(2.0, 10.0, -0.1)
        self.assertAlmostEqual(t, 10.0)
        self.assertAlmostEqual(v, -0.1)

    def test_a_NaN_is_not_read_as_zero(self) -> None:
        # Reading NaN as 0.0 would let an occlusion LOWER a T maximum or
        # RAISE a V minimum, which is evidence invented the other way round.
        w = PeakWindow(1.0)
        w.update(0.0, 150.0, -4.0)
        t, v = w.update(0.1, NAN, NAN)
        self.assertAlmostEqual(t, 150.0)
        self.assertAlmostEqual(v, -4.0)

    def test_nothing_measurable_returns_NaN(self) -> None:
        # "There was no movement" is not the same as "it could not be measured".
        w = PeakWindow(1.0)
        t, v = w.update(0.0, NAN, NAN)
        self.assertTrue(math.isnan(t))
        self.assertTrue(math.isnan(v))

    def test_the_window_moves_with_the_clock_not_the_frames(self) -> None:
        # Lesson C5: "24 frames" is 0.8 s at 30 fps and 1.6 s at 15 fps.
        # With NaN in between, counting measurable frames does not work either.
        w = PeakWindow(0.5)
        w.update(0.0, 170.0, -5.0)
        for i in range(1, 20):
            t, v = w.update(i * 0.1, 10.0, -0.1)
        self.assertAlmostEqual(t, 10.0)

    def test_a_window_that_is_not_positive_is_refused(self) -> None:
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                PeakWindow(bad)


class TestZeroIsThePreviousBehaviour(unittest.TestCase):
    """What makes the §3.7 ablation possible without touching code."""

    SCENES = [
        [(10.0, -0.1)] * 30,                                   # still
        [(170.0, -0.05)] * 30,                                 # already on the floor
        [(80.0, -0.5)] * 5 + [(20.0, -3.0)] * 5,               # out of step
        [(100.0, -2.0)] * 10,                                  # plain fall
        [(NAN, NAN)] * 10 + [(90.0, -2.0)] * 10,               # with an occlusion
    ]

    def test_zero_fires_the_same_as_without_the_parameter(self) -> None:
        for i, scene in enumerate(self.SCENES):
            with self.subTest(scene=i):
                without = drive(trigger(), scene)
                zero = drive(trigger(peak_window_s=0.0), scene)
                self.assertEqual(without, zero)

    def test_zero_does_not_build_the_window(self) -> None:
        # Not only the same result: it must not pay the cost or carry state
        # that would later have to be reset.
        self.assertIsNone(trigger(peak_window_s=0.0)._peaks)
        self.assertIsNotNone(trigger(peak_window_s=0.8)._peaks)

    def test_a_negative_window_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            trigger(peak_window_s=-0.1)


class TestItRecoversTheOutOfStepFall(unittest.TestCase):

    # The measured profile: V peaks first with the trunk still almost
    # vertical, T peaks half a second later with the body almost stopped, and
    # in between there is nothing notable. No frame reaches 2.2 on its own:
    #   0.20 s  20/45 + 2.5/1.5 = 2.11   (V at its peak, trunk vertical)
    #   0.40 s  20/45 + 0.4/1.5 = 0.71   (the transition, nothing extreme)
    #   0.17 s  85/45 + 0.2/1.5 = 2.09   (T at its peak, already stopped)
    # and no partial pair either: bridging the transition with the V peak
    # gives 2.11, and with the T peak 2.16. Only reaching from one end to the
    # other, 85/45 + 2.5/1.5 = 3.56, crosses the threshold — which is exactly
    # what makes the window's length a parameter with content.
    # The T peak is kept below 99 degrees on purpose: above that the trunk
    # term crosses 2.2 without help from V and the test would stop measuring
    # the window.
    FALL = ([(20.0, -2.5)] * 6 + [(20.0, -0.4)] * 12 + [(85.0, -0.2)] * 6)

    def test_without_the_window_it_does_not_fire(self) -> None:
        self.assertEqual(drive(trigger(), self.FALL), [])

    def test_with_the_window_it_fires(self) -> None:
        self.assertTrue(drive(trigger(peak_window_s=0.8), self.FALL))

    def test_a_window_shorter_than_the_gap_is_not_enough(self) -> None:
        # The window is the parameter, and it has to MEASURE something: if
        # any value would do, there would be nothing to calibrate.
        self.assertEqual(drive(trigger(peak_window_s=0.1), self.FALL), [])

    def test_it_also_works_with_the_simultaneous_formulation(self) -> None:
        # The window does not belong to 'score': it widens the instant for
        # any reading of the same §3.5 sentence.
        without = Stage1Trigger(formulation=SIMULTANEOUS, hold_s=0.0)
        with_window = Stage1Trigger(formulation=SIMULTANEOUS, hold_s=0.0,
                                    peak_window_s=0.8)
        self.assertEqual(drive(without, self.FALL), [])
        self.assertTrue(drive(with_window, self.FALL))


class TestItInventsNoEvidence(unittest.TestCase):

    def test_it_does_not_fire_on_someone_still(self) -> None:
        still = [(12.0, -0.08)] * 60
        self.assertEqual(drive(trigger(peak_window_s=0.8), still), [])

    def test_without_any_negative_V_it_does_not_fire(self) -> None:
        # §3.5: the trigger happens "while V remains negative (downward)".
        # With the trunk at 170 the T term is worth 3.8 on its own, so this
        # guard is all that separates a fall from someone GETTING UP from
        # the floor.
        rising = [(170.0, 2.0)] * 30
        self.assertEqual(drive(trigger(peak_window_s=0.8), rising), [])

    def test_reset_stops_it_crossing_a_discontinuity(self) -> None:
        """The T peak from before an occlusion cannot be paired with the V after it.

        It may be another body, or another point of the timeline. It is the
        same reason ``sequential`` does not let an armed event survive.
        """
        t = trigger(peak_window_s=0.8)
        drive(t, [(85.0, -0.1)] * 6)           # T only, not enough
        t.reset()                              # subject lost
        self.assertEqual(drive(t, [(20.0, -2.6)] * 3, start=0.2), [])

    def test_without_a_reset_they_would_combine(self) -> None:
        # The complement of the one above: it shows the reset is what stops
        # the combination, not that the scene could not fire.
        t = trigger(peak_window_s=0.8)
        drive(t, [(85.0, -0.1)] * 6)
        self.assertTrue(drive(t, [(20.0, -2.6)] * 3, start=0.2))


if __name__ == "__main__":
    unittest.main()
