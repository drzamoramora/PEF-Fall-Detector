"""Clip-level labelling: the four PEF-FallDB classes (§3.3).

These tests pin the two decisions that turn per-event severities into one
label per video, and the reason each was made the way it was:

* the §3.3 mapping itself, because it is the only place the paper's
  vocabulary and the dataset's vocabulary are allowed to meet;
* that an unjudged event is UNDETERMINED and never NoFall, because a
  confusion matrix that cannot separate "never fired" from "fired but could
  not judge" hides two different defects behind one number;
* the end-of-clip rule, whose whole reason to exist is clip A13 of this
  project's own dataset: the subject lies still for about six seconds and
  then stands up unaided, and a resolver that closes on the first branch to
  fire calls that NotRecovered — the opposite label.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from pef_fall_detector.classification import (
    NO_FALL,
    NOT_RECOVERED,
    PARTIALLY_RECOVERED,
    RECOVERED,
    UNDETERMINED,
    class_for_event,
    classify_clip,
)
from pef_fall_detector.state_machine import Stage3Evaluator, TriggerEvent


def event(verdict: str, severity: str = "", timestamp: float = 1.0) -> TriggerEvent:
    return TriggerEvent(frame_index=0, timestamp=timestamp, t_deg=60.0,
                        v_tps=-3.0, formulation="score",
                        verdict=verdict, severity=severity)


class TestSeverityMapping(unittest.TestCase):
    """§3.3's three tags, and the two outcomes it does not name."""

    def test_the_three_paper_tags_map_to_three_classes(self) -> None:
        self.assertEqual(class_for_event("stage3_nullified", "mild"), RECOVERED)
        self.assertEqual(class_for_event("stage3_confirmed", "moderate"),
                         PARTIALLY_RECOVERED)
        self.assertEqual(class_for_event("stage3_confirmed", "severe"),
                         NOT_RECOVERED)

    def test_a_geometrically_rejected_event_is_a_real_no_fall(self) -> None:
        # Stage 2 looked at the support geometry and found no topple. That is
        # a judgement, so it earns the NoFall label.
        self.assertEqual(class_for_event("stage2_rejected", ""), NO_FALL)

    def test_an_unjudged_event_is_undetermined_not_no_fall(self) -> None:
        # The distinction this whole label exists for. The trigger DID fire;
        # calling it NoFall would record a claim the funnel never made, and
        # would score as a correct negative on a NoFall clip.
        for verdict in ("stage1_only", "stage2_inconclusive", "stage3_unresolved"):
            self.assertEqual(class_for_event(verdict, ""), UNDETERMINED, verdict)

    def test_an_unknown_pairing_degrades_to_undetermined(self) -> None:
        # A verdict this module has never heard of must not be guessed at.
        self.assertEqual(class_for_event("stage9_teleported", "critical"),
                         UNDETERMINED)


class TestClipLabel(unittest.TestCase):

    def test_a_clip_with_no_events_is_no_fall(self) -> None:
        label, reason = classify_clip([])
        self.assertEqual(label, NO_FALL)
        self.assertIn("sin eventos", reason)

    def test_the_worst_event_names_the_clip(self) -> None:
        # Someone who gets up from one fall and stays down after a second one
        # is a NotRecovered clip; averaging or taking the first would lose the
        # outcome that matters.
        label, reason = classify_clip([
            event("stage3_nullified", "mild", 2.0),
            event("stage3_confirmed", "severe", 9.0),
        ])
        self.assertEqual(label, NOT_RECOVERED)
        self.assertIn("t=9.00", reason)
        self.assertIn("el peor de 2", reason)

    def test_undetermined_outranks_no_fall_but_not_a_real_outcome(self) -> None:
        # An event nobody could judge must not be outvoted by silence...
        self.assertEqual(classify_clip([event("stage2_inconclusive")])[0],
                         UNDETERMINED)
        # ...but must not outrank a stage that actually reached a verdict.
        label, _ = classify_clip([event("stage2_inconclusive", "", 1.0),
                                  event("stage3_confirmed", "severe", 5.0)])
        self.assertEqual(label, NOT_RECOVERED)

    def test_the_reason_carries_the_evidence(self) -> None:
        # A label without its evidence can be believed but not checked, and a
        # human annotator disagreeing with the machine needs to see why.
        _, reason = classify_clip([event("stage3_confirmed", "severe", 4.25)])
        self.assertIn("t=4.25", reason)
        self.assertIn("stage3_confirmed/severe", reason)


class TestEndOfClipRule(unittest.TestCase):
    """§3.3's tags are three ENDINGS, so the label reads the ending."""

    def build(self, **kw) -> Stage3Evaluator:
        stage3 = Stage3Evaluator(window_s=30.0, threshold_w_s=5.0,
                                 upright_t_deg=30.0, recovery_hold_s=1.0,
                                 standing_extension=1.1, defer_to_end=True, **kw)
        stage3.start(0.0)
        return stage3

    def feed(self, stage3, start, seconds, t_deg, extension, still=0.0):
        t = start
        while t < start + seconds:
            stage3.observe(t, t_deg, still, extension)
            t += 1 / 30.0
        return t

    def test_still_down_at_the_end_is_severe(self) -> None:
        stage3 = self.build()
        self.feed(stage3, 0.0, 8.0, t_deg=95.0, extension=1.5, still=6.0)
        stage3.finalise()
        self.assertEqual(stage3.verdict(), ("stage3_confirmed", "severe"))

    def test_standing_at_the_end_is_mild(self) -> None:
        stage3 = self.build()
        end = self.feed(stage3, 0.0, 6.0, t_deg=95.0, extension=1.5, still=6.0)
        self.feed(stage3, end, 2.0, t_deg=4.0, extension=1.4)   # se levanta
        stage3.finalise()
        self.assertEqual(stage3.verdict(), ("stage3_nullified", "mild"))

    def test_upright_but_folded_at_the_end_is_moderate(self) -> None:
        # Sat up, knelt, propped against furniture: §3.3's "partially
        # recovered". The trunk is up; the legs never reach standing range.
        stage3 = self.build()
        end = self.feed(stage3, 0.0, 6.0, t_deg=95.0, extension=1.5, still=6.0)
        self.feed(stage3, end, 2.0, t_deg=10.0, extension=0.6)
        stage3.finalise()
        self.assertEqual(stage3.verdict(), ("stage3_confirmed", "moderate"))

    def test_the_A13_case_immobility_then_a_recovery(self) -> None:
        """The regression this rule was written for.

        Six seconds motionless past the confirmation threshold W, then the
        subject stands. Greedy resolution closes this as ``severe`` at t=5;
        the ending says ``mild``. Truth: Recovered.
        """
        stage3 = self.build()
        end = self.feed(stage3, 0.0, 6.0, t_deg=95.0, extension=1.5, still=6.0)
        self.feed(stage3, end, 2.0, t_deg=3.0, extension=1.3)
        # Deferred: nothing may have closed it in the meantime, even though I
        # passed W several seconds ago.
        self.assertFalse(stage3.ready(end + 2.0))
        stage3.finalise()
        self.assertEqual(stage3.verdict(), ("stage3_nullified", "mild"))

    def test_greedy_mode_still_closes_early(self) -> None:
        # The live behaviour of §3.6 is unchanged: an alert that waits for the
        # end of an episode that may never end is worthless.
        stage3 = Stage3Evaluator(window_s=30.0, threshold_w_s=5.0)
        stage3.start(0.0)
        self.feed(stage3, 0.0, 8.0, t_deg=95.0, extension=1.5, still=6.0)
        self.assertTrue(stage3.ready(8.0))
        self.assertEqual(stage3.verdict(), ("stage3_confirmed", "severe"))

    def test_a_subject_lost_at_the_end_stays_unresolved(self) -> None:
        # No usable trunk angle in the final window. "Remained down" is the
        # outcome that matters most, and inventing it for someone the camera
        # stopped seeing is the one error with a real cost.
        stage3 = self.build()
        self.feed(stage3, 0.0, 6.0, t_deg=95.0, extension=1.5, still=6.0)
        self.feed(stage3, 6.0, 2.0, t_deg=float("nan"), extension=float("nan"))
        stage3.finalise()
        self.assertEqual(stage3.verdict()[0], "stage3_unresolved")
        self.assertEqual(class_for_event(*stage3.verdict()), UNDETERMINED)

    def test_one_collapsed_frame_cannot_decide_the_label(self) -> None:
        """Why the final trunk angle is a median, not a mean or a last frame.

        Clip A14 produced a trunk of 2 px with the pose model reporting 0.99
        confidence, and trunk angles of 176 deg next to 26 deg on consecutive
        frames. A mean over the final window crosses the upright threshold on
        one such frame; a median does not.
        """
        stage3 = self.build()
        end = self.feed(stage3, 0.0, 6.0, t_deg=95.0, extension=1.5, still=6.0)
        # A second of genuine standing, with one absurd frame in the middle.
        self.feed(stage3, end, 0.5, t_deg=4.0, extension=1.4)
        stage3.observe(end + 0.5, 176.0, 0.0, 1.4)
        self.feed(stage3, end + 0.55, 0.5, t_deg=4.0, extension=1.4)
        stage3.finalise()
        self.assertEqual(stage3.verdict(), ("stage3_nullified", "mild"))

    def test_finalising_twice_keeps_the_first_answer(self) -> None:
        stage3 = self.build()
        self.feed(stage3, 0.0, 6.0, t_deg=95.0, extension=1.5, still=6.0)
        stage3.finalise()
        first = stage3.verdict()
        stage3.finalise()
        self.assertEqual(stage3.verdict(), first)


if __name__ == "__main__":
    unittest.main()
