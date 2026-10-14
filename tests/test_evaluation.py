"""The metrics of §3.7, and the partition.

These tests protect numbers that go into Section 4 of the paper. A mistake
here breaks nothing visibly: it produces a plausible, wrong figure, which is
the worst kind of error a measuring instrument can have.

What is pinned, and why:

* the binary fall / no-fall, because accuracy over four classes mixes two
  different clinical questions and shows neither;
* that the undetermined case counts as negative — what the deployed system
  does — and is reported separately;
* the Wilson interval, because with 40 clips the normal approximation gives
  limits above 100 %;
* that an unknown actor is NOT assigned by default, because assigning it
  silently contaminates exactly what the partition protects.

Run with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from pef_fall_detector.classification import (
    NO_FALL,
    NOT_RECOVERED,
    PARTIALLY_RECOVERED,
    RECOVERED,
    UNDETERMINED,
)
from pef_fall_detector.config import PARTITION_PATH
from pef_fall_detector.evaluation import (
    actor_of,
    binary_metrics,
    fp_per_hour,
    load_split,
    pass_summary,
    per_class,
    report,
    split_rows,
    undetermined_count,
    wilson,
)

SPLIT = PARTITION_PATH


def row(video, truth, detected, duration="10.4", reviewed=""):
    return {"video": video, "true_class": truth, "detected_class": detected,
            "reviewed_class": reviewed, "duration_s": duration}


class TestActorOf(unittest.TestCase):

    def test_the_actor_is_the_first_field(self) -> None:
        self.assertEqual(actor_of("A01-S1-Recovered.mp4"), "A01")
        self.assertEqual(actor_of("B14-S3-NoFall.mp4"), "B14")

    def test_the_A3_typo_resolves_to_its_real_actor(self) -> None:
        # A13 has S1, S3 and S4; this one completes the four every actor has. Read
        # as an actor of its own, it would leave a one-clip actor outside the
        # partition.
        self.assertEqual(actor_of("A3-S2-Recovery.mp4"), "A13")

    def test_a_full_path_works(self) -> None:
        self.assertEqual(actor_of("/x/y/A07-S2-Recovered.mp4"), "A07")


class TestSplit(unittest.TestCase):

    def test_the_shipped_split_covers_every_actor_exactly_once(self) -> None:
        split = load_split(SPLIT)
        self.assertEqual(len(split), 32, "deberian ser 32 actores")
        self.assertEqual(sum(1 for s in split.values() if s == "test"), 10)
        self.assertEqual(sum(1 for s in split.values() if s == "train"), 22)

    def test_the_split_is_stratified(self) -> None:
        # Two actors of each fall class and four of NoFall in test. A partition
        # that left a whole class on one side would measure nothing about it.
        split = load_split(SPLIT)
        test = {a for a, s in split.items() if s == "test"}
        self.assertEqual(test, {"A07", "A16", "A08", "A17", "A09", "A18",
                                "B03", "B06", "B09", "B12"})

    def test_an_unknown_actor_is_not_silently_assigned(self) -> None:
        # New material, or a misspelt name. Sending it to one side by default would
        # contaminate exactly what the partition protects.
        groups = split_rows([row("Z99-S1-Recovered.mp4", RECOVERED, RECOVERED)],
                            load_split(SPLIT))
        self.assertEqual(len(groups["unassigned"]), 1)
        self.assertEqual(len(groups["train"]) + len(groups["test"]), 0)

    def test_rows_land_on_the_side_their_actor_belongs_to(self) -> None:
        groups = split_rows([row("A07-S1-Recovered.mp4", RECOVERED, RECOVERED),
                             row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED)],
                            load_split(SPLIT))
        self.assertEqual(len(groups["test"]), 1)
        self.assertEqual(len(groups["train"]), 1)

    def test_comments_are_not_read_as_entries(self) -> None:
        # The file carries one comment per actor (`- A07   # Recovered`) and a
        # block of dashed comments at the top. None of them is data.
        self.assertNotIn("Recovered", load_split(SPLIT))
        self.assertNotIn("#", "".join(load_split(SPLIT)))


class TestBinaryMetrics(unittest.TestCase):

    def test_severity_confusions_do_not_hurt_the_binary(self) -> None:
        # The whole point of reporting the binary separately: a system that confuses
        # severities but catches the falls works, and accuracy over four classes
        # does not show it.
        m = binary_metrics([
            row("A01-S1-Recovered.mp4", RECOVERED, NOT_RECOVERED),
            row("A02-S1-NotRecovered.mp4", NOT_RECOVERED, PARTIALLY_RECOVERED),
            row("B01-S1-NoFall.mp4", NO_FALL, NO_FALL),
        ])
        self.assertEqual(m["sensitivity"], 1.0)
        self.assertEqual(m["specificity"], 1.0)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (2, 0, 0, 1))

    def test_a_missed_fall_is_a_false_negative(self) -> None:
        m = binary_metrics([row("A01-S1-Recovered.mp4", RECOVERED, NO_FALL)])
        self.assertEqual((m["tp"], m["fn"]), (0, 1))
        self.assertEqual(m["sensitivity"], 0.0)

    def test_undetermined_counts_as_no_alarm(self) -> None:
        # It is what the deployed system does: `dispatch_verdicts` only dispatches
        # `stage3_confirmed`, so an undetermined event raises no alarm and nobody
        # comes. Counting it as positive would measure an intention.
        m = binary_metrics([
            row("A01-S1-Recovered.mp4", RECOVERED, UNDETERMINED),
            row("B01-S1-NoFall.mp4", NO_FALL, UNDETERMINED),
        ])
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (0, 1, 0, 1))

    def test_a_human_review_overrides_the_machine(self) -> None:
        m = binary_metrics([row("A01-S1-Recovered.mp4", RECOVERED, NO_FALL,
                                reviewed=RECOVERED)])
        self.assertEqual(m["tp"], 1)

    def test_f1_is_nan_rather_than_zero_when_undefined(self) -> None:
        # With no predicted and no real positive, F1 is not 0: it is undefined.
        # A 0 would read as "the system failed", which is a different claim from
        # "there is nothing to measure it with".
        m = binary_metrics([row("B01-S1-NoFall.mp4", NO_FALL, NO_FALL)])
        self.assertNotEqual(m["f1"], m["f1"])


    def test_every_ratio_on_a_mixed_set(self) -> None:
        # 2 TP, 2 FN, 1 FP, 3 TN: every count differs, so no ratio can
        # borrow another's denominator unnoticed.
        m = binary_metrics([
            row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED),
            row("A01-S2-Recovered.mp4", RECOVERED, NOT_RECOVERED),
            row("A01-S3-Recovered.mp4", RECOVERED, NO_FALL),
            row("A01-S4-Recovered.mp4", RECOVERED, NO_FALL),
            row("B01-S1-NoFall.mp4", NO_FALL, RECOVERED),
            row("B01-S2-NoFall.mp4", NO_FALL, NO_FALL),
            row("B01-S3-NoFall.mp4", NO_FALL, NO_FALL),
            row("B01-S4-NoFall.mp4", NO_FALL, NO_FALL),
        ])
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (2, 2, 1, 3))
        self.assertAlmostEqual(m["sensitivity"], 2 / 4)
        self.assertAlmostEqual(m["specificity"], 3 / 4)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 4 / 7)
        self.assertEqual(m["sens_ci"], wilson(2, 4))
        self.assertEqual(m["spec_ci"], wilson(3, 4))


class TestWilson(unittest.TestCase):

    def test_perfect_score_does_not_exceed_one(self) -> None:
        # What the normal approximation gets wrong: 40/40 gets an upper limit
        # above 100 %.
        lo, hi = wilson(40, 40)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 1.0)

    def test_zero_does_not_go_below_zero(self) -> None:
        lo, hi = wilson(0, 40)
        self.assertGreaterEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_the_interval_narrows_with_more_clips(self) -> None:
        narrow = wilson(360, 400)
        wide = wilson(36, 40)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_forty_clips_at_ninety_percent_span_about_eighteen_points(self) -> None:
        # The number that justifies reporting the interval: on the test partition,
        # 90 % and 81 % cannot be told apart.
        lo, hi = wilson(36, 40)
        self.assertGreater(hi - lo, 0.12)

    def test_an_empty_set_is_nan_not_zero(self) -> None:
        lo, hi = wilson(0, 0)
        self.assertNotEqual(lo, lo)
        self.assertNotEqual(hi, hi)


class TestFalsePositivesPerHour(unittest.TestCase):

    def test_only_non_fall_footage_is_in_the_denominator(self) -> None:
        # An alarm on a real fall is not a false alarm, and its footage does not
        # dilute the rate.
        rows = [row("B01-S1-NoFall.mp4", NO_FALL, RECOVERED, duration="3600"),
                row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED, duration="3600")]
        self.assertAlmostEqual(fp_per_hour(rows), 1.0, places=6)

    def test_no_footage_is_nan_not_zero(self) -> None:
        self.assertNotEqual(fp_per_hour([]), fp_per_hour([]))


class TestReport(unittest.TestCase):

    def test_the_report_carries_the_interval_and_the_undetermined_row(self) -> None:
        rows = [row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED),
                row("B01-S1-NoFall.mp4", NO_FALL, UNDETERMINED)]
        text = report(rows, "TEST")
        self.assertIn("95% CI", text)
        self.assertIn("undetermined      : 1", text)
        self.assertIn("TEST", text)

    def test_per_class_counts(self) -> None:
        counts = per_class([row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED),
                            row("A04-S1-Recovered.mp4", RECOVERED, NO_FALL)])
        self.assertEqual(counts[RECOVERED], (1, 2))

    def test_unlabelled_rows_are_excluded_not_failed(self) -> None:
        # A clip with no readable truth says nothing about the detector; counting it
        # as a failure would lower every figure by an amount that depends on the file
        # names.
        self.assertIn("no clips", report([row("x.mp4", "", NO_FALL)], "X"))
        self.assertEqual(undetermined_count([row("x.mp4", "", UNDETERMINED)]), 1)


class TestPassSummary(unittest.TestCase):

    def test_each_side_is_reported_with_its_own_rows(self) -> None:
        # One train clip (A01), two test clips (A07) and one clip of an actor
        # outside the partition (Z99): each block counts only its own.
        rows = [row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED),
                row("A07-S1-Recovered.mp4", RECOVERED, RECOVERED),
                row("A07-S2-Recovered.mp4", RECOVERED, NO_FALL),
                row("Z99-S1-Recovered.mp4", RECOVERED, RECOVERED)]
        text = pass_summary(rows, SPLIT)
        self.assertIn("CALIBRATION  (1 clips)", text)
        self.assertIn("TEST  (2 clips)", text)
        self.assertIn("UNASSIGNED  (1 clips)", text)

    def test_without_a_partition_file_only_the_matrix(self) -> None:
        rows = [row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED)]
        text = pass_summary(rows, SPLIT.parent / "no-such-partition.yaml")
        self.assertIn("correct: 1/1", text)
        self.assertNotIn("CALIBRATION", text)


if __name__ == "__main__":
    unittest.main()
