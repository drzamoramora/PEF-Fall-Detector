"""Reading PEF-FallDB's labels out of its file names.

The ground truth of this dataset lives in the file name, which makes the
parser a piece of measurement apparatus: a bug here does not crash anything,
it silently scores the detector against the wrong answer. So the tests pin
the exact spellings seen in the 128 clips, including the ones that are
wrong, and pin that nothing else is normalised.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pef_fall_detector.classification import (
    NOT_RECOVERED,
    NO_FALL,
    PARTIALLY_RECOVERED,
    RECOVERED,
)
from pef_fall_detector.dataset import (
    confusion,
    format_confusion,
    truth_from_name,
    video_files,
)


class TestTruthFromName(unittest.TestCase):

    def test_the_four_classes_are_read(self) -> None:
        self.assertEqual(truth_from_name("B05-S1-NoFall.mp4"), NO_FALL)
        self.assertEqual(truth_from_name("A01-S1-Recovered.mp4"), RECOVERED)
        self.assertEqual(truth_from_name("A02-S1-NotRecovered.mp4"), NOT_RECOVERED)
        self.assertEqual(truth_from_name("A03-S1-PartiallyRecovered.mp4"),
                         PARTIALLY_RECOVERED)

    def test_recovery_is_read_as_recovered(self) -> None:
        # Four clips are spelled "Recovery". Not a guess: the material has
        # exactly six actors per fall class, and this actor is what makes the
        # Recovered class reach six — the counts land on 24/24/24 only under
        # this reading.
        self.assertEqual(truth_from_name("A13-S1-Recovery.mp4"), RECOVERED)
        self.assertEqual(truth_from_name("A3-S2-Recovery.mp4"), RECOVERED)

    def test_a_middle_token_is_not_mistaken_for_a_class(self) -> None:
        # The same clip appears as "A13-S1-Recovery" in one generation of the
        # material and "A13-S1-Kitchen-Recovered" in another. Position-based
        # parsing reads "Kitchen" as the class.
        self.assertEqual(truth_from_name("A13-S1-Kitchen-Recovered.mp4"), RECOVERED)

    def test_an_unreadable_name_is_empty_never_a_guess(self) -> None:
        # And never NoFall: "I cannot read this name" and "this clip has no
        # fall" are different statements, and only one of them can be scored.
        for name in ("clip.mp4", "A01-S1-Levantado.mp4", "A01_S1_Recovered.mp4"):
            self.assertEqual(truth_from_name(name), "", name)

    def test_the_odd_indices_are_left_alone(self) -> None:
        # A06-S14 and A11-S14 are almost certainly S4 typos, and A3-S2 is
        # almost certainly A13-S2. None of them touches the CLASS, which is
        # all this parser claims to read; rewriting names to match a theory
        # is how a dataset acquires errors nobody can find later.
        self.assertEqual(truth_from_name("A06-S14-PartiallyRecovered.mp4"),
                         PARTIALLY_RECOVERED)
        self.assertEqual(truth_from_name("A11-S14-NotRecovered.mp4"), NOT_RECOVERED)

    def test_a_full_path_works_like_a_name(self) -> None:
        self.assertEqual(truth_from_name(Path("/x/y/B14-S3-NoFall.mp4")), NO_FALL)


class TestVideoFiles(unittest.TestCase):

    def test_only_videos_sorted_and_not_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("b.mp4", "a.MP4", "notes.txt", "c.mov"):
                (root / name).write_bytes(b"")
            (root / "sub").mkdir()
            (root / "sub" / "deep.mp4").write_bytes(b"")
            found = [p.name for p in video_files(root)]
        # Sorted, videos only, and the subdirectory deliberately skipped: a
        # stray copy of the material must not join a run unnoticed.
        self.assertEqual(found, ["a.MP4", "b.mp4", "c.mov"])


class TestConfusion(unittest.TestCase):

    def rows(self, *pairs):
        return [{"clase_verdad": t, "clase_detectada": d, "clase_revisada": ""}
                for t, d in pairs]

    def test_counts_land_in_the_right_cells(self) -> None:
        matrix, scored, correct = confusion(self.rows(
            (RECOVERED, RECOVERED), (RECOVERED, NOT_RECOVERED),
            (NO_FALL, NO_FALL)))
        self.assertEqual(matrix[RECOVERED][RECOVERED], 1)
        self.assertEqual(matrix[RECOVERED][NOT_RECOVERED], 1)
        self.assertEqual((scored, correct), (3, 2))

    def test_a_clip_with_no_readable_truth_is_not_scored(self) -> None:
        # Excluded, not counted as a failure: an unlabelled clip says nothing
        # about the detector, and counting it would depress every number by
        # an amount that depends on the file names.
        _, scored, correct = confusion(self.rows((RECOVERED, RECOVERED),
                                                 ("", NO_FALL)))
        self.assertEqual((scored, correct), (1, 1))

    def test_a_human_review_overrides_the_machine(self) -> None:
        rows = self.rows((RECOVERED, NOT_RECOVERED))
        rows[0]["clase_revisada"] = RECOVERED
        _, scored, correct = confusion(rows)
        self.assertEqual((scored, correct), (1, 1))

    def test_the_text_matrix_reports_the_tally(self) -> None:
        text = format_confusion(self.rows((RECOVERED, RECOVERED),
                                          (NO_FALL, NO_FALL),
                                          (NOT_RECOVERED, NO_FALL)))
        self.assertIn("aciertos: 2/3", text)
        self.assertIn(NOT_RECOVERED, text)

    def test_no_scorable_rows_says_so_instead_of_dividing_by_zero(self) -> None:
        self.assertIn("sin clips", format_confusion(self.rows(("", ""))))


if __name__ == "__main__":
    unittest.main()
