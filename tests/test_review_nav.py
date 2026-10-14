"""Moving between the queue's clips after the pass.

WHY IT EXISTS. PEF-FallDB's situations are recorded with four subjects, and
what explains why three clips fail and one succeeds is almost never inside
one clip: it is in the difference between them. Reviewing is comparing, and
comparing needs going back and forth.

What is pinned:

* that the arrows do nothing during the pass, because loading another clip
  mid-measurement would relabel a clip against a pipeline that saw half of
  another — the same accident the open-video button is already barred from;
* that they stop at the ends instead of wrapping around;
* that reviewing does NOT touch the pass results, which are exactly what one
  compares against while looking.

Run with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# PEF-Lab runs where Qt is installed; the rest of the project — the pipeline,
# the evaluation, the §3.6 deployment — does not need it and is sometimes
# tested on machines without it. Skipping is the right answer there: an import
# failure on a machine with no GUI says nothing about this window, and would
# leave the suite red for a reason that is not a defect.
try:
    from PySide6.QtWidgets import QApplication, QTableWidgetItem
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("PySide6 is not installed; PEF-Lab is not tested here") from None

from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.gui.lab_window import LabWindow  # noqa: E402
from tests.support import TEST_CFG  # noqa: E402

_app = QApplication.instance() or QApplication([])


class TestReviewNavigation(unittest.TestCase):

    def setUp(self) -> None:
        self.win = LabWindow(Config(TEST_CFG))
        self.win._queue = [Path(f"/no/existe/A14-S{i}-NotRecovered.mp4")
                           for i in (1, 2, 3, 4)]
        self.win.queue_panel.table.setRowCount(4)
        for r, p in enumerate(self.win._queue):
            self.win.queue_panel.table.setItem(r, 0, QTableWidgetItem(p.name))
        self.opened: list[int] = []
        # The real load opens the file; here only where it points matters.
        self.win._open_for_review = self._spy(self.win._open_for_review)

    def _spy(self, real):
        def wrapped(row):
            self.opened.append(row)
            self.win._review_index = row
        return wrapped

    def tearDown(self) -> None:
        self.win.close()

    def test_with_no_clip_open_the_right_arrow_enters_at_the_first(self) -> None:
        # Arriving at a folder without knowing where to start is the normal case,
        # not an error: the first arrow has to do something.
        self.win._review_step(+1)
        self.assertEqual(self.opened, [0])

    def test_with_no_clip_open_the_left_arrow_enters_at_the_last(self) -> None:
        self.win._review_step(-1)
        self.assertEqual(self.opened, [3])

    def test_moves_forward_and_back_one_at_a_time(self) -> None:
        self.win._review_index = 1
        self.win._review_step(+1)
        self.win._review_step(+1)
        self.win._review_step(-1)
        self.assertEqual(self.opened, [2, 3, 2])

    def test_stops_at_the_last_instead_of_wrapping_around(self) -> None:
        """Going back to the first one without notice reads as "nothing happened".

        With four subjects per situation, the point is comparing different clips;
        a silent wrap makes one believe the arrow did not respond and has to be
        pressed again.
        """
        self.win._review_index = 3
        self.win._review_step(+1)
        self.assertEqual(self.opened, [])

    def test_stops_at_the_first(self) -> None:
        self.win._review_index = 0
        self.win._review_step(-1)
        self.assertEqual(self.opened, [])

    def test_during_the_pass_the_arrows_do_nothing(self) -> None:
        # Loading another clip mid-measurement would relabel a clip against a
        # pipeline that saw half of another.
        self.win._queue_running = True
        self.win._review_index = 1
        self.win._review_step(+1)
        self.assertEqual(self.opened, [])

    def test_with_an_empty_queue_they_do_nothing(self) -> None:
        self.win._queue = []
        self.win._review_step(+1)
        self.assertEqual(self.opened, [])

    def test_clearing_the_queue_drops_the_anchor(self) -> None:
        # Otherwise the next folder would start from an index of the previous one,
        # which may not even exist.
        self.win._review_index = 2
        self.win._on_queue_clear()
        self.assertIsNone(self.win._review_index)


class TestWhatReviewingMustNotTouch(unittest.TestCase):

    def setUp(self) -> None:
        self.win = LabWindow(Config(TEST_CFG))
        self.win._queue = [Path("/no/existe/A14-S1-NotRecovered.mp4")]
        self.win.queue_panel.table.setRowCount(1)
        self.win.queue_panel.table.setItem(0, 0, QTableWidgetItem("A14-S1"))

    def tearDown(self) -> None:
        self.win.close()

    def test_the_pass_results_survive_the_review(self) -> None:
        """They are what one compares against while looking at the clip."""
        self.win._show_queue_result(0, "NoFall", "NotRecovered")
        self.win._review_index = 0
        self.win._update_review_label()
        self.assertEqual(self.win.queue_panel.table.item(0, 2).text(), "NoFall")
        self.assertEqual(self.win.queue_panel.table.item(0, 3).text(), "X")

    def test_the_counter_says_where_it_is_and_of_how_many(self) -> None:
        self.win._review_index = 0
        self.win._update_review_label()
        text = self.win.queue_panel.review_label.text()
        self.assertIn("1 of 1", text)
        self.assertIn("A14-S1", text)

    def test_with_no_clip_open_the_counter_explains_how_to_start(self) -> None:
        self.win._review_index = None
        self.win._update_review_label()
        self.assertIn("Alt", self.win.queue_panel.review_label.text())


class TestTheButtonsAreWired(unittest.TestCase):
    """B6: the Queue tab's buttons and double-click reach the window."""

    setUp = TestReviewNavigation.setUp
    tearDown = TestReviewNavigation.tearDown
    _spy = TestReviewNavigation._spy

    def test_previous_and_next(self) -> None:
        self.win._update_controls()             # what adding a folder does
        self.win._review_index = 2
        self.win.queue_panel.prev_btn.click()
        self.win.queue_panel.next_btn.click()
        self.win.queue_panel.next_btn.click()
        self.assertEqual(self.opened, [1, 2, 3])

    def test_double_click_opens_that_row(self) -> None:
        self.win.queue_panel.table.cellDoubleClicked.emit(2, 0)
        self.assertEqual(self.opened, [2])


if __name__ == "__main__":
    unittest.main()
