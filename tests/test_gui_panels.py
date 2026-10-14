"""PEF-Lab's panels on their own (B6): the queue table, the curves, the View tab.

The window tests (test_lab_frozen, test_review_nav) drive these panels through
LabWindow; these check what each panel does by itself, including the paths
the window tests reach only through a file dialog or a live camera.
"""

from __future__ import annotations

import copy
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("PySide6 is not installed; PEF-Lab is not tested here") from None

from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.gui import plots  # noqa: E402
from pef_fall_detector.gui.plots import PlotPanel, frame_rgb  # noqa: E402
from pef_fall_detector.gui.queue_panel import QueuePanel, metrics_text  # noqa: E402
from pef_fall_detector.gui.widgets import ViewPanel, format_time  # noqa: E402
from tests.support import TEST_CFG  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _result(stage="MONITORING", detected=True, reliable=True, **quantities):
    return SimpleNamespace(pose=SimpleNamespace(detected=detected), reliable=reliable,
                           stage=stage, quantities=quantities)


class TestQueuePanel(unittest.TestCase):

    def setUp(self) -> None:
        self.panel = QueuePanel(_app.style())

    def test_clips_are_appended_after_the_rows_already_there(self) -> None:
        self.panel.add_clips([Path("A01-S1-NotRecovered.mp4")])
        self.panel.add_clips([Path("B01-S1-NoFall.mp4"), Path("X-unlabelled.mp4")])
        t = self.panel.table
        self.assertEqual(t.rowCount(), 3)
        self.assertEqual([t.item(r, 0).text() for r in range(3)],
                         ["A01-S1-NotRecovered.mp4", "B01-S1-NoFall.mp4", "X-unlabelled.mp4"])
        self.assertEqual([t.item(r, 1).text() for r in range(3)], ["NotRecovered", "NoFall", "?"])
        self.assertEqual(self.panel.progress.maximum(), 3)
        self.assertEqual(self.panel.progress.value(), 0)

    def test_the_reviewed_column_reads_the_human_choice(self) -> None:
        self.panel.add_clips([Path("A01-S1-NotRecovered.mp4")])
        self.assertEqual(self.panel.reviewed(0), "")
        self.panel.table.cellWidget(0, 4).setCurrentText("Recovered")
        self.assertEqual(self.panel.reviewed(0), "Recovered")
        self.assertIsNone(self.panel.reviewed(5))

    def test_a_result_marks_the_row_and_moves_the_progress(self) -> None:
        self.panel.add_clips([Path("A01-S1-NotRecovered.mp4"), Path("A01-S2-NotRecovered.mp4")])
        self.panel.show_result(0, "NotRecovered", "NotRecovered", "score text")
        self.panel.show_result(1, "NoFall", "NotRecovered", "score text 2")
        t = self.panel.table
        self.assertEqual((t.item(0, 2).text(), t.item(0, 3).text()), ("NotRecovered", "OK"))
        self.assertEqual((t.item(1, 2).text(), t.item(1, 3).text()), ("NoFall", "X"))
        self.assertEqual(t.item(1, 3).foreground().color(), Qt.red)
        self.assertEqual(self.panel.progress.value(), 2)
        self.assertEqual(self.panel.score.text(), "score text 2")

    def test_no_truth_no_mark(self) -> None:
        self.panel.add_clips([Path("X-unlabelled.mp4")])
        self.panel.show_result(0, "NoFall", "", "")
        self.assertEqual(self.panel.table.item(0, 3).text(), "")

    def test_clear_empties_everything(self) -> None:
        self.panel.add_clips([Path("A01-S1-NotRecovered.mp4")])
        self.panel.show_result(0, "NotRecovered", "NotRecovered", "x")
        self.panel.clear()
        self.assertEqual(self.panel.table.rowCount(), 0)
        self.assertEqual(self.panel.progress.maximum(), 0)
        self.assertEqual(self.panel.score.text(), "")

    def test_the_metrics_line(self) -> None:
        rows = [{"true_class": "NotRecovered", "detected_class": "NotRecovered"},
                {"true_class": "NotRecovered", "detected_class": "Recovered"},
                {"true_class": "NoFall", "detected_class": "NoFall"},
                {"true_class": "", "detected_class": "NoFall"}]
        self.assertEqual(metrics_text(rows),
                         "4-class 2/3 (67%)  ·  exact falls 1/2  ·  sens 100%  ·  spec 100%")
        self.assertEqual(metrics_text([]), "")


class TestPlotPanel(unittest.TestCase):

    def setUp(self) -> None:
        self.panel = PlotPanel(Config(copy.deepcopy(TEST_CFG)))

    def test_with_the_curves_off_only_the_timeline_is_kept(self) -> None:
        self.panel.update(0, 0.0, _result(T_deg=10.0), curves=False, live=False)
        self.assertEqual(self.panel.timeline_rgb, {0: plots.STAGE_RGB["MONITORING"]})
        self.assertEqual(self.panel.plot_data, {})

    def test_live_keeps_only_the_last_frames(self) -> None:
        for i in range(plots.LIVE_FRAMES + 5):
            self.panel.update(i, i / 30.0, _result(T_deg=float(i)), curves=True, live=True)
        self.assertEqual(len(self.panel.plot_data), plots.LIVE_FRAMES)
        self.assertEqual(min(self.panel.plot_data), 5)
        self.assertEqual(min(self.panel.timeline_rgb), 5)

    def test_a_recorded_clip_keeps_every_frame(self) -> None:
        for i in range(plots.LIVE_FRAMES + 5):
            self.panel.update(i, i / 30.0, _result(T_deg=float(i)), curves=True, live=False)
        self.assertEqual(len(self.panel.plot_data), plots.LIVE_FRAMES + 5)

    def test_a_missing_quantity_is_a_gap_not_a_zero(self) -> None:
        self.panel.update(0, 0.0, _result(T_deg=10.0), curves=True, live=False)
        point = self.panel.plot_data[0]
        self.assertEqual(point["T_deg"], 10.0)
        self.assertNotEqual(point["V_tps"], point["V_tps"])          # NaN

    def test_clear_removes_data_and_marks(self) -> None:
        self.panel.update(0, 0.0, _result(T_deg=10.0), curves=True, live=False)
        self.panel.mark(0.0, "#ff1744", dash=True)
        self.assertEqual(len(self.panel.markers), len(self.panel.curve_plots) + 1)
        self.panel.clear()
        self.assertEqual((self.panel.plot_data, self.panel.timeline_rgb, self.panel.markers),
                         ({}, {}, []))

    def test_frame_colours(self) -> None:
        self.assertEqual(frame_rgb(_result(detected=False)), plots.LOST_RGB)
        self.assertEqual(frame_rgb(_result(reliable=False)), plots.UNRELIABLE_RGB)
        self.assertEqual(frame_rgb(_result(stage="OBSERVING")), plots.STAGE_RGB["OBSERVING"])
        self.assertEqual(frame_rgb(_result(stage="??")), plots.STAGE_RGB["MONITORING"])


class TestViewPanel(unittest.TestCase):

    def test_everything_is_drawn_by_default(self) -> None:
        v = ViewPanel()
        for chk in (v.skeleton, v.trunk, v.centroid, v.support, v.curves):
            self.assertTrue(chk.isChecked(), chk.text())

    def test_time_format(self) -> None:
        self.assertEqual(format_time(0), "0:00")
        self.assertEqual(format_time(75.9), "1:15")
        self.assertEqual(format_time(-3), "0:00")


if __name__ == "__main__":
    unittest.main()
