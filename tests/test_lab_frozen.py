"""PEF-Lab: frozen analysis, no CSV when browsing, re-analysing, speeds.

The real window, with a real video (synthetic, in a temporary folder) and
the real pipeline; the only thing replaced is MediaPipe, by scripted poses,
so the test is fast and deterministic. The calls to ``process`` are counted:
in frozen mode there must be none.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("PySide6 is not installed; PEF-Lab is not tested here") from None

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.gui import lab_window, plots  # noqa: E402
from pef_fall_detector.gui.lab_window import SPEEDS, LabWindow  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.sources import VideoFileSource  # noqa: E402
from tests.support import TEST_CFG, make_pose, toppling_pose  # noqa: E402

_app = QApplication.instance() or QApplication([])
FPS = 30.0
N = 120                                   # a 4 s clip


def _script():
    """Standing 1 s, falls in 0.67 s, stays on the floor until the end."""
    frames = [make_pose(i, i / FPS) for i in range(30)]
    frames += [toppling_pose(i, i / FPS, angle_deg=(i - 30) * 4.5,
                              hip_y=500.0 + (i - 30) * 12.0) for i in range(30, 50)]
    frames += [toppling_pose(i, i / FPS, angle_deg=90.0, hip_y=740.0)
               for i in range(50, N)]
    return frames


def _write_video(path: Path, n: int = N) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for _ in range(n):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()


class _Base(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.out = root / "logs"
        self.clips = root / "clips"
        self.clips.mkdir()
        self.a = self.clips / "A01-S1-NotRecovered.mp4"
        self.b = self.clips / "A01-S2-NotRecovered.mp4"
        _write_video(self.a)
        _write_video(self.b)
        script = _script()
        self.calls = [0]
        calls = self.calls

        class _Scripted(FramePipeline):
            def process(self, frame_bgr, frame_index, timestamp):
                calls[0] += 1
                pf = script[min(frame_index, len(script) - 1)]
                return self.analyze(pf)

        self._saved = lab_window.FramePipeline
        lab_window.FramePipeline = _Scripted
        cfg = copy.deepcopy(TEST_CFG)
        cfg["logging"]["output_dir"] = str(self.out)
        self.cfg = cfg
        self.win = LabWindow(Config(cfg))

    def tearDown(self) -> None:
        self.win.timer.stop()
        self.win._close_source()
        self.win.close()
        lab_window.FramePipeline = self._saved
        self.tmp.cleanup()

    # -- helpers -------------------------------------------------------------
    def drive(self, limit: int = 3 * N) -> None:
        """Run the self-scheduling loop by hand until the clip stops playing."""
        self.win.timer.stop()
        for _ in range(limit):
            if not self.win.playing:
                return
            self.win._process_next_frame()
            self.win.timer.stop()
        self.fail("the clip did not end")

    def drive_queue(self, start, check_started: bool = False) -> None:
        """Start the queue and run it to the end, by hand.

        The pass summary it prints goes to ``self.printed``, not the console.
        """
        w = self.win
        with contextlib.redirect_stdout(io.StringIO()) as out:
            start()
            if check_started:
                self.assertTrue(w._queue_running, "the Start button started nothing")
            for _ in range(4 * N):
                if not w._queue_running:
                    break
                if w.playing:
                    w._process_next_frame()
                w.timer.stop()
        self.printed = out.getvalue()
        if w._queue_running:
            self.fail("the queue did not finish")

    def records(self, kind: str = "frames") -> list[Path]:
        files = sorted(self.out.glob("*.csv")) if self.out.exists() else []
        if kind == "events":
            return [f for f in files if "-events-" in f.name]
        if kind == "landmarks":
            return [f for f in files if "-landmarks-" in f.name]
        return [f for f in files if "-events-" not in f.name and "-landmarks-" not in f.name
                and not f.name.startswith("dataset-labels")]

    def analyse(self, path: Path, **kw) -> None:
        self.win.load_video(str(path), **kw)
        self.drive()


class TestFrozenAfterAPass(_Base):

    def test_a_finished_pass_is_frozen_with_one_record(self) -> None:
        self.analyse(self.a)
        self.assertIsNotNone(self.win._frozen)
        self.assertEqual(self.win._frozen.frame_count, N)
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(len(self.records("events")), 1)
        self.assertEqual(self.calls[0], N)

    def test_a_pass_records_landmarks_and_every_event(self) -> None:
        # PEF-Lab always records (research), through the same writer as
        # main.py: the 33 landmarks too, and every event of the pass.
        self.analyse(self.a)
        self.assertEqual(len(self.records("landmarks")), 1)
        events = self.win._frozen.events
        files = self.records("events")
        rows = 0
        if files:
            with open(files[0], newline="", encoding="utf-8") as fh:
                rows = sum(1 for _ in csv.DictReader(fh))
        self.assertEqual(rows, len(events))

    def test_closing_writes_the_events_no_verdict_reached(self) -> None:
        # An event a detection loss abandoned never reaches _log_event; the
        # record still has to carry it, in its place (PEF-Lab used to drop it).
        from pef_fall_detector.records import RunRecords
        from pef_fall_detector.state_machine import TriggerEvent
        events = [TriggerEvent(frame_index=i, timestamp=i / 10, t_deg=50.0, v_tps=-2.0,
                               formulation="score", trigger_source="score")
                  for i in range(2)]
        self.win.records = RunRecords(self.out, "abandoned")
        self.win._run_events = events
        self.win._finalise_records("test")
        files = sorted(self.out.glob("abandoned-events-*.csv"))
        with open(files[0], newline="", encoding="utf-8") as fh:
            self.assertEqual([r["event_index"] for r in csv.DictReader(fh)], ["0", "1"])

    def test_pef_lab_records_even_when_the_config_says_deployment(self) -> None:
        # logging.mode governs main.py (the device); the laboratory records.
        cfg = copy.deepcopy(self.cfg)
        cfg["logging"]["mode"] = "deployment"
        self.win.close()
        self.win = LabWindow(Config(cfg))
        self.analyse(self.a)
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(len(self.records("landmarks")), 1)

    def test_moving_through_it_recomputes_nothing(self) -> None:
        self.analyse(self.a)
        before_calls = self.calls[0]
        before_curves = dict(self.win.plots.plot_data)
        for target in (10, 45, 0, N - 1, 60):
            self.win._seek_to(target)
        self.win._step(+1)
        self.win._step(-1)
        self.win._step_seconds(-1.0)
        self.assertEqual(self.calls[0], before_calls, "it recomputed in frozen mode")
        self.assertEqual(self.win.plots.plot_data, before_curves, "the curves changed")
        self.assertTrue(self.win.stage_label.text().startswith("[FROZEN]"))
        self.assertEqual(len(self.records()), 1)

    def test_the_stage_shown_is_the_one_the_pass_saw_on_that_frame(self) -> None:
        self.analyse(self.a)
        seen = self.win._frozen.frames[55].stage
        self.win._seek_to(55)
        self.assertIn(f"stage on this frame: {seen}", self.win.stage_label.text())

    def test_replaying_it_recomputes_nothing_and_writes_nothing(self) -> None:
        self.analyse(self.a)
        before = self.calls[0]
        self.win._on_play_pause()                      # at the end: starts over
        self.assertTrue(self.win.playing)
        self.drive()
        self.assertEqual(self.calls[0], before)
        self.assertEqual(len(self.records()), 1)

    def test_the_verdict_panel_carries_the_reason(self) -> None:
        self.analyse(self.a)
        text = self.win.verdict_label.text()
        self.assertIn("stage3_confirmed", text)
        self.assertIn("T final", text)

    def test_t_axis_reaches_180_where_inverted_skeletons_live(self) -> None:
        y0, y1 = self.win.plots.t_plot.getViewBox().viewRange()[1]
        self.assertLessEqual(y0, 0.0)
        self.assertGreaterEqual(y1, 180.0)

    def test_the_score_curve_is_what_the_trigger_compared(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["stage1"]["trigger_formulation"] = "score"
        self.win.close()
        self.win = LabWindow(Config(cfg))
        self.analyse(self.a)
        _, ys = self.win.plots.s_curve.getData()
        finite = ys[np.isfinite(ys)]
        self.assertTrue(len(finite), "the score curve stayed empty")
        self.assertGreaterEqual(finite.max(), float(cfg["stage1"]["trigger_score"]))

    def test_the_timeline_has_one_colour_per_frame(self) -> None:
        self.analyse(self.a)
        self.assertEqual(len(self.win.plots.timeline_rgb), N)
        self.assertIsNotNone(self.win.plots.timeline_img.image)


class TestNavigationDoesNotRecord(_Base):

    def test_reopening_in_the_same_session_is_frozen_and_writes_nothing(self) -> None:
        self.analyse(self.a)
        self.win.load_video(str(self.b))               # another clip: a real pass
        self.drive()
        calls, records = self.calls[0], len(self.records())
        self.win.load_video(str(self.a))               # back to the first one
        self.assertIsNotNone(self.win._frozen)
        self.assertFalse(self.win.playing, "frozen opens paused")
        self.assertIn("Frozen", self.win.status_label.text())
        self.assertEqual(self.calls[0], calls)
        self.assertEqual(len(self.records()), records)

    def test_a_different_configuration_is_reanalysed(self) -> None:
        self.analyse(self.a)
        self.win._cfg_key = "another-configuration"
        time.sleep(1.05)
        self.win.load_video(str(self.a))
        self.assertIsNone(self.win._frozen)
        self.assertTrue(self.win.playing)
        self.drive()
        self.assertEqual(len(self.records()), 2)

    def test_reanalyse_from_zero_writes_a_new_record(self) -> None:
        self.analyse(self.a)
        self.win._on_reanalyze()
        self.assertIsNone(self.win._frozen)
        self.drive()
        self.assertEqual(self.calls[0], 2 * N)
        self.assertEqual(len(self.records()), 2, "the second pass overwrote the first")

    def test_a_pass_with_a_jump_is_not_frozen(self) -> None:
        self.win.load_video(str(self.a))
        self.win.timer.stop()
        for _ in range(20):
            self.win._process_next_frame()
        self.win._seek_to(70)                          # a jump mid-pass
        self.win.playing = True
        self.drive()
        self.assertIsNone(self.win._cache.get(self.a, self.win._cfg_key))
        self.assertIn("Not frozen", self.win.status_label.text())

    def test_a_jump_that_lands_on_the_next_frame_is_still_a_jump(self) -> None:
        # Indices stay contiguous, but the pipeline was reset: the frames after
        # it did not come from one continuous observation.
        self.win.load_video(str(self.a))
        self.win.timer.stop()
        for _ in range(20):
            self.win._process_next_frame()        # frames 0..19 recorded
        self.win._seek_to(19)                     # re-processes 19 without recording;
        self.win.playing = True                   # the next one, 20, is contiguous
        self.drive()
        self.assertIsNone(self.win._cache.get(self.a, self.win._cfg_key))


class TestQueue(_Base):

    def _run_queue(self) -> None:
        w = self.win
        w._queue = [self.a, self.b]
        w.queue_panel.table.setRowCount(2)
        self.drive_queue(w._on_queue_start)

    def test_after_the_queue_every_clip_is_frozen_and_browsing_records_nothing(self) -> None:
        self._run_queue()
        records, calls = len(self.records()), self.calls[0]
        self.assertEqual(records, 2)
        for row in (0, 1, 0):
            self.win._open_for_review(row)
            self.assertIsNotNone(self.win._frozen)
        self.win._review_step(+1)
        self.assertEqual(self.calls[0], calls)
        self.assertEqual(len(self.records()), records)

    def test_starting_the_queue_again_reanalyses_everything(self) -> None:
        self._run_queue()
        time.sleep(1.05)
        self._run_queue()
        self.assertEqual(len(self.records()), 4)

    def test_the_queue_shows_the_37_metrics(self) -> None:
        self._run_queue()
        text = self.win.queue_panel.score.text()
        self.assertIn("4-class 2/2", text)
        self.assertIn("exact falls 2/2", text)
        self.assertIn("sens 100%", text)

    def test_the_summary_splits_train_and_test(self) -> None:
        # A01 is a train actor in config/partition.yaml; without the
        # partition the summary would silently lose this section.
        self._run_queue()
        self.assertIn("CALIBRATION  (2 clips)", self.printed)

    def test_the_reason_is_on_the_label_row_and_the_tooltip(self) -> None:
        self._run_queue()
        self.assertIn("T final", self.win._queue_rows[0]["reason"])
        self.assertIn("T final", self.win.queue_panel.table.item(0, 2).toolTip())


class TestSpeed(_Base):

    def test_every_speed_paces_the_loop(self) -> None:
        self.analyse(self.a)
        slots = {}
        started = []
        self.win.timer.start = lambda ms: started.append(ms)   # captures the pace
        for i, sp in enumerate(SPEEDS):
            self.win.speed_box.setCurrentIndex(i)
            self.assertEqual(self.win._speed, sp)
            self.win.source.seek(0)
            self.win.playing = True
            started.clear()
            self.win._on_tick()
            slots[sp] = started[-1]
        self.assertGreater(slots[0.25], slots[1.0])
        self.assertGreater(slots[1.0], slots[2.0])
        self.assertAlmostEqual(slots[0.25], 1000.0 / (FPS * 0.25), delta=20)


class TestExperimentalTab(_Base):
    """The experimental [EXP] measures show, in their tab, on the clip's time."""

    PAPER = {"T_deg", "V_tps", "P", "I"}
    EXP = ("H_ratio", "H_raw", "H_baseline", "extension_ratio",
           "reach_proximity", "Vh_tps", "I_displacement")

    def test_two_tabs_paper_and_experimental(self) -> None:
        tabs = self.win.plots.tabs
        self.assertEqual(tabs.count(), 2)
        self.assertIn("Paper", tabs.tabText(0))
        self.assertIn("EXP", tabs.tabText(1))

    def test_every_experimental_quantity_has_a_curve(self) -> None:
        for key in self.EXP + ("still_fraction",):
            self.assertIn(key, self.win.plots.series, key)

    def test_the_curves_carry_what_the_pipeline_measured(self) -> None:
        self.analyse(self.a)
        fed = {}
        for key in self.EXP + ("still_fraction",):
            _, ys = self.win.plots.series[key].getData()
            fed[key] = int(np.isfinite(np.asarray(ys, dtype=float)).sum())
        # With the scripted poses these exist on every frame (H does not: the
        # synthetic poses give no height baseline, so it stays NaN).
        for key in ("extension_ratio", "reach_proximity", "Vh_tps", "I_displacement"):
            self.assertGreater(fed[key], 0, f"{key} stayed empty")
        # The still share only exists while Stage 3 observes.
        self.assertGreater(fed["still_fraction"], 0)
        self.assertLess(fed["still_fraction"], N)

    def test_experimental_plots_share_the_clip_time_axis(self) -> None:
        self.analyse(self.a)
        x_paper = self.win.plots.t_plot.getViewBox().viewRange()[0]
        for w in (self.win.plots.h_plot, self.win.plots.hraw_plot, self.win.plots.ext_plot,
                  self.win.plots.r_plot, self.win.plots.vh_plot, self.win.plots.d_plot):
            x = w.getViewBox().viewRange()[0]
            self.assertAlmostEqual(x[0], x_paper[0], places=3)
            self.assertAlmostEqual(x[1], x_paper[1], places=3)

    def test_frozen_review_keeps_the_experimental_curves(self) -> None:
        self.analyse(self.a)
        before = self.win.plots.series["extension_ratio"].getData()[1].copy()
        self.win._seek_to(30)
        self.win._step(+1)
        after = self.win.plots.series["extension_ratio"].getData()[1]
        np.testing.assert_array_equal(before, after)


class TestWindowWiring(_Base):
    """B6: the window feeds the panels it no longer builds (plots, queue, View)."""

    def _run_queue(self) -> None:
        w = self.win
        w._queue = [self.a, self.b]
        w.queue_panel.add_clips([self.a, self.b])
        w._update_controls()                    # what adding a folder does
        self.drive_queue(w.queue_panel.start_btn.click, check_started=True)

    def test_the_curves_switch_hides_them_and_stops_drawing(self) -> None:
        self.win.view_panel.curves.setChecked(False)
        self.assertTrue(self.win.curves_panel.isHidden())
        self.assertTrue(self.win.plots.timeline.isHidden())
        self.win.load_video(str(self.a))
        for _ in range(10):
            self.win._process_next_frame()
        self.win.timer.stop()
        self.assertEqual(self.win.plots.plot_data, {})
        self.assertEqual(len(self.win.plots.timeline_rgb), 10)

    def test_a_live_source_keeps_a_bounded_window(self) -> None:
        clip = str(self.a)

        class _Camera(VideoFileSource):
            is_live = True

            def __init__(self, _index):
                super().__init__(clip)

        with mock.patch.object(lab_window, "CameraSource", _Camera), \
                mock.patch.object(plots, "LIVE_FRAMES", 10):
            self.win._on_toggle_camera()
            for _ in range(30):
                self.win._process_next_frame()
            self.win.timer.stop()
            self.assertEqual(len(self.win.plots.plot_data), 10)

    def test_a_recorded_clip_keeps_every_frame(self) -> None:
        with mock.patch.object(plots, "LIVE_FRAMES", 10):
            self.win.load_video(str(self.a))
            for _ in range(30):
                self.win._process_next_frame()
            self.win.timer.stop()
            self.assertEqual(len(self.win.plots.plot_data), 30)

    def test_the_frozen_review_marks_the_verdict_decided_at_the_end(self) -> None:
        # The verdict lands in finalise(), after the last frame: the frozen
        # curves must still carry its mark, at the end of the clip.
        self.analyse(self.a)
        self.assertIsNotNone(self.win._frozen)
        widths = [item.pen.width() for _w, item in self.win.plots.markers]
        self.assertIn(3, widths)

    def test_the_queue_table_follows_the_running_clip(self) -> None:
        self._run_queue()
        rows = [i.row() for i in self.win.queue_panel.table.selectionModel().selectedRows()]
        self.assertEqual(rows, [1])

    def test_save_reviews_writes_the_human_column(self) -> None:
        self._run_queue()
        self.win.queue_panel.table.cellWidget(0, 4).setCurrentText("Recovered")
        with contextlib.redirect_stdout(io.StringIO()):
            self.win.queue_panel.save_btn.click()
        saved = sorted(self.out.glob("dataset-labels-reviewed-*.csv"))
        self.assertEqual(len(saved), 1)
        with open(saved[0], newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["reviewed_class"] for r in rows], ["Recovered", ""])


if __name__ == "__main__":
    unittest.main()
