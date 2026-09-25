"""PEF-Lab: análisis congelado, sin CSV al navegar, re-analizar, velocidades.

La ventana real, con un video real (sintético, en un directorio temporal) y el
pipeline real; lo único sustituido es MediaPipe, por poses guionadas, para que
la prueba sea rápida y determinista. Se cuenta cuántas veces se llama a
``process``: en modo congelado tiene que ser cero.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("PySide6 no esta instalado; PEF-Lab no se prueba aqui")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.gui import lab_window  # noqa: E402
from pef_fall_detector.gui.lab_window import SPEEDS, LabWindow  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from tests.test_main_headless import _toppling_pose  # noqa: E402
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402

_app = QApplication.instance() or QApplication([])
FPS = 30.0
N = 120                                   # 4 s de clip


def _script():
    """De pie 1 s, cae en 0.67 s, queda en el suelo hasta el final."""
    frames = [make_pose(i, i / FPS) for i in range(30)]
    frames += [_toppling_pose(i, i / FPS, angle_deg=(i - 30) * 4.5,
                              hip_y=500.0 + (i - 30) * 12.0) for i in range(30, 50)]
    frames += [_toppling_pose(i, i / FPS, angle_deg=90.0, hip_y=740.0)
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
        self.fail("el clip no termino")

    def records(self, kind: str = "frames") -> list[Path]:
        files = sorted(self.out.glob("*.csv")) if self.out.exists() else []
        if kind == "events":
            return [f for f in files if "-events-" in f.name]
        return [f for f in files if "-events-" not in f.name
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

    def test_moving_through_it_recomputes_nothing(self) -> None:
        self.analyse(self.a)
        before_calls = self.calls[0]
        before_curves = dict(self.win._plot_data)
        for target in (10, 45, 0, N - 1, 60):
            self.win._seek_to(target)
        self.win._step(+1)
        self.win._step(-1)
        self.win._step_seconds(-1.0)
        self.assertEqual(self.calls[0], before_calls, "se recalculo en modo congelado")
        self.assertEqual(self.win._plot_data, before_curves, "las curvas cambiaron")
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
        self.win._on_play_pause()                      # al final: vuelve a empezar
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
        y0, y1 = self.win.t_plot.getViewBox().viewRange()[1]
        self.assertLessEqual(y0, 0.0)
        self.assertGreaterEqual(y1, 180.0)

    def test_the_score_curve_is_what_the_trigger_compared(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["stage1"]["trigger_formulation"] = "score"
        self.win.close()
        self.win = LabWindow(Config(cfg))
        self.analyse(self.a)
        _, ys = self.win.s_curve.getData()
        finite = ys[np.isfinite(ys)]
        self.assertTrue(len(finite), "la curva del puntaje quedo vacia")
        self.assertGreaterEqual(finite.max(), float(cfg["stage1"]["trigger_score"]))

    def test_the_timeline_has_one_colour_per_frame(self) -> None:
        self.analyse(self.a)
        self.assertEqual(len(self.win._timeline_rgb), N)
        self.assertIsNotNone(self.win.timeline_img.image)


class TestNavigationDoesNotRecord(_Base):

    def test_reopening_in_the_same_session_is_frozen_and_writes_nothing(self) -> None:
        self.analyse(self.a)
        self.win.load_video(str(self.b))               # otro clip: pasada real
        self.drive()
        calls, records = self.calls[0], len(self.records())
        self.win.load_video(str(self.a))               # volver al primero
        self.assertIsNotNone(self.win._frozen)
        self.assertFalse(self.win.playing, "congelado abre en pausa")
        self.assertIn("Frozen", self.win.status_label.text())
        self.assertEqual(self.calls[0], calls)
        self.assertEqual(len(self.records()), records)

    def test_a_different_configuration_is_reanalysed(self) -> None:
        self.analyse(self.a)
        self.win._cfg_key = "otra-configuracion"
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
        self.assertEqual(len(self.records()), 2, "la segunda pasada piso a la primera")

    def test_a_pass_with_a_jump_is_not_frozen(self) -> None:
        self.win.load_video(str(self.a))
        self.win.timer.stop()
        for _ in range(20):
            self.win._process_next_frame()
        self.win._seek_to(70)                          # salto a mitad de la pasada
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
            self.win._process_next_frame()        # cuadros 0..19 registrados
        self.win._seek_to(19)                     # re-procesa 19 sin registrar;
        self.win.playing = True                   # el siguiente, 20, es contiguo
        self.drive()
        self.assertIsNone(self.win._cache.get(self.a, self.win._cfg_key))


class TestQueue(_Base):

    def _run_queue(self) -> None:
        w = self.win
        w._queue = [self.a, self.b]
        w.queue_table.setRowCount(2)
        w._on_queue_start()
        for _ in range(4 * N):
            if not w._queue_running:
                return
            if w.playing:
                w._process_next_frame()
            w.timer.stop()
        self.fail("la cola no termino")

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
        text = self.win.queue_score.text()
        self.assertIn("4-class 2/2", text)
        self.assertIn("exact falls 2/2", text)
        self.assertIn("sens 100%", text)

    def test_the_reason_is_on_the_label_row_and_the_tooltip(self) -> None:
        self._run_queue()
        self.assertIn("T final", self.win._queue_rows[0]["motivo"])
        self.assertIn("T final", self.win.queue_table.item(0, 2).toolTip())


class TestSpeed(_Base):

    def test_every_speed_paces_the_loop(self) -> None:
        self.analyse(self.a)
        slots = {}
        started = []
        self.win.timer.start = lambda ms: started.append(ms)   # captura el ritmo
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
    """Las medidas experimentales [EXP] se ven, en su pestaña, con el tiempo del clip."""

    PAPER = {"T_deg", "V_tps", "P", "I"}
    EXP = ("H_ratio", "H_raw", "H_baseline", "extension_ratio",
           "reach_proximity", "Vh_tps", "I_displacement")

    def test_two_tabs_paper_and_experimental(self) -> None:
        tabs = self.win.curve_tabs
        self.assertEqual(tabs.count(), 2)
        self.assertIn("Paper", tabs.tabText(0))
        self.assertIn("EXP", tabs.tabText(1))

    def test_every_experimental_quantity_has_a_curve(self) -> None:
        for key in self.EXP + ("still_fraction",):
            self.assertIn(key, self.win._series, key)

    def test_the_curves_carry_what_the_pipeline_measured(self) -> None:
        self.analyse(self.a)
        fed = {}
        for key in self.EXP + ("still_fraction",):
            _, ys = self.win._series[key].getData()
            fed[key] = int(np.isfinite(np.asarray(ys, dtype=float)).sum())
        # Con las poses guionadas estas existen en todos los cuadros (H no:
        # las poses sintéticas no dan la base de altura, queda NaN).
        for key in ("extension_ratio", "reach_proximity", "Vh_tps", "I_displacement"):
            self.assertGreater(fed[key], 0, f"{key} quedo vacia")
        # La fracción quieta solo existe mientras la Etapa 3 observa.
        self.assertGreater(fed["still_fraction"], 0)
        self.assertLess(fed["still_fraction"], N)

    def test_experimental_plots_share_the_clip_time_axis(self) -> None:
        self.analyse(self.a)
        x_paper = self.win.t_plot.getViewBox().viewRange()[0]
        for w in (self.win.h_plot, self.win.hraw_plot, self.win.ext_plot,
                  self.win.r_plot, self.win.vh_plot, self.win.d_plot):
            x = w.getViewBox().viewRange()[0]
            self.assertAlmostEqual(x[0], x_paper[0], places=3)
            self.assertAlmostEqual(x[1], x_paper[1], places=3)

    def test_frozen_review_keeps_the_experimental_curves(self) -> None:
        self.analyse(self.a)
        before = self.win._series["extension_ratio"].getData()[1].copy()
        self.win._seek_to(30)
        self.win._step(+1)
        after = self.win._series["extension_ratio"].getData()[1]
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
