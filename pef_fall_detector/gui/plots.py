"""PEF-Lab's curves and timeline: what each frame measured, drawn against time.

One object owns every plot, the timeline strip under the video, their data,
the cursor and the firing / verdict marks, because all of them share one
time axis and are cleared together. The window places :attr:`PlotPanel.tabs`
and :attr:`PlotPanel.timeline` in its layout and feeds the panel frames; it
never draws on a plot itself.

Looks only: the panel reads the configuration for the threshold lines it
draws, never to decide anything.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QScrollArea, QTabWidget, QWidget

from ..config import Config
from . import theme

#: Minimum plot height. The curves live in a scroll area, so this never
#: adds up into the window's own minimum height.
PLOT_MIN_HEIGHT = 125

#: Timeline colours, per frame.
STAGE_RGB = {
    "MONITORING": (84, 110, 122),
    "CONFIRMING": (255, 152, 0),
    "OBSERVING": (30, 136, 229),
    "COOLDOWN": (142, 36, 170),
}
LOST_RGB = (0, 0, 0)            # no detection
UNRELIABLE_RGB = (60, 60, 60)   # detected but barely visible
SEVERITY_PEN = {"severe": "#e53935", "moderate": "#fb8c00", "mild": "#43a047"}
#: Kept below this many frames in live mode: ~30 s at 30 FPS.
LIVE_FRAMES = 900


def frame_rgb(result) -> tuple[int, int, int]:
    """The colour of one frame on the timeline."""
    if not result.pose.detected:
        return LOST_RGB
    if not result.reliable:
        return UNRELIABLE_RGB
    return STAGE_RGB.get(result.stage, STAGE_RGB["MONITORING"])


def _plot(title: str, y0: float | None = None, y1: float | None = None):
    w = pg.PlotWidget()
    w.setTitle(title, size="9pt", color=theme.TEXT)
    w.setMinimumHeight(PLOT_MIN_HEIGHT)
    w.setMinimumWidth(200)
    w.showGrid(x=True, y=True, alpha=0.12)
    tick_font = QFont()
    tick_font.setPointSize(8)
    for side in ("left", "bottom"):
        w.getAxis(side).setStyle(tickFont=tick_font, tickTextOffset=3)
    w.getAxis("left").setWidth(34)
    if y0 is not None:
        w.setYRange(y0, y1)
    return w


def _hline(w, y, label, colour, dash=True, width=1, at=0.15):
    line = pg.InfiniteLine(
        pos=float(y), angle=0,
        pen=pg.mkPen(colour, width=width,
                     style=Qt.DashLine if dash else Qt.SolidLine),
        label=label or None,
        labelOpts={"position": at, "color": colour} if label else None)
    w.addItem(line)


def _grid_of(plots, cols=3):
    """A tab page: the plots in a grid, inside a scroll area.

    The scroll area is what keeps the window's minimum height small:
    without it every plot row adds to it (a seventh plot once pushed
    the buttons off the Mac's screen).
    """
    inner = QWidget()
    g = QGridLayout(inner)
    g.setContentsMargins(6, 6, 6, 6)
    g.setSpacing(6)
    for i, w in enumerate(plots):
        g.addWidget(w, i // cols, i % cols)
    page = QScrollArea()
    page.setWidgetResizable(True)
    page.setFrameShape(QFrame.NoFrame)
    page.setWidget(inner)
    return page


class PlotPanel:
    """The curves (two tabs) and the timeline, with their data and marks."""

    def __init__(self, cfg: Config) -> None:
        # Two tabs sharing the time axis, the cursor and the firing / verdict
        # marks:
        #   «Paper»: what decides the funnel (T, V, the score actually
        #            compared, P, I and phase 3's still share);
        #   «Experimental [EXP]»: what is logged outside §3.4 (A, H, R, legs,
        #            Vh, I's displacement), with the config lines each one is
        #            compared against.
        pg.setConfigOptions(antialias=False, background=theme.PLOT_BG,
                            foreground=theme.PLOT_FG)  # cheap drawing at 30 FPS
        score_formulation = str(cfg.stage1.trigger_formulation) == "score"
        #: FrameResult.quantities key -> the curve that draws it
        self.series: dict[str, object] = {}
        #: frame index -> {quantity key: value, "t": time}; keyed by index so
        #: scrubbing overwrites cleanly instead of appending duplicates.
        self.plot_data: dict[int, dict[str, float]] = {}
        #: Timeline: one colour per frame, and the event marks that have to
        #: be removed when the clip changes.
        self.timeline_rgb: dict[int, tuple[int, int, int]] = {}
        self.timeline_t: dict[int, float] = {}
        self.markers: list[tuple[object, object]] = []

        def curve(w, key, colour, dash=False):
            c = w.plot(pen=pg.mkPen(colour, width=2,
                                    style=Qt.DashLine if dash else Qt.SolidLine))
            self.series[key] = c
            return c

        # ---- Paper ---------------------------------------------------------
        # T up to 180: above 120 live the inverted skeletons, this project's
        # central failure mode. The lines are §3.4's bands and Stage 3's cuts,
        # labels staggered because 30, 45 and 60 sit a few pixels apart.
        self.t_plot = _plot("T — trunk angle (deg)", 0, 180)
        _hline(self.t_plot, 15, "", (90, 90, 90))
        _hline(self.t_plot, 90, "", (90, 90, 90))
        _hline(self.t_plot, float(cfg.stage3.upright_T_deg),
               f"upright < {float(cfg.stage3.upright_T_deg):.0f}", (120, 180, 120), at=0.12)
        # With the "score" formulation threshold_T does not fire on its own:
        # it is the score's scale. Drawn faint, and named for what it is.
        _hline(self.t_plot, float(cfg.stage1.threshold_T_deg),
               "T scale" if score_formulation else "threshold_T",
               (130, 130, 130) if score_formulation else "orange", at=0.45)
        _hline(self.t_plot, float(cfg.state_display.lying_T_deg),
               f"floor ≥ {float(cfg.state_display.lying_T_deg):.0f}", (229, 57, 53),
               at=0.78)
        self.t_curve = curve(self.t_plot, "T_deg", "y")

        self.v_plot = _plot("V — vertical velocity (torso/s, neg = down)", -4, 4)
        _hline(self.v_plot, float(cfg.stage1.threshold_V),
               "V scale" if score_formulation else "threshold_V",
               (110, 110, 110) if score_formulation else "orange")
        _hline(self.v_plot, 0.0, "", (120, 120, 120), dash=False)
        self.v_curve = curve(self.v_plot, "V_tps", "m")

        # What Stage 1 actually compares: T/T_scale + V/V_scale over the peak
        # window. Empty where the trigger does not evaluate it.
        self.s_plot = _plot("Trigger score", 0, 5)
        _hline(self.s_plot, float(cfg.stage1.trigger_score),
               f"fires ≥ {float(cfg.stage1.trigger_score):.1f}", (229, 57, 53),
               dash=False, width=2)
        self.s_curve = curve(self.s_plot, "trigger_score", (255, 255, 255))

        # Quantity P. Zero is the whole decision line: the sign says inside
        # or outside the support, so it is solid, not a configurable threshold.
        self.p_plot = _plot("P — COM vs support (torso, >0 = outside)", -1.5, 1.5)
        _hline(self.p_plot, 0.0, "outside", "orange", dash=False, width=2)
        self.p_curve = curve(self.p_plot, "P_offset", "c")

        # Quantity I. The dashed line is W. It saws back to zero on every
        # movement, which is the quantity working, not a glitch.
        self.i_plot = _plot("I — immobility (s)", 0,
                            max(2.0, float(cfg.stage3.threshold_W_seconds) * 1.5))
        _hline(self.i_plot, float(cfg.stage3.threshold_W_seconds), "W", "orange")
        self.i_curve = curve(self.i_plot, "I_still_s", "g")

        # Phase 3: the share of Stage 3 that stayed still. At the end of the
        # clip, above the line and neither lying nor standing -> severe.
        self.q_plot = _plot("Stage 3 still share (phase 3)", 0, 1)
        frac = float(cfg.stage3.persistent_immobility_fraction)
        if frac > 0.0:
            _hline(self.q_plot, frac, f"severe if ≥ {100 * frac:.0f}%", (229, 57, 53),
                   dash=False, at=0.2)
        self.q_curve = curve(self.q_plot, "still_fraction", (255, 213, 79))

        paper = _grid_of((self.t_plot, self.v_plot, self.s_plot,
                          self.p_plot, self.i_plot, self.q_plot))

        # ---- Experimental --------------------------------------------------
        ex = cfg.experimental
        sd = cfg.state_display
        # H: the H trigger uses trigger_H_ratio (also the "lying" cut when T
        # is not measured); recovery uses recovery_H_ratio.
        self.h_plot = _plot("H — height vs baseline [EXP]", 0, 2)
        for value, label, colour, at in (
                (ex.trigger_H_ratio, "collapse", (229, 57, 53), 0.15),
                (ex.recovery_H_ratio, "recovered", (67, 160, 71), 0.45),
                (ex.trigger_H_erect, "erect", (120, 180, 120), 0.75)):
            if float(value) > 0.0:
                _hline(self.h_plot, float(value), f"{label} {float(value):.2f}", colour, at=at)
        curve(self.h_plot, "H_ratio", (255, 112, 67))

        # Raw H and its baseline: B05's poisoned baseline shows here (a raw H
        # spike the baseline absorbs and drags for seconds). Auto axis,
        # because that spike reached 98.
        self.hraw_plot = _plot("Raw H (white) and baseline (cyan) [EXP]")
        curve(self.hraw_plot, "H_raw", (236, 239, 241))
        curve(self.hraw_plot, "H_baseline", (0, 229, 255), dash=True)

        self.ext_plot = _plot("Legs: hip-ankle / torso [EXP]", 0,
                              float(sd.max_extension_ratio) * 1.1)
        _hline(self.ext_plot, float(sd.crouch_extension),
               f"crouched < {float(sd.crouch_extension):.1f}", (255, 152, 0), at=0.15)
        _hline(self.ext_plot, float(sd.standing_extension),
               f"extended ≥ {float(sd.standing_extension):.1f}", (67, 160, 71), at=0.45)
        _hline(self.ext_plot, float(sd.max_extension_ratio),
               f"plausible max {float(sd.max_extension_ratio):.0f}", (130, 130, 130), at=0.75)
        curve(self.ext_plot, "extension_ratio", (171, 71, 188))

        # R: logged only; the config has no threshold for it, so none is drawn.
        self.r_plot = _plot("R — wrist to ankle (torsos) [EXP]", 0, 3)
        curve(self.r_plot, "reach_proximity", (129, 212, 250))

        self.vh_plot = _plot("Vh — horizontal velocity (torso/s) [EXP]", -2, 2)
        walk = float(sd.walking_vh_tps)
        _hline(self.vh_plot, walk, f"walking ±{walk:.2f}", (130, 130, 130), at=0.15)
        _hline(self.vh_plot, -walk, "", (130, 130, 130))
        curve(self.vh_plot, "Vh_tps", (255, 241, 118))

        # What I compares every frame: how far the subject moved from where
        # it started to stay still. Above ε, I drops back to zero.
        eps = float(cfg.stage3.epsilon)
        self.d_plot = _plot("I displacement (torsos) [EXP]", 0, max(0.2, 4 * eps))
        _hline(self.d_plot, eps, f"ε {eps:.2f}", "orange", at=0.15)
        curve(self.d_plot, "I_displacement", (165, 214, 167))

        # A (phase 8): 3D head height against gravity, 1 = standing. Cyan the
        # head, dashed purple the hip. The lines are the severity bands (8.2)
        # and the "reached the floor" cut (8.3).
        qa = cfg.quantity_a
        self.a_plot = _plot("A — 3D head height (standing = 1) [EXP]", -0.3, 1.3)
        for key, label, colour, at in (
                ("band_standing", "standing", (67, 160, 71), 0.15),
                ("floor_reached", "reached floor", (255, 167, 38), 0.62),
                ("band_floor", "on the floor", (229, 57, 53), 0.12)):
            value = float(getattr(qa, key))
            _hline(self.a_plot, value, f"{label} {value:.2f}", colour, at=at)
        curve(self.a_plot, "A_head", (0, 229, 255))
        curve(self.a_plot, "A_hip", (186, 104, 200), dash=True)

        # Four columns here: seven plots in two rows, like the Paper tab.
        experimental = _grid_of((self.a_plot, self.h_plot, self.hraw_plot,
                                 self.ext_plot, self.r_plot, self.vh_plot,
                                 self.d_plot), cols=4)

        self.tabs = QTabWidget()
        self.tabs.addTab(paper, "Paper (§3.4–3.5)")
        self.tabs.addTab(experimental, "Experimental [EXP]")

        # No linked X axes: pyqtgraph aligns linked axes by screen pixel, and
        # with hidden tabs that gave nonsense ranges. The time range is set
        # by hand on every redraw.
        self.curve_plots = (self.t_plot, self.v_plot, self.s_plot, self.p_plot,
                            self.i_plot, self.q_plot, self.a_plot, self.h_plot,
                            self.hraw_plot, self.ext_plot, self.r_plot, self.vh_plot,
                            self.d_plot)
        self.cursors = []
        for w in self.curve_plots:
            c = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=1))
            w.addItem(c)
            self.cursors.append(c)

        # Timeline: stage per frame (grey = monitoring, orange = Stage 2,
        # blue = Stage 3, purple = cooldown), black where detection was lost.
        # Marks: firing (dashed red) and verdict (severity colour). Detection
        # gaps were the key to A08-S3; here they show without guessing them
        # from breaks in the curves.
        self.timeline = pg.PlotWidget()
        self.timeline.setFixedHeight(34)
        self.timeline.hideAxis("left")
        self.timeline.hideAxis("bottom")
        self.timeline.setMouseEnabled(x=False, y=False)
        self.timeline.setYRange(0, 1, padding=0)
        self.timeline.setToolTip("Stage per frame — grey: monitoring · orange: Stage 2 · "
                                 "blue: Stage 3 · purple: cooldown · black: no detection")
        # NOT linked to the curves: pyqtgraph aligns linked axes by screen
        # pixel, and a full-width strip linked to a one-third-width curve
        # showed 0-34 s for a 10 s clip. The range is set by hand.
        self.timeline_img = pg.ImageItem()
        self.timeline.addItem(self.timeline_img)
        self.timeline_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=2))
        self.timeline.addItem(self.timeline_cursor)
        self.cursors.append(self.timeline_cursor)

    # ------------------------------------------------------------ feeding --
    def update(self, index: int, timestamp: float, result, curves: bool,
               live: bool) -> None:
        """Store this frame and redraw the curves and the timeline.

        Data is keyed by frame index so scrubbing back over already-seen
        frames overwrites instead of duplicating; the curve is drawn in
        time order regardless of the order frames were visited. In live
        mode only the most recent window is kept so memory stays bounded.
        ``curves`` False (the View tab's switch) keeps only the timeline
        colour, and draws nothing.
        """
        self.timeline_rgb[index] = frame_rgb(result)
        self.timeline_t[index] = timestamp
        if not curves:
            return
        self.store(index, timestamp, result)
        if live and len(self.plot_data) > LIVE_FRAMES:
            for old_key in sorted(self.plot_data)[: len(self.plot_data) - LIVE_FRAMES]:
                del self.plot_data[old_key]
                self.timeline_rgb.pop(old_key, None)
                self.timeline_t.pop(old_key, None)
        self.redraw_curves()
        self.redraw_timeline()
        self.move_cursors(timestamp)

    def store(self, index: int, timestamp: float, result) -> None:
        q = result.quantities
        nan = float("nan")
        point = {key: q.get(key, nan) for key in self.series}
        point["t"] = timestamp
        self.plot_data[index] = point

    def show_analysis(self, analysis) -> None:
        """Every frame of a frozen pass at once, with its firing and verdict marks."""
        self.clear()
        marked: set[int] = set()
        for index in sorted(analysis.frames):
            res = analysis.frames[index]
            self.store(index, res.pose.timestamp, res)
            self.timeline_rgb[index] = frame_rgb(res)
            self.timeline_t[index] = res.pose.timestamp
            if res.event is not None:
                self.mark(res.event.timestamp, "#ff1744", dash=True)
            if res.resolved_event is not None:
                self.mark_verdict(res.pose.timestamp, res.resolved_event)
                marked.add(id(res.resolved_event))
        # In labelling mode almost everything is decided in finalise(), AFTER
        # the last frame: those verdicts hang from no frame and are marked
        # at the end of the clip, which is where the pass saw them land.
        if analysis.frames:
            t_end = analysis.frames[max(analysis.frames)].pose.timestamp
            for ev in analysis.events:
                if id(ev) not in marked and ev.verdict != "stage1_only":
                    self.mark_verdict(t_end, ev)
        self.redraw_curves()
        self.redraw_timeline()

    # ------------------------------------------------------------ drawing --
    def redraw_curves(self) -> None:
        if not self.plot_data:
            return
        keys = sorted(self.plot_data)
        times = np.array([self.plot_data[k]["t"] for k in keys], dtype=float)
        # pyqtgraph skips NaN gaps with connect="finite": the curve breaks
        # where there was no detection instead of drawing a fake bridge. For
        # P the break is doubly meaningful — it marks the stretches where the
        # feet were not visible and the quantity genuinely does not exist.
        for key, c in self.series.items():
            ys = np.array([self.plot_data[k].get(key, float("nan")) for k in keys],
                          dtype=float)
            c.setData(times, ys, connect="finite")
        # The time axis is set by hand on all of them: with linked axes the
        # X auto-range was off and the view stayed at -0.5..0.5, with the
        # curves out of frame (seen in the A17-S2 capture).
        span = max(times[-1] - times[0], 1.0 / 30.0)
        for w in self.curve_plots:
            w.setXRange(times[0], times[0] + span, padding=0.01)

    def redraw_timeline(self) -> None:
        if not self.timeline_rgb:
            self.timeline_img.clear()
            return
        keys = sorted(self.timeline_rgb)
        rgb = np.array([self.timeline_rgb[k] for k in keys], dtype=np.uint8)
        self.timeline_img.setImage(rgb.reshape(len(keys), 1, 3), levels=(0, 255))
        t0, t1 = self.timeline_t[keys[0]], self.timeline_t[keys[-1]]
        dt = (t1 - t0) / max(1, len(keys) - 1) if len(keys) > 1 else 1.0 / 30.0
        self.timeline_img.setRect(QRectF(t0, 0.0, (t1 - t0) + dt, 1.0))
        self.timeline.setXRange(t0, t1 + dt, padding=0.01)

    def move_cursors(self, timestamp: float) -> None:
        for c in self.cursors:
            c.setPos(timestamp)

    def clear(self) -> None:
        self.plot_data.clear()
        self.timeline_rgb.clear()
        self.timeline_t.clear()
        for curve in self.series.values():
            curve.setData([], [])
        self.timeline_img.clear()
        for w, item in self.markers:
            w.removeItem(item)
        self.markers.clear()

    def mark(self, t: float, colour, dash: bool, width: int = 2) -> None:
        """One vertical mark on every curve and on the timeline."""
        for w in self.curve_plots + (self.timeline,):
            item = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(
                colour, width=width, style=Qt.DashLine if dash else Qt.SolidLine))
            w.addItem(item)
            self.markers.append((w, item))

    def mark_verdict(self, t: float, ev) -> None:
        self.mark(t, SEVERITY_PEN.get(ev.severity, "#9e9e9e"), dash=False, width=3)
