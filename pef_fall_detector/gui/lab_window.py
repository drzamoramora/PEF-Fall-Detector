"""PEF-Lab main window (Phase 1).

A visualization workbench over the source-agnostic core: open a recorded
clip (e.g. from PEF-Video-Tool's ``recordings/``) or a live camera, watch
the MediaPipe skeleton and Step-0 anchors drawn in real time, and get a
per-frame audit CSV written automatically.

The window is strictly a *viewer*: every computation happens in the core
modules (``pose_frontend``, later ``quantities``/``state_machine``), which
also power the headless CLI used for batch evaluation and, eventually, the
Raspberry Pi deployment. Later phases extend this window with time-series
plots (T(t), V(t)), live threshold editing, and per-overlay toggles.

Video-mode controls: Play/Pause, single-frame step (◀ / ▶), and a seek bar —
falls last 10–15 frames, so being able to stop on the exact frame where a
stage fired is the whole point of this tool.
"""

from __future__ import annotations

import html
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..alerts import Alert
from ..audit_log import EVENT_FIELDS, AuditLogger
from ..classification import CLASSES, classify_clip
from ..config import Config
from ..dataset import LABEL_FIELDS, format_confusion, truth_from_name, video_files
from ..evaluation import binary_metrics, load_split, report, split_rows
from ..overlay import draw_overlay
from ..pipeline import FramePipeline
from ..sources import CameraSource, FrameSource, VideoFileSource
from .review_cache import ClipAnalysis, PassRecorder, ReviewCache, config_key

#: Velocidades de reproducción. En la revisión congelada no corre MediaPipe,
#: así que 2x es alcanzable; en una pasada de análisis el límite lo pone el
#: procesamiento, y la velocidad solo puede frenarla.
SPEEDS = (0.25, 0.5, 0.75, 1.0, 2.0)

#: Colores de la línea de tiempo, por cuadro.
_STAGE_RGB = {
    "MONITORING": (84, 110, 122),
    "CONFIRMING": (255, 152, 0),
    "OBSERVING": (30, 136, 229),
    "COOLDOWN": (142, 36, 170),
}
_LOST_RGB = (0, 0, 0)            # sin detección
_UNRELIABLE_RGB = (60, 60, 60)   # detectado pero poco visible
_SEVERITY_PEN = {"severe": "#e53935", "moderate": "#fb8c00", "mild": "#43a047"}


def _frame_rgb(result) -> tuple[int, int, int]:
    """El color de un cuadro en la línea de tiempo."""
    if not result.pose.detected:
        return _LOST_RGB
    if not result.reliable:
        return _UNRELIABLE_RGB
    return _STAGE_RGB.get(result.stage, _STAGE_RGB["MONITORING"])


def _p_side(p_offset: float | None) -> bool | None:
    """Whether the COM projects outside the support — or None if unknown.

    Kept as three states rather than a boolean: a NaN P (feet occluded) must
    not be drawn as "inside", which is what any ``> 0`` test on NaN would
    silently produce.
    """
    if p_offset is None or math.isnan(p_offset):
        return None
    return p_offset > 0.0


def _format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


class LabWindow(QWidget):
    """PEF-Lab: source selection + overlay display + audit logging."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.setWindowTitle("PEF-Lab — Fall Detector Workbench (Phase 1)")
        self.resize(1000, 760)

        self.cfg = cfg
        self.source: FrameSource | None = None
        self.pipeline: FramePipeline | None = None
        self.logger: AuditLogger | None = None
        self.event_logger: AuditLogger | None = None
        self.playing = False
        self.frame_index = 0
        self._user_scrubbing = False
        self._last_raw_frame = None
        self._last_perf_shown = 0.0
        # Per-clip counters, reset by _start_session; the label row reports
        # what fraction of the clip the pose front-end could actually use,
        # which is the first thing to look at when a label disagrees.
        self._last_timestamp: float | None = None
        self._logged_frames = 0
        self._reliable_frames = 0
        # Dataset pass
        self._queue: list[Path] = []
        #: Clip abierto para revisión, o None si todavía ninguno.
        #: Es el ancla de las flechas Alt+← / Alt+→; se mantiene
        #: aparte de ``_queue_index`` porque revisar y correr la
        #: pasada son dos recorridos distintos sobre la misma cola.
        self._review_index: int | None = None
        self._queue_rows: list[dict] = []
        self._queue_index = 0
        self._queue_running = False
        self._label_logger: AuditLogger | None = None
        # Plot data keyed by frame index so scrubbing overwrites cleanly
        # instead of appending duplicates:
        # index -> (t, T_deg, V_tps, P_offset, I_still_s, trigger_score).
        self._plot_data: dict[int, tuple[float, ...]] = {}
        # Análisis congelado (ver review_cache.py): lo que una pasada completa
        # calculó, para revisar sin recalcular ni escribir registros.
        self._cache = ReviewCache()
        self._cfg_key = config_key(cfg.as_dict())
        self._recorder: PassRecorder | None = None
        self._frozen: ClipAnalysis | None = None
        self._current_path: str | None = None
        self._speed = 1.0
        # Línea de tiempo: un color por cuadro, y las marcas de eventos que
        # hay que quitar al cambiar de clip.
        self._timeline_rgb: dict[int, tuple[int, int, int]] = {}
        self._timeline_t: dict[int, float] = {}
        self._markers: list[tuple[object, object]] = []

        self._build_ui()
        self._update_controls()

        # Single-shot, self-scheduling loop: after each frame is processed the
        # next one is scheduled for whatever time is left of its slot. A
        # repeating fixed-interval timer cannot do this — when processing is
        # slower than the interval (measured: 80 ms achieved against a 33 ms
        # timer on the author's machine) the events queue up, frames pile in
        # the capture buffer and latency grows without bound. Self-scheduling
        # degrades gracefully instead: it simply runs as fast as the machine
        # allows, and never accumulates a backlog.
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._on_tick)

        # The banner clears itself so a stale alarm cannot be mistaken for a
        # live one after the subject has recovered.
        self.alert_timer = QTimer(self)
        self.alert_timer.setSingleShot(True)
        self.alert_timer.timeout.connect(
            lambda: self.alert_banner.setVisible(False))

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.video_label = QLabel("Open a video or start the camera to begin.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 540)
        self.video_label.setStyleSheet("background-color: #202020; color: #aaaaaa;")
        layout.addWidget(self.video_label, stretch=1)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Stage-1 readout: current funnel position and the events raised so
        # far. Kept beside the curves rather than buried in the HUD because
        # while calibrating, WHEN it fired matters more than the frame it
        # fired on.
        # The alarm. A confirmed fall must be impossible to miss while
        # testing with the camera — a line of grey text is not an alarm.
        # Hidden until one fires, so its presence alone carries meaning.
        self.alert_banner = QLabel("")
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setVisible(False)
        layout.addWidget(self.alert_banner)

        self.stage_label = QLabel("Stage: MONITORING   |   events: 0")
        self.stage_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.stage_label)

        # --- Curvas ------------------------------------------------------
        # Dos pestañas con el mismo eje de tiempo, el mismo cursor y las mismas
        # marcas de disparo y veredicto:
        #   «Paper»: lo que decide el embudo (T, V, el puntaje que de verdad se
        #            compara, P, I y el % quieto de la fase 3);
        #   «Experimentales [EXP]»: lo que se registra fuera del §3.4 (H, R,
        #            piernas, Vh, el desplazamiento de I), con las líneas de la
        #            configuración contra las que cada una se compara.
        # Debajo, a todo el ancho: el veredicto con su motivo y la línea de
        # tiempo de la pasada.
        pg.setConfigOptions(antialias=False)  # cheap drawing at 30 FPS
        score_formulation = str(self.cfg.stage1.trigger_formulation) == "score"
        #: clave de FrameResult.quantities -> curva que la dibuja
        self._series: dict[str, object] = {}

        def plot(title: str, y0: float | None = None, y1: float | None = None):
            w = pg.PlotWidget(title=title)
            w.setFixedHeight(130)
            if y0 is not None:
                w.setYRange(y0, y1)
            return w

        def hline(w, y, label, colour, dash=True, width=1, at=0.15):
            line = pg.InfiniteLine(
                pos=float(y), angle=0,
                pen=pg.mkPen(colour, width=width,
                             style=Qt.DashLine if dash else Qt.SolidLine),
                label=label or None,
                labelOpts={"position": at, "color": colour} if label else None)
            w.addItem(line)

        def curve(w, key, colour, dash=False):
            c = w.plot(pen=pg.mkPen(colour, width=2,
                                    style=Qt.DashLine if dash else Qt.SolidLine))
            self._series[key] = c
            return c

        def grid_of(plots):
            page = QWidget()
            g = QGridLayout(page)
            g.setContentsMargins(0, 0, 0, 0)
            for i, w in enumerate(plots):
                g.addWidget(w, i // 3, i % 3)
            return page

        # ---- Paper ---------------------------------------------------------
        # T hasta 180: por encima de 120 viven los esqueletos invertidos, que
        # son el modo de falla central de este proyecto. Las líneas son las
        # bandas del §3.4 y los cortes de la Etapa 3, con las etiquetas
        # escalonadas porque 30, 45 y 60 quedan a pocos píxeles.
        self.t_plot = plot("T — tronco (grados)", 0, 180)
        hline(self.t_plot, 15, "", (90, 90, 90))
        hline(self.t_plot, 90, "", (90, 90, 90))
        hline(self.t_plot, float(self.cfg.stage3.upright_T_deg),
              f"erguido < {float(self.cfg.stage3.upright_T_deg):.0f}", (120, 180, 120), at=0.12)
        # Con la formulación "score" threshold_T no dispara por sí sola: es
        # la escala del puntaje. Se dibuja tenue y se dice qué es.
        hline(self.t_plot, float(self.cfg.stage1.threshold_T_deg),
              "escala T" if score_formulation else "threshold_T",
              (130, 130, 130) if score_formulation else "orange", at=0.45)
        hline(self.t_plot, float(self.cfg.state_display.lying_T_deg),
              f"suelo \u2265 {float(self.cfg.state_display.lying_T_deg):.0f}", (229, 57, 53),
              at=0.78)
        self.t_curve = curve(self.t_plot, "T_deg", "y")

        self.v_plot = plot("V — velocidad vertical (torso/s, neg = baja)", -4, 4)
        hline(self.v_plot, float(self.cfg.stage1.threshold_V),
              "escala V" if score_formulation else "threshold_V",
              (110, 110, 110) if score_formulation else "orange")
        hline(self.v_plot, 0.0, "", (120, 120, 120), dash=False)
        self.v_curve = curve(self.v_plot, "V_tps", "m")

        # Lo que la Etapa 1 compara de verdad: T/escala_T + V/escala_V sobre
        # la ventana de pico. Vacío donde el disparador no lo evalúa.
        self.s_plot = plot("Puntaje del disparador", 0, 5)
        hline(self.s_plot, float(self.cfg.stage1.trigger_score),
              f"dispara \u2265 {float(self.cfg.stage1.trigger_score):.1f}", (229, 57, 53),
              dash=False, width=2)
        provisional = float(getattr(self.cfg.stage1, "trigger_score_provisional", 0.0) or 0.0)
        if provisional > 0.0:
            hline(self.s_plot, provisional, f"banda {provisional:.1f}", (255, 152, 0), at=0.5)
        self.s_curve = curve(self.s_plot, "trigger_score", (255, 255, 255))

        # Quantity P. Zero is the whole decision line: the sign says inside
        # or outside the support, so it is solid, not a configurable threshold.
        self.p_plot = plot("P — COM vs apoyo (torso, >0 = fuera)", -1.5, 1.5)
        hline(self.p_plot, 0.0, "fuera", "orange", dash=False, width=2)
        self.p_curve = curve(self.p_plot, "P_offset", "c")

        # Quantity I. The dashed line is W. It saws back to zero on every
        # movement, which is the quantity working, not a glitch.
        self.i_plot = plot("I — inmovilidad (s)", 0,
                           max(2.0, float(self.cfg.stage3.threshold_W_seconds) * 1.5))
        hline(self.i_plot, float(self.cfg.stage3.threshold_W_seconds), "W", "orange")
        self.i_curve = curve(self.i_plot, "I_still_s", "g")

        # Fase 3: la parte de la Etapa 3 que estuvo quieta. Al final del
        # clip, si queda sobre la línea y la postura no es tumbada ni de pie,
        # el veredicto es severe (la inmovilidad persistió, §3.5).
        self.q_plot = plot("% de la Etapa 3 quieto (fase 3)", 0, 1)
        frac = float(self.cfg.stage3.persistent_immobility_fraction)
        if frac > 0.0:
            hline(self.q_plot, frac, f"severe si \u2265 {100 * frac:.0f}%", (229, 57, 53),
                  dash=False, at=0.2)
        self.q_curve = curve(self.q_plot, "still_fraction", (255, 213, 79))

        paper = grid_of((self.t_plot, self.v_plot, self.s_plot,
                         self.p_plot, self.i_plot, self.q_plot))

        # ---- Experimentales -----------------------------------------------
        ex = self.cfg.experimental
        sd = self.cfg.state_display
        # H: el disparo por H usa trigger_H_ratio (y también es el corte de
        # «tumbado» cuando T no está medida); la recuperación, recovery_H_ratio.
        self.h_plot = plot("H — altura vs su base [EXP]", 0, 2)
        for value, label, colour, at in (
                (ex.trigger_H_ratio, "colapso", (229, 57, 53), 0.15),
                (ex.recovery_H_ratio, "recuperado", (67, 160, 71), 0.45),
                (ex.trigger_H_erect, "erguido", (120, 180, 120), 0.75)):
            if float(value) > 0.0:
                hline(self.h_plot, float(value), f"{label} {float(value):.2f}", colour, at=at)
        curve(self.h_plot, "H_ratio", (255, 112, 67))

        # H cruda y su base: aquí se ve la base envenenada de B05 (un pico
        # de H cruda que la base absorbe y arrastra por segundos). Eje
        # automático, porque ese pico llegó a 98.
        self.hraw_plot = plot("H cruda (blanco) y su base (cian) [EXP]")
        curve(self.hraw_plot, "H_raw", (236, 239, 241))
        curve(self.hraw_plot, "H_baseline", (0, 229, 255), dash=True)

        self.ext_plot = plot("Piernas: cadera-tobillo / torso [EXP]", 0,
                             float(sd.max_extension_ratio) * 1.1)
        hline(self.ext_plot, float(sd.crouch_extension),
              f"agachado < {float(sd.crouch_extension):.1f}", (255, 152, 0), at=0.15)
        hline(self.ext_plot, float(sd.standing_extension),
              f"extendidas \u2265 {float(sd.standing_extension):.1f}", (67, 160, 71), at=0.45)
        hline(self.ext_plot, float(sd.max_extension_ratio),
              f"m\u00e1x plausible {float(sd.max_extension_ratio):.0f}", (130, 130, 130), at=0.75)
        curve(self.ext_plot, "extension_ratio", (171, 71, 188))

        # R: solo se registra; no hay umbral en la configuración, así que no
        # se dibuja ninguno inventado.
        self.r_plot = plot("R — mu\u00f1eca a tobillo (torsos) [EXP]", 0, 3)
        curve(self.r_plot, "reach_proximity", (129, 212, 250))

        self.vh_plot = plot("Vh — velocidad horizontal (torso/s) [EXP]", -2, 2)
        walk = float(sd.walking_vh_tps)
        hline(self.vh_plot, walk, f"caminando \u00b1{walk:.2f}", (130, 130, 130), at=0.15)
        hline(self.vh_plot, -walk, "", (130, 130, 130))
        curve(self.vh_plot, "Vh_tps", (255, 241, 118))

        # Lo que I compara cada cuadro: cuánto se alejó el sujeto del punto
        # donde empezó a quedarse quieto. Sobre ε, I vuelve a cero.
        eps = float(self.cfg.stage3.epsilon)
        self.d_plot = plot("Desplazamiento de I (torsos) [EXP]", 0, max(0.2, 4 * eps))
        hline(self.d_plot, eps, f"\u03b5 {eps:.2f}", "orange", at=0.15)
        curve(self.d_plot, "I_displacement", (165, 214, 167))

        experimental = grid_of((self.h_plot, self.hraw_plot, self.ext_plot,
                                self.r_plot, self.vh_plot, self.d_plot))

        self.curve_tabs = QTabWidget()
        self.curve_tabs.addTab(paper, "Paper (\u00a73.4\u20133.5)")
        self.curve_tabs.addTab(experimental, "Experimentales [EXP]")
        layout.addWidget(self.curve_tabs)

        # El veredicto y su motivo: la regla que decidió, con sus números.
        # A todo el ancho, visible con cualquiera de las dos pestañas.
        self.verdict_label = QLabel("Veredicto: —")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.verdict_label.setStyleSheet("padding: 4px;")
        self.verdict_label.setMaximumHeight(64)
        layout.addWidget(self.verdict_label)

        # Sin enlazar los ejes X: pyqtgraph alinea ejes enlazados por píxel en
        # pantalla, y con pestañas ocultas eso daba rangos sin sentido. El
        # rango de tiempo se fija a mano en todas al redibujar.
        self._curve_plots = (self.t_plot, self.v_plot, self.s_plot, self.p_plot,
                             self.i_plot, self.q_plot, self.h_plot, self.hraw_plot,
                             self.ext_plot, self.r_plot, self.vh_plot, self.d_plot)
        self._cursors = []
        for w in self._curve_plots:
            c = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=1))
            w.addItem(c)
            self._cursors.append(c)

        # Línea de tiempo: etapa por cuadro (gris = vigilando, naranja =
        # Etapa 2, azul = Etapa 3, violeta = enfriamiento), negro donde se
        # perdió la detección. Marcas: disparo (rojo punteado) y veredicto
        # (color de la severidad). Las lagunas de detección fueron la clave de
        # A08-S3; aquí se ven sin adivinarlas por cortes en las curvas.
        self.timeline = pg.PlotWidget()
        self.timeline.setFixedHeight(46)
        self.timeline.hideAxis("left")
        self.timeline.setMouseEnabled(x=False, y=False)
        self.timeline.setYRange(0, 1, padding=0)
        # SIN enlazar a las curvas: pyqtgraph alinea ejes enlazados por píxel
        # en pantalla, y una franja a todo el ancho enlazada a una curva de un
        # tercio mostraba 0-34 s para un clip de 10 s. El rango se fija a mano.
        self.timeline_img = pg.ImageItem()
        self.timeline.addItem(self.timeline_img)
        self.timeline_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=2))
        self.timeline.addItem(self.timeline_cursor)
        self._cursors.append(self.timeline_cursor)
        layout.addWidget(self.timeline)

        # --- Overlay toggles ------------------------------------------------
        toggles_row = QHBoxLayout()
        toggles_row.addWidget(QLabel("Show:"))
        self.chk_skeleton = QCheckBox("Skeleton")
        self.chk_skeleton.setChecked(True)
        toggles_row.addWidget(self.chk_skeleton)
        self.chk_trunk = QCheckBox("Trunk vector")
        self.chk_trunk.setChecked(True)
        toggles_row.addWidget(self.chk_trunk)
        self.chk_centroid = QCheckBox("Centroid")
        self.chk_centroid.setChecked(True)
        toggles_row.addWidget(self.chk_centroid)
        self.chk_support = QCheckBox("Support / COM")
        self.chk_support.setChecked(True)
        toggles_row.addWidget(self.chk_support)
        self.chk_curves = QCheckBox("Curves")
        self.chk_curves.setChecked(True)
        self.chk_curves.toggled.connect(self._on_curves_toggled)
        toggles_row.addWidget(self.chk_curves)
        toggles_row.addStretch(1)
        layout.addLayout(toggles_row)

        # Source row
        source_row = QHBoxLayout()
        self.open_video_btn = QPushButton("Open Video…")
        self.open_video_btn.clicked.connect(self._on_open_video)
        source_row.addWidget(self.open_video_btn)
        source_row.addSpacing(16)
        source_row.addWidget(QLabel("Camera index:"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        source_row.addWidget(self.camera_spin)
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self._on_toggle_camera)
        source_row.addWidget(self.camera_btn)
        self.close_btn = QPushButton("Close Source")
        self.close_btn.clicked.connect(self._close_source)
        source_row.addWidget(self.close_btn)
        # La única forma de volver a analizar un clip ya congelado en esta
        # sesión: una decisión explícita, que escribe un registro nuevo.
        self.reanalyze_btn = QPushButton("Re-analizar desde 0")
        self.reanalyze_btn.clicked.connect(self._on_reanalyze)
        source_row.addWidget(self.reanalyze_btn)
        source_row.addStretch(1)
        layout.addLayout(source_row)

        # Playback row (video mode only)
        playback_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._on_play_pause)
        playback_row.addWidget(self.play_btn)
        self.step_back_btn = QPushButton("◀ Frame")
        self.step_back_btn.clicked.connect(lambda: self._step(-1))
        playback_row.addWidget(self.step_back_btn)
        self.step_fwd_btn = QPushButton("Frame ▶")
        self.step_fwd_btn.clicked.connect(lambda: self._step(+1))
        playback_row.addWidget(self.step_fwd_btn)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.valueChanged.connect(self._on_slider_moved)
        playback_row.addWidget(self.seek_slider, stretch=1)
        self.time_label = QLabel("0:00 / 0:00")
        playback_row.addWidget(self.time_label)
        playback_row.addWidget(QLabel("Velocidad:"))
        self.speed_box = QComboBox()
        for sp in SPEEDS:
            self.speed_box.addItem(f"{sp:g}x", sp)
        self.speed_box.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_box.currentIndexChanged.connect(
            lambda i: setattr(self, "_speed", float(self.speed_box.itemData(i))))
        playback_row.addWidget(self.speed_box)
        layout.addLayout(playback_row)

        # Atajos: espacio reproduce/pausa, flechas un cuadro, Mayús+flechas
        # un segundo. Alt+flechas siguen siendo cambiar de clip en la cola.
        for keys, slot in (("Space", self._on_play_pause),
                           ("Right", lambda: self._step(+1)),
                           ("Left", lambda: self._step(-1)),
                           ("Shift+Right", lambda: self._step_seconds(+1.0)),
                           ("Shift+Left", lambda: self._step_seconds(-1.0))):
            sc = QShortcut(QKeySequence(keys), self)
            sc.activated.connect(slot)

        self._build_queue_panel(layout)

    # ---------------------------------------------------------------- queue --
    def _build_queue_panel(self, layout: QVBoxLayout) -> None:
        """The dataset pass: a folder of clips, labelled one after another.

        It reuses the ordinary playback path rather than running its own
        loop, so what you watch during the pass is exactly what produced the
        label — the same overlay, the same curves, the same pipeline. A
        separate fast path would label clips nobody ever sees, and the first
        disagreement with the annotation would have no way to be examined.
        """
        queue_row = QHBoxLayout()
        self.queue_add_btn = QPushButton("Add Folder…")
        self.queue_add_btn.clicked.connect(self._on_queue_add_folder)
        queue_row.addWidget(self.queue_add_btn)
        self.queue_start_btn = QPushButton("Start Queue")
        self.queue_start_btn.clicked.connect(self._on_queue_start)
        queue_row.addWidget(self.queue_start_btn)
        self.queue_stop_btn = QPushButton("Stop Queue")
        self.queue_stop_btn.clicked.connect(self._on_queue_stop)
        queue_row.addWidget(self.queue_stop_btn)
        self.queue_save_btn = QPushButton("Save Reviews")
        self.queue_save_btn.clicked.connect(self._on_queue_save_reviews)
        queue_row.addWidget(self.queue_save_btn)
        self.queue_clear_btn = QPushButton("Clear")
        self.queue_clear_btn.clicked.connect(self._on_queue_clear)
        queue_row.addWidget(self.queue_clear_btn)
        self.queue_progress = QProgressBar()
        self.queue_progress.setFormat("%v / %m")
        queue_row.addWidget(self.queue_progress, stretch=1)
        self.queue_score = QLabel("")
        self.queue_score.setStyleSheet("font-weight: bold;")
        queue_row.addWidget(self.queue_score)
        layout.addLayout(queue_row)

        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(
            ["video", "verdad", "detectada", "", "revisada"])
        header = self.queue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setMaximumHeight(190)
        # Double-click replays one clip on its own, outside the pass, so a
        # disagreement can be inspected frame by frame without re-running the
        # whole folder.
        self.queue_table.cellDoubleClicked.connect(self._on_queue_row_opened)
        layout.addWidget(self.queue_table)

        # Revisión: moverse entre los clips ya corridos.
        #
        # El doble clic ya abría un clip suelto, pero revisar una carpeta es
        # comparar: las situaciones de PEF-FallDB se graban con los cuatro
        # sujetos, y lo que explica por qué tres fallan y uno acierta casi
        # nunca está dentro de un clip — está en la diferencia entre ellos.
        # Volver a la tabla y buscar la fila cada vez rompe esa comparación,
        # así que aquí van los dos botones y sus atajos.
        nav = QHBoxLayout()
        self.review_prev_btn = QPushButton("◀ Clip anterior")
        self.review_prev_btn.clicked.connect(lambda: self._review_step(-1))
        self.review_prev_btn.setShortcut(QKeySequence("Alt+Left"))
        self.review_prev_btn.setToolTip("Alt+←   (o doble clic en la tabla)")
        nav.addWidget(self.review_prev_btn)

        self.review_next_btn = QPushButton("Clip siguiente ▶")
        self.review_next_btn.clicked.connect(lambda: self._review_step(+1))
        self.review_next_btn.setShortcut(QKeySequence("Alt+Right"))
        self.review_next_btn.setToolTip("Alt+→   (o doble clic en la tabla)")
        nav.addWidget(self.review_next_btn)

        self.review_label = QLabel("")
        self.review_label.setStyleSheet("color: #666;")
        nav.addWidget(self.review_label, stretch=1)
        layout.addLayout(nav)

    # -------------------------------------------------------------- sources --
    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str, reanalyze: bool = False) -> None:
        """Open a clip by path: frozen if this session already analysed it.

        Same clip, same configuration, PEF-Lab not closed since: the stored
        analysis is shown as it came out of the pass — nothing recomputed,
        nothing recorded. Otherwise (or with ``reanalyze``) a fresh pass runs
        and writes its own record. See review_cache.py for why.
        """
        self._close_source()
        try:
            self.source = VideoFileSource(path)
        except (IOError, FileNotFoundError) as exc:
            self._set_status(str(exc), error=True)
            return
        self._current_path = str(path)
        self.seek_slider.setRange(0, max(0, self.source.frame_count - 1))
        cached = None if reanalyze else self._cache.get(path, self._cfg_key)
        if reanalyze:
            self._cache.drop(path, self._cfg_key)
        if cached is not None:
            self._enter_frozen(cached)
            self._seek_to(0)
            self._set_status(
                f"Congelado: {Path(path).name} — el analisis de esta sesion, sin "
                f"recalcular y sin escribir registros. Espacio para reproducir; "
                f"\"Re-analizar desde 0\" para una pasada nueva.")
            self._update_controls()
            return
        self._start_session(Path(path).stem, metadata={
            "source": str(Path(path).resolve()),
            "source_kind": "video",
            "source_fps_declared": self.source.fps,
            "frame_count": self.source.frame_count,
        })
        self._recorder = PassRecorder(path, self._cfg_key, self.source.fps)
        self.playing = True
        self.play_btn.setText("Pause")
        self.timer.start(0)  # self-scheduling loop takes over from here
        self._set_status(
            f"Video: {Path(path).name} — {self.source.frame_count} frames "
            f"@ {self.source.fps:.1f} fps (declared). Logging to {self.logger.path.name}"
        )
        self._update_controls()

    def _on_reanalyze(self) -> None:
        """Descartar lo congelado y correr una pasada nueva, con registro nuevo."""
        if self._current_path is None or self._queue_running:
            return
        self.load_video(self._current_path, reanalyze=True)

    # ---------------------------------------------------------- congelado --
    def _enter_frozen(self, analysis: ClipAnalysis) -> None:
        """Mostrar un análisis guardado: curvas completas, marcas y veredicto."""
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None     # nada puede recalcular a partir de aquí
        self._recorder = None
        self._frozen = analysis
        self.playing = False
        self.play_btn.setText("Play")
        self._clear_curves()
        marked: set[int] = set()
        for index in sorted(analysis.frames):
            res = analysis.frames[index]
            self._store_plot_point(index, res.pose.timestamp, res)
            self._timeline_rgb[index] = _frame_rgb(res)
            self._timeline_t[index] = res.pose.timestamp
            if res.event is not None:
                self._mark(res.event.timestamp, "#ff1744", dash=True)
            if res.resolved_event is not None:
                self._mark_verdict(res.pose.timestamp, res.resolved_event)
                marked.add(id(res.resolved_event))
        # En modo etiquetado casi todo se decide en finalise(), DESPUÉS del
        # último cuadro: esos veredictos no cuelgan de ningún cuadro y se
        # marcan al final del clip, que es donde la pasada los vio caer.
        if analysis.frames:
            t_end = analysis.frames[max(analysis.frames)].pose.timestamp
            for ev in analysis.events:
                if id(ev) not in marked and ev.verdict != "stage1_only":
                    self._mark_verdict(t_end, ev)
        self._redraw_curves()
        self._redraw_timeline()
        self._show_verdict(analysis.events)

    def _show_frozen_frame(self, frame, index: int, timestamp: float) -> None:
        """Dibujar lo que la pasada vio en este cuadro. Sin pipeline."""
        res = self._frozen.frames.get(index)
        if res is None:          # cuadro que la pasada no alcanzó
            self._display(frame)
        else:
            self._display(draw_overlay(
                frame, res.pose,
                visibility_threshold=self.cfg.pose.visibility_threshold,
                show_skeleton=self.chk_skeleton.isChecked(),
                show_trunk=self.chk_trunk.isChecked(),
                show_centroid=self.chk_centroid.isChecked(),
                centroid_px=res.centroid_px,
                show_support=self.chk_support.isChecked(),
                com_px=res.com_px,
                support_hull_px=res.support_hull_px,
                com_outside=_p_side(res.quantities.get("P_offset")),
                hud_lines=["[CONGELADO] sin recalcular"] + list(res.hud_lines),
            ))
            n = len(self._frozen.events)
            self.stage_label.setText(
                f"[CONGELADO]  etapa en este cuadro: {res.stage}   |   eventos: {n}")
            self.stage_label.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        self._move_cursors(timestamp)

    def _on_toggle_camera(self) -> None:
        if self.source is not None and self.source.is_live:
            self._close_source()
            return
        self._close_source()
        try:
            self.source = CameraSource(self.camera_spin.value())
        except IOError as exc:
            self._set_status(str(exc), error=True)
            return
        self._start_session(f"camera{self.camera_spin.value()}", metadata={
            "source": f"camera index {self.camera_spin.value()}",
            "source_kind": "live camera",
            "source_fps_declared": self.source.fps,
        })
        self.playing = True
        self.camera_btn.setText("Stop Camera")
        self.timer.start(0)  # self-scheduling loop; pace comes from the source
        self._set_status(
            f"Live camera {self.camera_spin.value()} — logging to {self.logger.path.name}"
        )
        self._update_controls()

    def _start_session(self, source_name: str, metadata: dict | None = None) -> None:
        """Create the processing pipeline and the audit logger for a new source."""
        # A recorded clip is labelled from how the episode ENDED (§3.3); a
        # live camera has no end, so it keeps the greedy resolution the
        # §3.6 device needs. The flag is the difference between the two.
        self.pipeline = FramePipeline(
            self.cfg,
            labelling=self.source is not None and not self.source.is_live)
        self._last_timestamp = None
        self._logged_frames = 0
        self._reliable_frames = 0
        self.pipeline.source_name = source_name
        self.pipeline.alerts.add_sink(self._on_alert)
        self.alert_banner.setVisible(False)
        meta = dict(metadata or {})
        meta["config"] = self.cfg.as_dict()
        self.logger = AuditLogger(self.cfg.logging.output_dir, source_name, metadata=meta)
        # A second record, one row per resolved event. Live sessions used to
        # produce none at all: the events log existed only in the headless
        # runner, so a fall watched through this window left no trace of what
        # any stage decided about it.
        # flush_each_row: an event is rare and irreplaceable; see AuditLogger.
        self.event_logger = AuditLogger(
            self.cfg.logging.output_dir, source_name + "-events",
            fields=EVENT_FIELDS,
            metadata={**meta, "record_kind": "stage1_events"},
            flush_each_row=True)
        self.frame_index = 0
        self._clear_curves()
        self.verdict_label.setText("Veredicto: — (analizando)")

    def _log_event(self, ev) -> None:
        """Write one resolved event. Called when its verdict lands, not when
        it was raised: the verdict is the point of the record."""
        if self.event_logger is None:
            return
        self.event_logger.log({
            "event_index": len(self.pipeline.machine.events) - 1,
            "frame_index": ev.frame_index,
            "timestamp_s": f"{ev.timestamp:.3f}",
            "T_deg": f"{ev.t_deg:.2f}",
            "V_tps": f"{ev.v_tps:.3f}",
            "formulation": ev.formulation,
            "verdict": ev.verdict,
            "severity": ev.severity,
            "max_immobility_s": ("" if math.isnan(ev.max_immobility_s)
                                 else f"{ev.max_immobility_s:.2f}"),
            "p_outside_fraction": ("" if math.isnan(ev.p_outside_fraction)
                                   else f"{ev.p_outside_fraction:.3f}"),
            "p_samples": ev.p_samples,
            "H_ratio_EXP": "" if math.isnan(ev.h_ratio) else f"{ev.h_ratio:.3f}",
            "motivo": ev.reason,
        })

    # ----------------------------------------------------------- queue flow --
    def _on_queue_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dataset folder")
        if not folder:
            return
        clips = video_files(folder)
        if not clips:
            self._set_status(f"No videos in {folder}", error=True)
            return
        start = len(self._queue)
        self._queue.extend(clips)
        self.queue_table.setRowCount(len(self._queue))
        for offset, path in enumerate(clips):
            row = start + offset
            truth = truth_from_name(path)
            self.queue_table.setItem(row, 0, QTableWidgetItem(path.name))
            self.queue_table.setItem(row, 1, QTableWidgetItem(truth or "?"))
            self.queue_table.setItem(row, 2, QTableWidgetItem(""))
            self.queue_table.setItem(row, 3, QTableWidgetItem(""))
            box = QComboBox()
            box.addItems(("",) + tuple(CLASSES))
            # The human column. A dataset needs both opinions: without the
            # reviewed one there is nothing to score the machine against
            # except a file name, and file names are where this material
            # already carries typos.
            self.queue_table.setCellWidget(row, 4, box)
        self.queue_progress.setRange(0, len(self._queue))
        self.queue_progress.setValue(0)
        self._set_status(f"Queued {len(clips)} clips from {Path(folder).name} "
                         f"({len(self._queue)} total)")
        self._update_controls()

    def _on_queue_start(self) -> None:
        if not self._queue or self._queue_running:
            return
        self._close_source()
        self._queue_running = True
        self._queue_index = 0
        self._queue_rows = []
        self._label_logger = AuditLogger(
            self.cfg.logging.output_dir, "dataset-labels",
            fields=LABEL_FIELDS,
            metadata={"record_kind": "clip_labels", "clips": len(self._queue),
                      "config": self.cfg.as_dict()},
            flush_each_row=True)
        self._queue_advance()

    def _on_queue_stop(self) -> None:
        """Stop after the current clip, keeping every label written so far."""
        if not self._queue_running:
            return
        self._queue_running = False
        self._close_source()
        self._close_label_record("Queue stopped.")
        self._update_controls()

    def _on_queue_clear(self) -> None:
        if self._queue_running:
            return
        self._queue = []
        self._review_index = None
        self._queue_rows = []
        self.queue_table.setRowCount(0)
        self.queue_progress.setRange(0, 0)
        self.queue_score.setText("")
        self._update_controls()

    def _on_queue_save_reviews(self) -> None:
        """Write a fresh labels file with the human column filled in.

        A separate file, not an edit of the one the pass produced: the run's
        own output is a record of what the system decided, and rewriting it
        in place would destroy the only evidence of the disagreement being
        corrected. Each save is timestamped, so the sequence of review passes
        stays readable.
        """
        if self._queue_running or not self._queue_rows:
            return
        self._label_logger = AuditLogger(
            self.cfg.logging.output_dir, "dataset-labels-reviewed",
            fields=LABEL_FIELDS,
            metadata={"record_kind": "clip_labels_reviewed",
                      "clips": len(self._queue_rows),
                      "config": self.cfg.as_dict()},
            flush_each_row=True)
        for row, record in enumerate(self._queue_rows):
            box = self.queue_table.cellWidget(row, 4)
            record["clase_revisada"] = (box.currentText()
                                        if box is not None else "")
            self._label_logger.log(record)
        self._close_label_record("Reviews saved.")

    def _queue_advance(self) -> None:
        """Open the next clip, or finish the pass."""
        if not self._queue_running:
            return
        if self._queue_index >= len(self._queue):
            self._queue_running = False
            self._close_label_record("Queue finished.")
            self._update_controls()
            return
        path = self._queue[self._queue_index]
        self.queue_table.selectRow(self._queue_index)
        self.queue_table.scrollToItem(self.queue_table.item(self._queue_index, 0))
        # Iniciar la cola es correr de nuevo, a propósito: nunca congelado.
        self.load_video(str(path), reanalyze=True)

    def _label_current_clip(self) -> None:
        """Classify the clip that just ended and record one row for it.

        Called from the end-of-video path, after the pipeline has been asked
        to close any event still open — a clip that ends with the subject
        still on the floor has an unresolved event, and that event IS the
        answer.
        """
        if self.pipeline is None or self._queue_index >= len(self._queue):
            return
        path = self._queue[self._queue_index]
        events = self.pipeline.machine.events
        detected, reason = classify_clip(events)
        truth = truth_from_name(path)
        worst = events[-1] if events else None
        now = datetime.now()
        total = (self.source.frame_count / self.source.fps
                 if self.source is not None and self.source.fps else 0.0)
        reliable = self._reliable_frames
        frames = self._logged_frames
        row = {
            "video": path.name,
            "fecha": now.strftime("%Y-%m-%d"),
            "hora": now.strftime("%H:%M:%S"),
            "duracion_s": f"{total:.2f}",
            "frames": frames,
            "frames_confiables_pct": (f"{100.0 * reliable / frames:.1f}"
                                      if frames else ""),
            "clase_detectada": detected,
            "clase_verdad": truth,
            "clase_revisada": "",
            "acierto": ("" if not truth else int(truth == detected)),
            "eventos": len(events),
            "t_disparo_s": f"{events[0].timestamp:.2f}" if events else "",
            "verdict": worst.verdict if worst else "",
            "severity": worst.severity if worst else "",
            "I_max_s": (f"{worst.max_immobility_s:.2f}"
                        if worst and worst.max_immobility_s == worst.max_immobility_s
                        else ""),
            "motivo": (f"{reason} — {worst.reason}" if worst and worst.reason
                       else reason),
        }
        if self._label_logger is not None:
            self._label_logger.log(row)
        self._queue_rows.append(row)
        self._show_queue_result(self._queue_index, detected, truth)
        item = self.queue_table.item(self._queue_index, 2)
        if item is not None:
            item.setToolTip(row["motivo"])
        return row

    def _show_queue_result(self, row: int, detected: str, truth: str) -> None:
        self.queue_table.setItem(row, 2, QTableWidgetItem(detected))
        mark = "" if not truth else ("OK" if truth == detected else "X")
        item = QTableWidgetItem(mark)
        if mark == "X":
            item.setForeground(Qt.red)
        self.queue_table.setItem(row, 3, item)
        self.queue_progress.setValue(row + 1)
        self.queue_score.setText(self._queue_metrics_text())

    def _queue_metrics_text(self) -> str:
        """Las métricas del §3.7, con el mismo código que el reporte.

        Usa evaluation.binary_metrics: el número en pantalla y el del reporte
        no pueden diferir, que es lo que el docstring de evaluation.py pide.
        """
        scored = [r for r in self._queue_rows if r["clase_verdad"]]
        if not scored:
            return ""
        hits = sum(1 for r in scored if r["clase_verdad"] == r["clase_detectada"])
        falls = [r for r in scored if r["clase_verdad"] != "NoFall"]
        fall_hits = sum(1 for r in falls if r["clase_verdad"] == r["clase_detectada"])
        b = binary_metrics(scored)
        parts = [f"4 clases {hits}/{len(scored)} ({100.0 * hits / len(scored):.0f}%)"]
        if falls:
            parts.append(f"caidas exactas {fall_hits}/{len(falls)}")
        for key, name in (("sensibilidad", "sens"), ("especificidad", "espec")):
            v = b.get(key, float("nan"))
            if v == v:                              # no NaN
                parts.append(f"{name} {100.0 * v:.0f}%")
        return "  ·  ".join(parts)

    def _on_queue_row_opened(self, row: int, _column: int) -> None:
        if self._queue_running or row >= len(self._queue):
            return
        self._open_for_review(row)

    def _open_for_review(self, row: int) -> None:
        """Cargar el clip ``row`` de la cola para mirarlo, fuera de la pasada.

        Es la ruta de reproducción de siempre; lo único que agrega es
        recordar en qué clip quedó, para que las flechas sepan desde dónde
        moverse. No toca la tabla: los resultados de la pasada siguen ahí,
        que es contra lo que uno está comparando mientras mira.
        """
        if not (0 <= row < len(self._queue)):
            return
        self._review_index = row
        self.queue_table.selectRow(row)
        item = self.queue_table.item(row, 0)
        if item is not None:
            self.queue_table.scrollToItem(item)
        self.load_video(str(self._queue[row]))
        self._update_review_label()

    def _review_step(self, delta: int) -> None:
        """Moverse al clip anterior o siguiente de la cola.

        Sin clip abierto todavía, la primera flecha entra por un extremo en
        vez de no hacer nada: llegar a la carpeta y no saber por dónde
        empezar es el caso más común, no un error.

        Se detiene en los extremos en lugar de dar la vuelta. Con cuatro
        sujetos por situación, volver al primero sin avisar se confunde muy
        fácil con "no pasó nada", y el objetivo aquí es comparar clips
        distintos.
        """
        if self._queue_running or not self._queue:
            return
        if self._review_index is None:
            self._open_for_review(0 if delta > 0 else len(self._queue) - 1)
            return
        destino = self._review_index + delta
        if not (0 <= destino < len(self._queue)):
            extremo = "primer" if delta < 0 else "último"
            self._set_status(f"Ya está en el {extremo} clip de la cola.")
            return
        self._open_for_review(destino)

    def _update_review_label(self) -> None:
        """El contador junto a las flechas: en qué clip está, y de cuántos."""
        if not self._queue:
            self.review_label.setText("")
            return
        if self._review_index is None:
            self.review_label.setText(
                f"{len(self._queue)} clips en la cola — "
                f"Alt+← / Alt+→ para recorrerlos")
            return
        nombre = self._queue[self._review_index].name
        self.review_label.setText(
            f"clip {self._review_index + 1} de {len(self._queue)}   ·   {nombre}")

    def _close_label_record(self, reason: str) -> None:
        """Finish the labels file, folding in any human corrections."""
        if self._label_logger is None:
            return
        for row, record in enumerate(self._queue_rows):
            box = self.queue_table.cellWidget(row, 4)
            if box is not None and box.currentText():
                record["clase_revisada"] = box.currentText()
        path = self._label_logger.path
        self._label_logger.close()
        self._label_logger = None
        summary = format_confusion(self._queue_rows)
        print(f"\n{summary}")
        # Entrenamiento y prueba SIEMPRE por separado: un solo numero sobre el
        # conjunto entero no dice si los ajustes generalizan, y ese es el
        # unico numero que un revisor va a mirar dos veces.
        split_file = Path(__file__).resolve().parents[2] / "notas" / "particion.yaml"
        if split_file.exists():
            groups = split_rows(self._queue_rows, load_split(split_file))
            for side, title in (("train", "CALIBRACION"), ("test", "PRUEBA"),
                                ("sin_asignar", "SIN ASIGNAR")):
                if groups[side]:
                    print("\n" + report(groups[side], title))
        print(f"\n-> {path}")
        first = summary.splitlines()[-1]
        self._set_status(f"{reason} {len(self._queue_rows)} clips — {first} -> {path.name}")

    def _finalise_records(self, reason: str) -> None:
        """Close both records and write their metadata sidecars.

        Separate from :meth:`_close_source` because the two events are not the
        same: a video that reaches its end has produced a COMPLETE record and
        should have it on disk, but the source stays open so the run can be
        reviewed with the seek bar. Before this existed, playing a clip to the
        end and then quitting left the rows in Python's buffer — measured on
        this project's own material, a 0-byte event file that had actually
        recorded a confirmed fall.

        Both loggers are dropped afterwards. That is deliberate: AuditLogger
        opens in "w" mode, so continuing to write to a closed record would
        truncate it. It now raises instead, and this keeps the window from
        ever reaching that.
        """
        if self.logger is None and self.event_logger is None:
            return
        path, rows = None, 0
        if self.logger is not None:
            path, rows = self.logger.path, self.logger.rows_written
            self.logger.close()
            self.logger = None
        events = 0
        if self.event_logger is not None:
            events = self.event_logger.rows_written
            self.event_logger.close()
            self.event_logger = None
        if path is not None:
            self._set_status(f"{reason} {rows} frames, {events} events -> {path}")
        else:
            self._set_status(reason)

    def _close_source(self) -> None:
        self.timer.stop()
        self.playing = False
        if self.source is not None:
            self.source.release()
            self.source = None
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None
        self._frozen = None
        self._recorder = None
        self._finalise_records("Source closed.")
        self.camera_btn.setText("Start Camera")
        self.play_btn.setText("Play")
        self.seek_slider.setRange(0, 0)
        self._update_controls()

    # ------------------------------------------------------------- playback --
    def _on_play_pause(self) -> None:
        if self.source is None or self.source.is_live:
            return
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing and self._frozen is not None:
            # Al final del clip, reproducir vuelve a empezar.
            if self.frame_index >= self.source.frame_count - 1:
                self.source.seek(0)
            self.timer.start(0)
            return
        if self.playing:
            if self.logger is None:
                # The record was finalised when the clip ended, so replaying it
                # records nothing. Saying so is the point: a silent no-op here
                # would let someone believe a second pass was captured.
                self._set_status("Replaying — the record is closed; "
                                 "reopen the clip to record another pass.")
            self.timer.start(0)  # restart the self-scheduling chain

    def _step(self, delta: int) -> None:
        """Advance/rewind exactly one frame while paused (video mode)."""
        if self.source is None or self.source.is_live:
            return
        self.playing = False
        self.play_btn.setText("Play")
        last = max(0, self.source.frame_count - 1)
        self._seek_to(min(last, max(0, self.frame_index + delta)))

    def _step_seconds(self, seconds: float) -> None:
        """Moverse un segundo (Mayús+flecha): la escala de una caída."""
        if self.source is None or self.source.is_live or self._queue_running:
            return
        self._step(int(round(seconds * self.source.fps)))

    def _seek_to(self, frame_index: int) -> None:
        """Jump to a frame and process it, declaring the discontinuity.

        The pipeline cannot tell a jump from ordinary playback — the
        timestamps of two frames a few positions apart look like a normal
        interval — so the caller has to say so. Without this the velocity
        window would silently mix frames from two places in the recording.
        """
        self.source.seek(frame_index)
        if self._frozen is None and self.pipeline is not None:
            self.pipeline.reset()
        self._process_next_frame(log=False)

    def _on_slider_pressed(self) -> None:
        self._user_scrubbing = True
        self.playing = False
        self.play_btn.setText("Play")

    def _on_slider_released(self) -> None:
        self._user_scrubbing = False

    def _on_slider_moved(self, value: int) -> None:
        if not self._user_scrubbing or self.source is None or self.source.is_live:
            return
        self._seek_to(value)

    # ----------------------------------------------------------------- loop --
    def _on_tick(self) -> None:
        """Process one frame and schedule the next (see the timer comment)."""
        if self.source is None or not self.playing:
            return
        started = time.perf_counter()
        self._process_next_frame()
        # The source may have been closed or ended inside the call above.
        if self.source is None or not self.playing:
            return
        rate = self.source.fps * (1.0 if self.source.is_live else self._speed)
        slot_ms = 1000.0 / max(1.0, rate)
        spent_ms = (time.perf_counter() - started) * 1000.0
        self.timer.start(max(0, int(slot_ms - spent_ms)))

    def _process_next_frame(self, log: bool = True) -> None:
        """Read → pose → overlay → display → audit log, for one frame.

        Args:
            log: whether to record the frame. False for frames reached by a
                manual jump: the record is a chronological account of a
                continuous observation, and re-visiting a frame while
                scrubbing would append a duplicate, out-of-order row that
                any later analysis would count twice.
        """
        ok, frame, timestamp = self.source.read()
        if self._frozen is not None:
            # Revisión congelada: se muestra lo que la pasada vio. Nada se
            # recalcula, nada se registra, ningún evento puede dispararse.
            if not ok:
                self.playing = False
                self.play_btn.setText("Play")
                return
            index = self.source.current_index
            self.frame_index = index
            self._show_frozen_frame(frame, index, timestamp)
            self._sync_slider(index, timestamp)
            return
        if not ok:
            if not self.source.is_live:  # end of file
                self.playing = False
                self.play_btn.setText("Play")
                # Close any event still being judged BEFORE labelling: a clip
                # that ends with the subject on the floor has an open event,
                # and that event is precisely the answer the label needs.
                if self.pipeline is not None and self._last_timestamp is not None:
                    resolved = self.pipeline.finalise(self._last_timestamp)
                    if resolved is not None:
                        self._on_event_resolved(resolved)
                label = self._label_current_clip() if self._queue_running else None
                record = str(self.logger.path) if self.logger is not None else ""
                # The record is complete the moment the clip ends, so it goes
                # to disk now rather than waiting for a Close that may never
                # come. The source stays open: reviewing is the whole point of
                # stopping here, and reviewed frames are not logged anyway.
                self._finalise_records("End of video.")
                self._freeze_finished_pass(label, record)
                if self._queue_running:
                    self._queue_index += 1
                    self._queue_advance()
            return

        if self.source.is_live:
            index = self.frame_index  # 0-based, same convention as file mode
            self.frame_index += 1
        else:
            index = self.source.current_index
            self.frame_index = index

        # All computation happens in the shared core pipeline; this window
        # only displays the result (design rule: the GUI is a viewer).
        result = self.pipeline.process(frame, index, timestamp)

        if log and self.logger is not None:
            self.logger.log_pose_frame(result.pose, extra=result.csv_extra())
            self._logged_frames += 1
            self._reliable_frames += int(bool(result.reliable))
        if self._recorder is not None:
            if log:
                self._recorder.record(index, result)
            else:
                self._recorder.invalidate("salto durante la pasada")
        self._last_timestamp = timestamp

        t_overlay = time.perf_counter()
        shown = draw_overlay(
            frame,
            result.pose,
            visibility_threshold=self.cfg.pose.visibility_threshold,
            show_skeleton=self.chk_skeleton.isChecked(),
            show_trunk=self.chk_trunk.isChecked(),
            show_centroid=self.chk_centroid.isChecked(),
            centroid_px=result.centroid_px,
            show_support=self.chk_support.isChecked(),
            com_px=result.com_px,
            support_hull_px=result.support_hull_px,
            com_outside=_p_side(result.quantities.get("P_offset")),
            hud_lines=result.hud_lines,
        )
        t_display = time.perf_counter()
        self._display(shown)
        t_plots = time.perf_counter()
        self._update_plots(index, timestamp, result)
        t_end = time.perf_counter()
        perf = self.pipeline.timings
        perf.add("overlay", t_display - t_overlay)
        perf.add("display", t_plots - t_display)
        perf.add("curves", t_end - t_plots)
        if result.resolved_event is not None:
            self._on_event_resolved(result.resolved_event)
        self._update_stage(result)

        if not self.source.is_live:
            self._sync_slider(index, timestamp)

    def _sync_slider(self, index: int, timestamp: float) -> None:
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(index)
        self.seek_slider.blockSignals(False)
        total = self.source.frame_count / self.source.fps
        self.time_label.setText(f"{_format_time(timestamp)} / {_format_time(total)}")

    def _freeze_finished_pass(self, label: dict | None, record: str) -> None:
        """Congelar la pasada que acaba de terminar, si fue continua."""
        rec, self._recorder = self._recorder, None
        if rec is None or self.pipeline is None:
            return
        analysis = rec.finish(events=list(self.pipeline.machine.events),
                              record_path=record, label=label)
        if analysis is None:
            self._set_status(f"No se congelo: {rec.invalid_reason or 'pasada vacia'}. "
                             f"Se re-analizara al volver a abrirlo.")
            return
        self._cache.store(analysis)
        if not self._queue_running:
            # Quedarse en el clip, ya congelado: moverse por él desde aquí no
            # recalcula nada. En la cola, el siguiente clip se abre enseguida.
            self._enter_frozen(analysis)

    def _display(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    # ----------------------------------------------------------------- plots --
    def _update_plots(self, index: int, timestamp: float, result) -> None:
        """Guardar este cuadro y redibujar las curvas y la línea de tiempo.

        Data is keyed by frame index so scrubbing back over already-seen
        frames overwrites instead of duplicating; the curve is drawn in
        time order regardless of the order frames were visited. In live
        mode only the most recent window is kept so memory stays bounded.
        """
        self._timeline_rgb[index] = _frame_rgb(result)
        self._timeline_t[index] = timestamp
        if not self.chk_curves.isChecked():
            return
        self._store_plot_point(index, timestamp, result)
        if self.source is not None and self.source.is_live and len(self._plot_data) > 900:
            # Live mode: keep ~30 s at 30 FPS; drop the oldest entries.
            for old_key in sorted(self._plot_data)[: len(self._plot_data) - 900]:
                del self._plot_data[old_key]
                self._timeline_rgb.pop(old_key, None)
                self._timeline_t.pop(old_key, None)
        self._redraw_curves()
        self._redraw_timeline()
        self._move_cursors(timestamp)

    def _store_plot_point(self, index: int, timestamp: float, result) -> None:
        q = result.quantities
        nan = float("nan")
        point = {key: q.get(key, nan) for key in self._series}
        point["t"] = timestamp
        self._plot_data[index] = point

    def _redraw_curves(self) -> None:
        if not self._plot_data:
            return
        keys = sorted(self._plot_data)
        times = np.array([self._plot_data[k]["t"] for k in keys], dtype=float)
        # pyqtgraph skips NaN gaps with connect="finite": the curve breaks
        # where there was no detection instead of drawing a fake bridge. For
        # P the break is doubly meaningful — it marks the stretches where the
        # feet were not visible and the quantity genuinely does not exist.
        for key, c in self._series.items():
            ys = np.array([self._plot_data[k].get(key, float("nan")) for k in keys],
                          dtype=float)
            c.setData(times, ys, connect="finite")
        # El eje de tiempo se fija a mano en todas: con ejes enlazados el
        # autoajuste de X quedaba apagado y la vista se quedaba en -0.5..0.5,
        # con las curvas fuera de cuadro (visto en la captura de A17-S2).
        span = max(times[-1] - times[0], 1.0 / 30.0)
        for w in self._curve_plots:
            w.setXRange(times[0], times[0] + span, padding=0.01)

    def _redraw_timeline(self) -> None:
        if not self._timeline_rgb:
            self.timeline_img.clear()
            return
        keys = sorted(self._timeline_rgb)
        rgb = np.array([self._timeline_rgb[k] for k in keys], dtype=np.uint8)
        self.timeline_img.setImage(rgb.reshape(len(keys), 1, 3), levels=(0, 255))
        t0, t1 = self._timeline_t[keys[0]], self._timeline_t[keys[-1]]
        dt = (t1 - t0) / max(1, len(keys) - 1) if len(keys) > 1 else 1.0 / 30.0
        self.timeline_img.setRect(QRectF(t0, 0.0, (t1 - t0) + dt, 1.0))
        self.timeline.setXRange(t0, t1 + dt, padding=0.01)

    def _move_cursors(self, timestamp: float) -> None:
        for c in self._cursors:
            c.setPos(timestamp)

    def _clear_curves(self) -> None:
        self._plot_data.clear()
        self._timeline_rgb.clear()
        self._timeline_t.clear()
        for curve in self._series.values():
            curve.setData([], [])
        self.timeline_img.clear()
        for w, item in self._markers:
            w.removeItem(item)
        self._markers.clear()
        self.verdict_label.setText("Veredicto: —")

    def _mark(self, t: float, colour, dash: bool, width: int = 2) -> None:
        """Una marca vertical en todas las curvas y en la línea de tiempo."""
        for w in self._curve_plots + (self.timeline,):
            item = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(
                colour, width=width, style=Qt.DashLine if dash else Qt.SolidLine))
            w.addItem(item)
            self._markers.append((w, item))

    def _mark_verdict(self, t: float, ev) -> None:
        self._mark(t, _SEVERITY_PEN.get(ev.severity, "#9e9e9e"), dash=False, width=3)

    def _show_verdict(self, events) -> None:
        """El panel del veredicto: cada evento, con la regla que lo decidió."""
        if not events:
            self.verdict_label.setText("Veredicto: sin eventos (la Etapa 1 nunca disparo)")
            return
        lines = []
        for ev in events:
            tag = ev.verdict + (f" [{ev.severity}]" if ev.severity else "")
            # Escapado: el motivo lleva '<' y '>=', que como HTML se comían
            # el texto ("T final 3 deg = 50%" en vez de "< 30 ... >= 50%").
            lines.append(f"<b>t={ev.timestamp:.2f}s  {html.escape(tag)}</b><br>"
                         f"{html.escape(ev.reason or '—')}")
        self.verdict_label.setText("Veredicto:<br>" + "<br>".join(lines))

    def _on_alert(self, alert: Alert) -> None:
        """Screen sink for the §3.5 alert: the banner, in severity colour."""
        colour = {"severe": "#c62828", "moderate": "#ef6c00",
                  "mild": "#2e7d32"}.get(alert.severity, "#c62828")
        self._show_banner(alert.message(), colour, 15, 12_000)

    def _show_banner(self, text: str, colour: str, size: int, ms: int) -> None:
        self.alert_banner.setText(text)
        self.alert_banner.setStyleSheet(
            f"background-color: {colour}; color: white; font-weight: bold; "
            f"font-size: {size}px; padding: 10px; border-radius: 4px;")
        self.alert_banner.setVisible(True)
        self.alert_timer.start(ms)

    def _on_event_resolved(self, ev) -> None:
        """Every outcome gets shown, not only the ones that raise an alarm.

        A verdict of "rejected" or "inconclusive" is information while
        testing: it says the funnel looked at something and decided against
        it. Showing nothing in those cases is indistinguishable from the
        detector being asleep, which is exactly what a person testing in
        front of the camera needs to be able to tell apart. Confirmed falls
        still arrive through the alert sink, louder and in severity colour.
        """
        self._log_event(ev)
        if self._last_timestamp is not None:
            self._mark_verdict(self._last_timestamp, ev)
        if self.pipeline is not None:
            self._show_verdict(self.pipeline.machine.events)
        if self.pipeline is not None and self.pipeline.alerts.should_dispatch(ev.verdict):
            return                      # the alert sink already showed it
        text = (f"event at t={ev.timestamp:.2f}s  ->  {ev.verdict}"
                + (f" [{ev.severity}]" if ev.severity else "")
                + f"   (T={ev.t_deg:.1f} deg, V={ev.v_tps:+.2f} torso/s)"
                + (f"\nmotivo: {ev.reason}" if ev.reason else ""))
        self._show_banner(text, "#37474f", 13, 6_000)

    def _update_stage(self, result) -> None:
        """Reflect the §3.5 funnel state, and mark firings on the curves."""
        machine = self.pipeline.machine if self.pipeline is not None else None
        n = len(machine.events) if machine is not None else 0
        verdicts = ""
        if machine is not None and machine.events:
            tally: dict[str, int] = {}
            for e in machine.events:
                key = f"{e.verdict}/{e.severity}" if e.severity else e.verdict
                tally[key] = tally.get(key, 0) + 1
            verdicts = "   |   " + "  ".join(
                f"{k}={v}" for k, v in sorted(tally.items()))
        text = f"Stage: {result.stage}   |   events: {n}{verdicts}"
        if result.event is not None:
            ev = result.event
            text += f"   ->  FIRED at t={ev.timestamp:.2f}s  T={ev.t_deg:.1f}  V={ev.v_tps:+.2f}"
            # A vertical marker on every curve and on the timeline, so the
            # moment of the decision is readable against all quantities.
            self._mark(ev.timestamp, "#ff1744", dash=True)
        now = time.monotonic()
        if self.pipeline is not None and now - self._last_perf_shown > 1.0:
            self._last_perf_shown = now
            self._set_status(self.pipeline.timings.summary())
        self.stage_label.setText(text)
        self.stage_label.setStyleSheet(
            "font-weight: bold; color: #ff5555;" if result.event is not None
            else "font-weight: bold;")

    def _on_curves_toggled(self, checked: bool) -> None:
        for w in (self.curve_tabs, self.timeline, self.verdict_label):
            w.setVisible(checked)

    # -------------------------------------------------------------- helpers --
    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            # No colour on the normal case: the theme's own text colour is
            # readable in both light and dark, which a fixed green was not.
            "color: #cc3333; font-weight: bold;" if error else ""
        )

    def _update_controls(self) -> None:
        has_video = self.source is not None and not self.source.is_live
        running = self._queue_running
        self.play_btn.setEnabled(has_video and not running)
        self.step_back_btn.setEnabled(has_video and not running)
        self.step_fwd_btn.setEnabled(has_video and not running)
        self.seek_slider.setEnabled(has_video and not running)
        self.reanalyze_btn.setEnabled(has_video and not running)
        self.speed_box.setEnabled(has_video)
        # Opening another source mid-pass would silently relabel a clip
        # against a pipeline that had already seen half of a different one.
        self.open_video_btn.setEnabled(not running)
        self.camera_btn.setEnabled(not running)
        self.close_btn.setEnabled(not running)
        self.queue_add_btn.setEnabled(not running)
        self.queue_clear_btn.setEnabled(not running and bool(self._queue))
        self.queue_save_btn.setEnabled(not running and bool(self._queue_rows))
        self.queue_start_btn.setEnabled(not running and bool(self._queue))
        self.queue_stop_btn.setEnabled(running)
        # Revisar es lo contrario de correr la pasada: durante ella las
        # flechas cargarian otro clip a media medicion, que es el mismo
        # accidente que ya impide abrir un video con el boton de arriba.
        self.review_prev_btn.setEnabled(not running and bool(self._queue))
        self.review_next_btn.setEnabled(not running and bool(self._queue))
        self._update_review_label()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Release the camera, the pipeline and the record when the window closes.

        Qt does not do this for us: without it, closing the window would
        leave the capture device held and the audit CSV unflushed.
        """
        self._close_source()
        super().closeEvent(event)


def run_gui(cfg: Config, video: str | None = None, camera: int | None = None) -> None:
    """Launch PEF-Lab, optionally preloading a video file or a camera."""
    app = QApplication(sys.argv)
    window = LabWindow(cfg)
    window.show()
    if video is not None:
        window.load_video(video)
    elif camera is not None:
        window.camera_spin.setValue(camera)
        window._on_toggle_camera()
    sys.exit(app.exec())
