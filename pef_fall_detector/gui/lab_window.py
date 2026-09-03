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

import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
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
from ..overlay import draw_overlay
from ..pipeline import FramePipeline
from ..sources import CameraSource, FrameSource, VideoFileSource


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
        self._queue_rows: list[dict] = []
        self._queue_index = 0
        self._queue_running = False
        self._label_logger: AuditLogger | None = None
        # Plot data keyed by frame index so scrubbing overwrites cleanly
        # instead of appending duplicates: index -> (t, T_deg, V_tps, P_offset, I_still_s).
        self._plot_data: dict[int, tuple[float, float, float, float, float]] = {}

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

        # --- Live curves: T(t) and V(t) with threshold lines (2.5) ---------
        # The horizontal dashed lines are the CURRENT config thresholds, so
        # while calibrating you can see exactly where Stage 1 would fire.
        pg.setConfigOptions(antialias=False)  # cheap drawing at 30 FPS
        plots_row = QHBoxLayout()

        self.t_plot = pg.PlotWidget(title="T — trunk angle (deg)")
        self.t_plot.setFixedHeight(150)
        self.t_plot.setYRange(0, 120)
        self.t_plot.addItem(pg.InfiniteLine(
            pos=float(self.cfg.stage1.threshold_T_deg), angle=0,
            pen=pg.mkPen("orange", style=Qt.DashLine),
            label="threshold_T", labelOpts={"position": 0.05, "color": "orange"},
        ))
        self.t_curve = self.t_plot.plot(pen=pg.mkPen("y", width=2))
        self.t_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=1))
        self.t_plot.addItem(self.t_cursor)
        plots_row.addWidget(self.t_plot)

        self.v_plot = pg.PlotWidget(title="V — centroid velocity (torso/s, neg = down)")
        self.v_plot.setFixedHeight(150)
        self.v_plot.setYRange(-4, 4)
        self.v_plot.addItem(pg.InfiniteLine(
            pos=float(self.cfg.stage1.threshold_V), angle=0,
            pen=pg.mkPen("orange", style=Qt.DashLine),
            label="threshold_V", labelOpts={"position": 0.05, "color": "orange"},
        ))
        self.v_plot.addItem(pg.InfiniteLine(pos=0.0, angle=0,
                                            pen=pg.mkPen((120, 120, 120))))
        self.v_curve = self.v_plot.plot(pen=pg.mkPen("m", width=2))
        self.v_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=1))
        self.v_plot.addItem(self.v_cursor)
        plots_row.addWidget(self.v_plot)

        # Quantity P. Zero is the whole decision line here — the sign says
        # inside or outside the support — so it is drawn solid rather than as
        # a configurable threshold, because it is not configurable: it is
        # where §3.4's geometric condition changes.
        self.p_plot = pg.PlotWidget(title="P — COM vs support (torso, >0 = outside)")
        self.p_plot.setFixedHeight(150)
        self.p_plot.setYRange(-1.5, 1.5)
        self.p_plot.addItem(pg.InfiniteLine(
            pos=0.0, angle=0, pen=pg.mkPen("orange", width=2),
            label="outside", labelOpts={"position": 0.05, "color": "orange"},
        ))
        self.p_curve = self.p_plot.plot(pen=pg.mkPen("c", width=2))
        self.p_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=1))
        self.p_plot.addItem(self.p_cursor)
        plots_row.addWidget(self.p_plot)

        # Quantity I. The dashed line is the confirmation threshold W: while
        # calibrating, the curve climbing past it is the moment Stage 3 would
        # confirm a fall. It saws back to zero on every movement, which is the
        # quantity working, not a glitch.
        self.i_plot = pg.PlotWidget(title="I — immobility (s)")
        self.i_plot.setFixedHeight(150)
        self.i_plot.setYRange(0, max(2.0, float(self.cfg.stage3.threshold_W_seconds) * 1.5))
        self.i_plot.addItem(pg.InfiniteLine(
            pos=float(self.cfg.stage3.threshold_W_seconds), angle=0,
            pen=pg.mkPen("orange", style=Qt.DashLine),
            label="threshold_W", labelOpts={"position": 0.05, "color": "orange"},
        ))
        self.i_curve = self.i_plot.plot(pen=pg.mkPen("g", width=2))
        self.i_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=1))
        self.i_plot.addItem(self.i_cursor)
        plots_row.addWidget(self.i_plot)

        layout.addLayout(plots_row)

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
        layout.addLayout(playback_row)

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

    # -------------------------------------------------------------- sources --
    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str) -> None:
        """Open a clip by path (used by the file dialog and by --video)."""
        self._close_source()
        try:
            self.source = VideoFileSource(path)
        except (IOError, FileNotFoundError) as exc:
            self._set_status(str(exc), error=True)
            return
        self._start_session(Path(path).stem, metadata={
            "source": str(Path(path).resolve()),
            "source_kind": "video",
            "source_fps_declared": self.source.fps,
            "frame_count": self.source.frame_count,
        })
        self.seek_slider.setRange(0, max(0, self.source.frame_count - 1))
        self.playing = True
        self.play_btn.setText("Pause")
        self.timer.start(0)  # self-scheduling loop takes over from here
        self._set_status(
            f"Video: {Path(path).name} — {self.source.frame_count} frames "
            f"@ {self.source.fps:.1f} fps (declared). Logging to {self.logger.path.name}"
        )
        self._update_controls()

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
        self._plot_data.clear()
        self.t_curve.setData([], [])
        self.v_curve.setData([], [])
        self.p_curve.setData([], [])
        self.i_curve.setData([], [])

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
        self.load_video(str(path))

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
            "motivo": reason,
        }
        if self._label_logger is not None:
            self._label_logger.log(row)
        self._queue_rows.append(row)
        self._show_queue_result(self._queue_index, detected, truth)

    def _show_queue_result(self, row: int, detected: str, truth: str) -> None:
        self.queue_table.setItem(row, 2, QTableWidgetItem(detected))
        mark = "" if not truth else ("OK" if truth == detected else "X")
        item = QTableWidgetItem(mark)
        if mark == "X":
            item.setForeground(Qt.red)
        self.queue_table.setItem(row, 3, item)
        self.queue_progress.setValue(row + 1)
        scored = sum(1 for r in self._queue_rows if r["clase_verdad"])
        hits = sum(1 for r in self._queue_rows if r["acierto"] == 1)
        if scored:
            self.queue_score.setText(f"{hits}/{scored} ({100.0 * hits / scored:.0f}%)")

    def _on_queue_row_opened(self, row: int, _column: int) -> None:
        if self._queue_running or row >= len(self._queue):
            return
        self.load_video(str(self._queue[row]))

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
        print(f"\n{summary}\n-> {path}")
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
        self._seek_to(max(0, self.frame_index + delta))

    def _seek_to(self, frame_index: int) -> None:
        """Jump to a frame and process it, declaring the discontinuity.

        The pipeline cannot tell a jump from ordinary playback — the
        timestamps of two frames a few positions apart look like a normal
        interval — so the caller has to say so. Without this the velocity
        window would silently mix frames from two places in the recording.
        """
        self.source.seek(frame_index)
        if self.pipeline is not None:
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
        slot_ms = 1000.0 / max(1.0, self.source.fps)
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
                if self._queue_running:
                    self._label_current_clip()
                # The record is complete the moment the clip ends, so it goes
                # to disk now rather than waiting for a Close that may never
                # come. The source stays open: reviewing is the whole point of
                # stopping here, and reviewed frames are not logged anyway.
                self._finalise_records("End of video.")
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
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(index)
            self.seek_slider.blockSignals(False)
            total = self.source.frame_count / self.source.fps
            self.time_label.setText(f"{_format_time(timestamp)} / {_format_time(total)}")

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
        """Store this frame's T and V and redraw the curves.

        Data is keyed by frame index so scrubbing back over already-seen
        frames overwrites instead of duplicating; the curve is drawn in
        time order regardless of the order frames were visited. In live
        mode only the most recent window is kept so memory stays bounded.
        """
        if not self.chk_curves.isChecked():
            return
        t_deg = result.quantities.get("T_deg", float("nan"))
        v_tps = result.quantities.get("V_tps", float("nan"))
        p_off = result.quantities.get("P_offset", float("nan"))
        i_still = result.quantities.get("I_still_s", float("nan"))
        self._plot_data[index] = (timestamp, t_deg, v_tps, p_off, i_still)

        if self.source is not None and self.source.is_live and len(self._plot_data) > 900:
            # Live mode: keep ~30 s at 30 FPS; drop the oldest entries.
            for old_key in sorted(self._plot_data)[: len(self._plot_data) - 900]:
                del self._plot_data[old_key]

        data = sorted(self._plot_data.values())  # sorted by timestamp
        times = np.array([d[0] for d in data])
        t_vals = np.array([d[1] for d in data])
        v_vals = np.array([d[2] for d in data])
        p_vals = np.array([d[3] for d in data])
        i_vals = np.array([d[4] for d in data])
        # pyqtgraph skips NaN gaps with connect="finite": the curve breaks
        # where there was no detection instead of drawing a fake bridge. For
        # P the break is doubly meaningful — it marks the stretches where the
        # feet were not visible and the quantity genuinely does not exist.
        self.t_curve.setData(times, t_vals, connect="finite")
        self.v_curve.setData(times, v_vals, connect="finite")
        self.p_curve.setData(times, p_vals, connect="finite")
        self.i_curve.setData(times, i_vals, connect="finite")
        self.t_cursor.setPos(timestamp)
        self.v_cursor.setPos(timestamp)
        self.p_cursor.setPos(timestamp)
        self.i_cursor.setPos(timestamp)

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
        if self.pipeline is not None and self.pipeline.alerts.should_dispatch(ev.verdict):
            return                      # the alert sink already showed it
        text = (f"event at t={ev.timestamp:.2f}s  ->  {ev.verdict}"
                + (f" [{ev.severity}]" if ev.severity else "")
                + f"   (T={ev.t_deg:.1f} deg, V={ev.v_tps:+.2f} torso/s)")
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
            # A vertical marker on every curve, so the moment of the decision
            # is readable against all four quantities at once.
            for plot in (self.t_plot, self.v_plot, self.p_plot, self.i_plot):
                plot.addItem(pg.InfiniteLine(
                    pos=ev.timestamp, angle=90,
                    pen=pg.mkPen("r", width=2, style=Qt.DashLine)))
        now = time.monotonic()
        if self.pipeline is not None and now - self._last_perf_shown > 1.0:
            self._last_perf_shown = now
            self._set_status(self.pipeline.timings.summary())
        self.stage_label.setText(text)
        self.stage_label.setStyleSheet(
            "font-weight: bold; color: #ff5555;" if result.event is not None
            else "font-weight: bold;")

    def _on_curves_toggled(self, checked: bool) -> None:
        self.t_plot.setVisible(checked)
        self.v_plot.setVisible(checked)
        self.p_plot.setVisible(checked)
        self.i_plot.setVisible(checked)

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
