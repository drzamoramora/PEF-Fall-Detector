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

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..audit_log import AuditLogger
from ..config import Config
from ..overlay import draw_overlay
from ..pipeline import FramePipeline
from ..sources import CameraSource, FrameSource, VideoFileSource


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
        self.playing = False
        self.frame_index = 0
        self._user_scrubbing = False
        self._last_raw_frame = None
        # Plot data keyed by frame index so scrubbing overwrites cleanly
        # instead of appending duplicates: index -> (t, T_deg, V_tps).
        self._plot_data: dict[int, tuple[float, float, float]] = {}

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
        self.pipeline = FramePipeline(self.cfg)
        meta = dict(metadata or {})
        meta["config"] = self.cfg.as_dict()
        self.logger = AuditLogger(self.cfg.logging.output_dir, source_name, metadata=meta)
        self.frame_index = 0
        self._plot_data.clear()
        self.t_curve.setData([], [])
        self.v_curve.setData([], [])

    def _close_source(self) -> None:
        self.timer.stop()
        self.playing = False
        if self.source is not None:
            self.source.release()
            self.source = None
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None
        if self.logger is not None:
            rows = self.logger.rows_written
            self.logger.close()
            self._set_status(f"Source closed. {rows} frames logged to {self.logger.path}")
            self.logger = None
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
                self._set_status("End of video. Use the seek bar or ◀/▶ to review.")
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

        shown = draw_overlay(
            frame,
            result.pose,
            visibility_threshold=self.cfg.pose.visibility_threshold,
            show_skeleton=self.chk_skeleton.isChecked(),
            show_trunk=self.chk_trunk.isChecked(),
            show_centroid=self.chk_centroid.isChecked(),
            centroid_px=result.centroid_px,
            hud_lines=result.hud_lines,
        )
        self._display(shown)
        self._update_plots(index, timestamp, result)

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
        self._plot_data[index] = (timestamp, t_deg, v_tps)

        if self.source is not None and self.source.is_live and len(self._plot_data) > 900:
            # Live mode: keep ~30 s at 30 FPS; drop the oldest entries.
            for old_key in sorted(self._plot_data)[: len(self._plot_data) - 900]:
                del self._plot_data[old_key]

        data = sorted(self._plot_data.values())  # sorted by timestamp
        times = np.array([d[0] for d in data])
        t_vals = np.array([d[1] for d in data])
        v_vals = np.array([d[2] for d in data])
        # pyqtgraph skips NaN gaps with connect="finite": the curve breaks
        # where there was no detection instead of drawing a fake bridge.
        self.t_curve.setData(times, t_vals, connect="finite")
        self.v_curve.setData(times, v_vals, connect="finite")
        self.t_cursor.setPos(timestamp)
        self.v_cursor.setPos(timestamp)

    def _on_curves_toggled(self, checked: bool) -> None:
        self.t_plot.setVisible(checked)
        self.v_plot.setVisible(checked)

    # -------------------------------------------------------------- helpers --
    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            "color: #cc3333; font-weight: bold;" if error else "color: #2e7d32;"
        )

    def _update_controls(self) -> None:
        has_video = self.source is not None and not self.source.is_live
        self.play_btn.setEnabled(has_video)
        self.step_back_btn.setEnabled(has_video)
        self.step_fwd_btn.setEnabled(has_video)
        self.seek_slider.setEnabled(has_video)

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
