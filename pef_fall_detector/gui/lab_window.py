"""PEF-Lab main window.

A visualization workbench over the source-agnostic core: open a recorded
clip (e.g. from PEF-Video-Tool's ``recordings/``) or a live camera, watch
the MediaPipe skeleton and Step-0 anchors drawn in real time, and get the
run's records written automatically.

The window is strictly a *viewer*: every computation happens in the core
modules (``pose_frontend``, ``quantities``, ``state_machine``), which also
power the headless CLI used for batch evaluation and the Raspberry Pi
deployment. This module is the flow — sources, playback, the per-frame
loop, frozen review and the queue pass; what it shows lives beside it:
:mod:`.plots` (curves and timeline), :mod:`.queue_panel` (the Queue tab),
:mod:`.widgets` (the video view, the View tab, small building blocks).

Video-mode controls: Play/Pause, single-frame step (◀ / ▶), and a seek bar —
falls last 10–15 frames, so being able to stop on the exact frame where a
stage fired is the whole point of this tool.

Records schema 2 (0.3.0.dev0): every pass writes into a run folder
(``run_folder``), as ``main.py`` does. A clip opened on its own or the camera
is a run of its own; a queue pass is one run with all its clips, its labels
and its metrics, shown in the Results tab. "Open Run…" loads a run recorded
earlier (here or with ``pef-run``) and shows any of its clips frozen, rebuilt
from its landmarks without MediaPipe; "Save Reviews" then adds the human
column to that run. The status line names the run being written, and
"Export Figure" saves the video and the curves as a PNG into its folder.
"""

from __future__ import annotations

import csv
import html
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import pose_frontend, run_metrics
from ..alerts import Alert
from ..audit_log import AuditLogger
from ..config import Config
from ..dataset import LABEL_FIELDS, VIDEO_SUFFIXES, format_confusion, label_row, video_files
from ..overlay import draw_overlay
from ..pipeline import FramePipeline
from ..records import RunRecords
from ..run_folder import (
    CAMERA,
    DATASET,
    LABELS_FILE,
    REVIEWED_LABELS_FILE,
    VIDEO,
    RunFolder,
    file_sha256,
)
from ..sources import CameraSource, FrameSource, VideoFileSource
from . import theme
from .plots import SEVERITY_PEN, PlotPanel
from .queue_panel import QueuePanel, metrics_text
from .review_cache import ClipAnalysis, PassRecorder, ReviewCache, config_key
from .widgets import VideoView, ViewPanel, card, fit_to_screen, format_time, separator

#: Playback speeds. Frozen review does not run MediaPipe, so 2x is
#: reachable; in an analysis pass the processing sets the limit, and the
#: speed can only slow it down.
SPEEDS = (0.25, 0.5, 0.75, 1.0, 2.0)

PLAY_TEXT, PAUSE_TEXT = "Play", "Pause"
CAMERA_START, CAMERA_STOP = "Start Camera", "Stop Camera"


def _p_side(p_offset: float | None) -> bool | None:
    """Whether the COM projects outside the support — or None if unknown.

    Kept as three states rather than a boolean: a NaN P (feet occluded) must
    not be drawn as "inside", which is what any ``> 0`` test on NaN would
    silently produce.
    """
    if p_offset is None or math.isnan(p_offset):
        return None
    return p_offset > 0.0


class LabWindow(QWidget):
    """PEF-Lab: source selection + overlay display + audit logging."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.setWindowTitle("PEF-Lab — Fall Detection Workbench")
        self.setMinimumSize(980, 640)
        fit_to_screen(self)

        self.cfg = cfg
        self.source: FrameSource | None = None
        self.pipeline: FramePipeline | None = None
        #: The records of the pass in progress (records.RunRecords); None
        #: once they are closed, or while a finished pass is being reviewed.
        self.records: RunRecords | None = None
        #: The events list of the pass, kept to close its records after the
        #: pipeline itself is gone.
        self._run_events: list = []
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
        #: Clip open for review, or None if none yet. It anchors the
        #: Alt+← / Alt+→ arrows; kept apart from ``_queue_index`` because
        #: reviewing and running the pass are two different walks over
        #: the same queue.
        self._review_index: int | None = None
        self._queue_rows: list[dict] = []
        self._queue_index = 0
        self._queue_running = False
        self._label_logger: AuditLogger | None = None
        # Run folders (records schema 2). ``_session_run``: the run of a clip
        # opened on its own or of the camera, closed with its source.
        # ``_queue_run``: the run of the queue pass in progress. ``_last_run``:
        # the run the Results tab, "Save Reviews" and "Export Figure" refer to
        # (the last queue pass, or the run opened with "Open Run…").
        self._session_run: RunFolder | None = None
        self._queue_run: RunFolder | None = None
        self._last_run: RunFolder | None = None
        #: A run opened with "Open Run…": its records by clip stem and its config.
        self._opened_run: SimpleNamespace | None = None
        self._clip_sha = ""
        # Frozen analysis (see review_cache.py): what a full pass computed,
        # to review without recomputing or writing records.
        self._cache = ReviewCache()
        self._cfg_key = config_key(cfg.as_dict())
        self._recorder: PassRecorder | None = None
        self._frozen: ClipAnalysis | None = None
        self._current_path: str | None = None
        self._speed = 1.0

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
        """Window layout. Appearance only: no detection logic lives here.

        Sized for a laptop screen (checked at 1280x760): no block has a fixed
        height that adds up past that, and the curves sit in a scroll area,
        so they can never push the controls off screen again (what happened
        on the author's Mac on 25/09).

            header : sources (open video, add folder, camera, re-analyze, close)
            top    : video + playback + timeline  |  stage, verdict, queue / view
            bottom : curves (Paper / Experimental), full width, resizable
            footer : status line
        """
        self.setStyleSheet(theme.QSS)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(8)
        style = self.style()

        # --- Header: where the frames come from ----------------------------
        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 8, 10, 8)
        hl.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        app_title = QLabel("PEF-Lab")
        app_title.setObjectName("appTitle")
        app_sub = QLabel("Physics-informed fall detection · workbench")
        app_sub.setObjectName("appSubtitle")
        titles.addWidget(app_title)
        titles.addWidget(app_sub)
        hl.addLayout(titles)
        hl.addStretch(1)

        self.open_video_btn = QPushButton("Open Video…")
        self.open_video_btn.setObjectName("primary")
        self.open_video_btn.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
        self.open_video_btn.setToolTip("Analyze a recorded clip (.mp4, .avi, .mov, .mkv)")
        self.open_video_btn.clicked.connect(self._on_open_video)
        hl.addWidget(self.open_video_btn)
        self.queue_add_btn = QPushButton("Add Folder…")
        self.queue_add_btn.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        self.queue_add_btn.setToolTip("Load a folder of clips into the queue (see the Queue tab)")
        self.queue_add_btn.clicked.connect(self._on_queue_add_folder)
        hl.addWidget(self.queue_add_btn)
        self.open_run_btn = QPushButton("Open Run…")
        self.open_run_btn.setIcon(style.standardIcon(QStyle.SP_FileDialogContentsView))
        self.open_run_btn.setToolTip("Review a run recorded earlier: its clips frozen, "
                                     "rebuilt from its landmarks, and its metrics")
        self.open_run_btn.clicked.connect(self._on_open_run)
        hl.addWidget(self.open_run_btn)
        hl.addWidget(separator())
        hl.addWidget(QLabel("Camera"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.setToolTip("Camera index")
        hl.addWidget(self.camera_spin)
        self.camera_btn = QPushButton(CAMERA_START)
        self.camera_btn.setIcon(style.standardIcon(QStyle.SP_ComputerIcon))
        self.camera_btn.clicked.connect(self._on_toggle_camera)
        hl.addWidget(self.camera_btn)
        hl.addWidget(separator())
        # The only way to analyze an already frozen clip again in this
        # session: an explicit decision, which writes a new record.
        self.reanalyze_btn = QPushButton("Re-analyze")
        self.reanalyze_btn.setIcon(style.standardIcon(QStyle.SP_BrowserReload))
        self.reanalyze_btn.setToolTip("Discard the frozen analysis and run a new pass, "
                                      "with a new record")
        self.reanalyze_btn.clicked.connect(self._on_reanalyze)
        hl.addWidget(self.reanalyze_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.setObjectName("danger")
        self.close_btn.setIcon(style.standardIcon(QStyle.SP_DialogCloseButton))
        self.close_btn.setToolTip("Close the current source")
        self.close_btn.clicked.connect(self._close_source)
        hl.addWidget(self.close_btn)
        root.addWidget(header)

        # --- Video and the state readouts -----------------------------------
        self.video_label = VideoView("Open a video or start the camera to begin.")
        self.video_label.setObjectName("videoView")

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLine")

        # The alarm. A confirmed fall must be impossible to miss while
        # testing with the camera — a line of grey text is not an alarm.
        # Hidden until one fires, so its presence alone carries meaning.
        self.alert_banner = QLabel("")
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setWordWrap(True)
        self.alert_banner.setVisible(False)

        # Funnel position: a coloured chip for the stage, and the detail line
        # (events so far, the last firing) under it.
        self.stage_chip = QLabel(theme.STAGE_NAMES["MONITORING"])
        self.stage_chip.setStyleSheet(theme.chip_style(theme.STAGE_COLOURS["MONITORING"]))
        self.stage_label = QLabel("Stage: MONITORING   |   events: 0")
        self.stage_label.setWordWrap(True)
        self.stage_label.setObjectName("muted")

        # --- Curves and timeline (plots.PlotPanel) ----------------------------
        self.plots = PlotPanel(self.cfg)

        # The verdict and its reason: the rule that decided, with its numbers.
        self.verdict_label = QLabel("—")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setTextFormat(Qt.RichText)
        self.verdict_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.verdict_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # --- Playback bar (video only) ---------------------------------------
        playbar = QFrame()
        playbar.setObjectName("playbar")
        pl = QHBoxLayout(playbar)
        pl.setContentsMargins(10, 6, 10, 6)
        pl.setSpacing(6)
        self.play_btn = QPushButton(PLAY_TEXT)
        self.play_btn.setObjectName("primary")
        self.play_btn.setMinimumWidth(96)
        self.play_btn.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setToolTip("Play / pause  (Space)")
        self.play_btn.clicked.connect(self._on_play_pause)
        pl.addWidget(self.play_btn)
        self.step_back_btn = QPushButton()
        self.step_back_btn.setIcon(style.standardIcon(QStyle.SP_MediaSeekBackward))
        self.step_back_btn.setToolTip("Previous frame  (←)   ·   Shift+← one second back")
        self.step_back_btn.clicked.connect(lambda: self._step(-1))
        pl.addWidget(self.step_back_btn)
        self.step_fwd_btn = QPushButton()
        self.step_fwd_btn.setIcon(style.standardIcon(QStyle.SP_MediaSeekForward))
        self.step_fwd_btn.setToolTip("Next frame  (→)   ·   Shift+→ one second forward")
        self.step_fwd_btn.clicked.connect(lambda: self._step(+1))
        pl.addWidget(self.step_fwd_btn)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.valueChanged.connect(self._on_slider_moved)
        pl.addWidget(self.seek_slider, stretch=1)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("timeLabel")
        pl.addWidget(self.time_label)
        pl.addSpacing(6)
        speed_caption = QLabel("Speed")
        speed_caption.setObjectName("muted")
        pl.addWidget(speed_caption)
        self.speed_box = QComboBox()
        for sp in SPEEDS:
            self.speed_box.addItem(f"{sp:g}x", sp)
        self.speed_box.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_box.currentIndexChanged.connect(
            lambda i: setattr(self, "_speed", float(self.speed_box.itemData(i))))
        pl.addWidget(self.speed_box)
        self.export_btn = QPushButton("Export Figure")
        self.export_btn.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        self.export_btn.setToolTip("Save the video and the curves, as they are on screen, "
                                   "as a PNG in the run's figures/ folder")
        self.export_btn.clicked.connect(self.export_figure)
        pl.addWidget(self.export_btn)

        # Shortcuts: space plays/pauses, arrows one frame, Shift+arrows one
        # second. Alt+arrows stay "change clip" in the queue.
        for keys, slot in (("Space", self._on_play_pause),
                           ("Right", lambda: self._step(+1)),
                           ("Left", lambda: self._step(-1)),
                           ("Shift+Right", lambda: self._step_seconds(+1.0)),
                           ("Shift+Left", lambda: self._step_seconds(-1.0))):
            sc = QShortcut(QKeySequence(keys), self)
            sc.activated.connect(slot)

        # --- Overlay toggles (View tab) ---------------------------------------
        self.view_panel = ViewPanel()
        self.view_panel.curves.toggled.connect(self._on_curves_toggled)

        # --- Assemble -------------------------------------------------------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        ll.addWidget(self.alert_banner)
        ll.addWidget(self.video_label, stretch=1)
        ll.addWidget(playbar)
        ll.addWidget(self.plots.timeline)

        stage_card, sc_l = card("Stage")
        chip_row = QHBoxLayout()
        chip_row.addWidget(self.stage_chip)
        chip_row.addStretch(1)
        sc_l.addLayout(chip_row)
        sc_l.addWidget(self.stage_label)

        verdict_card, vc_l = card("Verdict")
        verdict_scroll = QScrollArea()
        verdict_scroll.setWidgetResizable(True)
        verdict_scroll.setFrameShape(QFrame.NoFrame)
        verdict_scroll.setWidget(self.verdict_label)
        verdict_scroll.setMinimumHeight(70)
        vc_l.addWidget(verdict_scroll)

        self.side_tabs = QTabWidget()
        queue_page = QueuePanel(style)
        self.queue_panel = queue_page
        queue_page.start_btn.clicked.connect(self._on_queue_start)
        queue_page.stop_btn.clicked.connect(self._on_queue_stop)
        queue_page.save_btn.clicked.connect(self._on_queue_save_reviews)
        queue_page.clear_btn.clicked.connect(self._on_queue_clear)
        queue_page.table.cellDoubleClicked.connect(self._on_queue_row_opened)
        queue_page.prev_btn.clicked.connect(lambda: self._review_step(-1))
        queue_page.next_btn.clicked.connect(lambda: self._review_step(+1))
        # Both side tabs scroll, so the side column never sets the window's
        # minimum height; the video column does.
        queue_scroll = QScrollArea()
        queue_scroll.setWidgetResizable(True)
        queue_scroll.setFrameShape(QFrame.NoFrame)
        queue_scroll.setWidget(queue_page)
        self.side_tabs.addTab(queue_scroll, "Queue")
        view_scroll = QScrollArea()
        view_scroll.setWidgetResizable(True)
        view_scroll.setFrameShape(QFrame.NoFrame)
        view_scroll.setWidget(self.view_panel)
        self.side_tabs.addTab(view_scroll, "View")
        # Every number of the last queue pass or of the opened run
        # (run_metrics): the same text as metrics.txt.
        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.results_text.setObjectName("results")
        self.results_text.setStyleSheet("font-family: Menlo, Consolas, 'DejaVu Sans Mono', "
                                        "monospace; font-size: 11px;")
        self.results_text.setPlaceholderText(
            "The metrics of a queue pass appear here when it ends, "
            "and those of a run opened with «Open Run…».")
        self.side_tabs.addTab(self.results_text, "Results")

        side = QWidget()
        side.setMinimumWidth(320)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        sl.addWidget(stage_card)
        sl.addWidget(verdict_card, stretch=2)
        sl.addWidget(self.side_tabs, stretch=3)

        self.top_split = QSplitter(Qt.Horizontal)
        self.top_split.setChildrenCollapsible(False)
        self.top_split.addWidget(left)
        self.top_split.addWidget(side)
        self.top_split.setStretchFactor(0, 3)
        self.top_split.setStretchFactor(1, 2)
        self.top_split.setSizes([900, 460])

        self.curves_panel = QWidget()
        cl = QVBoxLayout(self.curves_panel)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.plots.tabs)

        self.main_split = QSplitter(Qt.Vertical)
        self.main_split.setChildrenCollapsible(False)
        self.main_split.addWidget(self.top_split)
        self.main_split.addWidget(self.curves_panel)
        self.main_split.setStretchFactor(0, 3)
        self.main_split.setStretchFactor(1, 2)
        self.main_split.setSizes([450, 340])

        root.addWidget(self.main_split, stretch=1)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addWidget(self.status_label, stretch=1)
        # Which run the records go to: what a screenshot needs to say.
        self.run_label = QLabel("")
        self.run_label.setObjectName("muted")
        self.run_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        footer.addWidget(self.run_label)
        root.addLayout(footer)

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
        except OSError as exc:                  # FileNotFoundError is one too
            self._set_status(str(exc), error=True)
            return
        self._current_path = str(path)
        self.seek_slider.setRange(0, max(0, self.source.frame_count - 1))
        cached = None if reanalyze else self._cache.get(path, self._cfg_key)
        from_run = None if reanalyze or cached is not None else self._opened_analysis(path)
        if reanalyze:
            self._cache.drop(path, self._cfg_key)
        if cached is not None or from_run is not None:
            self._enter_frozen(cached or from_run)
            self._seek_to(0)
            where = ("this session's analysis" if cached is not None
                     else f"rebuilt from the landmarks of {self._opened_run.path.name}")
            self._set_status(
                f"Frozen: {Path(path).name} — {where}, not recomputed and writing no "
                f"records. Space to play; \"Re-analyze\" for a new pass.")
            self._update_controls()
            return
        self._clip_sha = file_sha256(path)
        self._start_session(Path(path).stem, kind=VIDEO, seen_at=str(Path(path).resolve().parent),
                            metadata={
            "source": Path(path).name,
            "source_kind": "video",
            "source_fps_declared": self.source.fps,
            "frame_count": self.source.frame_count,
            "video_sha256": self._clip_sha,
            "decoder_backend": self.source.backend,
        })
        self._recorder = PassRecorder(path, self._cfg_key, self.source.fps)
        self.playing = True
        self._set_play_state(True)
        self.timer.start(0)  # self-scheduling loop takes over from here
        self._set_status(
            f"Video: {Path(path).name} — {self.source.frame_count} frames "
            f"@ {self.source.fps:.1f} fps (declared). Recording into {self.records.run.path.name}"
        )
        self._update_controls()

    def _on_reanalyze(self) -> None:
        """Drop what is frozen and run a new pass, with new records."""
        if self._current_path is None or self._queue_running:
            return
        self.load_video(self._current_path, reanalyze=True)

    # --------------------------------------------------------------- frozen --
    def _enter_frozen(self, analysis: ClipAnalysis) -> None:
        """Show a saved analysis: full curves, marks and verdict."""
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None     # nothing can recompute from here on
        self._recorder = None
        self._frozen = analysis
        self.playing = False
        self._set_play_state(False)
        self.verdict_label.setText("—")
        self.plots.show_analysis(analysis)
        self._show_verdict(analysis.events)

    def _show_frozen_frame(self, frame, index: int, timestamp: float) -> None:
        """Draw what the pass saw on this frame. No pipeline."""
        res = self._frozen.frames.get(index)
        if res is None:          # a frame the pass never reached
            self._display(frame)
        else:
            self._display(draw_overlay(
                frame, res.pose,
                visibility_threshold=self.cfg.pose.visibility_threshold,
                show_skeleton=self.view_panel.skeleton.isChecked(),
                show_trunk=self.view_panel.trunk.isChecked(),
                show_centroid=self.view_panel.centroid.isChecked(),
                centroid_px=res.centroid_px,
                show_support=self.view_panel.support.isChecked(),
                com_px=res.com_px,
                support_hull_px=res.support_hull_px,
                com_outside=_p_side(res.quantities.get("P_offset")),
                hud_lines=["[FROZEN] not recomputed"] + list(res.hud_lines),
            ))
            n = len(self._frozen.events)
            self.stage_label.setText(
                f"[FROZEN]  stage on this frame: {res.stage}   |   events: {n}")
            self._set_stage_chip(res.stage, frozen=True)
        self.plots.move_cursors(timestamp)

    def _on_toggle_camera(self) -> None:
        if self.source is not None and self.source.is_live:
            self._close_source()
            return
        self._close_source()
        try:
            self.source = CameraSource(self.camera_spin.value())
        except OSError as exc:
            self._set_status(str(exc), error=True)
            return
        self._clip_sha = ""
        self._start_session(f"camera{self.camera_spin.value()}", kind=CAMERA, metadata={
            "source": f"camera index {self.camera_spin.value()}",
            "source_kind": "live camera",
            "source_fps_declared": self.source.fps,
        })
        self.playing = True
        self.camera_btn.setText(CAMERA_STOP)
        self.timer.start(0)  # self-scheduling loop; pace comes from the source
        self._set_status(
            f"Live camera {self.camera_spin.value()} — recording into "
            f"{self.records.run.path.name}"
        )
        self._update_controls()

    def _start_session(self, source_name: str, metadata: dict | None = None,
                       kind: str = VIDEO, seen_at: str = "") -> None:
        """Create the processing pipeline and the records for a new source.

        In a queue pass the clip's records go into the queue's run; otherwise
        the source opens a run of its own, closed with it.
        """
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
        self.pipeline.hash_frames = True
        self.pipeline.alerts.add_sink(self._on_alert)
        self.alert_banner.setVisible(False)
        # PEF-Lab is the laboratory, not the deployed device: it always
        # records everything (research mode), whatever logging.mode says.
        # logging.mode governs main.py, which is what runs on the Pi (§3.6).
        run = self._queue_run
        if run is None:
            run = RunFolder.new(self.cfg.logging.output_dir, source_name, kind=kind,
                                seen_at=seen_at, config=self.cfg.as_dict(), command="PEF-Lab")
            self._session_run = run
        self.records = RunRecords(run, source_name, metadata=dict(metadata or {}),
                                  research=True)
        self._show_run(run)
        self._run_events = self.pipeline.machine.events
        self.frame_index = 0
        self.plots.clear()
        self.verdict_label.setText("— analyzing…")

    def _log_event(self, ev) -> None:
        """Write one resolved event, and any earlier one a loss abandoned.

        Called when its verdict lands, not when it was raised: the verdict is
        the point of the record."""
        if self.records is None or self.pipeline is None:
            return
        self.records.event_settled(self.pipeline.machine.events, ev)

    # ----------------------------------------------------------- queue flow --
    def _on_queue_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dataset folder")
        if not folder:
            return
        clips = video_files(folder)
        if not clips:
            self._set_status(f"No videos in {folder}", error=True)
            return
        self._queue.extend(clips)
        self.queue_panel.add_clips(clips)
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
        self._opened_run = None              # a new pass is not a review of an old run
        folders = {Path(p).resolve().parent for p in self._queue}
        folder = folders.pop() if len(folders) == 1 else None
        self._queue_run = RunFolder.new(
            self.cfg.logging.output_dir, folder.name if folder else "queue", kind=DATASET,
            seen_at=str(folder) if folder else "", config=self.cfg.as_dict(),
            command="PEF-Lab queue")
        self._label_logger = AuditLogger(self._queue_run.path, "dataset-labels",
                                         fields=LABEL_FIELDS, filename=LABELS_FILE,
                                         sidecar=False, flush_each_row=True)
        self._show_run(self._queue_run)
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
        self.queue_panel.clear()
        self._update_controls()

    def _on_queue_save_reviews(self) -> None:
        """Write a fresh labels file with the human column filled in.

        A separate file, not an edit of the one the pass produced: the run's
        own output is a record of what the system decided, and rewriting it
        in place would destroy the only evidence of the disagreement being
        corrected. It goes into the run the labels belong to (the last queue
        pass, or the run opened with "Open Run…"); each save is a new file
        (``dataset-labels-reviewed.csv``, ``-2``, ...), so the sequence of
        review passes stays readable, and ``metrics-reviewed`` follows the last.
        """
        if self._queue_running or not self._queue_rows or self._last_run is None:
            return
        run = self._last_run
        name, n = REVIEWED_LABELS_FILE, 2
        while (run.path / name).exists():
            name, n = f"{Path(REVIEWED_LABELS_FILE).stem}-{n}.csv", n + 1
        logger = AuditLogger(run.path, "dataset-labels-reviewed", fields=LABEL_FIELDS,
                             filename=name, sidecar=False, flush_each_row=True)
        for row, record in enumerate(self._queue_rows):
            record["reviewed_class"] = self.queue_panel.reviewed(row) or ""
            logger.log({k: record.get(k, "") for k in LABEL_FIELDS})
        logger.close()
        text = run_metrics.for_run(run, self._queue_rows, reviewed=True)
        self._show_results(text)
        self._set_status(f"Reviews saved -> {run.path.name}/{name} "
                         f"(and metrics-reviewed.txt)")

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
        self.queue_panel.select(self._queue_index)
        # Starting the queue is running again, on purpose: never frozen.
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
        total = (self.source.frame_count / self.source.fps
                 if self.source is not None and self.source.fps else 0.0)
        row = label_row(path, self.pipeline.machine.events, self._logged_frames,
                        self._reliable_frames, total, datetime.now(),
                        dispatch=self.cfg.alerts.dispatch_verdicts, video_sha256=self._clip_sha)
        detected, truth = row["detected_class"], row["true_class"]
        if self._label_logger is not None:
            self._label_logger.log(row)
        self._queue_rows.append(row)
        self._show_queue_result(self._queue_index, detected, truth)
        item = self.queue_panel.table.item(self._queue_index, 2)
        if item is not None:
            item.setToolTip(row["reason"])
        return row

    def _show_queue_result(self, row: int, detected: str, truth: str) -> None:
        self.queue_panel.show_result(row, detected, truth, metrics_text(self._queue_rows))

    def _on_queue_row_opened(self, row: int, _column: int) -> None:
        if self._queue_running or row >= len(self._queue):
            return
        self._open_for_review(row)

    def _open_for_review(self, row: int) -> None:
        """Load queue clip ``row`` to look at it, outside the pass.

        It is the usual playback path; all it adds is remembering which
        clip it is on, so the arrows know where to move from. It does not
        touch the table: the pass results stay there, which is what one is
        comparing against while looking.
        """
        if not (0 <= row < len(self._queue)):
            return
        self._review_index = row
        self.queue_panel.select(row)
        self.load_video(str(self._queue[row]))
        self._update_review_label()

    def _review_step(self, delta: int) -> None:
        """Move to the previous or next clip of the queue.

        With no clip open yet, the first arrow enters at one end instead of
        doing nothing: arriving at the folder and not knowing where to
        start is the most common case, not an error.

        It stops at the ends instead of wrapping around. With four
        subjects per situation, going back to the first without notice is
        easily mistaken for "nothing happened", and the point here is to
        compare different clips.
        """
        if self._queue_running or not self._queue:
            return
        if self._review_index is None:
            self._open_for_review(0 if delta > 0 else len(self._queue) - 1)
            return
        target = self._review_index + delta
        if not (0 <= target < len(self._queue)):
            end = "first" if delta < 0 else "last"
            self._set_status(f"Already at the {end} clip of the queue.")
            return
        self._open_for_review(target)

    def _update_review_label(self) -> None:
        """The counter next to the arrows: which clip it is on, and of how many."""
        if not self._queue:
            self.queue_panel.review_label.setText("")
            return
        if self._review_index is None:
            self.queue_panel.review_label.setText(
                f"{len(self._queue)} clips queued — "
                f"Alt+← / Alt+→ to browse them")
            return
        name = self._queue[self._review_index].name
        self.queue_panel.review_label.setText(
            f"clip {self._review_index + 1} of {len(self._queue)}   ·   {name}")

    def _close_label_record(self, reason: str) -> None:
        """Finish the labels file, folding in any human corrections, and the run.

        The run's metrics (``metrics.csv``, ``metrics.txt``) are written and
        shown in the Results tab, and printed as ``pef-run`` prints them.
        """
        if self._label_logger is None:
            return
        for row, record in enumerate(self._queue_rows):
            reviewed = self.queue_panel.reviewed(row)
            if reviewed:
                record["reviewed_class"] = reviewed
        self._label_logger.close()
        self._label_logger = None
        run, self._queue_run = self._queue_run, None
        summary = format_confusion(self._queue_rows)
        first = summary.splitlines()[-1]
        if run is None:
            return
        finished = reason == "Queue finished."
        run.close("finished" if finished else "stopped")
        self._last_run = run
        text = run_metrics.for_run(run, self._queue_rows)
        print("\n" + text)
        print(f"-> {run.path}")
        self._show_results(text)
        self._set_status(f"{reason} {len(self._queue_rows)} clips — {first} -> {run.path.name}")

    def _finalise_records(self, reason: str, complete: bool = False) -> None:
        """Close the records and write their metadata sidecars.

        Separate from :meth:`_close_source` because the two events are not the
        same: a video that reaches its end has produced a COMPLETE record and
        should have it on disk, but the source stays open so the run can be
        reviewed with the seek bar. Before this existed, playing a clip to the
        end and then quitting left the rows in Python's buffer — measured on
        this project's own material, a 0-byte event file that had actually
        recorded a confirmed fall.

        The records are dropped afterwards. That is deliberate: AuditLogger
        opens in "w" mode, so continuing to write to a closed record would
        truncate it. It now raises instead, and this keeps the window from
        ever reaching that. Closing also writes the events no verdict ever
        reached (abandoned by a detection loss), which PEF-Lab used to leave
        out of its record.
        """
        if self.records is None:
            self._close_session_run(complete)
            return
        path, rows = self.records.path, self.records.frames_written
        self.records.close(self._run_events, complete=complete)
        events = self.records.events_written
        self.records = None
        self._close_session_run(complete)
        if path is not None:
            self._set_status(f"{reason} {rows} frames, {events} events -> {path}")
        else:
            self._set_status(reason)

    def _close_session_run(self, complete: bool) -> None:
        """Close the run of a clip opened on its own, or of the camera."""
        run, self._session_run = self._session_run, None
        if run is not None:
            run.close("finished" if complete else "stopped")
            self._last_run = run

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
        self.camera_btn.setText(CAMERA_START)
        self._set_play_state(False)
        self.seek_slider.setRange(0, 0)
        self._update_controls()

    # ------------------------------------------------------------- playback --
    def _on_play_pause(self) -> None:
        if self.source is None or self.source.is_live:
            return
        self.playing = not self.playing
        self._set_play_state(self.playing)
        if self.playing and self._frozen is not None:
            # At the end of the clip, play starts over.
            if self.frame_index >= self.source.frame_count - 1:
                self.source.seek(0)
            self.timer.start(0)
            return
        if self.playing:
            if self.records is None:
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
        self._set_play_state(False)
        last = max(0, self.source.frame_count - 1)
        self._seek_to(min(last, max(0, self.frame_index + delta)))

    def _step_seconds(self, seconds: float) -> None:
        """Move one second (Shift+arrow): the scale of a fall."""
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
        self._set_play_state(False)

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
        t_read = time.perf_counter()
        ok, frame, timestamp = self.source.read()
        decode_ms = 1000.0 * (time.perf_counter() - t_read)
        if self._frozen is not None:
            # Frozen review: what the pass saw is shown. Nothing is
            # recomputed, nothing recorded, no event can fire.
            if not ok:
                self.playing = False
                self._set_play_state(False)
                return
            index = self.source.current_index
            self.frame_index = index
            self._show_frozen_frame(frame, index, timestamp)
            self._sync_slider(index, timestamp)
            return
        if not ok:
            if not self.source.is_live:  # end of file
                self.playing = False
                self._set_play_state(False)
                # Close any event still being judged BEFORE labelling: a clip
                # that ends with the subject on the floor has an open event,
                # and that event is precisely the answer the label needs.
                if self.pipeline is not None and self._last_timestamp is not None:
                    resolved = self.pipeline.finalise(self._last_timestamp)
                    if resolved is not None:
                        self._on_event_resolved(resolved)
                label = self._label_current_clip() if self._queue_running else None
                record = (str(self.records.path)
                          if self.records is not None and self.records.path else "")
                # The record is complete the moment the clip ends, so it goes
                # to disk now rather than waiting for a Close that may never
                # come. The source stays open: reviewing is the whole point of
                # stopping here, and reviewed frames are not logged anyway.
                self._finalise_records("End of video.", complete=True)
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

        if log and self.records is not None:
            self.records.frame(result, decode_ms=decode_ms)
            self._logged_frames += 1
            self._reliable_frames += int(bool(result.reliable))
        if self._recorder is not None:
            if log:
                self._recorder.record(index, result)
            else:
                self._recorder.invalidate("jump during the pass")
        self._last_timestamp = timestamp

        t_overlay = time.perf_counter()
        shown = draw_overlay(
            frame,
            result.pose,
            visibility_threshold=self.cfg.pose.visibility_threshold,
            show_skeleton=self.view_panel.skeleton.isChecked(),
            show_trunk=self.view_panel.trunk.isChecked(),
            show_centroid=self.view_panel.centroid.isChecked(),
            centroid_px=result.centroid_px,
            show_support=self.view_panel.support.isChecked(),
            com_px=result.com_px,
            support_hull_px=result.support_hull_px,
            com_outside=_p_side(result.quantities.get("P_offset")),
            hud_lines=result.hud_lines,
        )
        t_display = time.perf_counter()
        self._display(shown)
        t_plots = time.perf_counter()
        self.plots.update(index, timestamp, result,
                          curves=self.view_panel.curves.isChecked(),
                          live=self.source.is_live)
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
        self.time_label.setText(f"{format_time(timestamp)} / {format_time(total)}")

    def _freeze_finished_pass(self, label: dict | None, record: str) -> None:
        """Freeze the pass that just ended, if it was continuous."""
        rec, self._recorder = self._recorder, None
        if rec is None or self.pipeline is None:
            return
        analysis = rec.finish(events=list(self.pipeline.machine.events),
                              record_path=record, label=label)
        if analysis is None:
            self._set_status(f"Not frozen: {rec.invalid_reason or 'empty pass'}. "
                             f"It will be re-analyzed when reopened.")
            return
        self._cache.store(analysis)
        if not self._queue_running:
            # Stay on the clip, now frozen: moving through it from here
            # recomputes nothing. In the queue, the next clip opens at once.
            self._enter_frozen(analysis)

    def _display(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.show_image(QPixmap.fromImage(image))

    # --------------------------------------------------------------- readouts --
    def _show_verdict(self, events) -> None:
        """The verdict panel: every event, with the rule that decided it."""
        if not events:
            self.verdict_label.setText("No events (Stage 1 never fired).")
            return
        lines = []
        for ev in events:
            tag = ev.verdict + (f" [{ev.severity}]" if ev.severity else "")
            # Escaped: the reason carries '<' and '>=', which as HTML ate
            # the text ("T final 3 deg = 50%" instead of "< 30 ... >= 50%").
            colour = SEVERITY_PEN.get(ev.severity, theme.MUTED)
            lines.append(f"<span style='color:{colour}'><b>t={ev.timestamp:.2f}s  "
                         f"{html.escape(tag)}</b></span><br>"
                         f"{html.escape(ev.reason or '—')}")
        self.verdict_label.setText("<br><br>".join(lines))

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
            self.plots.mark_verdict(self._last_timestamp, ev)
        if self.pipeline is not None:
            self._show_verdict(self.pipeline.machine.events)
        if self.pipeline is not None and self.pipeline.alerts.should_dispatch(ev.verdict):
            return                      # the alert sink already showed it
        text = (f"event at t={ev.timestamp:.2f}s  ->  {ev.verdict}"
                + (f" [{ev.severity}]" if ev.severity else "")
                + f"   (T={ev.t_deg:.1f} deg, V={ev.v_tps:+.2f} torso/s)"
                + (f"\nreason: {ev.reason}" if ev.reason else ""))
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
            self.plots.mark(ev.timestamp, "#ff1744", dash=True)
        now = time.monotonic()
        if self.pipeline is not None and now - self._last_perf_shown > 1.0:
            self._last_perf_shown = now
            run = self.records.run if self.records is not None else None
            prefix = f"{run.path.name}  ·  " if run is not None else ""
            self._set_status(prefix + self.pipeline.timings.summary())
        self.stage_label.setText(text)
        self._set_stage_chip(result.stage, fired=result.event is not None)

    def _on_curves_toggled(self, checked: bool) -> None:
        for w in (self.curves_panel, self.plots.timeline):
            w.setVisible(checked)

    def _set_play_state(self, playing: bool) -> None:
        """Play button text and icon for the current state."""
        self.play_btn.setText(PAUSE_TEXT if playing else PLAY_TEXT)
        self.play_btn.setIcon(self.style().standardIcon(
            QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))

    def _set_stage_chip(self, stage: str, fired: bool = False, frozen: bool = False) -> None:
        """The coloured stage chip: same colours as the timeline."""
        name = theme.STAGE_NAMES.get(stage, stage)
        if fired:
            key, name = "FIRED", "Stage 1 fired"
        elif frozen:
            key, name = "FROZEN", f"Frozen · {name}"
        else:
            key = stage if stage in theme.STAGE_COLOURS else "MONITORING"
        self.stage_chip.setText(name)
        self.stage_chip.setStyleSheet(theme.chip_style(theme.STAGE_COLOURS[key]))


    # --------------------------------------------------------- runs on disk --
    def _show_run(self, run: RunFolder | None) -> None:
        """The run named at the bottom right: what a screenshot needs to say."""
        if run is None:
            self.run_label.setText("")
            return
        self.run_label.setText(f"Run: {run.path.name}")
        self.run_label.setToolTip(str(run.path))

    def _show_results(self, text: str) -> None:
        self.results_text.setPlainText(text)
        self.side_tabs.setCurrentWidget(self.results_text)

    def _on_open_run(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Run folder (the one with run.json)", str(self.cfg.logging.output_dir))
        if folder:
            self.open_run(folder)

    def open_run(self, folder: str | Path, videos: str | Path | None = None) -> bool:
        """Load a recorded run to review it: its clips, labels and metrics.

        Each clip opens frozen, rebuilt from the run's landmarks with the
        run's own configuration, without MediaPipe (records schema 2 keeps
        them exact, so it is the run as it was). The videos are looked for
        where the run saw them; ``videos`` (or a dialog) when they moved.
        """
        from ..cli import tool
        folder = Path(folder)
        info = RunFolder.read_info(folder)
        if not info:
            self._set_status(f"{folder.name} is not a run folder (no run.json).", error=True)
            return False
        clips, _ = tool("_records").find_runs(folder)
        if not clips:
            self._set_status(f"{folder.name} has no clips.", error=True)
            return False
        if videos is None:
            seen = (info.get("dataset") or {}).get("seen_at") or ""
            videos = seen if seen and Path(seen).is_dir() else None
        if videos is None:
            videos = QFileDialog.getExistingDirectory(
                self, f"Where are the videos of {folder.name}?")
            if not videos:
                self._set_status("Open Run: the run's videos are needed to show it.",
                                 error=True)
                return False
        by_stem = {rec.stem: rec for rec in clips.values()}
        paths, missing = [], []
        for stem in sorted(by_stem):
            found = [Path(videos) / f"{stem}{ext}" for ext in VIDEO_SUFFIXES]
            found = [p for p in found if p.exists()]
            (paths.append(found[0]) if found else missing.append(stem))
        self._close_source()
        rows = _read_rows(folder / LABELS_FILE)
        by_video = {Path(r.get("video", "")).stem: r for r in rows}
        self._queue = paths
        self._review_index = None
        self.queue_panel.clear()
        self.queue_panel.add_clips(paths)
        self._queue_rows = []
        for i, path in enumerate(paths):
            row = by_video.get(path.stem)
            if row is None:
                continue
            self._queue_rows.append(row)
            self.queue_panel.show_result(i, row.get("detected_class", ""),
                                         row.get("true_class", ""),
                                         metrics_text(self._queue_rows))
            if row.get("reviewed_class"):
                self.queue_panel.set_reviewed(i, row["reviewed_class"])
        cfg = Config(info.get("config") or self.cfg.as_dict(),
                     source=f"the configuration of {folder.name}")
        self._opened_run = SimpleNamespace(path=folder, info=info, clips=by_stem, cfg=cfg,
                                           cfg_key=config_key(cfg.as_dict()))
        self._last_run = RunFolder(folder)
        self._show_run(self._last_run)
        txt = folder / run_metrics.METRICS_TXT
        if txt.exists():
            self._show_results(txt.read_text(encoding="utf-8"))
        elif rows:
            table = run_metrics.compute(rows, info.get("partition") or {})
            self._show_results(run_metrics.format_text(table, rows, folder.name))
        self._update_controls()
        note = f"; videos not found for {len(missing)}: {', '.join(missing[:5])}" if missing else ""
        self._set_status(f"Opened run {folder.name}: {len(paths)} clips, double-click one "
                         f"to review it frozen{note}", error=bool(missing))
        return True

    def _opened_analysis(self, path) -> ClipAnalysis | None:
        """The opened run's analysis of this clip, rebuilt from its landmarks."""
        run = self._opened_run
        if run is None:
            return None
        rec = run.clips.get(Path(path).stem)
        if rec is None:
            return None
        cached = self._cache.get(path, run.cfg_key)
        if cached is not None:
            return cached
        from ..cli import tool
        replay = tool("replay")
        cfg = run.cfg
        stats = replay.ReplayStats()
        frames = list(replay.rebuild_frames(
            SimpleNamespace(pose_frontend=pose_frontend), rec,
            float(cfg.pose.visibility_threshold), float(cfg.pose.reacquire_after_s), stats))
        if not frames:
            return None
        pipeline = FramePipeline(cfg, labelling=True)
        recorder = PassRecorder(path, run.cfg_key, float(rec.metadata()["source_fps_declared"]))
        for pf in frames:
            recorder.record(pf.frame_index, pipeline.analyze(pf))
        pipeline.finalise(frames[-1].timestamp)
        row = next((r for r in self._queue_rows
                    if Path(r.get("video", "")).stem == Path(path).stem), None)
        analysis = recorder.finish(events=list(pipeline.machine.events),
                                   record_path=str(rec.frames), label=row)
        pipeline.close()
        if analysis is not None:
            self._cache.store(analysis)
        return analysis

    def export_figure(self) -> Path | None:
        """Save the video and the curves, as on screen, into the run's figures/."""
        run = (self.records.run if self.records is not None else None) or self._last_run
        base = run.path if run is not None else Path(self.cfg.logging.output_dir)
        if not base.is_absolute():
            from ..config import REPO_ROOT
            base = REPO_ROOT / base
        folder = base / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        clip = Path(self._current_path).stem if self._current_path else "pef-lab"
        stem = f"{clip}-f{self.frame_index:05d}"
        path, n = folder / f"{stem}.png", 2
        while path.exists():
            path, n = folder / f"{stem}-{n}.png", n + 1
        if not self.main_split.grab().save(str(path), "PNG"):
            self._set_status(f"Could not write {path}", error=True)
            return None
        self._set_status(f"Figure -> {path}")
        return path

    # -------------------------------------------------------------- helpers --
    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            # No colour on the normal case: the theme's own text colour is
            # readable in both light and dark, which a fixed green was not.
            f"color: {theme.DANGER}; font-weight: bold;" if error else ""
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
        self.open_run_btn.setEnabled(not running)
        self.queue_panel.clear_btn.setEnabled(not running and bool(self._queue))
        self.queue_panel.save_btn.setEnabled(not running and bool(self._queue_rows))
        self.queue_panel.start_btn.setEnabled(not running and bool(self._queue))
        self.queue_panel.stop_btn.setEnabled(running)
        # Reviewing is the opposite of running the pass: during it the
        # arrows would load another clip mid-measurement, the same
        # accident that already stops opening a video with the button above.
        self.queue_panel.prev_btn.setEnabled(not running and bool(self._queue))
        self.queue_panel.next_btn.setEnabled(not running and bool(self._queue))
        self.queue_panel.hint.setVisible(not self._queue)
        self._update_review_label()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Release the camera, the pipeline and the record when the window closes.

        Qt does not do this for us: without it, closing the window would
        leave the capture device held and the audit CSV unflushed.
        """
        self._close_source()
        super().closeEvent(event)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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
