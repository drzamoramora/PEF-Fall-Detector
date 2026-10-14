"""The records of a run: the only writer of frames, events and landmarks.

A run of one source (a clip, a live camera, a replay) leaves, in research
mode, its records inside the run's folder (``run_folder``, records schema 2):

* ``clips/<source>/frames.csv``: one row per processed frame, every quantity
  of §3.4 and the funnel state (``audit_log.FRAME_FIELDS``), plus the hash of
  the decoded image and what the frame cost the computer. The §3.5 promise
  that any decision can be reconstructed from the frames rests on it.
* ``clips/<source>/events.csv``: one row per Stage-1 event, with its verdict,
  its evidence and the rule that decided it (:func:`event_row`).
* ``landmarks/npz/<source>.npz`` and ``landmarks/csv/<source>.csv``: the 33
  landmarks of every frame, exact, as binary arrays and as text. No decision
  reads them; they let a new measure be evaluated, and the run be replayed
  (``tools/replay.py``) exactly, without MediaPipe.
* ``clips/<source>/clip.json``: the source (name, frame rate, frame count,
  the SHA-256 of the video file, the decoder that read it), how many rows
  and events were written, and whether the pass reached the end.

In deployment mode a run records nothing: §3.6 says the deployed detector
persists no frames, landmark coordinates or derived quantity values. What a
reviewer needs travels with the alert itself (severity, quantities, reason).

Before 0.2.0.dev3 the terminal, PEF-Lab and the replay tool each wrote these
records with their own copy of the code, and they did not write the same
thing: PEF-Lab left out the events a detection loss abandoned, and never
wrote the landmarks. Now all three write through :class:`RunRecords`.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .audit_log import (
    EVENT_FIELDS,
    FRAME_FIELDS,
    LANDMARK_FIELDS,
    AuditLogger,
    csv_column,
    landmark_row,
)
from .landmark_arrays import LandmarkArrays
from .run_folder import CLIP_FILE, EVENTS_FILE, FRAMES_FILE, RunFolder

try:                                    # not on Windows; the column stays blank there
    import resource
except ImportError:                     # pragma: no cover
    resource = None


def event_row(index: int, ev) -> dict[str, Any]:
    """One row of the events record."""
    def fmt(value: float, spec: str) -> str:
        return "" if math.isnan(value) else format(value, spec)
    return {
        "event_index": index,
        "frame_index": ev.frame_index,
        "timestamp_s": f"{ev.timestamp:.3f}",
        "T_deg": f"{ev.t_deg:.2f}",
        "V_tps": f"{ev.v_tps:.3f}",
        "formulation": ev.formulation,
        "trigger_source": ev.trigger_source,
        "verdict": ev.verdict,
        "severity": ev.severity,
        "max_immobility_s": fmt(ev.max_immobility_s, ".2f"),
        "p_outside_fraction": fmt(ev.p_outside_fraction, ".3f"),
        "p_samples": ev.p_samples,
        csv_column("H_ratio"): fmt(ev.h_ratio, ".3f"),
        "reason": ev.reason,
    }


def peak_rss_mb() -> float:
    """The process's peak resident memory so far, in MB (NaN if unknown)."""
    if resource is None:
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Bytes on macOS, kilobytes on Linux.
    import sys
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


class _LoadSampler:
    """CPU use and peak memory, sampled once per second of wall time."""

    def __init__(self, every_s: float = 1.0) -> None:
        self.every_s = every_s
        self._wall = time.perf_counter()
        self._cpu = time.process_time()

    def sample(self) -> dict[str, str]:
        wall, cpu = time.perf_counter(), time.process_time()
        elapsed = wall - self._wall
        if elapsed < self.every_s:
            return {}
        pct = 100.0 * (cpu - self._cpu) / elapsed
        self._wall, self._cpu = wall, cpu
        out = {"cpu_pct": f"{pct:.1f}"}
        rss = peak_rss_mb()
        if not math.isnan(rss):
            out["peak_rss_mb"] = f"{rss:.1f}"
        return out


class RunRecords:
    """Every record of one source in one run.

    Args:
        run: the run's :class:`run_folder.RunFolder`, or the path of one (a
            run folder is opened there, created if missing).
        source_name: identifier of the source (the video's stem), used as the
            clip's folder and file names.
        metadata: facts about this source (name, frame rate, frame count,
            the file's SHA-256), written to ``clip.json``. A ``config`` key,
            if given, belongs to the run: it is used for a new ``run.json``
            and otherwise ignored.
        research: True records everything; False (deployment) records nothing.

    Events are written in their order, each one as soon as its verdict lands
    (:meth:`event_settled`), so a live session that loses power keeps every
    decision already taken. Events that never get a verdict of their own (a
    detection loss abandoned them, or the source ended first) are written in
    their place: before the next event, or when the run closes.

    Nothing is created until the first frame or event: opening a source and
    closing it without recording anything leaves no trace.
    """

    def __init__(self, run: RunFolder | str | Path | None, source_name: str,
                 metadata: dict[str, Any] | None = None, research: bool = True) -> None:
        meta = dict(metadata or {})
        config = meta.pop("config", None)
        self.research = bool(research) and run is not None
        self.source_name = source_name
        self._metadata = meta
        self._run: RunFolder | None = None
        self._clip_dir: Path | None = None
        self._frames: AuditLogger | None = None
        self._events: AuditLogger | None = None
        self._landmarks: AuditLogger | None = None
        self._arrays: LandmarkArrays | None = None
        self._npz_path: Path | None = None
        self._load: _LoadSampler | None = None
        self._written = 0
        self._started = datetime.now()
        self._closed = False
        if not self.research:
            return
        self._run = run if isinstance(run, RunFolder) else RunFolder(run, config=config)

    # ----------------------------------------------------------------- paths --
    @property
    def run(self) -> RunFolder | None:
        return self._run

    @property
    def clip_dir(self) -> Path | None:
        """The clip's folder, once something was recorded."""
        return self._clip_dir

    @property
    def path(self) -> Path | None:
        """The per-frame record (None in deployment mode)."""
        if self._run is None:
            return None
        return self._run.clip_path(self.source_name) / FRAMES_FILE

    @property
    def events_path(self) -> Path | None:
        if self._run is None:
            return None
        return self._run.clip_path(self.source_name) / EVENTS_FILE

    @property
    def landmarks_path(self) -> Path | None:
        return None if self._run is None else self._run.landmarks_csv_path(self.source_name)

    @property
    def landmarks_npz_path(self) -> Path | None:
        return None if self._run is None else self._run.landmarks_npz_path(self.source_name)

    @property
    def frames_written(self) -> int:
        return self._frames.rows_written if self._frames is not None else 0

    @property
    def events_written(self) -> int:
        return self._written

    # --------------------------------------------------------------- opening --
    def _open(self) -> None:
        """Create the clip's folder and its writers, on the first thing recorded."""
        if self._clip_dir is not None or self._run is None:
            return
        run, stem = self._run, self.source_name
        self._clip_dir = run.clip_dir(stem)          # refuses a clip already recorded
        self._frames = AuditLogger(self._clip_dir, stem, fields=FRAME_FIELDS,
                                   filename=FRAMES_FILE, sidecar=False)
        self._events = AuditLogger(self._clip_dir, stem, fields=EVENT_FIELDS,
                                   filename=EVENTS_FILE, sidecar=False, flush_each_row=True)
        marks = run.landmarks_csv_path(stem)
        self._landmarks = AuditLogger(marks.parent, stem, fields=LANDMARK_FIELDS,
                                      filename=marks.name, sidecar=False)
        self._arrays = LandmarkArrays()
        self._npz_path = run.landmarks_npz_path(stem)
        self._load = _LoadSampler()

    # ------------------------------------------------------------- recording --
    def frame(self, result, decode_ms: float | None = None) -> None:
        """Record one analysed frame (a ``pipeline.FrameResult``).

        ``decode_ms``: how long reading the frame from its source took, when
        the caller measured it.
        """
        if not self.research:
            return
        self._open()
        extra = result.csv_extra()
        if decode_ms is not None:
            extra["decode_ms"] = f"{decode_ms:.2f}"
        if result.timing_ms:                        # a replay measures nothing
            extra.update(self._load.sample())
        self._frames.log_pose_frame(result.pose, extra=extra)
        self._landmarks.log(landmark_row(result.pose))
        self._arrays.add(result.pose)

    def event_settled(self, events: list, ev) -> None:
        """``ev`` got its verdict: write it, and any earlier event not yet written."""
        upto = next((i for i, e in enumerate(events) if e is ev), None)
        if upto is not None:
            self._write(events, upto + 1)

    def _write(self, events: list, upto: int) -> None:
        while self._written < upto:
            if self.research:
                self._open()
                self._events.log(event_row(self._written, events[self._written]))
            self._written += 1

    def close(self, events: list, complete: bool = True) -> None:
        """Write the events still unwritten and close every record.

        ``complete``: the pass reached the end of its source. A pass that
        broke off (an error, the queue stopped) is closed too, so nothing
        written is lost, but its ``clip.json`` says it is incomplete, and
        ``--resume`` runs that clip again.
        """
        if self._closed:
            return
        self._write(events, len(events))
        self._closed = True
        for log in (self._frames, self._events, self._landmarks):
            if log is not None:
                log.close()
        if self._clip_dir is None:
            return
        if self._arrays is not None and len(self._arrays):
            self._arrays.save(self._npz_path)
        info = {
            **self._metadata,
            "source_name": self.source_name,
            "rows": self.frames_written,
            "events": self._written,
            "complete": bool(complete),
            "started_at": self._started.isoformat(timespec="seconds"),
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }
        (self._clip_dir / CLIP_FILE).write_text(json.dumps(info, indent=2, default=str),
                                                encoding="utf-8")
