"""The records of one run: the only writer of frames, events and landmarks.

A run of one source (a clip, a live camera, a replay) leaves, in research
mode, three records next to each other:

* ``<source>-<stamp>.csv``: one row per processed frame, every quantity of
  §3.4 and the funnel state (``audit_log.PHASE2_FIELDS``). The §3.5 promise
  that any decision can be reconstructed from the frames rests on it.
* ``<source>-events-<stamp>.csv``: one row per Stage-1 event, with its
  verdict, its evidence and the rule that decided it (:func:`event_row`).
* ``<source>-landmarks-<stamp>.csv``: the 33 landmarks of every frame, which
  no decision reads; they let a new measure be evaluated, and the run be
  replayed (``tools/replay.py``), without MediaPipe.

In deployment mode a run records nothing: §3.6 says the deployed detector
persists no frames, landmark coordinates or derived quantity values. What a
reviewer needs travels with the alert itself (severity, quantities, reason).

Before 0.2.0.dev3 the terminal, PEF-Lab and the replay tool each wrote these
records with their own copy of the code, and they did not write the same
thing: PEF-Lab left out the events a detection loss abandoned, and never
wrote the landmarks. Now all three write through :class:`RunRecords`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .audit_log import EVENT_FIELDS, LANDMARK_FIELDS, AuditLogger, csv_column, landmark_row


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


class RunRecords:
    """Every record of one run of one source.

    Args:
        output_dir: folder of the records; a relative path is taken from the
            repository root (see ``audit_log.AuditLogger``).
        source_name: identifier of the source, used in the file names.
        metadata: facts about the run (source, frame rate, configuration),
            written to each record's ``.meta.json``.
        research: True records everything; False (deployment) records nothing.

    Events are written in their order, each one as soon as its verdict lands
    (:meth:`event_settled`), so a live session that loses power keeps every
    decision already taken. Events that never get a verdict of their own (a
    detection loss abandoned them, or the source ended first) are written in
    their place: before the next event, or when the run closes.
    """

    def __init__(self, output_dir: str | Path, source_name: str,
                 metadata: dict[str, Any] | None = None, research: bool = True) -> None:
        meta = dict(metadata or {})
        self.research = bool(research)
        self._frames: AuditLogger | None = None
        self._events: AuditLogger | None = None
        self._landmarks: AuditLogger | None = None
        self._written = 0
        if self.research:
            self._frames = AuditLogger(output_dir, source_name, metadata=meta)
            self._events = AuditLogger(
                output_dir, source_name + "-events", fields=EVENT_FIELDS,
                metadata={**meta, "record_kind": "stage1_events"}, flush_each_row=True)
            self._landmarks = AuditLogger(
                output_dir, source_name + "-landmarks", fields=LANDMARK_FIELDS,
                metadata={**meta, "record_kind": "landmarks"})

    @property
    def path(self) -> Path | None:
        """The per-frame record (None in deployment mode)."""
        return self._frames.path if self._frames is not None else None

    @property
    def events_path(self) -> Path | None:
        return self._events.path if self._events is not None else None

    @property
    def landmarks_path(self) -> Path | None:
        return self._landmarks.path if self._landmarks is not None else None

    @property
    def frames_written(self) -> int:
        return self._frames.rows_written if self._frames is not None else 0

    @property
    def events_written(self) -> int:
        return self._written

    def frame(self, result) -> None:
        """Record one analysed frame (a ``pipeline.FrameResult``)."""
        if self._frames is None:
            return
        self._frames.log_pose_frame(result.pose, extra=result.csv_extra())
        self._landmarks.log(landmark_row(result.pose))

    def event_settled(self, events: list, ev) -> None:
        """``ev`` got its verdict: write it, and any earlier event not yet written."""
        upto = next((i for i, e in enumerate(events) if e is ev), None)
        if upto is not None:
            self._write(events, upto + 1)

    def _write(self, events: list, upto: int) -> None:
        while self._written < upto:
            if self._events is not None:
                self._events.log(event_row(self._written, events[self._written]))
            self._written += 1

    def close(self, events: list) -> None:
        """Write the events still unwritten and close every record."""
        self._write(events, len(events))
        for log in (self._frames, self._events, self._landmarks):
            if log is not None:
                log.close()
