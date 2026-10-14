"""What one analysed frame produces, and where the per-frame time goes.

``FrameResult`` is the single object every consumer reads (PEF-Lab, the
headless runner, the replay tool, the records); ``Timings`` keeps the
measured split of the per-frame budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .audit_log import csv_column
from .pose_frontend import PoseFrame


class Timings:
    """Per-section wall-clock stats, for finding where the frame budget goes.

    Exists because this project has already once optimised from a plausible
    guess and been wrong (see the EMA episode in ``notas/``). When the live
    frame rate collapsed to 5 fps the obvious suspect was MediaPipe — but
    inference had been measured at ~25 ms months earlier, so most of the
    200 ms had to be elsewhere. Guessing which "elsewhere" is exactly the
    mistake worth not repeating.

    Keeps a bounded window of recent samples so a long session cannot grow
    it without limit, and reports medians rather than means: one 400 ms
    stall while a model loads should not colour the whole picture.
    """

    def __init__(self, window: int = 300) -> None:
        self.window = int(window)
        self._samples: dict[str, list[float]] = {}

    def add(self, section: str, seconds: float) -> None:
        bucket = self._samples.setdefault(section, [])
        bucket.append(seconds * 1000.0)
        if len(bucket) > self.window:
            del bucket[: len(bucket) - self.window]

    def median_ms(self, section: str) -> float:
        bucket = self._samples.get(section)
        if not bucket:
            return float("nan")
        ordered = sorted(bucket)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    def sections(self) -> list[str]:
        return sorted(self._samples)

    def summary(self) -> str:
        """One compact line, ready for a status bar."""
        if not self._samples:
            return ""
        parts = [f"{name} {self.median_ms(name):.0f}ms" for name in self.sections()]
        total = sum(self.median_ms(n) for n in self.sections())
        fps = 1000.0 / total if total > 0 else float("nan")
        return " | ".join(parts) + f"  ->  {total:.0f}ms/frame ({fps:.1f} fps)"

    def reset(self) -> None:
        self._samples.clear()


@dataclass
class FrameResult:
    """Everything the pipeline knows about one processed frame.

    Attributes:
        pose: the front-end output (landmarks, Step-0 anchors, visibility).
        reliable: detection present AND core visibility above threshold.
            Statistics and calibration must use only reliable frames.
        quantities: named quantity values as floats (NaN = not computable).
            Keys match audit-log column names (e.g. ``T_deg``, ``V_tps``).
        state: person-state display label (``person_state`` vocabulary),
            or None when nothing was detected.
        centroid_px: smoothed centroid position in pixels (for the overlay),
            or None when nothing was detected.
        com_px: hip-weighted centre of mass in pixels — Quantity P's point,
            which §3.4 defines differently from Quantity V's centroid. None
            when the feet were not visible enough to compute P.
        support_hull_px: (N, 2) vertices of the support polygon in pixels,
            or None on the same condition.
        stage: where this frame sits in the §3.5 funnel (``MONITORING``,
            ``CONFIRMING``, ``COOLDOWN``).
        event: the Stage-1 event raised ON this frame, or None. Only the
            firing frame carries it, so a consumer can count events by
            counting non-None values instead of de-duplicating.
        resolved_event: the event whose verdict landed on THIS frame, or
            None. A different moment from ``event``: a record or an alarm
            belongs to the frame where the funnel decided, not to the frame
            where it started wondering.
        hud_lines: ready-to-draw text lines for the PEF-Lab overlay.
    """

    pose: PoseFrame
    reliable: bool
    quantities: dict[str, float] = field(default_factory=dict)
    state: str | None = None
    centroid_px: np.ndarray | None = None
    com_px: np.ndarray | None = None
    support_hull_px: np.ndarray | None = None
    stage: str = "MONITORING"
    event: object | None = None
    resolved_event: object | None = None
    hud_lines: list[str] = field(default_factory=list)

    def csv_extra(self) -> dict[str, str]:
        """Values formatted for the audit CSV (NaN -> empty cell).

        Names are translated to their record spelling on the way out — the
        ``_EXP`` mark on columns the paper does not define. That mapping
        lives in ``audit_log`` so this pipeline, the GUI and the tests keep
        using one vocabulary regardless of how a column is labelled in the
        record.
        """
        extra = {
            csv_column(name): f"{value:.3f}"
            for name, value in self.quantities.items()
            if not math.isnan(value)
        }
        extra["reliable"] = "1" if self.reliable else "0"
        extra["stage"] = self.stage
        extra["stage1_fired"] = "1" if self.event is not None else "0"
        if self.state is not None:
            extra[csv_column("state")] = self.state
        if self.centroid_px is not None:
            extra["centroid_x_px"] = f"{self.centroid_px[0]:.2f}"
            extra["centroid_y_px"] = f"{self.centroid_px[1]:.2f}"
        return extra
