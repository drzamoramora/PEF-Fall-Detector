"""Frame sources: the abstraction that makes the core source-agnostic.

The whole PEF pipeline consumes ``(frame, timestamp)`` pairs and never needs
to know where they came from. Two concrete sources are provided:

* :class:`VideoFileSource` — clips recorded with PEF-Video-Tool (offline,
  reproducible; the mode used for threshold calibration).
* :class:`CameraSource` — a live webcam (the mode the edge deployment in
  §3.6 ultimately runs).

Design note on timestamps (relevant to Quantity V, §3.4): V is a time
derivative, so Δt correctness matters. For video files, timestamps are
derived from the FPS *declared in the file*; :meth:`VideoFileSource.fps_report`
exposes the data needed to sanity-check that declaration against a known
wall-clock duration, because the recording tool's capture loop may write
fewer frames per second than the container claims.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class FrameSource(ABC):
    """Abstract provider of ``(ok, frame_bgr, timestamp_seconds)`` tuples."""

    #: True when frames come from a live device (no seeking, unbounded length).
    is_live: bool = False

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Return the next frame and its timestamp in seconds."""

    @abstractmethod
    def release(self) -> None:
        """Free the underlying capture device or file handle."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second of the source."""


class VideoFileSource(FrameSource):
    """Reads a recorded clip (e.g. produced by PEF-Video-Tool).

    Timestamps are computed as ``frame_index / declared_fps``. Supports
    seeking, which the PEF-Lab GUI uses for frame stepping and scrubbing.

    Note: seeking interacts with MediaPipe's temporal smoothing (the tracker
    keeps state between frames), so after a large seek the first few pose
    estimates may be transient. Acceptable for a debugging workbench;
    irrelevant for headless sequential runs.
    """

    is_live = False

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file: {self.path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next_index = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration_seconds(self) -> float:
        """Duration implied by the declared FPS and the frame count."""
        return self.frame_count / self._fps if self._fps else 0.0

    @property
    def current_index(self) -> int:
        """Index of the frame most recently returned by :meth:`read`.

        Exposed because callers legitimately need it — the GUI shows it and
        steps relative to it — and reconstructing it from internals is how
        off-by-one bugs are born: the frame-stepping controls once computed
        it as "next index minus one" and consequently advanced zero frames
        forward and two backwards.
        """
        return max(0, self._next_index - 1)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame = self._cap.read()
        if not ok:
            return False, None, 0.0
        timestamp = self._next_index / self._fps
        self._next_index += 1
        return True, frame, timestamp

    def seek(self, frame_index: int) -> None:
        """Position the reader so the next ``read()`` returns ``frame_index``."""
        frame_index = max(0, min(frame_index, self.frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._next_index = frame_index

    def fps_report(self, expected_wall_seconds: float | None = None) -> dict:
        """Data to audit the declared FPS (see implementation-plan §5.6).

        If the true wall-clock duration of the recording is known (e.g. the
        'Video seconds' value used in PEF-Video-Tool), the effective capture
        rate is ``frame_count / expected_wall_seconds``; a mismatch with the
        declared FPS would silently rescale every velocity, so it must be
        detected before calibration.
        """
        report = {
            "declared_fps": self._fps,
            "frame_count": self.frame_count,
            "implied_duration_s": self.duration_seconds,
        }
        if expected_wall_seconds:
            report["effective_fps"] = self.frame_count / expected_wall_seconds
            report["fps_mismatch_ratio"] = (
                report["effective_fps"] / self._fps if self._fps else float("nan")
            )
        return report

    def release(self) -> None:
        self._cap.release()


class CameraSource(FrameSource):
    """Reads a live camera. Timestamps come from a monotonic wall clock,
    which makes Quantity V immune to camera FPS drift in live mode."""

    is_live = True

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise IOError(f"Could not open camera index {index}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1 else 30.0
        self._t0 = time.monotonic()

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame = self._cap.read()
        if not ok:
            return False, None, 0.0
        return True, frame, time.monotonic() - self._t0

    def release(self) -> None:
        self._cap.release()
