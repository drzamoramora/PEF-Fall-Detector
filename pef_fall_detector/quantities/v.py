"""Quantity V (§3.4): vertical velocity of the trunk centroid, torso lengths per second."""

from __future__ import annotations

from collections import deque

import numpy as np


def centroid(mid_hip: np.ndarray, mid_shoulder: np.ndarray, hip_weight: float = 0.5) -> np.ndarray:
    """The body centroid C used by Quantity V (§3.4).

    §3.4 defines C as the weighted mean of the hip and shoulder landmark
    clusters. With ``hip_weight = 0.5`` this is the plain midpoint; the
    weight is exposed because the hip cluster dominates the true center of
    mass, and Phase 4 (Quantity P) may want a hip-heavier estimate. Pure
    function: 2D pixel vectors in, 2D pixel vector out.
    """
    w = float(np.clip(hip_weight, 0.0, 1.0))
    return w * np.asarray(mid_hip, dtype=float)[:2] + (1.0 - w) * np.asarray(
        mid_shoulder, dtype=float
    )[:2]


class VelocityEstimator:
    """Quantity V (§3.4): centroid velocity over a sliding **time** window.

    Keeps the samples of the (smoothed) centroid that fall inside the last
    ``window_seconds`` and computes the discrete derivative between the
    newest and the oldest — the guide's "subtract the position N frames
    back, divide by elapsed time", with the window expressed in seconds
    instead of frames. Output is **dimensionless**: torso-lengths per
    second, so the same threshold works at any subject scale or camera
    distance (§3.2).

    Why seconds and not frames: measured on this project's own material,
    a window of "5 frames" spanned 0.08 s in one clip and 0.40 s in another
    — the same configuration value producing a five-fold difference in
    what is actually measured. Since a longer window flattens the peak, the
    same physical fall reads −5.03 torso/s at 60 fps and −2.11 at 10 fps.
    Calibrating a shared threshold across clips measured with different
    rulers would produce a meaningless number, and a threshold tuned on
    30 fps footage would miss falls on a slower edge device (§3.6).

    Returns a pair ``(v_vertical, v_horizontal)``:

    * ``v_vertical`` — the paper's Quantity V. Negative = downward (see the
      module docstring's sign-convention note). A fall shows a large
      negative spike in a short window; sitting down shows a smaller,
      sustained negative value.
    * ``v_horizontal`` — EXPERIMENTAL, not part of the paper (§3.4 defines V
      as vertical only). Logged as a diagnostic: walking produces sustained
      horizontal motion, and trip-type or lateral falls carry a horizontal
      component. If calibration shows it separates cases the vertical
      component cannot, adding it to the paper becomes an evidence-backed
      proposal. Positive = rightward in the image.

    Both are NaN until the window has filled with real elapsed time. A gap
    larger than ``max_gap_s`` between consecutive samples (subject lost,
    big seek) clears the history: differentiating across a gap would
    produce a meaningless spike.
    """

    def __init__(self, window_seconds: float = 0.167, max_gap_s: float = 0.5) -> None:
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be > 0")
        self.window_seconds = float(window_seconds)
        self.max_gap_s = float(max_gap_s)
        self._samples: deque[tuple[float, float, float]] = deque()

    def update(self, timestamp: float, centroid_xy: np.ndarray, torso_length: float
               ) -> tuple[float, float]:
        """Feed one (smoothed) centroid sample; returns (v_vertical, v_horizontal)."""
        if self._samples and timestamp - self._samples[-1][0] > self.max_gap_s:
            self._samples.clear()

        cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
        self._samples.append((timestamp, cx, cy))

        # Drop samples older than the window, but keep the newest one that is
        # still at least a full window old — that is the derivative's anchor.
        while len(self._samples) > 2 and timestamp - self._samples[1][0] >= self.window_seconds:
            self._samples.popleft()

        if len(self._samples) < 2:
            return float("nan"), float("nan")
        if not torso_length or torso_length <= 0.0:
            return float("nan"), float("nan")

        t0, x0, y0 = self._samples[0]
        dt = timestamp - t0
        # Not enough elapsed time yet: reporting a velocity over a fraction of
        # the window would exaggerate whatever noise is present.
        if dt < self.window_seconds:
            return float("nan"), float("nan")

        # Negated: image y grows down, but V's convention is negative = down.
        v_vertical = -((cy - y0) / dt) / torso_length
        v_horizontal = ((cx - x0) / dt) / torso_length
        return v_vertical, v_horizontal

    def reset(self) -> None:
        """Forget all history (call when the subject is lost or on seek)."""
        self._samples.clear()
