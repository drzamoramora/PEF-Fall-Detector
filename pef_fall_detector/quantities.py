"""The four physics-informed quantities (§3.4).

Every function in this module is **pure**: it takes plain numbers or arrays
and returns numbers. No MediaPipe, no OpenCV, no Qt, no file I/O. That is
what makes each quantity unit-testable against synthetic skeletons, and what
lets the exact same code run inside PEF-Lab, in batch evaluation (§3.7) and
on the Raspberry Pi (§3.6).

Coordinate convention (established in ``pose_frontend``): all inputs are in
**pixel space**, where x grows to the right and **y grows downward** (the
image convention). The camera-relative vertical axis of §3.4 is therefore
the unit vector ``(0, -1)``.

Implemented so far:
    T — trunk inclination angle          [Phase 2.1]
    V — vertical centroid velocity       [Phase 2.2-2.3]
    P — COM vs. support polygon          [Phase 4]
    I — post-fall immobility duration    [Phase 5]

Sign convention for velocities (IMPORTANT): V follows the paper's physical
convention — **negative = moving down** (a fall produces a large negative V,
§3.4; the guide's Stage-1 trigger tests ``V < threshold_V`` with a negative
threshold). Since image y grows *downward*, the implementation negates the
raw dy/dt. Note that the guide's own pseudocode returns ``(y_now - y_prev)/dt``
unnegated, which yields POSITIVE values for downward motion — inconsistent
with its own negative threshold. This implementation resolves that
inconsistency in favor of the physical convention.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

#: Camera-relative vertical ("up") in image pixel space, where y grows down.
VERTICAL_AXIS = np.array([0.0, -1.0])

#: Below this trunk length (pixels) the trunk direction is numerically
#: meaningless — noise dominates — so T is reported as NaN rather than as an
#: angle amplified out of nothing. Well under any real detection (a torso
#: spans tens to hundreds of pixels), so it only catches degenerate frames.
MIN_TRUNK_PX = 1e-3


def trunk_inclination_deg(mid_hip: np.ndarray, mid_shoulder: np.ndarray) -> float:
    """Quantity T — trunk inclination angle in degrees (§3.4).

    The trunk vector runs from the mid-hip landmark (average of 23 and 24)
    to the mid-shoulder landmark (average of 11 and 12); T is the angle
    between that vector and the camera-relative vertical.

    Reference values (§3.4)::

        T ~ 0 deg        upright posture
        T 15-35 deg      moderate lean, e.g. reaching for an object
        T 60-90 deg      horizontal collapse; the signature of a fall when
                         reached within a short time window

    Args:
        mid_hip: hip midpoint, pixel coordinates (only x, y are used).
        mid_shoulder: shoulder midpoint, pixel coordinates.

    Returns:
        Angle in degrees within [0, 180], or NaN for a degenerate skeleton
        (zero-length trunk). 90 deg means the trunk is horizontal; values
        above 90 deg mean the shoulders sit *below* the hips.

    Note:
        The implementation guide computes this angle on MediaPipe's
        normalized coordinates against a 3D vertical. We compute it in 2D
        pixel space instead, for two reasons: normalized coordinates divide
        x by the frame width and y by its height, which distorts angles on
        any non-square frame (a true 45 deg lean reads ~29 deg at 16:9 —
        see ``tests/test_quantities.py``), and the monocular z component is
        too noisy to be trusted as geometry (§3.2).
    """
    trunk = np.asarray(mid_shoulder, dtype=float)[:2] - np.asarray(mid_hip, dtype=float)[:2]
    norm = float(np.linalg.norm(trunk))
    if norm < MIN_TRUNK_PX:
        return float("nan")
    # VERTICAL_AXIS is a unit vector, so dividing by |trunk| alone is enough.
    cos_theta = float(np.dot(trunk, VERTICAL_AXIS) / norm)
    # Clamp inherited from the implementation guide (§3.4, "evitar errores de
    # redondeo"). It is NOT reachable with this formulation and no test can
    # detect its removal: because the vertical axis is a unit vector we divide
    # by a single norm, and sqrt(vx^2 + vy^2) is never below |vy| in IEEE
    # arithmetic, so the quotient cannot exceed 1 (verified over 400 000
    # near-vertical trunks spanning 12 orders of magnitude: zero overflows).
    # It is kept because the *general* angle formula — normalising both
    # vectors, as the guide writes it and as a 3D version would — does
    # overflow: 13 % of parallel vector pairs produce a cosine above 1, and
    # arccos of those is NaN. A NaN here would propagate silently into an
    # empty CSV cell and a skipped Stage-1 evaluation.
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def trunk_band(t_deg: float) -> str:
    """Human-readable band for a T value, per the reference table of §3.4.

    This is a **display aid** for the PEF-Lab HUD and the audit log, not a
    decision: no stage of §3.5 consumes it. It exists so that a value can be
    read against the paper's own vocabulary while calibrating. The
    boundaries are paper reference values, NOT calibration parameters —
    which is why they live here and not in ``config.yaml`` (that file is
    reserved for numbers tuned against data and reported in Section 4).

    Bands: ``upright`` (< 15 deg), ``moderate-lean`` (15-60 deg),
    ``collapse-range`` (>= 60 deg), ``n/a`` for NaN. Note: §3.4 defines
    15-35 deg as moderate lean and 60-90 deg as the fall signature, leaving
    35-60 deg undefined; this display absorbs that gap into
    ``moderate-lean`` — a labeling choice for readability, with no effect
    on any decision.
    """
    if math.isnan(t_deg):
        return "n/a"
    if t_deg < 15.0:
        return "upright"
    if t_deg < 60.0:
        return "moderate-lean"
    return "collapse-range"


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


class ExponentialMovingAverage:
    """Time-constant EMA smoother, applied before differentiation.

    Differentiation amplifies noise: MediaPipe landmarks jitter a few pixels
    even on a still subject, and a raw derivative would turn that jitter
    into spurious velocity spikes that could fire the Stage-1 trigger with
    nobody moving (implementation-plan §5.4). Measured on real footage, this
    filter removes ~63 % of the landmark jitter at a cost of ~4 % peak
    attenuation.

    The smoothing strength is a **time constant in seconds**, not a
    per-sample coefficient. The blend factor is recomputed for every sample
    from the actual elapsed time::

        alpha = 1 - exp(-dt / tau)
        smoothed = alpha * new + (1 - alpha) * previous

    Why time and not samples: a fixed per-sample coefficient means a
    different physical memory at every frame rate — alpha = 0.3 is ~0.09 s
    of memory at 30 fps but ~0.19 s at 15 fps — and live cameras deliver
    frames at irregular intervals anyway (21-25 % variation measured on this
    project's own sessions). Deriving alpha from the real dt makes the
    filter behave identically regardless of capture rate or frame drops.

    ``tau = 0`` disables smoothing (pass-through), which is the honest way
    to turn the filter off for an experiment. Works on scalars and on numpy
    vectors alike.
    """

    def __init__(self, time_constant_s: float) -> None:
        if time_constant_s < 0.0:
            raise ValueError(f"EMA time constant must be >= 0, got {time_constant_s}")
        self.tau = float(time_constant_s)
        self._value: float | np.ndarray | None = None

    def update(self, value, dt: float):
        """Feed one sample taken ``dt`` seconds after the previous one."""
        if self._value is None or self.tau <= 0.0 or dt <= 0.0:
            self._value = value
        else:
            alpha = 1.0 - math.exp(-dt / self.tau)
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    def reset(self) -> None:
        """Forget all history (call when the subject is lost or on seek)."""
        self._value = None


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
