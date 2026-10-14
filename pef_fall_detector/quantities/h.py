"""Quantity H (EXPERIMENTAL, not in §3.4): body height against the subject's own standing
baseline."""

from __future__ import annotations

import math

import numpy as np

from ..pose_frontend import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from .filters import ExponentialMovingAverage

#: Same reasoning as MIN_TRUNK_PX, applied to Quantity H's scale reference:
#: below this shoulder width (pixels) it is too short to divide by.
MIN_SHOULDER_PX = 1e-3


def body_vertical_extent(spine_points: np.ndarray) -> float:
    """Quantity H's numerator (EXPERIMENTAL): observed vertical span of the
    spine chain — nose, mid-shoulder, mid-hip, mid-ankle.

    T asks for one fixed pair (hip, shoulder) and P asks for the feet
    specifically. This asks a looser question instead: how much vertical
    spread is there among *whatever* of the spine is currently visible. A
    subject collapsing to the floor loses that spread regardless of which
    part of the chain a camera can still see, or which way they fell —
    including a fall whose rotation axis points at the camera, where T's own
    trunk vector (hip-to-shoulder alone) projects to almost nothing while the
    wider chain (down to the ankles, when visible) still shows the collapse.

    Args:
        spine_points: (N, 2) or (N, 3) array, one row per candidate landmark
            in any order. A row of NaN means that landmark was not visible
            this frame (visibility threshold, out of frame) — the caller's
            job, same convention as ``feet_in_contact``.

    Returns:
        ``max(y) - min(y)`` in pixels over the rows with a finite y, or NaN
        when fewer than two landmarks were visible: a span needs two points,
        and a fabricated zero would read as "no height" rather than as
        "could not measure".
    """
    pts = np.asarray(spine_points, dtype=float)
    ys = pts[:, 1]
    finite = ys[np.isfinite(ys)]
    if finite.size < 2:
        return float("nan")
    return float(finite.max() - finite.min())


def shoulder_width(left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> float:
    """Quantity H's scale reference (EXPERIMENTAL): shoulder-to-shoulder distance.

    Everything else in this module normalises by torso length — exactly the
    measurement that collapses when a fall's rotation axis points at the
    camera (see ``body_vertical_extent``). Shoulder width is used for H
    instead because it responds to yaw (turning side-on to the camera), not
    to the forward/backward pitch that breaks the trunk vector in that
    failure mode: during the critical transition of a fall toward or away
    from the camera, the shoulders keep presenting roughly the same width
    even as the trunk foreshortens to nothing.

    This is not a universal fix. A subject who yaws AND pitches away from the
    camera at once degrades this reference too — the same class of
    camera-angle dependence §3.4 already names for Quantity P
    (``com_support_offset``). Stated here rather than hidden, same reasoning.

    Returns:
        Distance between the shoulder landmarks in pixels, or NaN below
        ``MIN_SHOULDER_PX`` — a numerator divided by noise is not a
        measurement, the same guard ``MIN_TRUNK_PX`` applies to the trunk.
    """
    diff = (np.asarray(left_shoulder, dtype=float)[:2]
            - np.asarray(right_shoulder, dtype=float)[:2])
    width = float(np.linalg.norm(diff))
    if width < MIN_SHOULDER_PX:
        return float("nan")
    return width


def body_height_ratio_raw(vertical_extent_px: float, shoulder_width_px: float) -> float:
    """This frame's (vertical extent / shoulder width), before baselining.

    Kept separate from the final :func:`body_height_ratio` because the same
    raw number serves two different purposes: it is what
    :class:`HeightBaseline` accumulates while the subject is confidently
    standing, and it is also the numerator of the final reading once a
    baseline exists. Splitting them keeps each function honest about which
    question it answers.

    Returns NaN when either input is unusable, rather than a ratio built on
    a division by an unmeasured or degenerate value.
    """
    if (math.isnan(vertical_extent_px) or math.isnan(shoulder_width_px)
            or shoulder_width_px <= 0.0):
        return float("nan")
    return vertical_extent_px / shoulder_width_px


class HeightBaseline:
    """Personal calibration for Quantity H (EXPERIMENTAL).

    H is defined relative to the subject's OWN geometry, not an absolute
    pixel count — the same reason T needs no such baseline (it is already an
    angle) but P and this quantity do (both are lengths that scale with
    distance to the camera). Anchoring to torso length, the usual ruler in
    this module, was rejected specifically for H: torso length is the
    quantity that fails in the camera-axis fall this measure exists to catch
    (see ``body_vertical_extent``), so the baseline is expressed in shoulder
    widths instead, which survive that failure mode.

    Mirrors the pattern behind ``trunk_reference_window_s`` in config.yaml:
    the baseline only moves on frames the CALLER has independently judged as
    confidently standing (low T, standing-range leg extension). Feeding it a
    value from a lean, a crouch, or the fall itself would calibrate the
    "upright" reference against a posture that is not upright. An
    :class:`ExponentialMovingAverage` underneath gives it a time constant in
    seconds rather than a frame count, same reasoning as every other timing
    parameter in this project.
    """

    def __init__(self, time_constant_s: float = 2.0) -> None:
        if time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be > 0")
        self._ema = ExponentialMovingAverage(time_constant_s)

    def update(self, dt: float, raw_ratio: float, is_standing: bool) -> float:
        """Feed one frame; returns the baseline's current value (NaN until set).

        Only a frame with ``is_standing=True`` and a measurable
        ``raw_ratio`` moves the baseline. Every other frame — including a
        frame from the very fall H is meant to catch — only reads whatever
        baseline already exists; it must not recalibrate against a collapsed
        posture.
        """
        if is_standing and not math.isnan(raw_ratio):
            self._ema.update(raw_ratio, dt)
        return self.value

    @property
    def value(self) -> float:
        return self._ema.value

    def reset(self) -> None:
        """Forget the calibration (subject lost, seek, discontinuity)."""
        self._ema.reset()


def body_height_ratio(raw_ratio: float, baseline_ratio: float) -> float:
    """Quantity H (EXPERIMENTAL): current body height relative to the
    subject's own calibrated standing baseline.

    H ~ 1     standing, at the same (vertical extent / shoulder width) as the
              calibrated baseline
    H -> 0    collapsed: little vertical spread left relative to shoulder
              width, whatever the trunk vector's 2D angle happens to read

    NOT part of §3.4 and NOT wired into any §3.5 stage. Proposed to cover two
    gaps measured on this project's own footage: P becomes unmeasurable the
    moment feet are occluded or leave frame, and T reads a falsely low,
    "upright"-looking angle when a fall's rotation axis points toward or away
    from the camera (the trunk vector's 2D projection nearly vanishes; see
    ``body_vertical_extent``). H depends on neither the feet specifically nor
    the direction any one segment rotated, only on how much vertical spread
    the visible spine chain has left — which a forward-or-backward collapse
    still destroys even when T cannot see it.

    Two limits to carry alongside any use of this number:

    * It shares T and P's monocular, camera-angle dependence — a subject who
      also turns side-on to the camera degrades the shoulder-width reference
      it is built on (see ``shoulder_width``).
    * It confirms POSTURE, not INTENT, exactly like T: a deliberate deep bend
      (tying a shoe) collapses H the same way a fall does. It is a candidate
      corroborating signal for Stage 3 when P is inconclusive, not a
      discriminator between a fall and a voluntary movement — that is a
      separate problem this quantity does not address. ``wrist_ankle_proximity``
      below is proposed for THAT problem instead.

    Returns:
        Dimensionless ratio, or NaN when either input is unusable — no
        baseline established yet, or an unmeasurable current frame — exactly
        as honest as every other NaN in this module.
    """
    if math.isnan(raw_ratio) or math.isnan(baseline_ratio) or baseline_ratio <= 0.0:
        return float("nan")
    return raw_ratio / baseline_ratio


def measure(baseline: HeightBaseline, pf, dt: float, is_standing: bool,
            visibility_threshold: float) -> tuple[float, float, float]:
    """Quantity H for one frame (EXPERIMENTAL, not in §3.4).

    NOT consumed by any §3.5 stage. Proposed to cover two gaps measured
    on this project's own footage that T and P share: P is undefined the
    moment feet are occluded or leave frame, and T reads a falsely low,
    "upright"-looking angle when a fall's rotation axis points toward or
    away from the camera (its trunk vector's 2D projection nearly
    vanishes). See ``quantities.body_height_ratio`` for the two limits it
    does NOT overcome — camera-angle dependence, and confirming posture
    rather than intent — before this number is read as more than a
    diagnostic.

    Assembled here, rather than from ``pf.mid_hip``/``pf.mid_shoulder``
    directly, because those midpoints do not carry per-landmark
    visibility: H needs to know WHICH of nose, shoulders, hips and
    ankles were actually seen this frame, so a point below the
    visibility threshold can be dropped (NaN) instead of silently
    trusted — the same policy already applied to the ankles in
    ``_hip_ankle_extension``.

    The calibration input (``is_standing``) reuses ``classify_state``'s
    own STANDING/WALKING verdict — both are upright with legs extended,
    which is exactly the posture ``HeightBaseline`` needs to calibrate
    against.

    Returns:
        ``(H, raw_ratio, baseline)`` — H is the candidate reading; the
        other two are logged alongside it so a calibration pass can see
        how the baseline converged and whether a given frame's raw ratio
        was itself trustworthy, without having to re-derive them.
    """
    vis = pf.visibility
    threshold = visibility_threshold

    def midpoint_or_nan(a: int, b: int) -> np.ndarray:
        if vis[a] < threshold or vis[b] < threshold:
            return np.array([np.nan, np.nan])
        return pf.landmarks[[a, b], :2].mean(axis=0)

    nose = (pf.landmarks[NOSE, :2] if vis[NOSE] >= threshold
            else np.array([np.nan, np.nan]))
    spine = np.array([
        nose,
        midpoint_or_nan(LEFT_SHOULDER, RIGHT_SHOULDER),
        midpoint_or_nan(LEFT_HIP, RIGHT_HIP),
        midpoint_or_nan(LEFT_ANKLE, RIGHT_ANKLE),
    ])
    extent = body_vertical_extent(spine)

    if vis[LEFT_SHOULDER] < threshold or vis[RIGHT_SHOULDER] < threshold:
        width = float("nan")
    else:
        width = shoulder_width(pf.landmarks[LEFT_SHOULDER, :2],
                               pf.landmarks[RIGHT_SHOULDER, :2])

    raw_ratio = body_height_ratio_raw(extent, width)
    reference = baseline.update(dt, raw_ratio, is_standing)
    return body_height_ratio(raw_ratio, reference), raw_ratio, reference
