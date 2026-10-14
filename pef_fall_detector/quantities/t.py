"""Quantity T (§3.4): trunk inclination from the vertical, in degrees."""

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


class TrunkGuard:
    """Whether the projected trunk is long enough for T to mean anything.

    This is the guard §3.2 already promises — *"depth ambiguity from the
    monocular input is mitigated by tracking within-frame and across-frame
    landmark displacement ratios rather than relying on absolute 3D
    distances"* — and which the code did not implement.

    WHY THE §3.4 GUARD IS NOT ENOUGH. That section discards T when
    |trunk| < 0.001 px. No real pose reaches that, so the guard never
    fires. Measured on clip A14: the trunk projected to **2 px** while
    MediaPipe reported 0.99 confidence, and T produced 176.9 deg next to
    26.0 deg on consecutive frames. Two pixels is 2000x the paper's
    threshold, so the frame sailed through and the noise reached Stage 1.

    WHY THE COMPARISON IS TO THE SUBJECT'S OWN RECENT TRUNK, not to the
    frame or to an absolute pixel count. A subject far from the camera
    legitimately has a short trunk in pixels; nothing is wrong with that
    frame. What is not legitimate is a trunk that collapses relative to
    what the SAME subject measured a moment ago. That is an across-frame
    ratio, which is what §3.2 asks for.

    The reference is a **median**, not the existing EMA: the collapse
    itself drags a mean down with it, so a mean would normalise away the
    very event being detected.

    WHAT THIS GUARD CANNOT DO, and it matters. A genuine fall also
    shortens the projected trunk — measured across the 56 falls that
    fired, the trunk drops to a median of 0.40 of its own recent value.
    So this cannot separate a fall from an artefact, and a tight ratio
    would reject real falls. It is set loose on purpose: it removes
    frames whose geometry is impossible (collapses to 0.05-0.07 were
    recorded), not frames that are merely foreshortened. Telling an axial
    fall from a collapsed estimate is not possible from a projected trunk
    at all — that is the limitation documented as D4.
    """

    def __init__(self, min_ratio: float, window_s: float) -> None:
        self.min_ratio = float(min_ratio)
        self.window_s = float(window_s)
        #: (timestamp, torso_px) of recent frames.
        self._history: deque[tuple[float, float]] = deque()

    def plausible(self, torso_px: float, timestamp: float) -> bool:
        """Whether the projected trunk is long enough for T to mean anything."""
        if self.min_ratio <= 0.0:
            return True
        if not torso_px or math.isnan(torso_px):
            return True                      # nothing to judge; §3.4 handles it
        cutoff = timestamp - self.window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        recent = [t for _ts, t in self._history]
        self._history.append((timestamp, float(torso_px)))
        if len(recent) < 3:
            return True                      # no reference yet
        recent.sort()
        reference = recent[len(recent) // 2]
        return torso_px >= self.min_ratio * reference

    def clear(self) -> None:
        """Forget the reference: it belongs to one continuous observation."""
        self._history.clear()
