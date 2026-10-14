"""Quantity A (EXPERIMENTAL, phase 8): head height against gravity, in 3D."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Quantity A — head height against gravity, in 3D (EXPERIMENTAL, phase 8)
# ---------------------------------------------------------------------------
#
# Not in §3.4, and in tension with §3.2, which evaluates all geometry in 2D
# pixel space and excludes "the monocular z-coordinate … because its estimated
# depth is not reliable as metric geometry". A does not use that image z. It
# uses MediaPipe's *world* landmarks (metric, hip-centred, the second output of
# the same inference), and only as a RATIO against the same subject's own
# standing height — in the spirit of the "displacement ratios rather than …
# absolute 3D distances" that §3.2 proposes. Whether that ratio is reliable
# enough is exactly what phase 8.1 measures, on every clip and both platforms.
#
# Why it is proposed: every remaining error in PEF-FallDB asks the same
# physical question — how high did the body end up above the floor? — and
# answers it with a 2D angle. T cannot tell lying toward the camera from an
# inverted skeleton (T ≈ 175° in both), nor falling away from the camera from
# standing (A14-S4, T 28°), nor sitting down or crouching from falling. A fall
# is, by definition, a loss of height against gravity.
#
# Measured offline on the 128 clips (25/09, notas/ANALISIS-82-90.md), A at the
# end of the clip: standing 0.93–1.09, sitting/kneeling 0.28–0.70, on the
# floor −0.21–0.25. Logged only in phase 8.1; since 8.2/8.3 Stage 3 reads it,
# and since 8.4 Stage 1 too (each with its quantity_a switch).

#: World-landmark indices used by A (MediaPipe Pose topology).
_W_NOSE = 0
_W_SHOULDERS = (11, 12)
_W_HIPS = (23, 24)
_W_KNEES = (25, 26)
_W_ANKLES = (27, 28)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    c = float(np.dot(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def knee_angle_3d(world: np.ndarray) -> float:
    """Mean hip-knee-ankle angle of both legs, degrees (180 = straight legs).

    From world landmarks (33×3, metres). NaN if the array is unusable.
    """
    if not _usable_world(world):
        return float("nan")
    angles = [
        _angle_deg(world[h] - world[k], world[a] - world[k])
        for h, k, a in zip(_W_HIPS, _W_KNEES, _W_ANKLES, strict=True)
    ]
    return float(np.mean(angles)) if not any(math.isnan(x) for x in angles) else float("nan")


def _usable_world(world) -> bool:
    return (world is not None and getattr(world, "shape", None) == (33, 3)
            and bool(np.all(np.isfinite(world))))


class GravityCalibrator:
    """Learns "up" and the subject's standing height from the subject itself.

    MediaPipe's world axes follow the CAMERA, not gravity, and every scene in
    PEF-FallDB tilts its camera differently. So "up" is calibrated from frames
    in which the subject is confidently standing — the caller decides that
    (trunk upright in the image AND straight legs in 3D) — as the median
    direction ankles→shoulders. The standing head height (nose above ankles
    along that direction) and hip height are measured on the same frames.

    Causal, like everything a live device can do: A on a frame uses only the
    calibration gathered BEFORE and ON that frame; until ``min_standing_s`` of
    standing has been seen, A is NaN — never a guess. It is NOT reset on a
    detection gap: the camera does not move, and in a single-person scene the
    subject does not change. On the deployed device this is what would let
    the calibration happen once per installation.
    """

    def __init__(self, min_standing_s: float = 0.5, max_samples: int = 300,
                 max_step_s: float = 0.1) -> None:
        if min_standing_s <= 0.0:
            raise ValueError("min_standing_s must be > 0")
        if max_samples < 5:
            raise ValueError("max_samples must be >= 5")
        self.min_standing_s = float(min_standing_s)
        self.max_step_s = float(max_step_s)
        self._ups: deque = deque(maxlen=int(max_samples))
        self._heads: deque = deque(maxlen=int(max_samples))
        self._hips: deque = deque(maxlen=int(max_samples))
        self._standing_s = 0.0
        self._last_t: float | None = None
        self._cache: tuple[np.ndarray, float, float] | None = None

    def observe(self, timestamp: float, world: np.ndarray, standing: bool) -> None:
        """Feed one reliable frame. Only ``standing`` frames calibrate."""
        t = float(timestamp)
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        if not standing or not _usable_world(world):
            return
        ankles = world[list(_W_ANKLES)].mean(axis=0)
        shoulders = world[list(_W_SHOULDERS)].mean(axis=0)
        axis = shoulders - ankles
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return
        self._ups.append(axis / n)
        self._heads.append(world[_W_NOSE] - ankles)
        self._hips.append(world[list(_W_HIPS)].mean(axis=0) - ankles)
        # Standing TIME, not a frame count: a gap is capped so a subject who
        # stood for one frame and reappeared ten seconds later has not
        # "stood for ten seconds".
        self._standing_s += min(dt, self.max_step_s)
        self._cache = None

    @property
    def standing_s(self) -> float:
        return self._standing_s

    @property
    def calibrated(self) -> bool:
        return self._standing_s >= self.min_standing_s and len(self._ups) >= 5

    def _solve(self) -> tuple[np.ndarray, float, float] | None:
        if not self.calibrated:
            return None
        if self._cache is None:
            up = np.median(np.array(self._ups), axis=0)
            up = up / np.linalg.norm(up)
            stature = float(np.median(np.array(self._heads) @ up))
            hip = float(np.median(np.array(self._hips) @ up))
            self._cache = (up, stature, hip)
        return self._cache

    @property
    def up(self) -> np.ndarray | None:
        s = self._solve()
        return None if s is None else s[0]

    @property
    def stature(self) -> float:
        s = self._solve()
        return float("nan") if s is None else s[1]

    def head_ratio(self, world: np.ndarray) -> float:
        """Quantity A: head height above the ankles / the same, standing.

        ~1 standing, ~0.3–0.7 sitting or kneeling, ~0 lying on the floor
        (negative when the head ends up below the feet). NaN until calibrated.
        """
        s = self._solve()
        if s is None or not _usable_world(world) or s[1] <= 0.05:
            return float("nan")
        up, stature, _ = s
        ankles = world[list(_W_ANKLES)].mean(axis=0)
        return float(np.dot(world[_W_NOSE] - ankles, up)) / stature

    def hip_ratio(self, world: np.ndarray) -> float:
        """Hip height above the ankles / the same, standing (sofa vs floor)."""
        s = self._solve()
        if s is None or not _usable_world(world) or s[2] <= 0.05:
            return float("nan")
        up, _, hip = s
        ankles = world[list(_W_ANKLES)].mean(axis=0)
        return float(np.dot(world[list(_W_HIPS)].mean(axis=0) - ankles, up)) / hip

    def reset(self) -> None:
        """Forget everything (a new camera, or a new source)."""
        self._ups.clear()
        self._heads.clear()
        self._hips.clear()
        self._standing_s = 0.0
        self._last_t = None
        self._cache = None


def measure(gravity: GravityCalibrator, pf, t_deg: float, reliable: bool, timestamp: float,
            standing_t_deg: float, standing_knee_deg: float) -> tuple[float, float]:
    """Quantity A and the hip ratio for one frame (NaN when unmeasurable).

    Only reliable frames with a full world skeleton are read. A frame
    calibrates "up" only when the subject is confidently standing: trunk
    upright in the image (T below ``standing_T_deg``) AND legs straight in
    3D (knee angle at least ``standing_knee_deg``). The image test keeps a
    subject lying with straight legs from calibrating "up" sideways.
    """
    world = pf.world_landmarks
    if not (reliable and pf.detected and getattr(world, "shape", None) == (33, 3)):
        return float("nan"), float("nan")
    knee = knee_angle_3d(world)
    standing = (not math.isnan(t_deg) and t_deg < standing_t_deg
                and not math.isnan(knee) and knee >= standing_knee_deg)
    gravity.observe(timestamp, world, standing)
    return gravity.head_ratio(world), gravity.hip_ratio(world)
