"""Hip-to-ankle extension (EXPERIMENTAL): legs extended or folded, in torso lengths.

Not one of §3.4's quantities. It labels posture for display (person_state)
and, with ``stage3.severity_uses_leg_extension``, tells a recovery that
stood up (mild) from one that only sat up (moderate). Recorded as
``extension_ratio_EXP``.
"""

from __future__ import annotations

import numpy as np

from ..pose_frontend import LEFT_ANKLE, RIGHT_ANKLE
from .filters import ExponentialMovingAverage


class LegExtension:
    """Hip-to-ankle DISTANCE in torso units (posture feature, 2.4).

    Was the vertical extent — ``(ankle_y - hip_y) / torso`` — which is a
    signed drop, not a length, and it was wrong in a way the dataset made
    obvious: clip A15 recorded ``extension_ratio_EXP = -0.433``. A ratio
    of two lengths cannot be negative. Worse, the quantity is asked "are
    the legs extended or folded?" and a vertical drop cannot answer that:
    a subject lying prone with legs perfectly straight has ankles level
    with the hips, so the vertical extent reads ~0 — indistinguishable
    from kneeling. Since this ratio is what separates a full recovery
    (mild) from a partial one (moderate) in §3.3, the confusion landed
    straight on the severity.

    The Euclidean distance has neither problem: it is a length, so it is
    positive, and it measures leg extension in any orientation. It is
    read only while the trunk is upright — in the Stage-3 latch and, for
    display labels, after ``classify_state`` has already sent horizontal
    trunks to LYING — so "extended while upright" still means standing and
    "folded while upright" still means sitting or kneeling.

    NOTE: the numeric scale shifts for non-upright postures (a prone
    subject now reads ~1.5 instead of ~0), so ``standing_extension`` and
    ``crouch_extension`` need re-checking against the dataset. For
    upright postures — the only ones that consult it — the two
    definitions nearly coincide, because the ankles sit almost directly
    below the hips.

    Computed entirely from **raw** landmark positions — including a raw
    torso length — and then smoothed as a finished number. Mixing
    smoothing stages (a raw numerator over a filtered denominator, as
    this did originally) is close to useless: the noise is dominated by
    the ankles in the numerator, so filtering only the denominator cuts
    the jitter by ~16 %, while filtering the finished ratio cuts it by
    ~80 % (simulated across realistic jitter levels). Without the
    output filter, a perfectly still
    subject positioned near the standing/sitting boundary flips label
    tens of times per minute even in good conditions.

    Returns NaN when the ankles are below the visibility threshold —
    occluded ankles must read as "unknown", never as a posture (same
    principle as the Stage-2 occlusion rule, implementation-plan §5.3).
    A NaN also resets the smoother: blending across a period where the
    ankles were invisible would carry stale posture into the present.
    """

    def __init__(self, max_ratio: float, time_constant_s: float) -> None:
        self.max_ratio = float(max_ratio)
        self._ema = ExponentialMovingAverage(float(time_constant_s))

    def update(self, pf, dt: float, visibility_threshold: float) -> float:
        """The smoothed ratio for one frame, or NaN when it is not a leg."""
        ankle_vis = float(pf.visibility[[LEFT_ANKLE, RIGHT_ANKLE]].min())
        torso_raw = pf.torso_length
        if ankle_vis < visibility_threshold or not torso_raw:
            self._ema.reset()
            return float("nan")
        mean_ankle = pf.landmarks[[LEFT_ANKLE, RIGHT_ANKLE], :2].mean(axis=0)
        raw_ratio = float(np.linalg.norm(mean_ankle - pf.mid_hip[:2])) / torso_raw
        if raw_ratio > self.max_ratio:
            # Not a leg: the torso is foreshortened to a few pixels and the
            # ratio is dividing by it. Values up to 183 torso lengths appear
            # in this project's own dataset, in 20 of 128 clips. NaN — the
            # same answer occluded ankles get — rather than a clamped number,
            # because "implausible" is not a measurement and a clamped value
            # would sit in the record looking like one.
            self._ema.reset()
            return float("nan")
        return float(self._ema.update(raw_ratio, dt))

    def reset(self) -> None:
        self._ema.reset()
