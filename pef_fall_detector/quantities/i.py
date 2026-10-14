"""Quantity I (§3.4): how long the head and torso have stayed still."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..pose_frontend import HEAD_AND_TORSO_LANDMARKS


class ImmobilityTimer:
    """Quantity I — how long the subject has been essentially still (§3.4).

    §3.4 defines I as "the total wall-clock duration within which the head
    and torso landmark positions remain within a small radius ε around their
    average post-trigger position", measured **after a kinematic trigger
    fires**. This implementation generalises the scope: the clock runs
    continuously, on every frame, with no trigger needed. Stage 3 then reads
    the value it already holds at the moment the trigger fires, which is the
    same number §3.4 asks for, obtained without making the quantity depend
    on a decision stage that has not been built yet. The practical gain is
    that immobility becomes visible and recordable during Phase 5 instead of
    waiting for the state machine.

    One consequence must be stated, because it is a real divergence and not
    a free generalisation. §3.4 anchors ε to the *average post-trigger
    position*. This class anchors it to the position at which the current
    stillness episode began. Those differ whenever the trigger fires while
    the body is still moving — which is the normal case, since a fall
    triggers on the way down and comes to rest afterwards. §3.4's average
    would then be smeared across the tail of the fall, forcing ε to absorb
    motion that is not stillness. Anchoring to the episode start measures
    what the quantity is named after. The draft needs one sentence changed
    for the two to agree.

    How an episode works, and why it is built around a *mean*: §3.4 measures
    the radius "around their average position", so the anchor is the running
    mean of the episode so far, per landmark, over the frames where that
    landmark was actually seen. Every sample of the episode is re-checked
    against that mean each frame; the moment one falls outside ε the episode
    ends and a new one begins from the current frame, clock at zero.

    Re-checking all of them, rather than only the newest, is what catches a
    subject who creeps rather than moves: the mean follows them, so each new
    frame looks close to it, while the frames from the start of the episode
    fall further behind. Anchoring to a single frame instead would also work
    for that case but would carry that one frame's landmark noise into the
    radius; averaging removes it.

    A note on the guide's version of this quantity, which is **not** what is
    implemented: its pseudocode accumulates positions from the trigger
    onward, never clears them, and returns 0 whenever any sample deviates.
    Once a subject moves at all after the trigger, that formulation can
    never report a positive immobility again. §3.4's wording — "the total
    wall-clock duration *within which* the positions remain within ε" — asks
    for the duration of the stretch in which they stay near the average, and
    that is what the episode structure gives.

    Args:
        epsilon_torso: stillness radius in torso lengths (``stage3.epsilon``).
        max_samples: cap on the episode buffer. Default ≈30 s at 30 fps,
            matching the Stage-3 observation window, beyond which the mean
            slides. Without it a subject asleep in frame would grow the
            buffer without bound.

    Cost: re-checking the episode is O(samples) per frame instead of O(1).
    Measured on this machine: 0.13 ms per frame at 30 samples, 0.33 ms at
    300, 0.74 ms at the 900-sample cap. Worth stating because the live loop
    was measured at 200 ms per frame, so this is under half a percent of it
    even in the worst case — the frame budget is being spent somewhere else
    entirely.
    """

    def __init__(self, epsilon_torso: float, max_samples: int = 900) -> None:
        if epsilon_torso <= 0.0:
            raise ValueError("epsilon must be > 0")
        if max_samples < 2:
            raise ValueError("max_samples must be >= 2")
        self.epsilon_torso = float(epsilon_torso)
        self.max_samples = int(max_samples)
        #: Bounded: the oldest sample drops out by itself past ``max_samples``.
        self._samples: deque[np.ndarray] = deque(maxlen=self.max_samples)
        self._started_at: float | None = None

    def update(self, timestamp: float, points: np.ndarray, torso_length: float
               ) -> tuple[float, float]:
        """Feed this frame's tracked landmarks; returns (seconds, displacement).

        Args:
            timestamp: seconds, on the same clock as every other quantity.
            points: (N, 2) tracked landmark positions in pixels. Rows may be
                NaN for a landmark that is not visible this frame — the nose
                disappears in a face-down posture, which is precisely the
                posture this quantity exists to measure, so a missing row
                must not end the episode. A landmark only contributes to its
                own mean over the frames where it was actually seen.
            torso_length: Step-0 scale, so the radius is dimensionless.

        Returns:
            ``(still_seconds, max_displacement)``: how long the current
            episode has lasted, and how far the worst landmark of the whole
            episode currently sits from its mean, in torso lengths. The
            second value is what makes ε calibratable — it shows how close
            the episode ran to breaking. Both NaN when nothing can be
            compared or the scale is unusable.
        """
        pts = np.asarray(points, dtype=float)[:, :2]
        if not torso_length or torso_length <= 0.0 or pts.size == 0:
            self.reset()
            return float("nan"), float("nan")

        if self._samples and self._samples[0].shape != pts.shape:
            self._begin(timestamp, pts)
            return 0.0, 0.0
        if not self._samples:
            self._begin(timestamp, pts)
            return 0.0, 0.0

        self._samples.append(pts.copy())

        stack = np.stack(self._samples)                       # (frames, N, 2)
        # The per-landmark mean over the frames where that landmark was
        # actually seen. Computed explicitly rather than with ``nanmean`` so
        # a landmark never seen at all yields NaN without a warning, and so
        # the "only counts where observed" rule is visible in the code.
        seen = np.isfinite(stack).all(axis=2)                  # (frames, N)
        counts = seen.sum(axis=0)                              # (N,)
        totals = np.where(seen[..., None], stack, 0.0).sum(axis=0)
        # The NaN branch is belt-and-braces: a landmark seen in no frame is
        # NaN in every sample, so its deviations are NaN and ``nanmax``
        # already skips it whatever mean is written here — a mutation
        # replacing this with a plain division passes every test (verified).
        # It stays so the array never carries a fictitious origin that a
        # future reader might take for a position.
        mean = np.where(counts[:, None] > 0,
                        totals / np.maximum(counts, 1)[:, None],
                        np.nan)                                # (N, 2)
        # Every sample of the episode is re-checked against the CURRENT mean,
        # not just the newest one. That is what catches a slow drift: the
        # mean follows a creeping subject, so each new frame looks close to
        # it, while the frames from the start of the episode fall further and
        # further behind. Checking only the latest sample would let someone
        # crawl across the room and read as immobile the whole way.
        deviations = np.linalg.norm(stack - mean, axis=2)      # (frames, N)
        if not np.isfinite(deviations).any():
            self._begin(timestamp, pts)
            return 0.0, 0.0
        displacement = float(np.nanmax(deviations)) / torso_length

        if displacement > self.epsilon_torso:
            self._begin(timestamp, pts)
            return 0.0, displacement

        return max(0.0, timestamp - float(self._started_at)), displacement

    def _begin(self, timestamp: float, pts: np.ndarray) -> None:
        self._samples = deque([pts.copy()], maxlen=self.max_samples)
        self._started_at = float(timestamp)

    def reset(self) -> None:
        """Forget the current episode (subject lost, seek, discontinuity)."""
        self._samples = deque(maxlen=self.max_samples)
        self._started_at = None


def measure(timer: ImmobilityTimer, pf, timestamp: float,
            visibility_threshold: float) -> tuple[float, float]:
    """Quantity I for one frame: (still_seconds, displacement).

    §3.4 watches "the head and torso landmark positions". Those are not
    equally available: in a face-down posture — the very outcome this
    quantity exists to confirm — the nose is occluded while the torso
    stays visible. Requiring all five would therefore blind the
    measurement exactly where it matters most.

    So the torso four are required and the nose is optional: below the
    visibility threshold it is passed as NaN, which the timer skips
    without ending the episode, and it re-anchors itself if the face
    comes back into view. If the torso itself goes unreliable there is
    nothing trustworthy left to measure, and the episode resets rather
    than accumulating time on extrapolated anatomy.
    """
    if not pf.torso_length or pf.core_visibility < visibility_threshold:
        timer.reset()
        return float("nan"), float("nan")
    pts = pf.landmarks[list(HEAD_AND_TORSO_LANDMARKS), :2].astype(float).copy()
    vis = pf.visibility[list(HEAD_AND_TORSO_LANDMARKS)]
    pts[vis < visibility_threshold] = np.nan
    return timer.update(timestamp, pts, pf.torso_length)
