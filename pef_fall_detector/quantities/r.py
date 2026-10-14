"""Quantity R (EXPERIMENTAL, not in §3.4): how close the hands come to the feet."""

from __future__ import annotations

import numpy as np

from ..pose_frontend import LEFT_ANKLE, LEFT_WRIST, RIGHT_ANKLE, RIGHT_WRIST


def wrist_ankle_proximity(wrists: np.ndarray, ankles: np.ndarray,
                          torso_length: float) -> float:
    """Quantity R (EXPERIMENTAL, not in §3.4): how close the hands are to the feet.

    Proposed for a problem T, V, P, I and H all share and none of them
    solve: each of those reads POSTURE (how the body is arranged), and a
    deliberate bend — tying a shoe, picking something off the floor —
    produces close to the same posture as a fall. None of them encode INTENT,
    because none of them look at what the hands are doing. A person tying a
    shoe reaches for their feet; a person falling does not. This is a
    genuinely new degree of freedom, not a reformulation of T/V/P/I/H — it is
    the first quantity in this module that reads the arms at all.

    Defined as the SMALLEST wrist-to-ankle distance among every wrist/ankle
    pair that is actually visible (up to four: two wrists by two ankles),
    normalised by torso length so it means the same thing at any camera
    distance, per §3.2's convention for every other quantity here. Smaller
    means a hand is closer to a foot.

    Args:
        wrists: (2, 2) array, one row per wrist (left, right), pixel
            coordinates. A row of NaN means that wrist was not visible —
            the caller's job (visibility threshold), same convention as
            every other quantity in this module.
        ankles: (2, 2) array, one row per ankle (left, right), same
            convention.
        torso_length: Step-0 scale.

    Returns:
        Distance in torso lengths, or NaN when no wrist/ankle pair is
        measurable or the torso length is unusable — an unobserved quantity
        reads as unknown, never as "far" or "close".

    NOT a discriminator by itself. This module stays pure measurement, same
    as T/V/P/I/H: it reports a distance, not a verdict on whether that
    distance means "reaching". What counts as close enough to indicate
    intent is a threshold decision for whoever calibrates it against
    labelled clips — this quantity is logged, not wired into any §3.5 stage,
    for exactly that reason.
    """
    if not torso_length or torso_length <= 0.0:
        return float("nan")
    w = np.asarray(wrists, dtype=float)[:, :2]
    a = np.asarray(ankles, dtype=float)[:, :2]
    w = w[np.isfinite(w).all(axis=1)]
    a = a[np.isfinite(a).all(axis=1)]
    if w.size == 0 or a.size == 0:
        return float("nan")
    # Small (<=4) pairwise distance set: a loop is clearer than broadcasting
    # for four numbers at most, and this runs once per frame, not per pixel.
    dists = [float(np.linalg.norm(wp - ap)) for wp in w for ap in a]
    return min(dists) / torso_length


def measure(pf, visibility_threshold: float) -> float:
    """Quantity R for one frame (EXPERIMENTAL, not in §3.4): how close
    the hands are to the feet.

    NOT consumed by any §3.5 stage — logged so it can be evaluated
    against labelled clips before it is. See
    ``quantities.wrist_ankle_proximity`` for what it is proposed to
    cover: T, V, P, I and H all read posture, and a deliberate bend
    (tying a shoe) produces close to the same posture as a fall by every
    one of them. This is the only quantity in the module that looks at
    the hands at all, which is what makes it a candidate for telling the
    two apart instead of just confirming that "something collapsed"
    happened again.

    Per-landmark visibility gating, same policy as every other quantity
    here: a wrist or ankle below the threshold is passed as NaN rather
    than trusted.
    """
    vis = pf.visibility
    threshold = visibility_threshold
    wrists = pf.landmarks[[LEFT_WRIST, RIGHT_WRIST], :2].copy()
    wrists[vis[[LEFT_WRIST, RIGHT_WRIST]] < threshold] = np.nan
    ankles = pf.landmarks[[LEFT_ANKLE, RIGHT_ANKLE], :2].copy()
    ankles[vis[[LEFT_ANKLE, RIGHT_ANKLE]] < threshold] = np.nan
    return wrist_ankle_proximity(wrists, ankles, pf.torso_length)
