"""Quantity P (§3.4): the centre of mass against the support polygon of the feet."""

from __future__ import annotations

import numpy as np

from ..pose_frontend import FOOT_LANDMARKS
from .v import centroid


def feet_in_contact(
    foot_points: np.ndarray,
    frame_size: tuple[int, int],
    torso_length: float,
    contact_band_torso: float = 0.15,
) -> np.ndarray:
    """The foot landmarks actually **in bipedal contact** (§3.4).

    §3.4 builds the support polygon from the ankles "plus their subsequent
    foot landmarks **in bipedal contact**" — not from every foot landmark
    that happens to be reported. Ignoring that clause is what let a lifted
    foot produce a support polygon 165 torso-lengths wide in this project's
    own live footage: MediaPipe keeps emitting a position for a foot it can
    no longer see, with a visibility score above threshold, and the geometry
    is built on a guess.

    Two filters, in order, and neither is a new invention:

    **Outside the frame is not an observation.** A landmark placed beyond
    the image border is the model extrapolating anatomy it cannot see. This
    project established that in Phase 2, when hips were being reported at
    y = 1958 in a 960-pixel-tall frame; the reliability policy already
    refuses to trust those for the core landmarks. Applying the same rule to
    the feet is consistency, not a new rule.

    **Contact means near the ground.** With no calibrated ground plane, the
    floor in image space is the horizontal line through the lowest foot
    landmark still standing after the first filter. A landmark more than
    ``contact_band_torso`` above that line has left the floor. The band is
    in torso lengths so it means the same thing at any camera distance.

    Args:
        foot_points: (N, 2) foot landmark positions in pixels.
        frame_size: (width, height) of the source frame.
        torso_length: Step-0 scale.
        contact_band_torso: how far above the lowest point still counts as
            touching the ground.

    Returns:
        (M, 2) subset in contact, possibly empty. An empty result is the
        honest answer when no foot can be located on the floor, and
        downstream it becomes an undefined P rather than an invented one.

    Note:
        Losing points here is **information**, not failure: §3.4 names "a
        sudden contraction of the support polygon from a full two-foot
        footprint to a heel-only contact pair" as a fall signature in its own
        right. Before this filter existed that contraction could not be seen,
        because lifted feet stayed in the polygon and widened it instead.
    """
    pts = np.asarray(foot_points, dtype=float)[:, :2]
    if pts.size == 0 or not torso_length or torso_length <= 0.0:
        return np.empty((0, 2))

    width, height = frame_size
    inside = (
        (pts[:, 0] >= 0.0) & (pts[:, 0] <= float(width))
        & (pts[:, 1] >= 0.0) & (pts[:, 1] <= float(height))
    )
    pts = pts[inside]
    if len(pts) == 0:
        return np.empty((0, 2))

    # y grows downward, so the largest y is the point nearest the floor.
    ground_y = float(pts[:, 1].max())
    band_px = contact_band_torso * torso_length
    return pts[pts[:, 1] >= ground_y - band_px]


def support_polygon(foot_points: np.ndarray) -> np.ndarray:
    """Convex hull of the foot landmarks — the support polygon of §3.4.

    §3.4 defines the support polygon as "the convex hull of the ankle
    landmark positions (left ankle landmark 27 and right ankle landmark 28)
    plus their subsequent foot landmarks in bipedal contact" — that is, the
    six points 27-32 (ankles, heels, foot indices).

    Args:
        foot_points: (N, 2) array of foot landmark positions in pixels.

    Returns:
        (M, 2) array with the hull vertices in counter-clockwise order, or
        the input points themselves when there are fewer than three (a hull
        is not defined for a point or a segment; both are still legitimate
        supports — a single visible foot, or two feet seen edge-on).

    Implemented as a monotone chain rather than pulled from scipy: the whole
    inference path must stay within the dependency budget of the §3.6 edge
    device, and six points do not justify a linear-algebra dependency.
    """
    pts = np.asarray(foot_points, dtype=float)[:, :2]
    if len(pts) < 3:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def half(points: np.ndarray) -> list[np.ndarray]:
        chain: list[np.ndarray] = []
        for p in points:
            # Drop the previous vertex while it makes a non-left turn: it lies
            # inside the hull being built.
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                if cross > 0:
                    break
                chain.pop()
            chain.append(p)
        return chain

    lower = half(pts)
    upper = half(pts[::-1])
    # First point of each half repeats the last of the other.
    return np.array(lower[:-1] + upper[:-1])


def com_support_offset(com_xy: np.ndarray, hull: np.ndarray, torso_length: float) -> float:
    """Quantity P — signed COM-to-support offset, in torso lengths (§3.4).

    §3.4 characterises a fall geometrically by "projection of COM outside the
    support polygon". The projection direction matters and is worth stating:
    a monocular frame carries no calibrated ground plane, so the only
    projection available is the **vertical one in image space** — the COM's
    x-coordinate against the horizontal extent of the feet. This is precisely
    why §3.4 notes that, of the four quantities, P "depends most directly on
    camera height and angle": the approximation degrades as the camera looks
    more steeply down. It is stated here rather than hidden so that the
    §3.7 camera-height sensitivity results have something concrete to vary.

    Returns:
        Signed horizontal distance from the COM to the support interval,
        divided by torso length: **negative inside** (with magnitude equal to
        the margin to the nearest edge), **positive outside**, so that a
        single threshold at 0 separates the two and the sign carries the
        meaning. NaN when the hull is empty or the torso length is unusable.

    The result is dimensionless by torso-length normalisation, per §3.4's
    statement that P "is reported in the dimensionless normalization provided
    by torso length".
    """
    hull = np.asarray(hull, dtype=float)
    if hull.size == 0 or not torso_length or torso_length <= 0.0:
        return float("nan")
    com_x = float(np.asarray(com_xy, dtype=float)[0])
    left, right = float(hull[:, 0].min()), float(hull[:, 0].max())
    if com_x < left:
        return (left - com_x) / torso_length
    if com_x > right:
        return (com_x - right) / torso_length
    # Inside: report the margin to the nearest edge as a negative number, so
    # that "how safely inside" is visible instead of collapsing to zero.
    return -min(com_x - left, right - com_x) / torso_length


def support_width(hull: np.ndarray, torso_length: float) -> float:
    """Horizontal extent of the support polygon, in torso lengths (§3.4).

    The second geometric signature named in §3.4: a fall may show "a sudden
    contraction of the support polygon from a full two-foot footprint to a
    heel-only contact pair". That is a change in the *size* of the support,
    which the COM offset alone cannot express — a subject can keep the COM
    well inside a support that is collapsing underneath them. Recording it
    separately is what gives that half of §3.4 something to be tested with.

    Returns NaN when the hull is empty or the torso length is unusable.
    """
    hull = np.asarray(hull, dtype=float)
    if hull.size == 0 or not torso_length or torso_length <= 0.0:
        return float("nan")
    return float(hull[:, 0].max() - hull[:, 0].min()) / torso_length


def measure(pf, min_foot_visibility: float, contact_band_torso: float, hip_weight: float):
    """Quantity P for one frame: (com_px, hull_px, offset, width).

    Two decisions worth reading before trusting the numbers.

    **The COM is hip-weighted, not the centroid of V.** §3.4 approximates
    the centre of mass as a projection "in which the hip cluster carries
    the dominant contribution", which is anatomically where the body's
    mass sits (near the sacrum). Quantity V's centroid is a plain 0.5/0.5
    midpoint by its own definition in the same section. They are two
    different points on purpose, and reusing one for the other would
    quietly redefine a published quantity.

    **Occluded feet make P undefined, never zero.** Without the feet
    there is no support polygon, and an assumed one would decide a fall
    on invented geometry. NaN is the same policy §3.4 states for a
    degenerate trunk in T, and the same one already applied to the
    extension ratio: an unobserved quantity reads as unknown.
    """
    vis = pf.visibility[list(FOOT_LANDMARKS)]
    if float(vis.min()) < min_foot_visibility or not pf.torso_length:
        return None, None, float("nan"), float("nan")
    # §3.4 builds the polygon from the foot landmarks *in bipedal
    # contact*, not from every one MediaPipe reports. Skipping that
    # clause is what produced supports 165 torso-lengths wide in this
    # project's own footage when a foot was lifted.
    feet = feet_in_contact(
        pf.landmarks[list(FOOT_LANDMARKS), :2],
        pf.frame_size, pf.torso_length, contact_band_torso,
    )
    if len(feet) == 0:
        # Defence in depth rather than the mechanism: an empty hull
        # already makes both P quantities NaN downstream, so a mutation
        # removing this early return passes every test (verified). It
        # stays because "no foot could be located on the floor" is a
        # distinct fact from "the arithmetic happened to yield NaN", and
        # the next person to touch the hull code should not have to
        # rediscover that the two coincide today.
        return None, None, float("nan"), float("nan")
    com = centroid(pf.mid_hip, pf.mid_shoulder, hip_weight=hip_weight)
    hull = support_polygon(feet)
    return (
        com,
        hull,
        com_support_offset(com, hull, pf.torso_length),
        support_width(hull, pf.torso_length),
    )
