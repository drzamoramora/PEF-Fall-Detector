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
    H — body height vs. personal baseline [EXPERIMENTAL, not in §3.4]
    R — wrist-to-ankle proximity          [EXPERIMENTAL, not in §3.4]

Quantity H is not part of the paper. It is proposed to cover two measured
gaps shared by T and P: both degrade exactly when a fall needs them most.
P becomes unmeasurable the moment feet are occluded or leave frame — the
most common failure of this project's own sensor. T reads a falsely low,
"upright"-looking angle when a fall's rotation axis points toward or away
from the camera, because the trunk vector's 2D projection nearly vanishes in
that orientation — a projection limit, not a tracking bug (§3.2 already
documents that this project does not trust MediaPipe's monocular z, which is
the only channel that could recover that rotation directly). H sidesteps
both: it reads how much vertical spread is left among whatever spine
landmarks are visible, relative to the subject's own calibrated standing
baseline, using shoulder width (not torso length) as the scale reference
because shoulder width survives the exact rotation that breaks T. Like every
``_EXP`` signal in this project, it must not enter a decision before the
draft — or here, evidence from PEF-FallDB — says so; see ``HeightBaseline``
and ``body_height_ratio`` below for the caveats that come with it.

Quantity R is not part of the paper either, and answers a different question
than H does. T, V, P, I and H all read POSTURE — how the body is arranged —
and a deliberate bend (tying a shoe, picking something off the floor)
produces close to the same posture as a fall by every one of them. None
reads INTENT, because none looks at the hands. R does: it is the distance
from wrist to ankle, and a person tying a shoe reaches for their feet while
a person falling does not. Logged only, same discipline as H, pending
calibration against labelled clips; see ``wrist_ankle_proximity`` below.

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

#: Same reasoning as MIN_TRUNK_PX, applied to Quantity H's scale reference:
#: below this shoulder width (pixels) it is too short to divide by.
MIN_SHOULDER_PX = 1e-3


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

    @property
    def value(self):
        """Current smoothed value, or NaN before the first sample."""
        return float("nan") if self._value is None else self._value

    def reset(self) -> None:
        """Forget all history (call when the subject is lost or on seek)."""
        self._value = None


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
        self._samples: list[np.ndarray] = []
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
        if len(self._samples) > self.max_samples:
            del self._samples[: len(self._samples) - self.max_samples]

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
        self._samples = [pts.copy()]
        self._started_at = float(timestamp)

    def reset(self) -> None:
        """Forget the current episode (subject lost, seek, discontinuity)."""
        self._samples = []
        self._started_at = None


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


# ---------------------------------------------------------------------------
# Quantity A — head height against gravity, in 3D (EXPERIMENTAL, fase 8)
# ---------------------------------------------------------------------------
#
# Not in §3.4, and in tension with §3.2, which evaluates all geometry in 2D
# pixel space and excludes "the monocular z-coordinate … because its estimated
# depth is not reliable as metric geometry". A does not use that image z. It
# uses MediaPipe's *world* landmarks (metric, hip-centred, the second output of
# the same inference), and only as a RATIO against the same subject's own
# standing height — in the spirit of the "displacement ratios rather than …
# absolute 3D distances" that §3.2 proposes. Whether that ratio is reliable
# enough is exactly what fase 8.1 measures, on every clip and both platforms.
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
# floor −0.21–0.25. Logged only in fase 8.1: no stage reads it.

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
        for h, k, a in zip(_W_HIPS, _W_KNEES, _W_ANKLES)
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
        self._ups.clear(); self._heads.clear(); self._hips.clear()
        self._standing_s = 0.0
        self._last_t = None
        self._cache = None
