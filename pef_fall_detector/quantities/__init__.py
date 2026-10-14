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
and ``body_height_ratio`` in :mod:`.h` for the caveats that come with it.
**That rule is not met today:** ``config.yaml`` ships H's thresholds on,
with values chosen for plausibility and never swept on PEF-FallDB, and H
decides — in the phase-8.4 records it raised 25 of the 90 Stage-1 events in
the cloud and 21 of 90 on the Mac (``trigger_source`` "H"). Whether to
describe it in the paper or switch it off is open
(``notas/POSIBLES-CAMBIOS-DEL-DRAFT.md`` C19).

Quantity R is not part of the paper either, and answers a different question
than H does. T, V, P, I and H all read POSTURE — how the body is arranged —
and a deliberate bend (tying a shoe, picking something off the floor)
produces close to the same posture as a fall by every one of them. None
reads INTENT, because none looks at the hands. R does: it is the distance
from wrist to ankle, and a person tying a shoe reaches for their feet while
a person falling does not. Logged only, pending calibration against
labelled clips; see ``wrist_ankle_proximity`` in :mod:`.r`.

Sign convention for velocities (IMPORTANT): V follows the paper's physical
convention — **negative = moving down** (a fall produces a large negative V,
§3.4; the guide's Stage-1 trigger tests ``V < threshold_V`` with a negative
threshold). Since image y grows *downward*, the implementation negates the
raw dy/dt. Note that the guide's own pseudocode returns ``(y_now - y_prev)/dt``
unnegated, which yields POSITIVE values for downward motion — inconsistent
with its own negative threshold. This implementation resolves that
inconsistency in favor of the physical convention.

**Layout (0.2.0.dev4).** One module per quantity: :mod:`.t`, :mod:`.v`,
:mod:`.p` and :mod:`.i` are the four of §3.4 (:data:`PAPER`); :mod:`.h`,
:mod:`.r`, :mod:`.a` and the leg :mod:`.extension` are the experimental ones
(:data:`EXPERIMENTAL`),
whose record columns carry the ``_EXP`` mark (``audit_log``). Adopting one
into the paper means moving it from one list to the other, and dropping its
``_EXP`` mark in ``audit_log.EXPERIMENTAL_QUANTITIES``. :mod:`.filters` holds
the smoothing they share. Every name is re-exported here, so
``from pef_fall_detector.quantities import X`` reads as it always did.
"""

from __future__ import annotations

from . import a, extension, h, i, p, r, t, v
from .a import GravityCalibrator, knee_angle_3d
from .extension import LegExtension
from .filters import ExponentialMovingAverage
from .h import (
    MIN_SHOULDER_PX,
    HeightBaseline,
    body_height_ratio,
    body_height_ratio_raw,
    body_vertical_extent,
    shoulder_width,
)
from .i import ImmobilityTimer
from .p import com_support_offset, feet_in_contact, support_polygon, support_width
from .r import wrist_ankle_proximity
from .t import MIN_TRUNK_PX, VERTICAL_AXIS, TrunkGuard, trunk_band, trunk_inclination_deg
from .v import VelocityEstimator, centroid

#: The quantities §3.4 defines, and the ones it does not (yet), by letter.
PAPER = {"T": t, "V": v, "P": p, "I": i}
EXPERIMENTAL = {"H": h, "R": r, "A": a, "extension": extension}

__all__ = [
    "EXPERIMENTAL", "MIN_SHOULDER_PX", "MIN_TRUNK_PX", "PAPER", "VERTICAL_AXIS",
    "ExponentialMovingAverage", "GravityCalibrator", "HeightBaseline", "ImmobilityTimer",
    "LegExtension", "TrunkGuard",
    "VelocityEstimator", "body_height_ratio", "body_height_ratio_raw",
    "body_vertical_extent", "centroid", "com_support_offset", "feet_in_contact",
    "knee_angle_3d", "shoulder_width", "support_polygon", "support_width", "trunk_band",
    "trunk_inclination_deg", "wrist_ankle_proximity",
]
