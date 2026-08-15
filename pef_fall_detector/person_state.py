"""Person-state display labels (Phase 2.4).

Rule-based classification of what the subject is doing, derived from the
same continuous features the pipeline already computes — no extra sensing,
no learning. The labels are **display and audit-log aids only**: no stage
of the paper's decision logic (§3.5) consumes them. Their value is
twofold: they let a human verify at a glance that the quantities are
measuring what they should ("it says SITTING while I sit"), and they give
every alert a human-readable prelude ("fall; subject was WALKING before"),
which strengthens the explainability story.

Features used (all dimensionless or angular — subject- and distance-invariant):

* ``t_deg`` — Quantity T, trunk inclination.
* ``extension_ratio`` — vertical hip-to-ankle extent divided by torso
  length. This, and NOT the centroid's height in the image, is the correct
  posture feature: image height depends on where the person stands in the
  frame, while this ratio collapses when the legs fold (sitting, crouching)
  regardless of position or distance. Standing ≈ 1.3-2.0; sitting reduces
  it; a deep crouch or lying collapses it toward 0.
* ``v_tps`` / ``vh_tps`` — vertical and horizontal centroid velocity in
  torso-lengths/second (NaN while the velocity window fills).

Classification is ordered by *specificity*: transition (fast vertical
motion) beats every static posture; lying (trunk horizontal) beats the
leg-based labels; leg extension separates crouch / sit / upright; and only
an otherwise-upright subject can be WALKING. When the ankles are not
visible, ``extension_ratio`` is NaN and the classifier falls back to
trunk-only labels — coarser, but honest.

All thresholds live in ``config.yaml`` under ``state_display`` (they will
be tuned against the calibration clips in 2.6), and the section is
explicitly marked as display-only so the §3.5 calibration surface stays
clean. The label vocabulary and boundaries are provisional until then.
"""

from __future__ import annotations

import math

# Label vocabulary (English in code/CSV; the project logbook maps them to
# the Spanish working vocabulary: DE_PIE, CAMINANDO, INCLINADO, AGACHADO,
# SENTADO, ACOSTADO, EN_TRANSICION, NO_DETECTADO).
STANDING = "STANDING"
WALKING = "WALKING"
LEANING = "LEANING"
CROUCHING = "CROUCHING"
SITTING = "SITTING"
LYING = "LYING"
TRANSITION = "TRANSITION"
NOT_DETECTED = "NOT_DETECTED"


def classify_state(
    t_deg: float,
    extension_ratio: float,
    v_tps: float,
    vh_tps: float,
    *,
    transition_v_tps: float,
    lying_t_deg: float,
    leaning_t_deg: float,
    standing_extension: float,
    crouch_extension: float,
    walking_vh_tps: float,
) -> str:
    """Map the frame's continuous features to one display label.

    Pure function; thresholds arrive as explicit keyword arguments (the
    pipeline reads them from ``config.yaml``'s ``state_display`` section).
    NaN inputs are legitimate — T NaN means no detection, extension NaN
    means occluded ankles, velocity NaN means the window is still filling —
    and each has a defined fallback path.
    """
    if math.isnan(t_deg):
        return NOT_DETECTED

    # Fast vertical motion overrides any static posture: the body is
    # between postures (sitting down, standing up, dropping...).
    if not math.isnan(v_tps) and abs(v_tps) >= transition_v_tps:
        return TRANSITION

    # Trunk horizontal: lying, whatever the legs are doing.
    if t_deg >= lying_t_deg:
        return LYING

    if math.isnan(extension_ratio):
        # Ankles occluded: trunk-only fallback (coarse but honest).
        return LEANING if t_deg >= leaning_t_deg else STANDING

    if extension_ratio < crouch_extension:
        return CROUCHING
    if extension_ratio < standing_extension:
        return SITTING

    # Legs extended, trunk not horizontal: upright family.
    if t_deg >= leaning_t_deg:
        return LEANING
    if not math.isnan(vh_tps) and abs(vh_tps) >= walking_vh_tps:
        return WALKING
    return STANDING
