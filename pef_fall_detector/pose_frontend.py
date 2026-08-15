"""Pose-estimation front-end (§3.2) and Step-0 normalization.

This module wraps MediaPipe Pose and produces, for every frame, a
:class:`PoseFrame`: the 33 landmarks in **pixel coordinates** plus the
derived anchors used by every downstream quantity — mid-hip, mid-shoulder,
and torso length (§3.2 / implementation guide "Paso 0").

Dependency decision (reproducibility): MediaPipe is pinned to ``0.10.14``,
the last release that ships the classic ``mp.solutions.pose`` interface used
by the implementation guide and by most of the cited literature (Saraswat &
Malathi 2024; Sirikongtham & Nimkoompai 2025; Tran & Huynh 2025). Newer
releases (>= 0.10.2x) removed that interface in favor of the Tasks API,
which requires downloading a separate ``.task`` model file at install time.
The pinned version bundles its models inside the wheel, so the system works
fully offline — a property the edge deployment in §3.6 explicitly claims
(no cloud dependency) — and keeps the experimental setup reproducible from
``requirements.txt`` alone. Migrating to the Tasks API is a contained,
future change: only this module would be touched.

Coordinate-system decision (important, see implementation-plan §5.1):
MediaPipe returns landmarks normalized to [0, 1] by dividing x by the image
*width* and y by the image *height*. Angles computed directly on those
values are distorted by the aspect ratio (a true 45° trunk lean reads ≈29°
on a 16:9 frame), which would invalidate the paper's threshold table for
Quantity T (§3.4). We therefore de-normalize to pixels immediately —
``x_px = x * width``, ``y_px = y * height`` — and do all geometry in pixel
space. The camera-relative vertical axis of §3.4 is then simply -y.
MediaPipe's z has roughly the same scale as x, so it is multiplied by
width; it is carried for future use but no Phase-1 computation depends on
it (monocular depth is treated as low-trust, per §3.2).

Scale invariance is restored by Step-0: distances are divided by the torso
length, making downstream quantities dimensionless (§3.2). Torso length is
computed in the 2D image plane, consistent with the pixel-space geometry.

Landmark indices used throughout (MediaPipe Pose topology):
    0 nose · 11/12 shoulders (L/R) · 23/24 hips (L/R) · 27/28 ankles (L/R)
    29/30 heels · 31/32 foot index
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# MediaPipe Pose landmark indices (kept as named constants for auditability).
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

#: Landmarks every core quantity depends on; used for the reliability check.
CORE_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)


@dataclass
class PoseFrame:
    """Pose data for a single frame, in pixel coordinates.

    Attributes:
        frame_index: 0-based index within the source.
        timestamp: seconds (file time for videos, wall clock for cameras).
        detected: whether MediaPipe found a person in this frame.
        landmarks: (33, 3) array of x, y, z in pixels (empty if not detected).
        visibility: (33,) array of MediaPipe per-landmark visibility scores.
        frame_size: (width, height) of the source frame in pixels.
        mid_hip: average of landmarks 23 and 24 (Step-0), pixels.
        mid_shoulder: average of landmarks 11 and 12 (Step-0), pixels.
        torso_length: 2D distance mid_hip↔mid_shoulder, pixels (Step-0).
        core_visibility: minimum visibility among shoulders and hips —
            the single number that says whether this frame's geometry
            can be trusted.
        world_landmarks: (33, 3) array in METERS — MediaPipe's metric 3D
            estimate, hip-centered. Same inference call, second output;
            captured as **diagnostic data** (no §3.5 stage consumes it).
            Empty if not detected.
        world_torso_length: 3D mid-hip↔mid-shoulder distance in meters.
            Should stay roughly constant regardless of camera distance,
            making it a scale-invariant Step-0 sanity check.
    """

    frame_index: int
    timestamp: float
    detected: bool
    frame_size: tuple[int, int]
    landmarks: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    visibility: np.ndarray = field(default_factory=lambda: np.empty(0))
    mid_hip: np.ndarray | None = None
    mid_shoulder: np.ndarray | None = None
    torso_length: float | None = None
    core_visibility: float = 0.0
    world_landmarks: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    world_torso_length: float | None = None

    def normalize(self, distance_px: float) -> float:
        """Express a pixel distance in torso lengths (Step-0 normalization).

        Dimensionless output makes the system invariant to subject scale
        and camera distance (§3.2).
        """
        if not self.torso_length:
            return float("nan")
        return distance_px / self.torso_length


def compute_step0(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Step-0 anchors from a (33, ≥2) landmark array in pixel coordinates.

    Pure function (no MediaPipe dependency) so it can be unit-tested with
    synthetic skeletons.

    Returns:
        (mid_hip, mid_shoulder, torso_length) with midpoints as 2D pixel
        vectors and torso length as the 2D euclidean distance between them.
    """
    mid_hip = (landmarks[LEFT_HIP, :2] + landmarks[RIGHT_HIP, :2]) / 2.0
    mid_shoulder = (landmarks[LEFT_SHOULDER, :2] + landmarks[RIGHT_SHOULDER, :2]) / 2.0
    torso_length = float(np.linalg.norm(mid_shoulder - mid_hip))
    return mid_hip, mid_shoulder, torso_length


class PoseFrontend:
    """MediaPipe Pose wrapper producing :class:`PoseFrame` objects (§3.2).

    Args map 1:1 to the ``pose:`` section of ``config.yaml``.
    ``model_complexity`` is the paper's tunable complexity setting
    (0=lite, 1=full — the §3.2 default, 2=heavy for low-light scenes).
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
    ) -> None:
        # Imported here so that pure-math consumers of this module
        # (tests, quantities) never need MediaPipe installed.
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=smooth_landmarks,
        )

    def process(self, frame_bgr: np.ndarray, frame_index: int, timestamp: float) -> PoseFrame:
        """Run pose estimation on one BGR frame and derive Step-0 anchors."""
        import cv2

        height, width = frame_bgr.shape[:2]
        results = self._pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks is None:
            return PoseFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                detected=False,
                frame_size=(width, height),
            )

        raw = results.pose_landmarks.landmark
        # De-normalize to pixel space (see module docstring for rationale).
        landmarks = np.array([[lm.x * width, lm.y * height, lm.z * width] for lm in raw])
        visibility = np.array([lm.visibility for lm in raw])

        mid_hip, mid_shoulder, torso_length = compute_step0(landmarks)
        core_visibility = float(visibility[list(CORE_LANDMARKS)].min())

        # Second output of the same inference: metric 3D landmarks (meters,
        # hip-centered). Captured as diagnostic data from day one so the
        # depth signal can be evaluated empirically without re-running
        # experiments; the decision pipeline stays 2D monocular (§3.2).
        world_landmarks = np.empty((0, 3))
        world_torso_length = None
        if results.pose_world_landmarks is not None:
            world_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]
            )
            w_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
            w_shoulder = (
                world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]
            ) / 2.0
            # Full 3D distance here: world coordinates are metric, so the z
            # component is meaningful, unlike the monocular image-space z.
            world_torso_length = float(np.linalg.norm(w_shoulder - w_hip))

        return PoseFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            detected=True,
            frame_size=(width, height),
            landmarks=landmarks,
            visibility=visibility,
            mid_hip=mid_hip,
            mid_shoulder=mid_shoulder,
            torso_length=torso_length,
            core_visibility=core_visibility,
            world_landmarks=world_landmarks,
            world_torso_length=world_torso_length,
        )

    def close(self) -> None:
        """Release MediaPipe's inference graph and its native resources."""
        self._pose.close()
