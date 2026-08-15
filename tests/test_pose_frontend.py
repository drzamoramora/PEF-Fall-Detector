"""Unit tests for Step-0 normalization with synthetic skeletons.

These tests exercise the pure-math path (no MediaPipe, no video): we build
33-landmark arrays by hand and verify that mid-points and torso length
behave as §3.2 / the implementation guide's "Paso 0" prescribe. This is the
"sub-part by sub-part" verification style of the project: every quantity
must be provable on a synthetic skeleton before it ever sees a real video.

Run with:  python -m pytest tests/  (or: python -m unittest discover tests)
"""

from __future__ import annotations

import unittest

import numpy as np

from pef_fall_detector.pose_frontend import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PoseFrame,
    compute_step0,
)


def synthetic_landmarks(
    left_shoulder: tuple[float, float],
    right_shoulder: tuple[float, float],
    left_hip: tuple[float, float],
    right_hip: tuple[float, float],
) -> np.ndarray:
    """A (33, 3) landmark array with only the four core points set."""
    lms = np.zeros((33, 3))
    lms[LEFT_SHOULDER, :2] = left_shoulder
    lms[RIGHT_SHOULDER, :2] = right_shoulder
    lms[LEFT_HIP, :2] = left_hip
    lms[RIGHT_HIP, :2] = right_hip
    return lms


class TestStep0(unittest.TestCase):
    def test_midpoints_are_averages(self) -> None:
        lms = synthetic_landmarks((90, 100), (110, 100), (95, 300), (105, 300))
        mid_hip, mid_shoulder, _ = compute_step0(lms)
        np.testing.assert_allclose(mid_shoulder, [100, 100])
        np.testing.assert_allclose(mid_hip, [100, 300])

    def test_torso_length_upright(self) -> None:
        # Upright skeleton: shoulders 200px above hips -> torso = 200px.
        lms = synthetic_landmarks((90, 100), (110, 100), (95, 300), (105, 300))
        _, _, torso = compute_step0(lms)
        self.assertAlmostEqual(torso, 200.0)

    def test_torso_length_lying_down(self) -> None:
        # Same skeleton rotated 90° (lying): torso length must be identical,
        # because Step-0 is a distance, not a direction.
        lms = synthetic_landmarks((100, 90), (100, 110), (300, 95), (300, 105))
        _, _, torso = compute_step0(lms)
        self.assertAlmostEqual(torso, 200.0)

    def test_normalize_makes_distances_dimensionless(self) -> None:
        lms = synthetic_landmarks((90, 100), (110, 100), (95, 300), (105, 300))
        mid_hip, mid_shoulder, torso = compute_step0(lms)
        pf = PoseFrame(
            frame_index=0,
            timestamp=0.0,
            detected=True,
            frame_size=(640, 480),
            landmarks=lms,
            visibility=np.ones(33),
            mid_hip=mid_hip,
            mid_shoulder=mid_shoulder,
            torso_length=torso,
        )
        # A distance equal to the torso itself must normalize to exactly 1.0:
        # this is the scale invariance §3.2 claims.
        self.assertAlmostEqual(pf.normalize(torso), 1.0)
        self.assertAlmostEqual(pf.normalize(torso / 2), 0.5)


if __name__ == "__main__":
    unittest.main()
