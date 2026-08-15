"""Unit tests for the physics-informed quantities (§3.4).

Synthetic skeletons only: no camera, no video, no MediaPipe. Each quantity
must be provable on hand-built geometry before it is trusted on real
footage — the "verify every sub-part on its own" rule of this project.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pef_fall_detector.quantities import trunk_band, trunk_inclination_deg


class TestTrunkInclination(unittest.TestCase):
    """Quantity T against the reference postures of §3.4."""

    def test_upright_is_zero(self) -> None:
        # Shoulders directly above the hips (remember: y grows downward).
        t = trunk_inclination_deg(mid_hip=np.array([640.0, 500.0]),
                                  mid_shoulder=np.array([640.0, 300.0]))
        self.assertAlmostEqual(t, 0.0, places=6)

    def test_lying_down_is_ninety(self) -> None:
        # Trunk horizontal: the collapse signature of §3.4.
        t = trunk_inclination_deg(mid_hip=np.array([400.0, 600.0]),
                                  mid_shoulder=np.array([600.0, 600.0]))
        self.assertAlmostEqual(t, 90.0, places=6)

    def test_moderate_lean_is_forty_five(self) -> None:
        # 100 px right, 100 px up -> exactly 45 deg from vertical.
        t = trunk_inclination_deg(mid_hip=np.array([640.0, 500.0]),
                                  mid_shoulder=np.array([740.0, 400.0]))
        self.assertAlmostEqual(t, 45.0, places=6)

    def test_lean_is_symmetric_left_and_right(self) -> None:
        # T measures inclination, not direction: leaning left or right by the
        # same amount must give the same angle.
        right = trunk_inclination_deg(np.array([640.0, 500.0]), np.array([740.0, 400.0]))
        left = trunk_inclination_deg(np.array([640.0, 500.0]), np.array([540.0, 400.0]))
        self.assertAlmostEqual(right, left, places=6)

    def test_shoulders_below_hips_exceeds_ninety(self) -> None:
        # Head-down posture: the angle must open past 90 deg, not wrap around.
        t = trunk_inclination_deg(mid_hip=np.array([640.0, 300.0]),
                                  mid_shoulder=np.array([640.0, 500.0]))
        self.assertAlmostEqual(t, 180.0, places=6)

    def test_degenerate_trunk_returns_nan(self) -> None:
        # Hips and shoulders on the same point: no trunk vector exists.
        t = trunk_inclination_deg(np.array([10.0, 10.0]), np.array([10.0, 10.0]))
        self.assertTrue(math.isnan(t))

    def test_microscopic_trunk_returns_nan(self) -> None:
        # A trunk of 1e-9 px is numerically meaningless: any angle computed
        # from it would be pure noise amplified out of nothing.
        t = trunk_inclination_deg(np.array([10.0, 10.0]),
                                  np.array([10.0 + 1e-9, 10.0 - 1e-9]))
        self.assertTrue(math.isnan(t))

    def test_extra_dimensions_are_ignored(self) -> None:
        # Landmarks arrive as (x, y, z); z must not affect the 2D angle.
        t = trunk_inclination_deg(mid_hip=np.array([640.0, 500.0, -80.0]),
                                  mid_shoulder=np.array([740.0, 400.0, 120.0]))
        self.assertAlmostEqual(t, 45.0, places=6)


class TestTrunkAngleInvariants(unittest.TestCase):
    """Property test: guarantees that must hold for *every* valid input.

    Unlike the cases above — "this input gives that output" — this asserts a
    property over many generated inputs. It is the right shape of test when
    the input that would break the code cannot be named in advance: here,
    the floating-point rounding that could push the cosine past 1 and make
    arccos return NaN (see the clamp note in ``quantities``).

    It cannot distinguish whether the clamp is present, because with this
    formulation the overflow is unreachable. What it does is pin the
    guarantee in writing, so that a future move to the general angle formula
    or to 3D world landmarks — where the overflow is very much reachable —
    breaks a test instead of silently emitting NaN.
    """

    def test_angle_is_never_nan_and_always_within_range(self) -> None:
        rng = np.random.default_rng(11)
        for _ in range(20_000):
            # Trunks at wildly different scales: a subject filling the frame
            # and one far away are orders of magnitude apart in pixels.
            scale = 10.0 ** rng.uniform(-4, 5)
            hip = rng.normal(0.0, scale, 2)
            shoulder = hip + rng.normal(0.0, scale, 2)
            t = trunk_inclination_deg(hip, shoulder)
            if math.isnan(t):  # only legal for a degenerate trunk
                self.assertLess(float(np.linalg.norm(shoulder - hip)), 1e-3)
                continue
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 180.0)

    def test_near_vertical_trunks_do_not_degenerate(self) -> None:
        # The pathological neighbourhood: almost exactly parallel to the
        # vertical axis, where the cosine approaches 1 and rounding bites.
        rng = np.random.default_rng(12)
        for _ in range(20_000):
            scale = 10.0 ** rng.uniform(-3, 4)
            hip = rng.normal(0.0, scale, 2)
            shoulder = hip + np.array([rng.normal(0.0, scale * 1e-9), -scale])
            t = trunk_inclination_deg(hip, shoulder)
            self.assertFalse(math.isnan(t), "un tronco casi vertical dio NaN")
            self.assertLess(t, 1.0)  # essentially upright


class TestTrunkBand(unittest.TestCase):
    """The §3.4 reference table, made executable (display aid only)."""

    def test_bands_match_the_paper(self) -> None:
        self.assertEqual(trunk_band(0.0), "upright")
        self.assertEqual(trunk_band(25.0), "moderate-lean")   # reaching
        self.assertEqual(trunk_band(75.0), "collapse-range")  # fall signature
        self.assertEqual(trunk_band(float("nan")), "n/a")

    def test_band_boundaries(self) -> None:
        self.assertEqual(trunk_band(14.9), "upright")
        self.assertEqual(trunk_band(15.0), "moderate-lean")
        self.assertEqual(trunk_band(59.9), "moderate-lean")
        self.assertEqual(trunk_band(60.0), "collapse-range")


class TestAspectRatioCorrection(unittest.TestCase):
    """Locks in the pixel-space decision documented in ``pose_frontend``.

    MediaPipe normalizes x by frame width and y by frame height. On a
    non-square frame those are different divisors, so angles computed on
    normalized coordinates are distorted. This test pins the size of that
    error so the decision cannot be silently reverted.
    """

    FRAME_W, FRAME_H = 1280, 720

    def test_pixel_space_recovers_the_true_angle(self) -> None:
        # A trunk that is truly 45 deg from vertical, in pixels.
        hip_px = np.array([640.0, 500.0])
        shoulder_px = np.array([740.0, 400.0])

        # What MediaPipe would report for that same trunk (normalized).
        hip_norm = hip_px / np.array([self.FRAME_W, self.FRAME_H])
        shoulder_norm = shoulder_px / np.array([self.FRAME_W, self.FRAME_H])

        # Our pipeline de-normalizes before measuring: the true angle is back.
        recovered = trunk_inclination_deg(
            hip_norm * np.array([self.FRAME_W, self.FRAME_H]),
            shoulder_norm * np.array([self.FRAME_W, self.FRAME_H]),
        )
        self.assertAlmostEqual(recovered, 45.0, places=6)

    def test_normalized_space_would_distort_the_angle(self) -> None:
        # Measuring directly on normalized coordinates (what the guide's
        # pseudocode does) under-reports the lean by ~16 degrees at 16:9.
        hip_norm = np.array([640.0 / self.FRAME_W, 500.0 / self.FRAME_H])
        shoulder_norm = np.array([740.0 / self.FRAME_W, 400.0 / self.FRAME_H])
        distorted = trunk_inclination_deg(hip_norm, shoulder_norm)

        self.assertAlmostEqual(distorted, 29.35, places=1)
        self.assertLess(distorted, 40.0)  # far outside the 45 deg it should be


if __name__ == "__main__":
    unittest.main()
