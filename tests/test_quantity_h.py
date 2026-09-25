"""Unit tests for Quantity H — body height vs. personal baseline (EXPERIMENTAL).

Synthetic geometry only, same rule as T and P: every property worth trusting
must be provable on hand-built points before any video is involved. H is not
part of §3.4 and is not wired into any decision — these tests only pin the
arithmetic and the honesty of its NaN handling.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pef_fall_detector.quantities import (
    MIN_SHOULDER_PX,
    HeightBaseline,
    body_height_ratio,
    body_height_ratio_raw,
    body_vertical_extent,
    shoulder_width,
)

#: A standing subject's spine chain: nose, mid-shoulder, mid-hip, mid-ankle,
#: evenly spaced going down the image (y grows downward).
STANDING_SPINE = np.array([
    [640.0, 200.0],   # nose
    [640.0, 300.0],   # mid-shoulder
    [640.0, 500.0],   # mid-hip
    [640.0, 700.0],   # mid-ankle
])


class TestBodyVerticalExtent(unittest.TestCase):
    def test_full_chain_spans_nose_to_ankle(self) -> None:
        self.assertAlmostEqual(body_vertical_extent(STANDING_SPINE), 500.0)

    def test_missing_ankle_still_measures_from_what_remains(self) -> None:
        # Ankle occluded (out of frame or below visibility threshold): the
        # caller marks that row NaN, the span still comes from nose-to-hip.
        partial = STANDING_SPINE.copy()
        partial[3] = [np.nan, np.nan]
        self.assertAlmostEqual(body_vertical_extent(partial), 300.0)

    def test_single_visible_point_is_undefined(self) -> None:
        # A span needs two points; one point alone must not read as zero
        # height, which would look like "lying flat" rather than "unknown".
        one_point = np.full((4, 2), np.nan)
        one_point[0] = [640.0, 200.0]
        self.assertTrue(math.isnan(body_vertical_extent(one_point)))

    def test_no_visible_points_is_nan(self) -> None:
        self.assertTrue(math.isnan(body_vertical_extent(np.full((4, 2), np.nan))))

    def test_collapsed_chain_has_little_spread(self) -> None:
        # A body lying flat across the image: the spine chain's y-coordinates
        # bunch together instead of stacking vertically.
        fallen = np.array([
            [500.0, 600.0], [520.0, 604.0], [560.0, 606.0], [610.0, 608.0],
        ])
        self.assertLess(body_vertical_extent(fallen), 10.0)

    def test_extra_dimensions_are_ignored(self) -> None:
        spine_3d = np.array([
            [640.0, 200.0, -50.0], [640.0, 300.0, 10.0],
            [640.0, 500.0, 30.0], [640.0, 700.0, -20.0],
        ])
        self.assertAlmostEqual(body_vertical_extent(spine_3d), 500.0)


class TestShoulderWidth(unittest.TestCase):
    def test_plain_distance(self) -> None:
        w = shoulder_width(np.array([600.0, 300.0]), np.array([680.0, 300.0]))
        self.assertAlmostEqual(w, 80.0)

    def test_degenerate_width_is_nan(self) -> None:
        w = shoulder_width(np.array([10.0, 10.0]), np.array([10.0, 10.0]))
        self.assertTrue(math.isnan(w))

    def test_microscopic_width_is_nan(self) -> None:
        w = shoulder_width(np.array([10.0, 10.0]),
                            np.array([10.0 + MIN_SHOULDER_PX / 2, 10.0]))
        self.assertTrue(math.isnan(w))


class TestBodyHeightRatioRaw(unittest.TestCase):
    def test_combines_extent_and_width(self) -> None:
        self.assertAlmostEqual(body_height_ratio_raw(500.0, 100.0), 5.0)

    def test_nan_extent_propagates(self) -> None:
        self.assertTrue(math.isnan(body_height_ratio_raw(float("nan"), 100.0)))

    def test_nan_width_propagates(self) -> None:
        self.assertTrue(math.isnan(body_height_ratio_raw(500.0, float("nan"))))

    def test_non_positive_width_is_nan(self) -> None:
        self.assertTrue(math.isnan(body_height_ratio_raw(500.0, 0.0)))


class TestHeightBaseline(unittest.TestCase):
    """The personal calibration only moves on frames judged confidently
    standing — the same discipline as ``trunk_reference_window_s``."""

    def test_rejects_non_positive_time_constant(self) -> None:
        with self.assertRaises(ValueError):
            HeightBaseline(time_constant_s=0.0)

    def test_value_is_nan_before_any_sample(self) -> None:
        b = HeightBaseline(time_constant_s=1.0)
        self.assertTrue(math.isnan(b.value))

    def test_converges_to_a_steady_standing_ratio(self) -> None:
        b = HeightBaseline(time_constant_s=0.5)
        for _ in range(200):
            value = b.update(dt=0.033, raw_ratio=5.0, is_standing=True)
        self.assertAlmostEqual(value, 5.0, places=2)

    def test_non_standing_frames_do_not_move_the_baseline(self) -> None:
        b = HeightBaseline(time_constant_s=0.5)
        for _ in range(50):
            b.update(dt=0.033, raw_ratio=5.0, is_standing=True)
        established = b.value
        # A fall's own collapsed ratio must not recalibrate "standing".
        for _ in range(50):
            b.update(dt=0.033, raw_ratio=0.3, is_standing=False)
        self.assertAlmostEqual(b.value, established)

    def test_nan_raw_ratio_on_a_standing_frame_does_not_move_the_baseline(self) -> None:
        b = HeightBaseline(time_constant_s=0.5)
        for _ in range(50):
            b.update(dt=0.033, raw_ratio=5.0, is_standing=True)
        established = b.value
        b.update(dt=0.033, raw_ratio=float("nan"), is_standing=True)
        self.assertAlmostEqual(b.value, established)

    def test_reset_forgets_calibration(self) -> None:
        b = HeightBaseline(time_constant_s=0.5)
        b.update(dt=0.033, raw_ratio=5.0, is_standing=True)
        b.reset()
        self.assertTrue(math.isnan(b.value))


class TestBodyHeightRatio(unittest.TestCase):
    def test_standing_reads_close_to_one(self) -> None:
        self.assertAlmostEqual(body_height_ratio(5.0, 5.0), 1.0)

    def test_collapsed_reads_far_below_one(self) -> None:
        self.assertLess(body_height_ratio(0.3, 5.0), 0.1)

    def test_nan_raw_ratio_propagates(self) -> None:
        self.assertTrue(math.isnan(body_height_ratio(float("nan"), 5.0)))

    def test_unestablished_baseline_is_nan(self) -> None:
        self.assertTrue(math.isnan(body_height_ratio(5.0, float("nan"))))

    def test_non_positive_baseline_is_nan(self) -> None:
        self.assertTrue(math.isnan(body_height_ratio(5.0, 0.0)))


class TestQuantityHEndToEnd(unittest.TestCase):
    """A small synthetic scenario tying the pieces together: a subject
    calibrates while standing, then falls with their trunk pitched almost
    straight at the camera — the failure mode T cannot see (its projected
    vector nearly vanishes) but H, built from the wider spine chain and a
    yaw-resistant scale reference, still reads as collapsed.
    """

    def test_axis_aligned_fall_is_still_visible_to_h(self) -> None:
        baseline = HeightBaseline(time_constant_s=0.5)
        left_shoulder, right_shoulder = np.array([600.0, 300.0]), np.array([680.0, 300.0])
        for _ in range(100):
            raw = body_height_ratio_raw(
                body_vertical_extent(STANDING_SPINE),
                shoulder_width(left_shoulder, right_shoulder),
            )
            baseline.update(dt=0.033, raw_ratio=raw, is_standing=True)

        # The fall: the spine chain bunches together in y (foreshortened
        # trunk, the case that fools T), while shoulder width barely moves
        # because the rotation is pitch, not yaw.
        fallen_spine = np.array([
            [500.0, 590.0], [520.0, 596.0], [560.0, 602.0], [610.0, 606.0],
        ])
        fallen_shoulders = (np.array([510.0, 590.0]), np.array([530.0, 594.0]))
        raw_fallen = body_height_ratio_raw(
            body_vertical_extent(fallen_spine),
            shoulder_width(*fallen_shoulders),
        )
        h = body_height_ratio(raw_fallen, baseline.value)
        self.assertFalse(math.isnan(h))
        self.assertLess(h, 0.2)
