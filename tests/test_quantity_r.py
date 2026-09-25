"""Unit tests for Quantity R — wrist-to-ankle proximity (EXPERIMENTAL).

Synthetic geometry only, same rule as every other quantity in this project:
provable on hand-built points before any video is involved. R is not part of
§3.4 and is not wired into any decision — these tests only pin the
arithmetic and the honesty of its NaN handling.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pef_fall_detector.quantities import wrist_ankle_proximity

TORSO = 200.0


class TestWristAnkleProximity(unittest.TestCase):
    def test_reaching_for_a_foot_reads_close(self) -> None:
        # Bent forward, right hand near the right foot: a shoelace-tying
        # posture.
        wrists = np.array([[300.0, 400.0], [615.0, 690.0]])   # left far, right near
        ankles = np.array([[400.0, 700.0], [610.0, 700.0]])
        r = wrist_ankle_proximity(wrists, ankles, TORSO)
        self.assertAlmostEqual(r, np.hypot(5.0, 10.0) / TORSO, places=6)

    def test_standing_normally_reads_far(self) -> None:
        # Arms at the sides, nowhere near the feet.
        wrists = np.array([[400.0, 450.0], [880.0, 450.0]])
        ankles = np.array([[410.0, 700.0], [870.0, 700.0]])
        r = wrist_ankle_proximity(wrists, ankles, TORSO)
        self.assertGreater(r, 1.0)   # more than a full torso length away

    def test_takes_the_closest_of_the_four_pairs(self) -> None:
        wrists = np.array([[0.0, 0.0], [610.0, 695.0]])   # left useless, right close
        ankles = np.array([[1000.0, 1000.0], [610.0, 700.0]])
        r = wrist_ankle_proximity(wrists, ankles, TORSO)
        self.assertAlmostEqual(r, 5.0 / TORSO, places=6)

    def test_one_wrist_occluded_still_measures_from_the_other(self) -> None:
        wrists = np.array([[np.nan, np.nan], [610.0, 695.0]])
        ankles = np.array([[400.0, 700.0], [610.0, 700.0]])
        r = wrist_ankle_proximity(wrists, ankles, TORSO)
        self.assertAlmostEqual(r, 5.0 / TORSO, places=6)

    def test_one_ankle_occluded_still_measures_from_the_other(self) -> None:
        wrists = np.array([[610.0, 695.0], [np.nan, np.nan]])
        ankles = np.array([[610.0, 700.0], [np.nan, np.nan]])
        r = wrist_ankle_proximity(wrists, ankles, TORSO)
        self.assertAlmostEqual(r, 5.0 / TORSO, places=6)

    def test_no_wrist_visible_is_nan(self) -> None:
        wrists = np.full((2, 2), np.nan)
        ankles = np.array([[400.0, 700.0], [610.0, 700.0]])
        self.assertTrue(math.isnan(wrist_ankle_proximity(wrists, ankles, TORSO)))

    def test_no_ankle_visible_is_nan(self) -> None:
        wrists = np.array([[400.0, 450.0], [880.0, 450.0]])
        ankles = np.full((2, 2), np.nan)
        self.assertTrue(math.isnan(wrist_ankle_proximity(wrists, ankles, TORSO)))

    def test_unusable_torso_length_is_nan(self) -> None:
        wrists = np.array([[610.0, 695.0], [610.0, 695.0]])
        ankles = np.array([[610.0, 700.0], [610.0, 700.0]])
        self.assertTrue(math.isnan(wrist_ankle_proximity(wrists, ankles, 0.0)))

    def test_extra_dimensions_are_ignored(self) -> None:
        wrists = np.array([[610.0, 695.0, 12.0], [610.0, 695.0, -8.0]])
        ankles = np.array([[610.0, 700.0, 3.0], [610.0, 700.0, 40.0]])
        r = wrist_ankle_proximity(wrists, ankles, TORSO)
        self.assertAlmostEqual(r, 5.0 / TORSO, places=6)
