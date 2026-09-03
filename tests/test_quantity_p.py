"""Unit tests for Quantity P — COM vs. support polygon (§3.4).

Synthetic geometry only. The support polygon and the COM offset are pure
functions of six foot landmarks and two body anchors, so every property
worth trusting can be proven on hand-built points before any video is
involved — the same rule applied to T and V.

What these tests pin, and why each matters:

* the hull is a hull (it contains its inputs, and drops interior points);
* the sign convention, because a single threshold at 0 carries the entire
  inside/outside decision of Stage 2;
* torso normalisation, because §3.4 reports P dimensionless and a P that
  scaled with camera distance would be uncalibratable;
* degenerate supports (one foot, collinear feet), which are legitimate
  observations and must not raise or silently produce a wrong hull.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pef_fall_detector.quantities import (
    com_support_offset,
    feet_in_contact,
    support_polygon,
    support_width,
)

#: A plausible two-foot stance in pixels: ankles 60 px apart, heels behind
#: and toes ahead, as a standing subject seen from the front.
FEET = np.array([
    [610.0, 700.0], [670.0, 700.0],     # ankles
    [600.0, 705.0], [660.0, 705.0],     # heels
    [630.0, 708.0], [690.0, 708.0],     # toes
])
TORSO = 200.0


class TestBipedalContact(unittest.TestCase):
    """§3.4's "in bipedal contact" clause, which the code used to ignore.

    The measured consequence of ignoring it: a lifted foot in this project's
    own live session produced a support polygon 165 torso-lengths wide,
    because MediaPipe kept reporting a position for a foot it could no longer
    see and the geometry was built on that guess.
    """

    FRAME = (1280, 960)

    def test_both_feet_on_the_floor_are_all_kept(self) -> None:
        kept = feet_in_contact(FEET, self.FRAME, TORSO)
        self.assertEqual(len(kept), len(FEET))

    def test_a_lifted_foot_leaves_the_polygon(self) -> None:
        # Right foot raised 60 px = 0.3 torso, twice the contact band.
        lifted = FEET.copy()
        lifted[[1, 3, 5], 1] -= 60.0
        kept = feet_in_contact(lifted, self.FRAME, TORSO)
        self.assertEqual(len(kept), 3, "solo el pie apoyado deberia quedar")
        self.assertLess(kept[:, 0].max(), 665.0, "quedo un punto del pie alzado")

    def test_the_polygon_contracts_when_a_foot_lifts(self) -> None:
        # §3.4's second signature, now actually observable: "a sudden
        # contraction of the support polygon from a full two-foot footprint
        # to a heel-only contact pair". Before this filter the same event
        # made the polygon WIDER, which is the opposite of the truth.
        both = support_width(support_polygon(feet_in_contact(FEET, self.FRAME, TORSO)), TORSO)
        lifted = FEET.copy()
        lifted[[1, 3, 5], 1] -= 60.0
        one = support_width(support_polygon(feet_in_contact(lifted, self.FRAME, TORSO)), TORSO)
        self.assertLess(one, both)

    def test_landmarks_outside_the_frame_are_discarded(self) -> None:
        # The failure that motivated this: a foot placed far outside the
        # image is the model extrapolating, not an observation. Same rule
        # already applied to the core landmarks since Phase 2.
        stray = FEET.copy()
        stray[1] = [40000.0, 30000.0]          # nowhere near the image
        kept = feet_in_contact(stray, self.FRAME, TORSO)
        self.assertEqual(len(kept), len(FEET) - 1)
        self.assertLess(support_width(support_polygon(kept), TORSO), 3.0)

    def test_an_extrapolated_point_cannot_redefine_the_floor(self) -> None:
        # Nastier variant: the stray point is BELOW the real feet, so a naive
        # "lowest point is the floor" rule would treat it as the ground and
        # discard the real feet instead.
        stray = FEET.copy()
        stray[1] = [700.0, 5000.0]             # below the frame
        kept = feet_in_contact(stray, self.FRAME, TORSO)
        self.assertEqual(len(kept), len(FEET) - 1)

    def test_no_locatable_foot_gives_an_empty_set(self) -> None:
        # Which downstream becomes an undefined P, never an invented one.
        gone = np.array([[-500.0, -500.0], [40000.0, 40000.0]])
        self.assertEqual(len(feet_in_contact(gone, self.FRAME, TORSO)), 0)

    def test_the_band_is_in_torso_lengths(self) -> None:
        # The same posture filmed twice as large must classify identically.
        lifted = FEET.copy()
        lifted[[1, 3, 5], 1] -= 60.0
        near = feet_in_contact(lifted, self.FRAME, TORSO)
        far = feet_in_contact(lifted * 2.0, (2560, 1920), TORSO * 2.0)
        self.assertEqual(len(near), len(far))

    def test_unusable_scale_gives_an_empty_set(self) -> None:
        self.assertEqual(len(feet_in_contact(FEET, self.FRAME, 0.0)), 0)


class TestSupportPolygon(unittest.TestCase):
    """The convex hull of the foot landmarks."""

    def test_hull_contains_the_extremes(self) -> None:
        hull = support_polygon(FEET)
        self.assertAlmostEqual(hull[:, 0].min(), FEET[:, 0].min())
        self.assertAlmostEqual(hull[:, 0].max(), FEET[:, 0].max())

    def test_interior_point_is_dropped(self) -> None:
        # A square plus its centre: the centre is inside, so a correct hull
        # has four vertices, not five.
        square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
                           [5.0, 5.0]])
        self.assertEqual(len(support_polygon(square)), 4)

    def test_single_foot_is_returned_as_is(self) -> None:
        # One visible foot is a legitimate support (standing on one leg, or
        # the other foot cropped). A hull is undefined; the point stands.
        one = np.array([[100.0, 200.0]])
        self.assertEqual(len(support_polygon(one)), 1)

    def test_collinear_feet_do_not_produce_a_degenerate_hull(self) -> None:
        # Feet seen edge-on project onto a line. The extent must survive.
        line = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        hull = support_polygon(line)
        self.assertAlmostEqual(hull[:, 0].min(), 0.0)
        self.assertAlmostEqual(hull[:, 0].max(), 10.0)


class TestComSupportOffset(unittest.TestCase):
    """Quantity P proper: where the COM falls relative to the feet."""

    def setUp(self) -> None:
        self.hull = support_polygon(FEET)

    def test_com_over_the_feet_is_negative(self) -> None:
        # Standing balanced: COM inside, so the offset is negative and its
        # magnitude is the margin to the nearest edge.
        p = com_support_offset(np.array([640.0, 400.0]), self.hull, TORSO)
        self.assertLess(p, 0.0)

    def test_com_beyond_the_toes_is_positive(self) -> None:
        # Toppling forward past the front of the support: the §3.4 fall
        # signature.
        p = com_support_offset(np.array([800.0, 400.0]), self.hull, TORSO)
        self.assertGreater(p, 0.0)
        # 800 is 110 px past the rightmost foot point (690), over a 200 px
        # torso.
        self.assertAlmostEqual(p, 110.0 / TORSO, places=6)

    def test_the_sign_flips_exactly_at_the_edge(self) -> None:
        edge = float(self.hull[:, 0].max())
        self.assertLessEqual(com_support_offset(np.array([edge - 1.0, 0.0]),
                                                self.hull, TORSO), 0.0)
        self.assertGreater(com_support_offset(np.array([edge + 1.0, 0.0]),
                                              self.hull, TORSO), 0.0)

    def test_offset_is_symmetric_left_and_right(self) -> None:
        left, right = float(self.hull[:, 0].min()), float(self.hull[:, 0].max())
        a = com_support_offset(np.array([left - 50.0, 0.0]), self.hull, TORSO)
        b = com_support_offset(np.array([right + 50.0, 0.0]), self.hull, TORSO)
        self.assertAlmostEqual(a, b, places=6)

    def test_p_is_invariant_to_camera_distance(self) -> None:
        # Step-0's whole purpose: the same posture filmed twice as large must
        # give the same dimensionless P. Scale every pixel by 2 and the torso
        # with it.
        near = com_support_offset(np.array([800.0, 400.0]), self.hull, TORSO)
        far = com_support_offset(np.array([1600.0, 800.0]),
                                 support_polygon(FEET * 2.0), TORSO * 2.0)
        self.assertAlmostEqual(near, far, places=6)

    def test_missing_support_is_nan_not_zero(self) -> None:
        # No feet means no geometry. Zero would read as "COM exactly on the
        # edge", which is a decision; NaN is the absence of one.
        self.assertTrue(math.isnan(
            com_support_offset(np.array([640.0, 400.0]), np.empty((0, 2)), TORSO)))

    def test_unusable_torso_is_nan(self) -> None:
        for bad in (0.0, -5.0):
            self.assertTrue(math.isnan(
                com_support_offset(np.array([640.0, 400.0]), self.hull, bad)))


class TestSupportWidth(unittest.TestCase):
    """§3.4's second signature: the support contracting under the subject."""

    def test_width_matches_the_foot_extent(self) -> None:
        w = support_width(support_polygon(FEET), TORSO)
        self.assertAlmostEqual(w, (690.0 - 600.0) / TORSO, places=6)

    def test_contraction_to_heels_only_shrinks_the_width(self) -> None:
        # The transition §3.4 names: "from a full two-foot footprint to a
        # heel-only contact pair".
        full = support_width(support_polygon(FEET), TORSO)
        heels = support_width(support_polygon(FEET[2:4]), TORSO)
        self.assertLess(heels, full)

    def test_width_is_invariant_to_camera_distance(self) -> None:
        near = support_width(support_polygon(FEET), TORSO)
        far = support_width(support_polygon(FEET * 3.0), TORSO * 3.0)
        self.assertAlmostEqual(near, far, places=6)

    def test_missing_support_is_nan(self) -> None:
        self.assertTrue(math.isnan(support_width(np.empty((0, 2)), TORSO)))


if __name__ == "__main__":
    unittest.main()
