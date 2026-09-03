"""Unit tests for Quantity I — immobility duration (§3.4).

Synthetic landmark tracks only: no camera, no video. The timer is a pure
function of positions and timestamps, so everything worth trusting can be
proven on hand-built motion.

The properties pinned here, and why each one earns a test:

* time accumulates while still, and only while still — the whole quantity;
* the clock is **wall-clock**, not a frame count, so a slow device and a
  fast one report the same immobility for the same event (the C5 lesson,
  applied to a quantity that did not exist when C5 was found);
* leaving the radius resets to zero, so a subject who stirs cannot bank
  earlier stillness toward a confirmation threshold;
* the radius is dimensionless in torso lengths, so the same motion at two
  camera distances gives the same verdict;
* an occluded nose does not end the episode — that is the face-down case
  the quantity exists for.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pef_fall_detector.quantities import ImmobilityTimer

TORSO = 200.0
EPS = 0.05                      # torso lengths -> 10 px at this scale


def still_points(x: float = 640.0, y: float = 500.0) -> np.ndarray:
    """Five head-and-torso landmarks in a fixed arrangement."""
    return np.array([
        [x, y - 120.0],                       # nose
        [x - 40.0, y - 60.0], [x + 40.0, y - 60.0],   # shoulders
        [x - 30.0, y], [x + 30.0, y],                 # hips
    ])


class TestStillnessAccumulates(unittest.TestCase):

    def test_a_motionless_subject_accumulates_wall_clock_time(self) -> None:
        timer = ImmobilityTimer(EPS)
        pts = still_points()
        for i in range(31):
            secs, disp = timer.update(i / 10.0, pts, TORSO)
        self.assertAlmostEqual(secs, 3.0, places=6)
        self.assertAlmostEqual(disp, 0.0, places=6)

    def test_the_clock_is_time_not_frames(self) -> None:
        # The same three seconds of stillness sampled at two rates must
        # report the same I. A frame counter would report 30 and 90.
        def run(fps: float) -> float:
            timer = ImmobilityTimer(EPS)
            pts, secs = still_points(), 0.0
            for i in range(int(3.0 * fps) + 1):
                secs, _ = timer.update(i / fps, pts, TORSO)
            return secs
        self.assertAlmostEqual(run(10.0), run(30.0), places=6)
        self.assertAlmostEqual(run(10.0), 3.0, places=6)

    def test_jitter_under_the_radius_does_not_stop_the_clock(self) -> None:
        # Landmark noise is always present; only real motion should count.
        rng = np.random.default_rng(3)
        timer = ImmobilityTimer(EPS)
        base, secs = still_points(), 0.0
        for i in range(51):
            noisy = base + rng.normal(0.0, 1.5, base.shape)   # ~1.5 px
            secs, _ = timer.update(i / 10.0, noisy, TORSO)
        self.assertGreater(secs, 4.0)


class TestMotionResets(unittest.TestCase):

    def test_leaving_the_radius_zeroes_the_clock(self) -> None:
        timer = ImmobilityTimer(EPS)
        for i in range(21):
            secs, _ = timer.update(i / 10.0, still_points(), TORSO)
        self.assertAlmostEqual(secs, 2.0, places=6)
        # Move 20 px = 0.10 torso, twice the radius.
        secs, disp = timer.update(2.1, still_points(x=660.0), TORSO)
        self.assertEqual(secs, 0.0)
        self.assertGreater(disp, EPS)

    def test_stillness_cannot_be_banked_across_a_movement(self) -> None:
        # Four seconds still, a step at t=4.1, four more still. I must report
        # only the time since the step — never the sum of both episodes,
        # which would let a subject who stirred reach a confirmation
        # threshold on stillness they had already spent.
        timer = ImmobilityTimer(EPS)
        for i in range(41):
            timer.update(i / 10.0, still_points(), TORSO)
        moved_at = 4.1
        timer.update(moved_at, still_points(x=700.0), TORSO)
        for i in range(41):
            end = 4.2 + i / 10.0
            secs, _ = timer.update(end, still_points(x=700.0), TORSO)
        # The new episode is anchored at the frame where the new position was
        # first seen, so the clock runs from there.
        self.assertAlmostEqual(secs, end - moved_at, places=6)
        self.assertLess(secs, 8.0)

    def test_slow_drift_eventually_breaks_the_episode(self) -> None:
        # A creeping subject must not read as immobile forever. The anchor is
        # fixed at the episode start precisely so that total displacement is
        # what counts, not per-frame displacement.
        timer = ImmobilityTimer(EPS)
        broke = False
        for i in range(60):
            secs, _ = timer.update(i / 10.0, still_points(x=640.0 + i * 0.5), TORSO)
            if i > 0 and secs == 0.0:
                broke = True
                break
        self.assertTrue(broke, "una deriva lenta deberia terminar el episodio")

    def test_the_radius_is_measured_around_the_mean_not_the_first_frame(self) -> None:
        """§3.4: "within a small radius ε around their **average** position".

        The observable difference: a first frame that is itself an outlier.
        Anchored to it, the rest of a perfectly still episode sits a full
        outlier away and the clock keeps restarting. Anchored to the mean,
        that one frame is diluted and the stillness is seen for what it is.
        """
        timer = ImmobilityTimer(EPS)
        # One frame 9 px off (just inside ε = 10 px), then 40 still frames.
        timer.update(0.0, still_points(x=649.0), TORSO)
        secs = 0.0
        for i in range(1, 41):
            secs, _ = timer.update(i / 10.0, still_points(), TORSO)
        self.assertAlmostEqual(secs, 4.0, places=6)

    def test_averaging_absorbs_landmark_noise_better_than_one_frame(self) -> None:
        # With the anchor averaged over the episode, the reported deviation
        # of a noisy but motionless subject stays well under the radius —
        # the property that lets ε be set close to the real motion scale
        # instead of being padded for one frame's bad luck.
        rng = np.random.default_rng(5)
        timer = ImmobilityTimer(EPS)
        base, worst = still_points(), 0.0
        for i in range(80):
            _, disp = timer.update(i / 10.0, base + rng.normal(0.0, 2.0, base.shape),
                                   TORSO)
            worst = max(worst, disp)
        self.assertLess(worst, EPS)

    def test_drifting_then_holding_still_breaks_the_episode(self) -> None:
        """Why EVERY sample is re-checked, not only the newest.

        A subject who slides a little and then holds the new position: the
        running mean converges onto where they ended up, so the newest frame
        looks perfectly still from then on. It is the frames from the start
        of the episode that are now far from the mean. Comparing only the
        latest sample would report this as uninterrupted immobility and let
        someone shift position without ever restarting the clock.
        """
        timer = ImmobilityTimer(EPS)
        broke = False
        for i in range(4):                       # slide 12 px in 4 frames
            timer.update(i / 10.0, still_points(x=640.0 + i * 4.0), TORSO)
        for i in range(4, 60):                   # then hold, perfectly still
            secs, _ = timer.update(i / 10.0, still_points(x=652.0), TORSO)
            if secs == 0.0:
                broke = True
                break
        self.assertTrue(broke, "el episodio deberia romperse al alejarse el promedio")

    def test_the_episode_buffer_is_bounded(self) -> None:
        # A subject asleep in frame keeps one episode alive indefinitely; the
        # buffer must not grow with it.
        timer = ImmobilityTimer(EPS, max_samples=10)
        for i in range(200):
            timer.update(i / 10.0, still_points(), TORSO)
        self.assertLessEqual(len(timer._samples), 10)

    def test_reset_forgets_the_episode(self) -> None:
        timer = ImmobilityTimer(EPS)
        for i in range(21):
            timer.update(i / 10.0, still_points(), TORSO)
        timer.reset()
        secs, _ = timer.update(2.1, still_points(), TORSO)
        self.assertEqual(secs, 0.0)


class TestScaleInvariance(unittest.TestCase):

    def test_same_motion_at_two_distances_gives_the_same_verdict(self) -> None:
        # A subject twice as close is twice as large in pixels and moves
        # twice as many pixels for the same real displacement.
        def run(scale: float) -> float:
            timer = ImmobilityTimer(EPS)
            secs = 0.0
            for i in range(31):
                pts = still_points(x=640.0 + i * 0.3) * scale
                secs, _ = timer.update(i / 10.0, pts, TORSO * scale)
            return secs
        self.assertAlmostEqual(run(1.0), run(2.5), places=6)


class TestOcclusion(unittest.TestCase):

    def test_a_hidden_nose_does_not_end_the_episode(self) -> None:
        # Face-down after a forward fall: the head is invisible, the torso is
        # not. This is the posture the quantity exists to time.
        timer = ImmobilityTimer(EPS)
        secs = 0.0
        for i in range(31):
            pts = still_points()
            pts[0] = np.nan                      # nose occluded
            secs, _ = timer.update(i / 10.0, pts, TORSO)
        self.assertAlmostEqual(secs, 3.0, places=6)

    def test_a_nose_that_becomes_occluded_does_not_end_the_episode(self) -> None:
        # The sequence that actually happens in a forward fall: the face is
        # visible on the way down, then buried once the subject is prone.
        # The anchor holds a real nose position, and the landmark then
        # disappears. Anything other than "skip it" — a zero, a last-known
        # value, a reset — turns the confirmation of a face-down fall into
        # a restart of its clock.
        timer = ImmobilityTimer(EPS)
        for i in range(21):
            timer.update(i / 10.0, still_points(), TORSO)     # nose visible
        secs = 0.0
        for i in range(21, 41):
            pts = still_points()
            pts[0] = np.nan                                   # face buried
            secs, _ = timer.update(i / 10.0, pts, TORSO)
        self.assertAlmostEqual(secs, 4.0, places=6)

    def test_a_reappearing_nose_is_re_anchored_not_treated_as_motion(self) -> None:
        # The face comes back into view mid-episode. Its position was never
        # anchored, so comparing it would be meaningless; it must not read as
        # a jump and reset a legitimate stillness.
        timer = ImmobilityTimer(EPS)
        for i in range(21):
            pts = still_points()
            pts[0] = np.nan
            timer.update(i / 10.0, pts, TORSO)
        secs, _ = timer.update(2.1, still_points(), TORSO)
        self.assertAlmostEqual(secs, 2.1, places=6)

    def test_a_re_anchored_nose_can_break_the_episode_again(self) -> None:
        # Re-anchoring is not bookkeeping: it puts the head back under
        # observation. §3.4 watches head AND torso, and a subject lifting
        # their head while the body stays put is exactly the early sign of
        # the recovery attempt Stage 3 looks for. A nose that reappeared but
        # was never re-anchored would stay permanently uncomparable, and that
        # movement would go unseen.
        timer = ImmobilityTimer(EPS)
        for i in range(11):
            pts = still_points()
            pts[0] = np.nan                       # face down, head unseen
            timer.update(i / 10.0, pts, TORSO)
        for i in range(11, 21):
            secs, _ = timer.update(i / 10.0, still_points(), TORSO)   # face back
        self.assertAlmostEqual(secs, 2.0, places=6)

        # Now only the head moves, well past the radius; the torso is frozen.
        lifted = still_points()
        lifted[0] = lifted[0] + np.array([0.0, -40.0])   # 0.2 torso
        secs, disp = timer.update(2.1, lifted, TORSO)
        self.assertEqual(secs, 0.0)
        self.assertGreater(disp, EPS)

    def test_unusable_scale_is_nan(self) -> None:
        timer = ImmobilityTimer(EPS)
        secs, disp = timer.update(0.0, still_points(), 0.0)
        self.assertTrue(math.isnan(secs))
        self.assertTrue(math.isnan(disp))


class TestConstruction(unittest.TestCase):

    def test_a_non_positive_radius_is_rejected(self) -> None:
        # Zero radius would make every frame a reset: the quantity would
        # silently read 0 forever instead of failing loudly.
        for bad in (0.0, -0.1):
            with self.assertRaises(ValueError):
                ImmobilityTimer(bad)


if __name__ == "__main__":
    unittest.main()
