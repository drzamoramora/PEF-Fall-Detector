"""Unit tests for Phase 2.2-2.4: EMA smoothing, Quantity V, person state.

Synthetic data only, as always: each sub-part must be provable on
hand-built inputs before it touches real footage.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pef_fall_detector.person_state import (
    CROUCHING,
    LEANING,
    LYING,
    NOT_DETECTED,
    SITTING,
    STANDING,
    TRANSITION,
    WALKING,
    classify_state,
)
from pef_fall_detector.quantities import ExponentialMovingAverage, VelocityEstimator

#: The provisional thresholds from config.yaml's state_display section,
#: mirrored here as explicit kwargs (tests must not read config files).
STATE_KW = dict(
    transition_v_tps=0.7,
    lying_t_deg=60.0,
    leaning_t_deg=15.0,
    standing_extension=1.1,
    crouch_extension=0.6,
    walking_vh_tps=0.25,
)

NAN = float("nan")

#: Un frame a 30 fps, y la ventana equivalente a los antiguos 5 frames.
DT = 1 / 30.0
WINDOW_S = 5 / 30.0


class TestEMA(unittest.TestCase):
    def test_first_sample_passes_through(self) -> None:
        ema = ExponentialMovingAverage(time_constant_s=0.093)
        self.assertEqual(ema.update(10.0, dt=DT), 10.0)

    def test_constant_input_stays_constant(self) -> None:
        ema = ExponentialMovingAverage(time_constant_s=0.093)
        for _ in range(20):
            value = ema.update(5.0, dt=DT)
        self.assertAlmostEqual(value, 5.0)

    def test_step_converges_to_new_level(self) -> None:
        # After a jump, the smoothed value must approach the new level.
        ema = ExponentialMovingAverage(time_constant_s=0.048)
        ema.update(0.0, dt=DT)
        for _ in range(20):
            value = ema.update(10.0, dt=DT)
        self.assertAlmostEqual(value, 10.0, places=3)

    def test_zero_time_constant_is_passthrough(self) -> None:
        ema = ExponentialMovingAverage(time_constant_s=0.0)
        ema.update(3.0, dt=DT)
        self.assertEqual(ema.update(7.0, dt=DT), 7.0)

    def test_smooths_vectors_too(self) -> None:
        # tau chosen so that one 30 fps step blends exactly half and half.
        tau = DT / math.log(2.0)
        ema = ExponentialMovingAverage(time_constant_s=tau)
        ema.update(np.array([0.0, 0.0]), dt=DT)
        smoothed = ema.update(np.array([10.0, 20.0]), dt=DT)
        np.testing.assert_allclose(smoothed, [5.0, 10.0])

    def test_negative_time_constant_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExponentialMovingAverage(time_constant_s=-1.0)
        with self.assertRaises(ValueError):
            ExponentialMovingAverage(time_constant_s=-0.5)


class TestVelocityEstimator(unittest.TestCase):
    """Quantity V on synthetic trajectories. Torso fixed at 100 px, frames
    at 30 FPS, window of 0.167 s."""

    FPS = 30.0
    TORSO = 100.0

    def _run(self, ys, xs=None):
        est = VelocityEstimator(window_seconds=WINDOW_S, max_gap_s=0.5)
        xs = xs if xs is not None else [0.0] * len(ys)
        out = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            out.append(est.update(i / self.FPS, np.array([x, y]), self.TORSO))
        return out

    def test_nan_while_window_fills(self) -> None:
        results = self._run([100.0] * 5)
        for v, vh in results:
            self.assertTrue(math.isnan(v))
            self.assertTrue(math.isnan(vh))

    def test_still_subject_reads_zero(self) -> None:
        v, vh = self._run([100.0] * 10)[-1]
        self.assertAlmostEqual(v, 0.0)
        self.assertAlmostEqual(vh, 0.0)

    def test_falling_reads_negative_two_torso_per_second(self) -> None:
        # y grows 200 px/s downward = 2 torso/s down -> V must be -2.0
        # (paper convention: negative = downward).
        ys = [100.0 + 200.0 * (i / self.FPS) for i in range(10)]
        v, _ = self._run(ys)[-1]
        self.assertAlmostEqual(v, -2.0, places=6)

    def test_rising_reads_positive(self) -> None:
        ys = [500.0 - 150.0 * (i / self.FPS) for i in range(10)]
        v, _ = self._run(ys)[-1]
        self.assertAlmostEqual(v, +1.5, places=6)

    def test_horizontal_walk_reads_in_vh_not_v(self) -> None:
        xs = [100.0 * (i / self.FPS) for i in range(10)]  # 1 torso/s rightward
        v, vh = self._run([300.0] * 10, xs=xs)[-1]
        self.assertAlmostEqual(v, 0.0)
        self.assertAlmostEqual(vh, +1.0, places=6)

    def test_gap_clears_history(self) -> None:
        # A subject at y=100 disappears for 1 s and reappears at y=400.
        # Without the gap guard this would read as a huge fake fall.
        est = VelocityEstimator(window_seconds=WINDOW_S, max_gap_s=0.5)
        for i in range(10):
            est.update(i / self.FPS, np.array([0.0, 100.0]), self.TORSO)
        v, _ = est.update(10 / self.FPS + 1.0, np.array([0.0, 400.0]), self.TORSO)
        self.assertTrue(math.isnan(v))  # history cleared, window refilling

    def test_zero_torso_reads_nan(self) -> None:
        est = VelocityEstimator(window_seconds=2/30.0, max_gap_s=0.5)
        out = [est.update(i / self.FPS, np.array([0.0, 100.0]), 0.0) for i in range(5)]
        self.assertTrue(math.isnan(out[-1][0]))


class TestPersonState(unittest.TestCase):
    """The provisional display-label rules (2.4) on representative feature
    combinations. These thresholds get tuned in 2.6; the tests pin the
    *logic*, not the final numbers."""

    def test_not_detected(self) -> None:
        self.assertEqual(classify_state(NAN, NAN, NAN, NAN, **STATE_KW), NOT_DETECTED)

    def test_standing(self) -> None:
        # Upright trunk, extended legs, still.
        self.assertEqual(classify_state(5.0, 1.5, 0.0, 0.0, **STATE_KW), STANDING)

    def test_walking(self) -> None:
        # Upright + sustained horizontal motion.
        self.assertEqual(classify_state(8.0, 1.5, 0.05, 0.6, **STATE_KW), WALKING)

    def test_leaning(self) -> None:
        # Reaching for something: tilted trunk, legs extended.
        self.assertEqual(classify_state(30.0, 1.4, 0.1, 0.0, **STATE_KW), LEANING)

    def test_sitting(self) -> None:
        # Upright trunk, legs folded to chair height.
        self.assertEqual(classify_state(10.0, 0.8, 0.0, 0.0, **STATE_KW), SITTING)

    def test_crouching(self) -> None:
        # Picking something up: legs fully collapsed.
        self.assertEqual(classify_state(40.0, 0.3, 0.1, 0.0, **STATE_KW), CROUCHING)

    def test_lying(self) -> None:
        # Trunk horizontal beats everything except transition.
        self.assertEqual(classify_state(80.0, 0.1, 0.0, 0.0, **STATE_KW), LYING)

    def test_transition_beats_static_postures(self) -> None:
        # Dropping fast: even with lying-range trunk angle, this frame is
        # TRANSITION — the body is between postures.
        self.assertEqual(classify_state(70.0, 0.5, -1.5, 0.0, **STATE_KW), TRANSITION)

    def test_occluded_ankles_fall_back_to_trunk_only(self) -> None:
        # extension NaN: coarse but honest labels from the trunk alone.
        self.assertEqual(classify_state(5.0, NAN, 0.0, 0.0, **STATE_KW), STANDING)
        self.assertEqual(classify_state(30.0, NAN, 0.0, 0.0, **STATE_KW), LEANING)
        self.assertEqual(classify_state(80.0, NAN, 0.0, 0.0, **STATE_KW), LYING)

    def test_velocity_nan_never_blocks_static_labels(self) -> None:
        # While the velocity window fills, static postures must still label.
        self.assertEqual(classify_state(5.0, 1.5, NAN, NAN, **STATE_KW), STANDING)


class TestFrameRateInvariance(unittest.TestCase):
    """P-01: the same physical fall must read the same at any capture rate.

    This is the test the time-based window exists for. With the window
    expressed in frames it failed badly — the same fall read −4.49 torso/s
    at 60 fps and −2.75 at 10 fps, a 63 % swing — which meant a threshold
    calibrated on one source would silently mis-fire on another, and in
    particular that thresholds tuned on 30 fps footage would miss falls on
    a slower edge device (§3.6).
    """

    WINDOW_S = 0.167
    TAU = 0.093

    def _peak_of_synthetic_fall(self, fps: float) -> float:
        """1.5 torso-lengths of drop in 0.5 s, constant acceleration."""
        torso, peak = 100.0, 0.0
        ema = ExponentialMovingAverage(self.TAU)
        est = VelocityEstimator(window_seconds=self.WINDOW_S)
        for i in range(int(2.0 * fps)):
            t = i / fps
            y = 300.0 if t < 0.5 else 300.0 + 150.0 * min(1.0, (t - 0.5) / 0.5) ** 2
            smoothed = ema.update(np.array([200.0, y]), dt=1.0 / fps)
            v, _ = est.update(t, smoothed, torso)
            if not math.isnan(v) and v < peak:
                peak = v
        return peak

    def test_peak_velocity_is_stable_across_frame_rates(self) -> None:
        peaks = [self._peak_of_synthetic_fall(fps) for fps in (60, 30, 25, 15, 10)]
        spread = (max(peaks) - min(peaks)) / abs(sum(peaks) / len(peaks))
        self.assertLess(
            spread, 0.15,
            f"V varía {100 * spread:.0f}% entre 60 y 10 fps: {peaks}",
        )

    def test_every_rate_would_cross_a_common_threshold(self) -> None:
        # The practical consequence: one threshold has to work everywhere.
        for fps in (60, 30, 25, 15, 10):
            with self.subTest(fps=fps):
                self.assertLess(self._peak_of_synthetic_fall(fps), -1.5)


if __name__ == "__main__":
    unittest.main()
