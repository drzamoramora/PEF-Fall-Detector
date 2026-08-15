"""Tests for the stateful per-frame logic of :class:`FramePipeline`.

These exercise ``FramePipeline.analyze()`` with **synthetic PoseFrame
objects**: no MediaPipe, no camera, no video. That is possible because the
pose front-end is created lazily, so the analysis path can be driven with
hand-built poses — which is the only practical way to test cross-frame
behaviour (smoothing, velocity history, discontinuity handling) at unit
speed.

Covers P-12 of the test catalogue (extension-ratio smoothing, M3).
"""

from __future__ import annotations

import math
import statistics
import unittest

import numpy as np

from pef_fall_detector.audit_log import PHASE2_FIELDS, csv_column
from pef_fall_detector.config import Config
from pef_fall_detector.pipeline import FramePipeline
from pef_fall_detector.pose_frontend import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PoseFrame,
    compute_step0,
)

#: Minimal configuration mirroring config.yaml's structure.
TEST_CFG = {
    "pose": {
        "model_complexity": 1,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "smooth_landmarks": True,
        "visibility_threshold": 0.5,
    },
    "logging": {"output_dir": "logs"},
    "stage1": {
        "threshold_T_deg": 45.0,
        "threshold_V": -1.5,
        "velocity_window_s": 5 / 30.0,
        "ema_time_constant_s": 0.093,
        "history_max_gap_s": 0.5,
        "min_consecutive_frames": 3,
    },
    "state_display": {
        "transition_v_tps": 0.7,
        "lying_T_deg": 60.0,
        "leaning_T_deg": 15.0,
        "standing_extension": 1.1,
        "crouch_extension": 0.6,
        "walking_vh_tps": 0.25,
        "ratio_time_constant_s": 0.093,
    },
}


def make_pose(
    frame_index: int,
    timestamp: float,
    hip_y: float = 500.0,
    shoulder_y: float = 300.0,
    ankle_y: float = 700.0,
    ankle_visible: bool = True,
    x: float = 640.0,
) -> PoseFrame:
    """A synthetic upright skeleton with the landmarks the pipeline reads."""
    lms = np.zeros((33, 3))
    for idx, y in ((LEFT_SHOULDER, shoulder_y), (RIGHT_SHOULDER, shoulder_y),
                   (LEFT_HIP, hip_y), (RIGHT_HIP, hip_y),
                   (LEFT_ANKLE, ankle_y), (RIGHT_ANKLE, ankle_y)):
        lms[idx, 0] = x
        lms[idx, 1] = y
    vis = np.ones(33)
    if not ankle_visible:
        vis[[LEFT_ANKLE, RIGHT_ANKLE]] = 0.1  # below the 0.5 threshold
    mid_hip, mid_shoulder, torso = compute_step0(lms)
    return PoseFrame(
        frame_index=frame_index,
        timestamp=timestamp,
        detected=True,
        frame_size=(1280, 960),
        landmarks=lms,
        visibility=vis,
        mid_hip=mid_hip,
        mid_shoulder=mid_shoulder,
        torso_length=torso,
        core_visibility=1.0,
    )


def jitter(values: list[float]) -> float:
    """High-frequency content: deviation from local linearity."""
    return statistics.median(
        abs(values[i] - 0.5 * (values[i - 1] + values[i + 1]))
        for i in range(1, len(values) - 1)
    )


class TestExtensionRatioSmoothing(unittest.TestCase):
    """M3: the ratio is computed raw and smoothed as a finished number."""

    def _run(self, ankle_sigma: float, hip_sigma: float, n: int = 300) -> list[float]:
        rng = np.random.default_rng(7)
        pipe = FramePipeline(Config(TEST_CFG))
        out = []
        for i in range(n):
            pf = make_pose(
                i, i / 30.0,
                hip_y=500.0 + rng.normal(0, hip_sigma),
                shoulder_y=300.0 + rng.normal(0, hip_sigma),
                ankle_y=700.0 + rng.normal(0, ankle_sigma),
            )
            out.append(pipe.analyze(pf).quantities["extension_ratio"])
        return out

    def test_still_subject_gives_the_expected_ratio(self) -> None:
        # Ankles 200 px below the hips, torso 200 px -> ratio 1.0.
        ratios = self._run(ankle_sigma=0.0, hip_sigma=0.0)
        self.assertAlmostEqual(ratios[-1], 1.0, places=6)

    def test_smoothing_suppresses_ankle_jitter(self) -> None:
        # With realistic ankle noise the smoothed ratio must be far quieter
        # than the raw quotient would be. Raw jitter is reconstructed from
        # the same noise model for comparison.
        smoothed = self._run(ankle_sigma=5.0, hip_sigma=2.0)
        rng = np.random.default_rng(7)
        raw = []
        for _ in range(300):
            hip = 500.0 + rng.normal(0, 2.0)
            sh = 300.0 + rng.normal(0, 2.0)
            ank = 700.0 + rng.normal(0, 5.0)
            raw.append((ank - hip) / abs(hip - sh))
        self.assertLess(jitter(smoothed), jitter(raw) / 2.0)

    def test_occluded_ankles_return_nan(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        result = pipe.analyze(make_pose(0, 0.0, ankle_visible=False))
        self.assertTrue(math.isnan(result.quantities["extension_ratio"]))

    def test_occlusion_does_not_poison_the_smoother(self) -> None:
        # Sit at ratio 1.0, lose the ankles, then reappear at ratio 1.5.
        # The smoother must NOT blend across the blind period: the first
        # value after recovery has to reflect the new posture, not the old.
        pipe = FramePipeline(Config(TEST_CFG))
        for i in range(30):
            pipe.analyze(make_pose(i, i / 30.0, ankle_y=700.0))          # ratio 1.0
        for i in range(30, 40):
            pipe.analyze(make_pose(i, i / 30.0, ankle_visible=False))
        after = pipe.analyze(make_pose(40, 40 / 30.0, ankle_y=800.0))    # ratio 1.5
        self.assertAlmostEqual(after.quantities["extension_ratio"], 1.5, places=6)

    def test_ratio_is_immune_to_distance_changes(self) -> None:
        # A subject walking toward the camera: the torso doubles in pixels
        # while the posture never changes, so the ratio must stay at 1.0.
        # This is what forbids mixing smoothing stages — dividing by a
        # *filtered* torso makes the lagging denominator bias the ratio
        # upward by ~3 % for as long as the subject keeps approaching,
        # which is a systematic error in a quantity whose whole purpose is
        # to be scale-invariant.
        pipe = FramePipeline(Config(TEST_CFG))
        ratios = []
        for i in range(61):
            t = i / 30.0
            torso = 100.0 + 100.0 * min(1.0, t / 2.0)
            hip_y = 500.0
            pf = make_pose(
                i, t,
                hip_y=hip_y,
                shoulder_y=hip_y - torso,     # torso grows as the subject nears
                ankle_y=hip_y + 1.0 * torso,  # posture held constant at 1.0
            )
            ratios.append(pipe.analyze(pf).quantities["extension_ratio"])
        worst = max(abs(r - 1.0) for r in ratios)
        self.assertLess(worst, 0.01, f"el ratio se desvió {100 * worst:.1f}%")

    def _count_label_flips(self, tau: float, n: int = 300) -> int:
        """Label changes for a STILL subject sitting on the posture boundary."""
        cfg = {k: dict(v) for k, v in TEST_CFG.items()}
        cfg["state_display"]["ratio_time_constant_s"] = tau
        rng = np.random.default_rng(3)
        pipe = FramePipeline(Config(cfg))
        labels = []
        for i in range(n):
            hip = 500.0 + rng.normal(0, 2.0)
            pf = make_pose(
                i, i / 30.0,
                hip_y=hip,
                shoulder_y=300.0 + rng.normal(0, 2.0),
                ankle_y=hip + 1.08 * 200.0 + rng.normal(0, 5.0),
            )
            labels.append(pipe.analyze(pf).state)
        return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])

    def test_smoothing_substantially_reduces_label_flicker(self) -> None:
        # A perfectly still subject positioned right at the standing/sitting
        # boundary is the worst case for label stability. Output smoothing
        # cannot eliminate flicker there — a value sitting exactly on a
        # threshold will always cross it under noise; that would need
        # hysteresis on the label itself (catalogued, not yet implemented).
        # What it must do is reduce it by a large factor.
        sin_filtro = self._count_label_flips(tau=0.0)    # 0 = passthrough
        con_filtro = self._count_label_flips(tau=0.093)
        self.assertGreater(sin_filtro, 100, "el caso de prueba no es exigente")
        self.assertLess(
            con_filtro, sin_filtro / 5.0,
            f"el filtro solo redujo de {sin_filtro} a {con_filtro} cambios",
        )


class TestDiscontinuity(unittest.TestCase):
    """Lote B (C1, C2): any break in continuity must reset motion history.

    Velocity is a difference between two moments. It only means something if
    the same body was tracked continuously between them. Two events break
    that guarantee — the detector losing the subject, and the operator
    jumping elsewhere in the video — and both must clear the history rather
    than differentiate across the discontinuity.
    """

    FPS = 30.0

    def _feed_steady(self, pipe: FramePipeline, frames: int, start: int = 0,
                     y: float = 500.0) -> None:
        for i in range(start, start + frames):
            pipe.analyze(make_pose(i, i / self.FPS, hip_y=y, shoulder_y=y - 200.0,
                                   ankle_y=y + 200.0))

    def test_velocity_is_available_during_continuous_tracking(self) -> None:
        # Control case: with no interruption the velocity must be computed.
        pipe = FramePipeline(Config(TEST_CFG))
        self._feed_steady(pipe, 20)
        result = pipe.analyze(make_pose(20, 20 / self.FPS))
        self.assertFalse(math.isnan(result.quantities["V_tps"]))

    def test_single_lost_frame_resets_the_velocity_window(self) -> None:
        # A gap of ONE frame (0.033 s) is far below the 0.5 s gap guard, yet
        # it is enough to lose track of who is being followed.
        pipe = FramePipeline(Config(TEST_CFG))
        self._feed_steady(pipe, 20)
        undetected = PoseFrame(frame_index=20, timestamp=20 / self.FPS,
                               detected=False, frame_size=(1280, 960))
        pipe.analyze(undetected)
        after = pipe.analyze(make_pose(21, 21 / self.FPS))
        self.assertTrue(
            math.isnan(after.quantities["V_tps"]),
            "la ventana de velocidad sobrevivió a una pérdida de detección",
        )

    def test_skeleton_jump_after_micro_gap_yields_no_impossible_velocity(self) -> None:
        # The real failure observed in a two-person clip: the subject is
        # tracked, one frame is lost, and the tracker re-attaches to a
        # different body 400 px away. Differentiating across that produced
        # +26 torso/s in production data — physically impossible.
        pipe = FramePipeline(Config(TEST_CFG))
        self._feed_steady(pipe, 20, y=500.0)
        pipe.analyze(PoseFrame(frame_index=20, timestamp=20 / self.FPS,
                               detected=False, frame_size=(1280, 960)))
        peak = 0.0
        for i in range(21, 31):  # the "other person", far up the frame
            r = pipe.analyze(make_pose(i, i / self.FPS, hip_y=100.0,
                                       shoulder_y=-100.0, ankle_y=300.0))
            v = r.quantities["V_tps"]
            if not math.isnan(v):
                peak = max(peak, abs(v))
        self.assertLess(peak, 8.0, f"velocidad implausible tras el salto: {peak:.1f}")

    def test_long_gap_still_resets(self) -> None:
        # The original guard must keep working: a long absence also resets.
        pipe = FramePipeline(Config(TEST_CFG))
        self._feed_steady(pipe, 20)
        after = pipe.analyze(make_pose(200, 200 / self.FPS + 5.0))
        self.assertTrue(math.isnan(after.quantities["V_tps"]))

    def test_public_reset_clears_the_velocity_window(self) -> None:
        # C2: the GUI must be able to declare a discontinuity when the user
        # jumps in the video. Without this, scrubbing forward by less than
        # the gap threshold mixes frames from two places in the timeline.
        pipe = FramePipeline(Config(TEST_CFG))
        self._feed_steady(pipe, 20)
        self.assertFalse(math.isnan(pipe.analyze(make_pose(20, 20 / self.FPS))
                                    .quantities["V_tps"]))
        pipe.reset()
        after = pipe.analyze(make_pose(21, 21 / self.FPS))
        self.assertTrue(math.isnan(after.quantities["V_tps"]))

    def test_public_reset_clears_the_smoothers(self) -> None:
        # After a reset the filters must seed from the new position instead
        # of blending it with wherever the subject used to be.
        pipe = FramePipeline(Config(TEST_CFG))
        self._feed_steady(pipe, 30, y=500.0)
        pipe.reset()
        after = pipe.analyze(make_pose(31, 31 / self.FPS, hip_y=900.0,
                                       shoulder_y=700.0, ankle_y=1100.0))
        # The smoothed centroid must sit on the NEW body, not between both.
        self.assertAlmostEqual(float(after.centroid_px[1]), 800.0, delta=1.0)

    def test_seeking_backwards_resets(self) -> None:
        # Timestamps going backwards can only mean the operator moved.
        pipe = FramePipeline(Config(TEST_CFG))
        self._feed_steady(pipe, 20)
        after = pipe.analyze(make_pose(5, 5 / self.FPS))
        self.assertTrue(math.isnan(after.quantities["V_tps"]))


class TestAnalyzeBasics(unittest.TestCase):
    def test_undetected_frame_is_never_reliable(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        pf = PoseFrame(frame_index=0, timestamp=0.0, detected=False, frame_size=(640, 480))
        result = pipe.analyze(pf)
        self.assertFalse(result.reliable)
        self.assertEqual(result.quantities, {})
        self.assertIsNone(result.state)

    def test_low_core_visibility_is_not_reliable_but_still_computed(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        pf = make_pose(0, 0.0)
        pf.core_visibility = 0.02  # face-only footage, as in test-1.mp4
        result = pipe.analyze(pf)
        self.assertFalse(result.reliable)
        # Data is never silently discarded: the quantity is still there.
        self.assertIn("T_deg", result.quantities)


class TestRowMatchesTheSchema(unittest.TestCase):
    """Every key the pipeline emits must exist as a column.

    ``extrasaction="raise"`` already refuses an unknown key — but only when
    a row is actually written, which means on a real video run, not in the
    suite. That is a slow way to find a typo. This closes the loop at unit
    speed, and it is what keeps the ``_EXP`` renaming honest: a stale
    unsuffixed key left behind in the pipeline fails here instead of in the
    middle of a calibration session.
    """

    def test_every_emitted_key_is_a_column(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        # Drive enough frames that the velocity window fills and every
        # quantity — not just the ones available on frame 0 — is present.
        result = None
        for i in range(40):
            result = pipe.analyze(make_pose(i, i / 30.0, ankle_y=700.0 + i))
        assert result is not None
        emitted = set(result.csv_extra())
        self.assertTrue(emitted, "el frame no produjo ninguna columna")
        self.assertLessEqual(emitted, set(PHASE2_FIELDS),
                             f"claves fuera del esquema: {emitted - set(PHASE2_FIELDS)}")

    def test_the_experimental_quantities_actually_reach_the_row(self) -> None:
        # Guards against the mark being applied to a name nothing emits —
        # a suffix on a column that is always empty would be decoration.
        pipe = FramePipeline(Config(TEST_CFG))
        result = None
        for i in range(40):
            result = pipe.analyze(make_pose(i, i / 30.0, ankle_y=700.0 + i))
        assert result is not None
        emitted = set(result.csv_extra())
        for name in ("Vh_tps", "extension_ratio", "state"):
            self.assertIn(csv_column(name), emitted)


if __name__ == "__main__":
    unittest.main()
