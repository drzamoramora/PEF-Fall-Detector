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
    NOSE,
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
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
        "trigger_formulation": "sequential",
        "trigger_score": 2.2,
        "trigger_hold_s": 0.1,
        "confirm_window_s": 1.0,
    },
    "state_display": {
        "transition_v_tps": 0.7,
        "lying_T_deg": 60.0,
        "leaning_T_deg": 15.0,
        "standing_extension": 1.1,
        "crouch_extension": 0.6,
        "max_extension_ratio": 3.0,
        "walking_vh_tps": 0.25,
        "ratio_time_constant_s": 0.093,
    },
    "alerts": {"dispatch_verdicts": ["stage3_confirmed"]},
    "stage2": {
        "com_hip_weight": 0.65,
        "com_eval_window_s": 0.33,
        "com_outside_fraction": 0.5,
        "com_min_samples": 3,
        "min_foot_visibility": 0.5,
        "contact_band_torso": 0.15,
    },
    "stage3": {
        "epsilon": 0.05,
        "threshold_W_seconds": 5.0,
        "observation_window_seconds": 30.0,
        "upright_T_deg": 30.0,
        "recovery_hold_seconds": 1.0,
        "severity_uses_leg_extension": True,
        "cooldown_seconds": 3.0,
    },
}


def make_pose(
    frame_index: int,
    timestamp: float,
    hip_y: float = 500.0,
    shoulder_y: float = 300.0,
    ankle_y: float = 700.0,
    ankle_visible: bool = True,
    nose_visible: bool = True,
    x: float = 640.0,
    feet_x: float | None = None,
    stance_px: float = 60.0,
) -> PoseFrame:
    """A synthetic upright skeleton with the landmarks the pipeline reads.

    ``feet_x`` defaults to ``x`` (feet under the body). Passing it separately
    shifts the support polygon away from the trunk, which is how the
    Quantity-P tests put the COM outside the feet without contorting the
    rest of the skeleton.
    """
    lms = np.zeros((33, 3))
    # The nose is a real position, not the origin: Quantity I anchors on it
    # while it is visible, so a test that leaves it at (0, 0) cannot tell a
    # skipped landmark from a landmark that never moved.
    lms[NOSE, 0], lms[NOSE, 1] = x, shoulder_y - 80.0
    for idx, y in ((LEFT_SHOULDER, shoulder_y), (RIGHT_SHOULDER, shoulder_y),
                   (LEFT_HIP, hip_y), (RIGHT_HIP, hip_y),
                   (LEFT_ANKLE, ankle_y), (RIGHT_ANKLE, ankle_y)):
        lms[idx, 0] = x
        lms[idx, 1] = y
    # Feet: ankles plus heels and toes, the six landmarks of §3.4's support
    # polygon. Without them the hull would be built from unset (0, 0) points.
    fx = x if feet_x is None else feet_x
    for ankle, heel, toe, side in (
        (LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX, -1.0),
        (RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX, +1.0),
    ):
        foot_x = fx + side * stance_px / 2.0
        lms[ankle, 0], lms[ankle, 1] = foot_x, ankle_y
        lms[heel, 0], lms[heel, 1] = foot_x - 10.0, ankle_y + 5.0
        lms[toe, 0], lms[toe, 1] = foot_x + 20.0, ankle_y + 8.0
    vis = np.ones(33)
    if not ankle_visible:
        vis[[LEFT_ANKLE, RIGHT_ANKLE]] = 0.1  # below the 0.5 threshold
    if not nose_visible:
        vis[NOSE] = 0.1                       # face buried / turned away
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


class TestQuantityPInThePipeline(unittest.TestCase):
    """P as the pipeline produces it, on synthetic skeletons.

    The pure geometry is covered in ``test_quantity_p``. What is left to
    prove here is the wiring: that the COM is the hip-weighted one and not
    Quantity V's midpoint, and that occluded feet reach the record as NaN
    rather than as a number.
    """

    def test_balanced_stance_puts_the_com_inside(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        q = pipe.analyze(make_pose(0, 0.0)).quantities
        self.assertLess(q["P_offset"], 0.0)
        self.assertGreater(q["P_support_width"], 0.0)

    def test_feet_far_from_the_body_put_the_com_outside(self) -> None:
        # Body at x=640, feet a metre to the left: the COM overhangs.
        pipe = FramePipeline(Config(TEST_CFG))
        q = pipe.analyze(make_pose(0, 0.0, feet_x=300.0)).quantities
        self.assertGreater(q["P_offset"], 0.0)

    def test_occluded_feet_make_p_undefined(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        q = pipe.analyze(make_pose(0, 0.0, ankle_visible=False)).quantities
        self.assertTrue(math.isnan(q["P_offset"]))
        self.assertTrue(math.isnan(q["P_support_width"]))

    def test_toppling_drives_p_from_inside_to_outside(self) -> None:
        """The §3.4 fall signature, end to end: "projection of COM outside
        the support polygon".

        A single frame with the COM outside proves the arithmetic; what the
        paper actually claims is a *transition*. This walks the body away
        from planted feet and asserts that P crosses zero once, in the right
        direction — the shape Stage 2 will eventually test for.
        """
        pipe = FramePipeline(Config(TEST_CFG))
        offsets = []
        for i in range(40):
            body_x = 640.0 + i * 6.0          # torso drifts right, feet stay
            pf = make_pose(i, i / 30.0, x=body_x, feet_x=640.0)
            offsets.append(pipe.analyze(pf).quantities["P_offset"])

        self.assertLess(offsets[0], 0.0, "de pie equilibrado deberia dar dentro")
        self.assertGreater(offsets[-1], 0.0, "volcado deberia dar fuera")
        crossings = sum(
            1 for a, b in zip(offsets, offsets[1:]) if (a <= 0.0) != (b <= 0.0)
        )
        self.assertEqual(crossings, 1, "P deberia cruzar el cero una sola vez")

    def test_com_is_hip_weighted_not_the_velocity_centroid(self) -> None:
        # §3.4 defines two different points: V's centroid is a 0.5/0.5
        # midpoint, P's COM is hip-dominant. With the trunk leaning, the two
        # sit at different x, and P must use the hip-weighted one. A pose
        # whose shoulders are offset from the hips makes the difference
        # measurable.
        cfg = Config(TEST_CFG)
        pipe = FramePipeline(cfg)
        pf = make_pose(0, 0.0)
        pf.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 0] = 400.0   # lean left
        pf.mid_hip, pf.mid_shoulder, pf.torso_length = compute_step0(pf.landmarks)
        q = pipe.analyze(pf).quantities

        w = float(cfg.stage2.com_hip_weight)
        hip_x, sh_x = float(pf.mid_hip[0]), float(pf.mid_shoulder[0])
        expected_com_x = w * hip_x + (1.0 - w) * sh_x
        midpoint_x = 0.5 * (hip_x + sh_x)
        self.assertNotAlmostEqual(expected_com_x, midpoint_x, places=3)

        hull_right = 640.0 + 60.0 / 2.0 + 20.0        # rightmost toe
        hull_left = 640.0 - 60.0 / 2.0 - 10.0         # leftmost heel
        margin = min(expected_com_x - hull_left, hull_right - expected_com_x)
        self.assertAlmostEqual(q["P_offset"], -margin / pf.torso_length, places=6)


class TestQuantityIInThePipeline(unittest.TestCase):
    """I as the pipeline produces it. The timer itself is covered in
    ``test_quantity_i``; what is left is the wiring and the reset policy."""

    def test_a_still_subject_accumulates_time(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        secs = float("nan")
        for i in range(31):
            secs = pipe.analyze(make_pose(i, i / 10.0)).quantities["I_still_s"]
        self.assertAlmostEqual(secs, 3.0, places=6)

    def test_moving_zeroes_the_clock(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        for i in range(21):
            pipe.analyze(make_pose(i, i / 10.0))
        # Torso is 200 px here, so 40 px is 0.2 torso — well past epsilon.
        q = pipe.analyze(make_pose(21, 2.1, x=680.0)).quantities
        self.assertEqual(q["I_still_s"], 0.0)

    def test_losing_the_subject_restarts_the_clock(self) -> None:
        # Not merely a pause: the clock must not resume where it left off.
        # A body reappearing inside the same radius is equally consistent
        # with a different person standing there, and nothing yet
        # distinguishes the two.
        pipe = FramePipeline(Config(TEST_CFG))
        for i in range(31):
            pipe.analyze(make_pose(i, i / 10.0))
        pipe.analyze(PoseFrame(frame_index=31, timestamp=3.1, detected=False,
                               frame_size=(1280, 960)))
        q = pipe.analyze(make_pose(32, 3.2)).quantities
        self.assertEqual(q["I_still_s"], 0.0)

    def test_the_face_getting_buried_does_not_restart_the_clock(self) -> None:
        # A forward fall in two acts: the face is visible on the way down and
        # buried once the subject is prone. The nose must drop out of the
        # comparison, not be replaced by a stand-in position — a substituted
        # value looks like the head teleporting and resets the very clock
        # that confirms the fall.
        pipe = FramePipeline(Config(TEST_CFG))
        for i in range(21):
            pipe.analyze(make_pose(i, i / 10.0))
        secs = 0.0
        for i in range(21, 41):
            secs = pipe.analyze(
                make_pose(i, i / 10.0, nose_visible=False)
            ).quantities["I_still_s"]
        self.assertAlmostEqual(secs, 4.0, places=6)

    def test_unreliable_frames_report_i_as_unknown(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        pf = make_pose(0, 0.0)
        pf.core_visibility = 0.02
        q = pipe.analyze(pf).quantities
        self.assertTrue(math.isnan(q["I_still_s"]))


class TestAlertDispatchFromThePipeline(unittest.TestCase):
    """The seam end to end: a confirmed fall reaches a sink."""

    @staticmethod
    def _toppling_pose(index: int, timestamp: float, angle_deg: float,
                       hip_y: float) -> PoseFrame:
        """A skeleton with the trunk rotated ``angle_deg`` from vertical.

        ``make_pose`` always stacks the shoulders directly above the hips, so
        its T is identically 0 — fine for the quantities it was written for,
        useless for driving a trigger that needs the trunk to rotate. Here
        the trunk keeps its 200 px length and swings, which is what a fall
        actually does to the geometry.
        """
        pf = make_pose(index, timestamp, hip_y=hip_y, feet_x=640.0)
        rad = math.radians(angle_deg)
        pf.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 0] = 640.0 + 200.0 * math.sin(rad)
        pf.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 1] = hip_y - 200.0 * math.cos(rad)
        pf.landmarks[NOSE, 0] = 640.0 + 260.0 * math.sin(rad)
        pf.landmarks[NOSE, 1] = hip_y - 260.0 * math.cos(rad)
        pf.mid_hip, pf.mid_shoulder, pf.torso_length = compute_step0(pf.landmarks)
        return pf

    def _fall_frames(self):
        """Topple past 45 deg, then lie still past W — a confirmed severe fall."""
        frames = []
        for i in range(20):                       # 0.67 s of toppling
            frames.append(self._toppling_pose(
                i, i / 30.0, angle_deg=i * 4.5, hip_y=500.0 + i * 12.0))
        for i in range(20, 300):                  # motionless on the floor
            frames.append(self._toppling_pose(
                i, i / 30.0, angle_deg=90.0, hip_y=740.0))
        return frames

    def test_a_confirmed_fall_reaches_the_sink_with_its_evidence(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        received = []
        pipe.alerts.add_sink(received.append)
        pipe.source_name = "clip.mp4"
        for pf in self._fall_frames():
            pipe.analyze(pf)
        self.assertEqual(len(pipe.machine.events), 1, "la Etapa 1 no disparo")
        self.assertEqual(len(received), 1, "no se despacho la alerta")
        a = received[0]
        self.assertEqual(a.source, "clip.mp4")
        self.assertEqual(a.verdict, "stage3_confirmed")
        self.assertEqual(a.severity, "severe")
        # The alert must arrive AFTER the funnel resolved, never on the frame
        # the event was merely raised on.
        self.assertNotEqual(a.verdict, "stage1_only")
        # And it must carry the evidence, not just the verdict (§3.5).
        for fragment in ("SEVERE", "T=", "V=", "COM outside", "still"):
            self.assertIn(fragment, a.message())
        # The prelude must name the posture the subject was in BEFORE the
        # event. Left updating during the fall it would report LYING —
        # "subject was lying down before they fell", which is both useless
        # to a caregiver and wrong.
        self.assertTrue(a.prior_state)
        self.assertNotIn(a.prior_state, ("LYING", "NOT_DETECTED"))

    def test_the_resolved_event_surfaces_once_on_the_deciding_frame(self) -> None:
        """What PEF-Lab's event record and outcome banner hang on.

        ``event`` marks the frame the funnel started wondering; this marks
        the frame it decided. Surfacing the verdict on the raising frame
        would record ``stage1_only`` for everything; surfacing it on every
        subsequent frame would write the same event to the record dozens of
        times.
        """
        pipe = FramePipeline(Config(TEST_CFG))
        resolved = []
        for pf in self._fall_frames():
            result = pipe.analyze(pf)
            if result.resolved_event is not None:
                resolved.append((result.pose.frame_index, result.resolved_event))
        self.assertEqual(len(resolved), 1, "el veredicto debe aparecer una sola vez")
        deciding_frame, ev = resolved[0]
        self.assertNotEqual(ev.verdict, "stage1_only")
        self.assertGreater(deciding_frame, ev.frame_index,
                           "el veredicto no puede caer en el frame del disparo")

    def test_nothing_is_dispatched_when_nothing_happens(self) -> None:
        pipe = FramePipeline(Config(TEST_CFG))
        received = []
        pipe.alerts.add_sink(received.append)
        for i in range(120):
            pipe.analyze(make_pose(i, i / 30.0))
        self.assertEqual(received, [])

    def test_the_alert_names_what_the_subject_was_doing_before(self) -> None:
        # §3.5's explainability, in the one line a caregiver reads. The state
        # must be the pre-event one, not the posture the fall produced.
        pipe = FramePipeline(Config(TEST_CFG))
        for i in range(40):
            pipe.analyze(make_pose(i, i / 30.0))
        self.assertTrue(pipe._recent_state)
        self.assertNotEqual(pipe._recent_state, "LYING")


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
