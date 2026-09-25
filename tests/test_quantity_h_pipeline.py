"""Integration tests for Quantity H (EXPERIMENTAL) inside FramePipeline.

``tests/test_quantity_h.py`` proves the arithmetic in isolation; this file
proves the wiring — that ``FramePipeline.analyze`` assembles H from a full
synthetic pose the same way a real MediaPipe frame would arrive, including
the one thing the pure-function tests cannot exercise: per-landmark
visibility gating read off a real (33, 3) landmark array.

Landmarks here get real left/right shoulder separation, unlike
``tests.test_pipeline``'s ``make_pose`` helper (which places both shoulders
at the same x — irrelevant for T, V, P and I, but it would make Quantity H's
shoulder-width reference collapse to zero for every existing pipeline test).
Building poses directly, the same way ``tests/test_quantity_p.py`` builds its
own foot geometry rather than reusing another file's fixture, avoids
changing a shared helper that many unrelated tests already depend on.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pef_fall_detector.config import Config
from pef_fall_detector.pipeline import FramePipeline
from pef_fall_detector.pose_frontend import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PoseFrame,
    compute_step0,
)
from tests.test_pipeline import TEST_CFG

DT = 1.0 / 30.0


def _pose_frame(frame_index: int, timestamp: float, landmarks: np.ndarray,
                 visibility: np.ndarray) -> PoseFrame:
    mid_hip, mid_shoulder, torso_length = compute_step0(landmarks)
    core_visibility = float(visibility[[LEFT_SHOULDER, RIGHT_SHOULDER,
                                        LEFT_HIP, RIGHT_HIP]].min())
    return PoseFrame(
        frame_index=frame_index,
        timestamp=timestamp,
        detected=True,
        frame_size=(1280, 960),
        landmarks=landmarks,
        visibility=visibility,
        mid_hip=mid_hip,
        mid_shoulder=mid_shoulder,
        torso_length=torso_length,
        core_visibility=core_visibility,
    )


def _standing_landmarks() -> np.ndarray:
    """An upright subject, seen roughly side-on: shoulders visibly apart."""
    lms = np.zeros((33, 3))
    lms[NOSE, :2] = (640.0, 220.0)
    lms[LEFT_SHOULDER, :2] = (600.0, 300.0)
    lms[RIGHT_SHOULDER, :2] = (680.0, 300.0)
    lms[LEFT_HIP, :2] = (610.0, 500.0)
    lms[RIGHT_HIP, :2] = (670.0, 500.0)
    lms[LEFT_ANKLE, :2] = (610.0, 730.0)
    lms[RIGHT_ANKLE, :2] = (670.0, 730.0)
    return lms


def _axis_aligned_fall_landmarks() -> np.ndarray:
    """A collapse whose rotation axis points at the camera.

    The hip-shoulder vector stays close to VERTICAL but shrinks to a few
    tens of pixels -- short enough to be almost meaningless, long enough to
    clear the plausibility guard (``stage1.min_trunk_ratio``) and read as a
    real, falsely low angle instead of NaN. This is the failure mode itself:
    a rotation into the depth axis does not reliably rotate the 2D
    projection of hip-to-shoulder, it mostly just shortens it, so T stays
    "upright"-looking. The wider spine chain (nose to ankle) still shows the
    whole body compressed into a shallow vertical band, and the shoulders
    keep a plausible horizontal separation because the rotation is pitch,
    not yaw -- which is what H is built to read instead.
    """
    lms = np.zeros((33, 3))
    lms[NOSE, :2] = (575.0, 578.0)
    lms[LEFT_SHOULDER, :2] = (545.0, 585.0)
    lms[RIGHT_SHOULDER, :2] = (605.0, 585.0)
    lms[LEFT_HIP, :2] = (560.0, 625.0)
    lms[RIGHT_HIP, :2] = (590.0, 625.0)
    lms[LEFT_ANKLE, :2] = (568.0, 633.0)
    lms[RIGHT_ANKLE, :2] = (600.0, 637.0)
    return lms


def _make_pipeline(trigger_h_ratio: float = 0.0,
                    recovery_h_ratio: float = 0.0,
                    trigger_h_erect: float = 0.0) -> FramePipeline:
    cfg = dict(TEST_CFG)
    cfg["experimental"] = {
        "height_baseline_time_constant_s": 0.3,
        "trigger_H_ratio": trigger_h_ratio,
        "trigger_H_erect": trigger_h_erect,
        "trigger_H_sequence_window_s": 2.0,
        "recovery_H_ratio": recovery_h_ratio,
    }
    return FramePipeline(Config(cfg))


class TestQuantityHConvergesWhileStanding(unittest.TestCase):
    def test_baseline_is_nan_before_any_standing_frame_is_seen(self) -> None:
        pipe = _make_pipeline()
        lms = _standing_landmarks()
        vis = np.ones(33)
        result = pipe.analyze(_pose_frame(0, 0.0, lms, vis))
        # First frame: dt is 0 (nothing to differentiate from yet), and the
        # EMA has had no sample. The baseline seeds on THIS call, so the
        # ratio it reports is 1.0 (raw over itself) rather than NaN.
        self.assertIn("H_baseline", result.quantities)
        self.assertFalse(math.isnan(result.quantities["H_baseline"]))

    def test_h_settles_near_one_for_a_standing_subject(self) -> None:
        pipe = _make_pipeline()
        lms = _standing_landmarks()
        vis = np.ones(33)
        result = None
        for i in range(60):
            result = pipe.analyze(_pose_frame(i, i * DT, lms, vis))
        self.assertAlmostEqual(result.quantities["H_ratio"], 1.0, places=2)
        self.assertEqual(result.state, "STANDING")


class TestQuantityHCatchesTheAxisAlignedFall(unittest.TestCase):
    """The scenario reported in the design discussion: a subject falling
    toward/away from the camera, where T reads a falsely low angle because
    its trunk vector's 2D projection nearly vanishes."""

    def test_h_collapses_where_t_does_not(self) -> None:
        pipe = _make_pipeline()
        standing = _standing_landmarks()
        vis = np.ones(33)
        for i in range(60):
            pipe.analyze(_pose_frame(i, i * DT, standing, vis))

        fallen = _axis_aligned_fall_landmarks()
        result = pipe.analyze(_pose_frame(60, 60 * DT, fallen, vis))

        # T's own trunk vector (mid-hip to mid-shoulder) is nearly vertical
        # and very short in this geometry -- the projection limit this
        # quantity is proposed to cover.
        self.assertLess(result.quantities["T_deg"], 20.0)
        # H, built from the wider spine chain and the yaw-resistant shoulder
        # width, still reads the collapse.
        self.assertLess(result.quantities["H_ratio"], 0.3)


class TestQuantityHDegradesGracefully(unittest.TestCase):
    def test_occluded_ankles_still_yield_a_reading(self) -> None:
        # §3.4's own occlusion problem, applied to H: ankles below the
        # visibility threshold must not blank the quantity when nose,
        # shoulders and hips are still visible.
        pipe = _make_pipeline()
        lms = _standing_landmarks()
        vis = np.ones(33)
        for i in range(60):
            pipe.analyze(_pose_frame(i, i * DT, lms, vis))

        occluded_vis = vis.copy()
        occluded_vis[[LEFT_ANKLE, RIGHT_ANKLE]] = 0.1
        result = pipe.analyze(_pose_frame(60, 60 * DT, lms, occluded_vis))
        self.assertFalse(math.isnan(result.quantities["H_ratio"]))

    def test_shoulders_occluded_makes_h_undefined(self) -> None:
        # Without the shoulders there is no scale reference at all -- an
        # honest NaN, not a fabricated ratio against a stale width.
        pipe = _make_pipeline()
        lms = _standing_landmarks()
        vis = np.ones(33)
        for i in range(60):
            pipe.analyze(_pose_frame(i, i * DT, lms, vis))

        occluded_vis = vis.copy()
        occluded_vis[[LEFT_SHOULDER, RIGHT_SHOULDER]] = 0.1
        result = pipe.analyze(_pose_frame(60, 60 * DT, lms, occluded_vis))
        self.assertTrue(math.isnan(result.quantities["H_ratio"]))

    def test_undetected_frame_carries_no_h_reading(self) -> None:
        pipe = _make_pipeline()
        result = pipe.analyze(PoseFrame(
            frame_index=0, timestamp=0.0, detected=False, frame_size=(1280, 960),
        ))
        self.assertNotIn("H_ratio", result.quantities)


class TestQuantityHResetsOnDiscontinuity(unittest.TestCase):
    def test_a_gap_forgets_the_calibrated_baseline(self) -> None:
        pipe = _make_pipeline()
        lms = _standing_landmarks()
        vis = np.ones(33)
        for i in range(60):
            pipe.analyze(_pose_frame(i, i * DT, lms, vis))

        # A gap far past history_max_gap_s: a different body may now be in
        # frame, so the old "standing" calibration must not carry over.
        far_later = 60 * DT + 10.0
        result = pipe.analyze(_pose_frame(60, far_later, lms, vis))
        # Baseline reseeds on this frame (first sample after reset), so the
        # ratio reads 1.0 again rather than carrying the old calibration
        # forward or reading NaN.
        self.assertAlmostEqual(result.quantities["H_ratio"], 1.0, places=2)
