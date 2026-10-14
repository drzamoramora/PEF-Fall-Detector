"""What several test modules share: the test configuration and synthetic clips.

Nothing here is a test. Before 0.2.0.dev7 these lived inside test modules and
were imported from one another (``from tests.test_pipeline import TEST_CFG``),
which also made unittest collect some test classes twice.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.pose_frontend import (  # noqa: E402
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PoseFrame,
    compute_step0,
)
from pef_fall_detector.state_machine import SEQUENTIAL, Stage1Trigger  # noqa: E402

# ------------------------------------------------------------ configuration --
#: The configuration the unit tests are written against. Every key is required
#: (config.SCHEMA), and its values are deliberately NOT config.yaml's: each
#: test switches on the one behaviour it checks over a plain baseline —
#: sequential trigger at 2.2, no peak window, H and A off, no persistence,
#: no get-up check, no re-acquisition, cooldown after a rejection, and EMA
#: and velocity windows at the values the code fell back to before
#: 0.2.0.dev2. Research mode, so the runs these tests drive leave records to
#: check. The shipped combination (config.yaml) is checked end to end by
#: tests/test_end_to_end.py.
TEST_CFG = {
    "pose": {
        "model_complexity": 1,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "smooth_landmarks": True,
        "visibility_threshold": 0.5,
        "reacquire_after_s": 0.0,
    },
    "logging": {"output_dir": "logs", "mode": "research"},
    "stage1": {
        "threshold_T_deg": 45.0,
        "threshold_V": -1.5,
        "velocity_window_s": 5 / 30.0,
        "ema_time_constant_s": 0.093,
        "history_max_gap_s": 0.5,
        "trigger_formulation": "sequential",
        "trigger_score": 2.2,
        "trigger_hold_s": 0.1,
        "min_trunk_ratio": 0.15,
        "trunk_reference_window_s": 1.0,
        "confirm_window_s": 1.0,
        "trigger_peak_window_s": 0.0,
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
        "cooldown_after_rejection": True,
        "persistent_still_s": 0.5,
        "persistent_immobility_fraction": 0.0,
        "getup_rise_torsos": 0.0,
        "getup_window_s": 2.0,
    },
    "quantity_a": {
        "standing_T_deg": 20.0,
        "standing_knee_deg": 150.0,
        "min_standing_s": 0.5,
        "max_samples": 300,
        "severity_from_a": False,
        "band_floor": 0.26,
        "band_standing": 0.82,
        "reach_check": False,
        "floor_reached": 0.31,
        "hip_raised": 0.22,
        "reach_before_s": 1.0,
        "reach_after_s": 4.0,
        "trigger_from_a": False,
        "trigger_high": 0.7,
        "trigger_low": 0.3,
        "trigger_window_s": 1.5,
    },
    "experimental": {
        "height_baseline_time_constant_s": 2.0,
        "trigger_H_ratio": 0.0,
        "trigger_H_erect": 0.0,
        "trigger_H_sequence_window_s": 2.0,
        "recovery_H_ratio": 0.0,
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


# --------------------------------------------------------- synthetic clips --
FPS = 30.0


def toppling_pose(index: int, timestamp: float, angle_deg: float, hip_y: float):
    """Same geometry as test_pipeline's fall fixture: a 200 px trunk that swings."""
    pf = make_pose(index, timestamp, hip_y=hip_y, feet_x=640.0)
    rad = math.radians(angle_deg)
    pf.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 0] = 640.0 + 200.0 * math.sin(rad)
    pf.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 1] = hip_y - 200.0 * math.cos(rad)
    pf.landmarks[NOSE, 0] = 640.0 + 260.0 * math.sin(rad)
    pf.landmarks[NOSE, 1] = hip_y - 260.0 * math.cos(rad)
    pf.mid_hip, pf.mid_shoulder, pf.torso_length = compute_step0(pf.landmarks)
    return pf


def short_fall():
    """Topple, then lie still 4 s — under W — and the clip ends."""
    frames = [toppling_pose(i, i / FPS, angle_deg=i * 4.5, hip_y=500.0 + i * 12.0)
              for i in range(20)]
    frames += [toppling_pose(i, i / FPS, angle_deg=90.0, hip_y=740.0)
               for i in range(20, 140)]
    return frames


def down_past_w_then_up():
    """The A13 case: still on the floor past W, then stands up unaided.

    Greedy resolution closes the event at W as ``severe`` and never sees the
    recovery; §3.3 labels by how the episode ENDED, which is standing.
    Ankles at 760 px give a 1.3 hip-to-ankle ratio: legs extended, so the
    recovery is verifiable (a 1.0 ratio would read as "could not verify").
    """
    frames = [toppling_pose(i, i / FPS, angle_deg=i * 4.5, hip_y=500.0 + i * 12.0)
              for i in range(20)]
    frames += [toppling_pose(i, i / FPS, angle_deg=90.0, hip_y=740.0)
               for i in range(20, 220)]            # 6.7 s still: past W = 5 s
    frames += [make_pose(i, i / FPS, hip_y=500.0, shoulder_y=300.0,
                         ankle_y=760.0, feet_x=640.0)
               for i in range(220, 300)]           # 2.7 s standing
    return frames


class FakeSource:
    """A recorded clip: finite, not live."""

    is_live = False
    fps = FPS

    def __init__(self, n_frames: int) -> None:
        self.frame_count = n_frames
        self._i = 0

    def read(self):
        if self._i >= self.frame_count:
            return False, None, None
        t = self._i / FPS
        self._i += 1
        return True, object(), t

    def release(self) -> None:
        pass

    def fps_report(self, expected=None) -> dict:
        return {"declared_fps": FPS, "frame_count": self.frame_count,
                "implied_duration_s": self.frame_count / FPS}


class HeadlessRun:
    """Runs main.py's headless path on a scripted clip (mixin for a TestCase).

    Only MediaPipe and the video file are replaced; the pipeline, the funnel
    and the records are the real ones. ``scenario`` returns the frames.
    """

    scenario = staticmethod(short_fall)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.frames = self.scenario()
        self.built_with: list[bool] = []
        self.finalised_at: list[float] = []
        frames, built, fin = self.frames, self.built_with, self.finalised_at

        class _ScriptedPipeline(FramePipeline):
            """The real pipeline; only MediaPipe is replaced by scripted poses."""

            def __init__(self, cfg, labelling: bool = False) -> None:
                super().__init__(cfg, labelling=labelling)
                built.append(labelling)

            def process(self, frame_bgr, frame_index, timestamp):
                return self.analyze(frames[frame_index])

            def finalise(self, timestamp):
                fin.append(timestamp)
                return super().finalise(timestamp)

        cfg = copy.deepcopy(TEST_CFG)
        cfg["logging"]["output_dir"] = self.tmp.name
        self._saved = (main.load_config, main.VideoFileSource, main.FramePipeline)
        main.load_config = lambda _path: Config(cfg)
        main.VideoFileSource = lambda _path: FakeSource(len(frames))
        main.FramePipeline = _ScriptedPipeline

    def tearDown(self) -> None:
        main.load_config, main.VideoFileSource, main.FramePipeline = self._saved
        self.tmp.cleanup()

    def _run(self) -> list[dict]:
        args = argparse.Namespace(video="A00-S1-NotRecovered.mp4", config=None,
                                  expected_seconds=None, headless=True)
        with contextlib.redirect_stdout(io.StringIO()):
            main.run_headless(args)
        events = sorted(Path(self.tmp.name).glob("*-events-*.csv"))
        self.assertEqual(len(events), 1, "no events record was written")
        with events[0].open(newline="") as fh:
            return list(csv.DictReader(fh))


# ----------------------------------------------------------- Stage 1 alone --
T_TH, V_TH = 45.0, -1.5
DT = 1.0 / 30.0
HOLD = 0.1                       # 3 frames at 30 fps


def trigger(formulation: str = SEQUENTIAL, **kw) -> Stage1Trigger:
    kw.setdefault("threshold_t_deg", T_TH)
    kw.setdefault("threshold_v_tps", V_TH)
    kw.setdefault("hold_s", HOLD)
    return Stage1Trigger(formulation=formulation, **kw)


def feed(trg: Stage1Trigger, samples, t0: float = 0.0) -> list[float]:
    """Feed (T, V) pairs at 30 fps; return the times at which it fired."""
    fired = []
    for i, (t_deg, v_tps) in enumerate(samples):
        ts = t0 + i * DT
        if trg.update(ts, t_deg, v_tps):
            fired.append(ts)
    return fired


# ------------------------------------------ 3D world landmarks (Quantity A) --
def world_body(points: dict[int, tuple[float, float, float]],
               pitch_deg: float = 30.0) -> np.ndarray:
    """33×3 world landmarks from a gravity-frame body, seen by a pitched camera.

    Gravity frame: y DOWN (MediaPipe's convention), hips at the origin. The
    camera looks down by ``pitch_deg``, so its axes are rotated about x: the
    calibration must recover "up" from the subject, not assume (0, -1, 0).
    """
    w = np.zeros((33, 3))
    for i, p in points.items():
        w[i] = p
    a = math.radians(pitch_deg)
    rot = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    return w @ rot.T


def world_pose(ankles, knees, shoulders_y, nose, hips=((-0.1, 0, 0), (0.1, 0, 0)), shoulders_z=0.0):
    return {0: nose, 11: (-0.18, shoulders_y, shoulders_z), 12: (0.18, shoulders_y, shoulders_z),
            23: hips[0], 24: hips[1], 25: knees[0], 26: knees[1], 27: ankles[0], 28: ankles[1]}


POSE_STANDING = world_pose(ankles=((-0.1, 0.9, 0), (0.1, 0.9, 0)),
                           knees=((-0.1, 0.45, 0), (0.1, 0.45, 0)),
                           shoulders_y=-0.5, nose=(0, -0.7, 0))         # head 1.6 m above ankles
POSE_LYING = world_pose(ankles=((-0.1, 0, 0.9), (0.1, 0, 0.9)),
                        knees=((-0.1, 0, 0.45), (0.1, 0, 0.45)),
                        shoulders_y=0.0, shoulders_z=-0.5,
                        nose=(0, -0.05, -0.7))                  # head 0.05 m above
POSE_KNEELING = world_pose(ankles=((-0.1, 0.45, 0.45), (0.1, 0.45, 0.45)),
                 knees=((-0.1, 0.45, 0), (0.1, 0.45, 0)),
                 shoulders_y=-0.5, nose=(0, -0.7, 0))                   # 1.15 m → 0.72
POSE_SITTING_FLOOR = world_pose(ankles=((-0.1, 0.05, 0.9), (0.1, 0.05, 0.9)),
                      knees=((-0.1, 0.0, 0.45), (0.1, 0.0, 0.45)),
                      shoulders_y=-0.5, nose=(0, -0.7, 0))              # 0.75 m → 0.47


def attach_world(pf, world):
    pf.world_landmarks = world
    return pf
