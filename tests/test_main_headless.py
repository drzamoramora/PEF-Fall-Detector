"""The headless runner labels a recorded clip the way PEF-Lab does.

Until 23/09 ``main.py --headless`` built ``FramePipeline(cfg)`` — greedy,
live-device resolution — and never called ``finalise()``. An event still
open when the clip ran out was never judged, so every clip that ends with
the subject on the floor came out wrong. Measured on A05-S1 with the same
clip and config: ``stage2_inconclusive`` headless, ``stage3_confirmed`` /
``severe`` in PEF-Lab. These tests pin the two halves of the fix.

The fall used here lies still for 4 s, under W = 5 s, and then the clip
ends. Greedy resolution never reaches W, so only ``finalise()`` — reading
how the episode ENDED (§3.3) — can close it. That is the case the bug hid.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.pose_frontend import (  # noqa: E402
    LEFT_SHOULDER,
    NOSE,
    RIGHT_SHOULDER,
    compute_step0,
)
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402

FPS = 30.0


def _toppling_pose(index: int, timestamp: float, angle_deg: float, hip_y: float):
    """Same geometry as test_pipeline's fall fixture: a 200 px trunk that swings."""
    pf = make_pose(index, timestamp, hip_y=hip_y, feet_x=640.0)
    rad = math.radians(angle_deg)
    pf.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 0] = 640.0 + 200.0 * math.sin(rad)
    pf.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 1] = hip_y - 200.0 * math.cos(rad)
    pf.landmarks[NOSE, 0] = 640.0 + 260.0 * math.sin(rad)
    pf.landmarks[NOSE, 1] = hip_y - 260.0 * math.cos(rad)
    pf.mid_hip, pf.mid_shoulder, pf.torso_length = compute_step0(pf.landmarks)
    return pf


def _short_fall():
    """Topple, then lie still 4 s — under W — and the clip ends."""
    frames = [_toppling_pose(i, i / FPS, angle_deg=i * 4.5, hip_y=500.0 + i * 12.0)
              for i in range(20)]
    frames += [_toppling_pose(i, i / FPS, angle_deg=90.0, hip_y=740.0)
               for i in range(20, 140)]
    return frames


def _down_past_w_then_up():
    """The A13 case: still on the floor past W, then stands up unaided.

    Greedy resolution closes the event at W as ``severe`` and never sees the
    recovery; §3.3 labels by how the episode ENDED, which is standing.
    Ankles at 760 px give a 1.3 hip-to-ankle ratio: legs extended, so the
    recovery is verifiable (a 1.0 ratio would read as "could not verify").
    """
    frames = [_toppling_pose(i, i / FPS, angle_deg=i * 4.5, hip_y=500.0 + i * 12.0)
              for i in range(20)]
    frames += [_toppling_pose(i, i / FPS, angle_deg=90.0, hip_y=740.0)
               for i in range(20, 220)]            # 6.7 s still: past W = 5 s
    frames += [make_pose(i, i / FPS, hip_y=500.0, shoulder_y=300.0,
                         ankle_y=760.0, feet_x=640.0)
               for i in range(220, 300)]           # 2.7 s standing
    return frames


class _FakeSource:
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


class TestHeadlessLabelsLikePefLab(unittest.TestCase):

    scenario = staticmethod(_short_fall)

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
        main.VideoFileSource = lambda _path: _FakeSource(len(frames))
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
        self.assertEqual(len(events), 1, "no se escribio el registro de eventos")
        with events[0].open(newline="") as fh:
            return list(csv.DictReader(fh))

    def test_a_recorded_clip_builds_the_pipeline_in_labelling_mode(self) -> None:
        self._run()
        self.assertEqual(self.built_with, [True])

    def test_the_open_event_is_closed_at_the_last_frame(self) -> None:
        self._run()
        self.assertEqual(self.finalised_at, [self.frames[-1].timestamp])

    def test_a_fall_that_ends_on_the_floor_is_recorded_as_severe(self) -> None:
        """The behaviour the two halves above exist for.

        Greedy resolution leaves this event open (immobility never reaches
        W), and a record written without finalise() says nothing useful.
        """
        rows = self._run()
        self.assertEqual(len(rows), 1, "la Etapa 1 no disparo")
        self.assertEqual(rows[0]["verdict"], "stage3_confirmed")
        self.assertEqual(rows[0]["severity"], "severe")


class TestHeadlessLabelsByHowTheEpisodeEnded(TestHeadlessLabelsLikePefLab):
    """Where labelling mode, not finalise() alone, changes the answer."""

    scenario = staticmethod(_down_past_w_then_up)

    def test_a_fall_that_ends_on_the_floor_is_recorded_as_severe(self) -> None:
        self.skipTest("escenario distinto: este sujeto se levanta")

    def test_a_subject_who_gets_up_after_w_is_labelled_as_recovered(self) -> None:
        rows = self._run()
        self.assertEqual(len(rows), 1, "la Etapa 1 no disparo")
        self.assertEqual((rows[0]["verdict"], rows[0]["severity"]),
                         ("stage3_nullified", "mild"))


if __name__ == "__main__":
    unittest.main()
