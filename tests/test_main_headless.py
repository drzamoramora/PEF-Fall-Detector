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
import io
import unittest

import main
from tests.support import FakeSource, HeadlessRun, down_past_w_then_up


class _LabellingMode:
    """The two halves of the fix, checked on every scenario below."""

    def test_a_recorded_clip_builds_the_pipeline_in_labelling_mode(self) -> None:
        self._run()
        self.assertEqual(self.built_with, [True])

    def test_the_open_event_is_closed_at_the_last_frame(self) -> None:
        self._run()
        self.assertEqual(self.finalised_at, [self.frames[-1].timestamp])


class TestHeadlessLabelsLikePefLab(_LabellingMode, HeadlessRun, unittest.TestCase):

    def test_a_fall_that_ends_on_the_floor_is_recorded_as_severe(self) -> None:
        """The behaviour the two halves above exist for.

        Greedy resolution leaves this event open (immobility never reaches
        W), and a record written without finalise() says nothing useful.
        """
        rows = self._run()
        self.assertEqual(len(rows), 1, "Stage 1 did not fire")
        self.assertEqual(rows[0]["verdict"], "stage3_confirmed")
        self.assertEqual(rows[0]["severity"], "severe")


class TestHeadlessLabelsByHowTheEpisodeEnded(_LabellingMode, HeadlessRun, unittest.TestCase):
    """Where labelling mode, not finalise() alone, changes the answer."""

    scenario = staticmethod(down_past_w_then_up)

    def test_a_subject_who_gets_up_after_w_is_labelled_as_recovered(self) -> None:
        rows = self._run()
        self.assertEqual(len(rows), 1, "Stage 1 did not fire")
        self.assertEqual((rows[0]["verdict"], rows[0]["severity"]),
                         ("stage3_nullified", "mild"))



class TestAFailingRunStillClosesItsFiles(HeadlessRun, unittest.TestCase):
    """A frame that raises must not leave the video open or a record unclosed.

    ``run_clip`` releases the source and closes the records in a ``finally``:
    an open capture keeps the file busy, and a record that is never closed
    loses the rows still in its buffer.
    """

    FAIL_AT = 50

    def setUp(self) -> None:
        super().setUp()
        released, closed = [], []
        self.released, self.closed = released, closed
        fail_at, scripted = self.FAIL_AT, main.FramePipeline

        class _Breaks(scripted):
            def process(self, frame_bgr, frame_index, timestamp):
                if frame_index == fail_at:
                    raise RuntimeError("frame decoder exploded")
                return super().process(frame_bgr, frame_index, timestamp)

        class _Source(FakeSource):
            def release(self) -> None:
                released.append(True)

        class _Records(main.RunRecords):
            def close(self, events) -> None:
                closed.append(True)
                super().close(events)

        n = len(self.frames)
        self._more = (main.RunRecords,)
        main.FramePipeline = _Breaks
        main.VideoFileSource = lambda _path: _Source(n)
        main.RunRecords = _Records

    def tearDown(self) -> None:
        (main.RunRecords,) = self._more
        super().tearDown()

    def test_the_source_is_released_and_the_records_closed(self) -> None:
        args = argparse.Namespace(video="A00-S1-NotRecovered.mp4", config=None,
                                  expected_seconds=None, headless=True)
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            main.run_headless(args)
        self.assertEqual(self.released, [True])
        self.assertEqual(self.closed, [True])


if __name__ == "__main__":
    unittest.main()
