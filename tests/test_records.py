"""One writer for the records of a run, and the batch run of main.py (0.2.0.dev3).

The terminal, PEF-Lab and the replay tool write through records.RunRecords,
so a run leaves the same records whoever drives it. Research records the
frames, the events and the 33 landmarks; deployment records nothing (§3.6).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import main  # noqa: E402
from pef_fall_detector.audit_log import EVENT_FIELDS  # noqa: E402
from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.dataset import LABEL_FIELDS, label_row  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.records import RunRecords, event_row  # noqa: E402
from pef_fall_detector.state_machine import TriggerEvent  # noqa: E402
from tests.support import TEST_CFG, FakeSource, make_pose, short_fall  # noqa: E402


def _event(i: int, verdict: str = "stage2_rejected") -> TriggerEvent:
    return TriggerEvent(frame_index=10 * i, timestamp=i / 3, t_deg=50.0, v_tps=-2.0,
                        formulation="score", trigger_source="score", verdict=verdict,
                        reason=f"event {i}")


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestRunRecords(unittest.TestCase):

    def test_research_writes_three_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = RunRecords(tmp, "A01-S1-Recovered", metadata={"source_fps_declared": 30.0})
            pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)))
            for i in range(5):
                run.frame(pipe.analyze(make_pose(i, i / 30)))
            run.close([_event(0)])
            names = sorted(p.name for p in Path(tmp).glob("*.csv"))
            self.assertEqual(len(names), 3)
            self.assertEqual(len(_rows(run.path)), 5)
            self.assertEqual(len(_rows(run.landmarks_path)), 5)
            self.assertEqual(len(_rows(run.events_path)), 1)
            meta = json.loads(run.events_path.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["record_kind"], "stage1_events")
            self.assertEqual(meta["source_fps_declared"], 30.0)

    def test_deployment_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = RunRecords(tmp, "camera0", research=False)
            pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)))
            run.frame(pipe.analyze(make_pose(0, 0.0)))
            run.close([_event(0)])
            self.assertIsNone(run.path)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_events_are_written_in_order_abandoned_ones_included(self) -> None:
        events = [_event(0), _event(1, "stage1_only"), _event(2), _event(3, "stage1_only")]
        with tempfile.TemporaryDirectory() as tmp:
            run = RunRecords(tmp, "clip")
            run.event_settled(events[:1], events[0])
            self.assertEqual(run.events_written, 1)
            # Event 1 never got a verdict (a detection loss abandoned it):
            # it is written in its place when event 2 settles ...
            run.event_settled(events[:3], events[2])
            self.assertEqual(run.events_written, 3)
            # ... and event 3, still open when the source ended, at close.
            run.close(events)
            rows = _rows(run.events_path)
        self.assertEqual([r["event_index"] for r in rows], ["0", "1", "2", "3"])
        self.assertEqual([r["reason"] for r in rows],
                         ["event 0", "event 1", "event 2", "event 3"])

    def test_the_event_row_is_the_record_schema(self) -> None:
        self.assertEqual(list(event_row(0, _event(0))), EVENT_FIELDS)


class TestLabelRow(unittest.TestCase):

    def test_columns_in_english_and_in_order(self) -> None:
        row = label_row(Path("A01-S1-Recovered.mp4"), [_event(0)], 30, 27, 1.0,
                        datetime(2026, 10, 6, 9, 30))
        self.assertEqual(list(row), LABEL_FIELDS)
        self.assertEqual((row["date"], row["time"], row["reliable_frames_pct"]),
                         ("2026-10-06", "09:30:00", "90.0"))
        self.assertEqual((row["detected_class"], row["true_class"], row["correct"]),
                         ("NoFall", "Recovered", 0))


class _Batch(unittest.TestCase):
    """main.py --folder over two fake clips."""

    def _run(self, mode: str, quiet: bool = True) -> tuple[Path, str]:
        out = Path(tempfile.mkdtemp())
        clips = Path(tempfile.mkdtemp())
        for name in ("A00-S1-NotRecovered.mp4", "B00-S1-NoFall.mp4"):
            (clips / name).touch()
        frames = short_fall()
        still = [make_pose(i, i / 30) for i in range(60)]

        class _Scripted(FramePipeline):
            def process(self, frame_bgr, frame_index, timestamp):
                script = frames if "A00" in self.source_name else still
                return self.analyze(script[min(frame_index, len(script) - 1)])

        cfg = copy.deepcopy(TEST_CFG)
        lengths = {"A00": len(frames), "B00": len(still)}
        saved = (main.load_config, main.VideoFileSource, main.FramePipeline)
        main.load_config = lambda _p: Config(cfg)
        main.VideoFileSource = lambda p: FakeSource(lengths[Path(p).name[:3]])
        main.FramePipeline = _Scripted
        try:
            args = argparse.Namespace(folder=str(clips), config=None, mode=mode,
                                      output_dir=str(out), quiet=quiet)
            with contextlib.redirect_stdout(io.StringIO()) as text:
                code = main.run_folder(args)
        finally:
            main.load_config, main.VideoFileSource, main.FramePipeline = saved
        self.assertEqual(code, 0)
        return out, text.getvalue()


class TestFolder(_Batch):

    def test_research_labels_every_clip_and_records_each_run(self) -> None:
        out, text = self._run("research")
        labels = sorted(out.glob("dataset-labels-*.csv"))
        self.assertEqual(len(labels), 1)
        rows = _rows(labels[0])
        self.assertEqual([r["video"] for r in rows],
                         ["A00-S1-NotRecovered.mp4", "B00-S1-NoFall.mp4"])
        self.assertEqual([r["true_class"] for r in rows], ["NotRecovered", "NoFall"])
        self.assertEqual(len(list(out.glob("A00-S1-NotRecovered-landmarks-*.csv"))), 1)
        self.assertIn("[1/2] A00-S1-NotRecovered", text)
        self.assertIn("[2/2] B00-S1-NoFall", text)
        # The same closing summary as PEF-Lab's queue.
        self.assertIn("correct:", text)
        self.assertIn(str(labels[0]), text)
        meta = json.loads(labels[0].with_suffix(".meta.json").read_text())
        self.assertEqual(meta["config"]["logging"]["output_dir"], str(out.resolve()))
        self.assertEqual(meta["config"]["logging"]["mode"], "research")

    def test_deployment_records_nothing_but_still_reports(self) -> None:
        out, text = self._run("deployment")
        self.assertEqual(list(out.iterdir()), [])
        self.assertIn("correct:", text)
        self.assertIn("deployment mode writes nothing", text)

    def test_not_quiet_shows_every_event_and_its_reason(self) -> None:
        _, text = self._run("research", quiet=False)
        self.assertIn("reason:", text)
        self.assertIn("-> stage3_", text)


class TestOptions(unittest.TestCase):

    def test_without_options_the_config_is_the_file(self) -> None:
        args = argparse.Namespace(config=None)
        self.assertEqual(main.run_config(args).logging.mode, "deployment")

    def test_mode_and_output_dir_go_into_the_config(self) -> None:
        args = argparse.Namespace(config=None, mode="research", output_dir="somewhere")
        cfg = main.run_config(args)
        self.assertEqual(cfg.logging.mode, "research")
        self.assertEqual(cfg.logging.output_dir, str(Path("somewhere").resolve()))


if __name__ == "__main__":
    unittest.main()
