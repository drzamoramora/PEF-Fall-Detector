"""One writer for the records of a run, and the batch run of main.py.

The terminal, PEF-Lab and the replay tool write through records.RunRecords
(0.2.0.dev3), so a run leaves the same records whoever drives it. Research
records the frames, the events and the 33 landmarks; deployment records
nothing (§3.6). Since records schema 2 (0.3.0.dev0) a run writes into a
folder of its own (run_folder.RunFolder), with run.json, one folder per clip,
the landmarks exact in two formats and, for a folder run, its metrics.
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
from pef_fall_detector.run_folder import RunFolder, dataset_sha256, run_name  # noqa: E402
from pef_fall_detector.state_machine import TriggerEvent  # noqa: E402
from tests.support import (  # noqa: E402
    TEST_CFG,
    FakeSource,
    make_pose,
    recorded,
    short_fall,
)


def _event(i: int, verdict: str = "stage2_rejected") -> TriggerEvent:
    return TriggerEvent(frame_index=10 * i, timestamp=i / 3, t_deg=50.0, v_tps=-2.0,
                        formulation="score", trigger_source="score", verdict=verdict,
                        reason=f"event {i}")


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestRunRecords(unittest.TestCase):

    def test_research_writes_the_records_of_a_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = RunRecords(tmp, "A01-S1-Recovered", metadata={"source_fps_declared": 30.0})
            pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)))
            for i in range(5):
                run.frame(pipe.analyze(make_pose(i, i / 30)))
            run.close([_event(0)])
            root = Path(tmp)
            self.assertEqual(run.path, root / "clips" / "A01-S1-Recovered" / "frames.csv")
            self.assertEqual(len(_rows(run.path)), 5)
            self.assertEqual(len(_rows(run.landmarks_path)), 5)
            self.assertEqual(len(_rows(run.events_path)), 1)
            self.assertEqual(run.landmarks_npz_path,
                             root / "landmarks" / "npz" / "A01-S1-Recovered.npz")
            self.assertTrue(run.landmarks_npz_path.exists())
            clip = json.loads((root / "clips" / "A01-S1-Recovered" / "clip.json").read_text())
            self.assertEqual(clip["source_fps_declared"], 30.0)
            self.assertEqual((clip["rows"], clip["events"], clip["complete"]), (5, 1, True))
            self.assertEqual(list(root.glob("**/*.meta.json")), [])   # no sidecars
            info = json.loads((root / "run.json").read_text())
            self.assertEqual(info["schema_version"], 2)

    def test_a_clip_is_never_recorded_twice_in_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FramePipeline(Config(copy.deepcopy(TEST_CFG)))
            first = RunRecords(tmp, "A01-S1-Recovered")
            first.frame(pipe.analyze(make_pose(0, 0.0)))
            first.close([])
            again = RunRecords(tmp, "A01-S1-Recovered")
            with self.assertRaises(FileExistsError):
                again.frame(pipe.analyze(make_pose(0, 0.0)))

    def test_an_interrupted_pass_is_marked_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = RunRecords(tmp, "clip")
            run.frame(FramePipeline(Config(copy.deepcopy(TEST_CFG))).analyze(make_pose(0, 0)))
            run.close([], complete=False)
            clip = json.loads((Path(tmp) / "clips" / "clip" / "clip.json").read_text())
            self.assertFalse(clip["complete"])
            self.assertEqual(RunFolder(tmp).complete_clips(), set())

    def test_nothing_recorded_leaves_no_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = RunRecords(tmp, "clip")
            run.close([])
            self.assertFalse((Path(tmp) / "clips").exists())

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

    def test_alarm_and_trigger_sources(self) -> None:
        events = [_event(0), _event(1, "stage3_confirmed")]
        events[1].trigger_source = "A"
        row = label_row(Path("A01-S1-Recovered.mp4"), events, 30, 30, 1.0, datetime.now(),
                        video_sha256="ab" * 32)
        self.assertEqual((row["alarm"], row["trigger_sources"]), (1, "score;A"))
        self.assertEqual(row["video_sha256"], "ab" * 32)
        quiet = label_row(Path("x.mp4"), events, 30, 30, 1.0, datetime.now(), dispatch=())
        self.assertEqual(quiet["alarm"], 0)


class TestRunFolder(unittest.TestCase):

    def test_the_name_says_when_where_and_on_what(self) -> None:
        name = run_name(datetime(2026, 10, 14, 0, 43), "my recordings", "macos-arm64")
        self.assertEqual(name, "2026-10-14_0043-macos-arm64-my-recordings")

    def test_a_new_run_never_reuses_a_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            when = datetime(2026, 10, 14, 0, 43)
            a = RunFolder.new(tmp, "recordings", kind="dataset", started=when)
            b = RunFolder.new(tmp, "recordings", kind="dataset", started=when)
            self.assertNotEqual(a.path, b.path)
            self.assertTrue(b.path.name.endswith("-recordings-2"))

    def test_run_json_says_what_the_run_was(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = RunFolder.new(tmp, "recordings", kind="dataset", seen_at="/somewhere",
                                config={"stage1": {"trigger_score": 2.6}})
            info = json.loads((run.path / "run.json").read_text())
            for key in ("schema_version", "run_name", "kind", "started_at", "code_version",
                        "git_commit", "platform", "libraries", "config", "partition",
                        "schemas", "dataset"):
                self.assertIn(key, info)
            self.assertEqual(info["status"], "running")
            self.assertEqual(info["dataset"]["seen_at"], "/somewhere")
            self.assertIn("machine", info["platform"])
            self.assertIn("numpy", info["libraries"])
            self.assertIn("frame_sha1", info["schemas"]["frames"])
            run.close()
            self.assertEqual(json.loads((run.path / "run.json").read_text())["status"],
                             "finished")

    def test_the_dataset_hash_is_by_content_not_name_or_order(self) -> None:
        self.assertEqual(dataset_sha256(["b", "a"]), dataset_sha256(["a", "b"]))
        self.assertNotEqual(dataset_sha256(["a", "b"]), dataset_sha256(["a", "c"]))


class TestLandmarkArrays(unittest.TestCase):

    def test_the_npz_holds_mediapipe_numbers_and_gives_back_the_same_pixels(self) -> None:
        import numpy as np

        from pef_fall_detector import landmark_arrays
        pf = make_pose(3, 0.1)
        w, h = pf.frame_size
        norm = (np.random.default_rng(1).random((33, 3)) * 0.9).astype(np.float32)
        pf.landmarks_norm = norm
        pf.landmarks = np.array([[float(x) * w, float(y) * h, float(z) * w]
                                 for x, y, z in norm])             # as pose_frontend does
        arrays = landmark_arrays.LandmarkArrays()
        arrays.add(pf)
        arrays.add(PoseFrameUndetected())
        with tempfile.TemporaryDirectory() as tmp:
            path = arrays.save(Path(tmp) / "x.npz")
            data = landmark_arrays.load(path)
        self.assertTrue(np.array_equal(data["image"][0], norm))
        pixels = landmark_arrays.pixel_landmarks(data["image"][0], w, h)
        self.assertTrue(np.array_equal(pixels, pf.landmarks))      # bit for bit
        self.assertFalse(data["detected"][1])
        self.assertTrue(np.isnan(data["image"][1]).all())


def PoseFrameUndetected():
    from pef_fall_detector.pose_frontend import PoseFrame
    return PoseFrame(frame_index=4, timestamp=0.2, detected=False, frame_size=(64, 48))


class _Batch(unittest.TestCase):
    """main.py --folder over two fake clips."""

    def _run(self, mode: str, quiet: bool = True, **options) -> tuple[Path, str]:
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
                                      output_dir=str(out), quiet=quiet, **options)
            with contextlib.redirect_stdout(io.StringIO()) as text:
                self.code = main.run_folder(args)
        finally:
            main.load_config, main.VideoFileSource, main.FramePipeline = saved
        self.clips_dir = clips
        return out, text.getvalue()


class TestFolder(_Batch):

    def test_research_labels_every_clip_and_records_each_run(self) -> None:
        out, text = self._run("research")
        self.assertEqual(self.code, 0)
        labels = recorded(out, "labels")
        self.assertEqual(len(labels), 1)
        run = labels[0].parent
        self.assertTrue(run.name.endswith(f"-{self.clips_dir.name}"))
        rows = _rows(labels[0])
        self.assertEqual([r["video"] for r in rows],
                         ["A00-S1-NotRecovered.mp4", "B00-S1-NoFall.mp4"])
        self.assertEqual([r["true_class"] for r in rows], ["NotRecovered", "NoFall"])
        self.assertEqual([p.stem for p in recorded(out, "landmarks")],
                         ["A00-S1-NotRecovered", "B00-S1-NoFall"])
        self.assertIn("[1/2] A00-S1-NotRecovered", text)
        self.assertIn("[2/2] B00-S1-NoFall", text)
        self.assertIn("ETA", text)
        # The same confusion matrix as PEF-Lab's queue, and every metric.
        self.assertIn("correct:", text)
        self.assertIn("Accuracy, 4 classes", text)
        self.assertIn(str(run), text)
        self.assertTrue((run / "metrics.csv").exists())
        self.assertTrue((run / "metrics.txt").exists())
        info = json.loads((run / "run.json").read_text())
        self.assertEqual(info["kind"], "dataset")
        self.assertEqual(info["status"], "finished")
        self.assertEqual(info["dataset"]["clips"], 2)
        self.assertEqual(info["dataset"]["seen_at"], str(self.clips_dir.resolve()))
        self.assertEqual(info["config"]["logging"]["output_dir"], str(out.resolve()))
        self.assertEqual(info["config"]["logging"]["mode"], "research")
        # The clips' content, empty files here: the same hash, and the dataset's.
        clip = json.loads((run / "clips" / "B00-S1-NoFall" / "clip.json").read_text())
        self.assertEqual(clip["video_sha256"], rows[1]["video_sha256"])
        self.assertEqual(len(info["dataset"]["sha256"]), 64)

    def test_only_the_clips_asked_for(self) -> None:
        out, text = self._run("research", clips="B00-S1")
        rows = _rows(recorded(out, "labels")[0])
        self.assertEqual([r["video"] for r in rows], ["B00-S1-NoFall.mp4"])
        self.assertIn("[1/1] B00-S1-NoFall", text)

    def test_a_clip_that_fails_does_not_stop_the_others(self) -> None:
        saved = main.run_clip

        def flaky(cfg, video, *a, **kw):
            if "A00" in str(video):
                raise RuntimeError("broken file")
            return saved(cfg, video, *a, **kw)

        main.run_clip = flaky
        try:
            out, text = self._run("research")
        finally:
            main.run_clip = saved
        self.assertEqual(self.code, main.EXIT_CLIP_FAILED)
        rows = _rows(recorded(out, "labels")[0])
        self.assertEqual([r["video"] for r in rows], ["B00-S1-NoFall.mp4"])
        self.assertIn("1 clip(s) failed", text)
        self.assertIn("A00-S1-NotRecovered.mp4: RuntimeError: broken file", text)
        info = json.loads(recorded(out, "runs")[0].read_text())
        self.assertEqual(info["status"], "finished with failures")

    def test_compare_to_a_reference_run(self) -> None:
        out, _ = self._run("research")
        reference = recorded(out, "runs")[0].parent
        _, text = self._run("research", compare_to=str(reference))
        self.assertEqual(self.code, 0, text)
        self.assertIn("identical clips: 2/2", text)
        self.assertIn("-> IDENTICAL", text)
        # A reference with another verdict: exit status 2.
        events = reference / "clips" / "A00-S1-NotRecovered" / "events.csv"
        events.write_text(events.read_text().replace("stage3_confirmed", "stage2_rejected"))
        _, text = self._run("research", compare_to=str(reference))
        self.assertEqual(self.code, main.EXIT_DIFFERENT, text)
        self.assertIn("-> DIFFERENT", text)

    def test_resume_runs_only_what_is_missing(self) -> None:
        out, _ = self._run("research", clips="A00-S1")
        run = recorded(out, "runs")[0].parent
        # A pass of B00 that broke off: an incomplete clip folder is left behind.
        (run / "clips" / "B00-S1-NoFall").mkdir()
        (run / "clips" / "B00-S1-NoFall" / "frames.csv").write_text("partial")
        _, text = self._resume(run)
        self.assertIn("1 clips done, 1 to run", text)
        rows = _rows(run / "dataset-labels.csv")
        self.assertEqual([r["video"] for r in rows],
                         ["A00-S1-NotRecovered.mp4", "B00-S1-NoFall.mp4"])
        self.assertEqual(json.loads((run / "run.json").read_text())["status"], "finished")
        self.assertEqual(len(recorded(out, "runs")), 1)        # the same run, not a new one

    def test_resume_refuses_another_configuration(self) -> None:
        out, _ = self._run("research", clips="A00-S1")
        run = recorded(out, "runs")[0].parent
        with self.assertRaises(SystemExit):
            self._resume(run, trigger_score=9.9)

    def test_resume_needs_no_mode(self) -> None:
        # The run was recorded, so resuming it records: no --mode research needed,
        # even with a config.yaml that says deployment.
        out, _ = self._run("research", clips="A00-S1")
        run = recorded(out, "runs")[0].parent
        code, text = self._resume(run, mode=None, file_mode="deployment")
        self.assertEqual(code, 0, text)
        self.assertIn("1 clips done, 1 to run", text)
        self.assertEqual(len(_rows(run / "dataset-labels.csv")), 2)

    def _resume(self, run: Path, trigger_score: float | None = None,
                mode: str | None = "research", file_mode: str | None = None) -> tuple[int, str]:
        frames = short_fall()
        still = [make_pose(i, i / 30) for i in range(60)]

        class _Scripted(FramePipeline):
            def process(self, frame_bgr, frame_index, timestamp):
                script = frames if "A00" in self.source_name else still
                return self.analyze(script[min(frame_index, len(script) - 1)])

        info = json.loads((run / "run.json").read_text())
        cfg = copy.deepcopy(info["config"])
        if trigger_score is not None:
            cfg["stage1"]["trigger_score"] = trigger_score
        if file_mode is not None:
            cfg["logging"]["mode"] = file_mode
        lengths = {"A00": len(frames), "B00": len(still)}
        saved = (main.load_config, main.VideoFileSource, main.FramePipeline)
        main.load_config = lambda _p: Config(cfg)
        main.VideoFileSource = lambda p: FakeSource(lengths[Path(p).name[:3]])
        main.FramePipeline = _Scripted
        try:
            args = argparse.Namespace(folder=None, resume=str(run), config=None,
                                      mode=mode, output_dir=None, quiet=True)
            with contextlib.redirect_stdout(io.StringIO()) as text:
                code = main.run_folder(args)
        finally:
            main.load_config, main.VideoFileSource, main.FramePipeline = saved
        return code, text.getvalue()

    def test_deployment_records_nothing_but_still_reports(self) -> None:
        out, text = self._run("deployment")
        self.assertEqual(list(out.iterdir()), [])
        self.assertIn("correct:", text)
        self.assertIn("Accuracy, 4 classes", text)
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
