"""The analysis tools in ``tools/``, and the provenance every record carries.

* ``_records.find_runs``: which files belong to which clip and run;
* ``replay.py``: rebuilding frames from the landmark record and running them
  through the detector must reproduce a run exactly when the landmarks are
  exact (they are here: every synthetic coordinate is representable at the
  precision the record keeps);
* ``compare_runs.py``: finds every difference, and forgives exactly the ones a
  map declares;
* ``metrics.py``: the headline numbers of a run;
* ``audit_log.git_provenance``: the commit a record came from.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import _records  # noqa: E402
import compare_runs  # noqa: E402
import metrics  # noqa: E402
import replay  # noqa: E402
from pef_fall_detector import __version__, audit_log  # noqa: E402
from pef_fall_detector.audit_log import (  # noqa: E402
    EVENT_FIELDS,
    LANDMARK_FIELDS,
    AuditLogger,
    landmark_row,
)
from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.state_machine import A_FALLBACK, TriggerEvent  # noqa: E402
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402
from tests.test_quantity_a import STANDING, _body, _pose  # noqa: E402

FPS = 30.0
STEM = "A99-S1-Recovered"
SLUMPED = _pose(ankles=((-0.1, 0.05, 0.9), (0.1, 0.05, 0.9)),
                knees=((-0.1, 0.0, 0.45), (0.1, 0.0, 0.45)),
                shoulders_y=-0.25, nose=(0, -0.35, 0))          # head 0.40 m up -> A 0.25


def _cfg() -> Config:
    cfg = copy.deepcopy(TEST_CFG)
    cfg["quantity_a"] = {"trigger_from_a": True}
    return Config(cfg)


def _frames(n: int = 150, collapse: bool = True):
    """Stand 1.5 s; then the hips drop 10 px a frame with the trunk vertical
    while the head goes to floor level in 3D: one event, raised by A.
    ``collapse=False``: the subject just stands.

    Exact at the recorded precision: integer pixels, world coordinates with
    two decimals (camera pitch 0), timestamps ``index / fps``.
    """
    stand, down = _body(STANDING, pitch_deg=0.0), _body(SLUMPED, pitch_deg=0.0)
    out = []
    for i in range(n):
        k = min(max(i - 45, 0), 19) if collapse else 0
        hip = 500.0 + 10.0 * k
        pf = make_pose(i, i / FPS, hip_y=hip, shoulder_y=hip - 150.0, ankle_y=900.0)
        world = stand if (i < 50 or not collapse) else down
        pf.world_landmarks = world
        pf.world_torso_length = float(np.linalg.norm(
            (world[11] + world[12]) / 2.0 - (world[23] + world[24]) / 2.0))
        out.append(pf)
    return out


def _write_run(folder: Path, frames, cfg: Config, stem: str = STEM) -> FramePipeline:
    """Records as ``main.py --headless`` writes them, landmarks included."""
    pipe = FramePipeline(cfg, labelling=True)
    meta = {"source_fps_declared": FPS, "config": cfg.as_dict()}
    log = AuditLogger(folder, stem, metadata=meta)
    marks = AuditLogger(folder, stem + "-landmarks", fields=LANDMARK_FIELDS, metadata=meta)
    for pf in frames:
        result = pipe.analyze(pf)
        log.log_pose_frame(result.pose, extra=result.csv_extra())
        marks.log(landmark_row(result.pose))
    pipe.finalise(frames[-1].timestamp)
    log.close()
    marks.close()
    events = AuditLogger(folder, stem + "-events", fields=EVENT_FIELDS, metadata=meta)
    for i, ev in enumerate(pipe.machine.events):
        events.log(replay.event_row(i, ev))
    events.close()
    return pipe


def _rewrite(path: Path, edit) -> None:
    """Apply ``edit(header, rows)`` to a CSV record in place."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header, rows = list(reader.fieldnames), list(reader)
    header, rows = edit(header, rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kw)
    return result, out.getvalue()


class TestFindRuns(unittest.TestCase):

    def test_records_are_matched_to_their_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            names = ["A01-S1-Recovered-20260925-192230.csv",
                     "A01-S1-Recovered-events-20260925-192231.csv",      # a second later
                     "A01-S1-Recovered-landmarks-20260925-192231.csv",
                     "A01-S1-Recovered-20260925-193000.csv",             # the batch restarted
                     "A01-S1-Recovered-events-20260925-193000.csv",
                     "B02-S3-NoFall-20260925-192300.csv",                # no events at all
                     "dataset-labels-20260925-192230.csv"]
            for name in names:
                (Path(tmp) / name).touch()
            runs, extra = _records.find_runs(tmp)
        self.assertEqual(sorted(runs), ["A01-S1", "B02-S3"])
        a = runs["A01-S1"]
        self.assertEqual((a.stem, a.stamp), ("A01-S1-Recovered", "20260925-192230"))
        self.assertEqual(a.events.name, "A01-S1-Recovered-events-20260925-192231.csv")
        self.assertEqual(a.landmarks.name, "A01-S1-Recovered-landmarks-20260925-192231.csv")
        self.assertEqual(a.meta.name, "A01-S1-Recovered-20260925-192230.meta.json")
        self.assertEqual(extra, {"A01-S1": 1})
        self.assertIsNone(runs["B02-S3"].events)

    def test_clip_id(self) -> None:
        self.assertEqual(_records.clip_id("A06-S14-PartiallyRecovered"), "A06-S14")


class TestReplay(unittest.TestCase):

    def test_a_replay_reproduces_the_run_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, out = Path(tmp) / "run", Path(tmp) / "replay"
            pipe = _write_run(run, _frames(), _cfg())
            self.assertEqual([e.trigger_source for e in pipe.machine.events], [A_FALLBACK])
            cfg_file = Path(tmp) / "config.yaml"
            import yaml
            cfg_file.write_text(yaml.safe_dump(_cfg().as_dict()), encoding="utf-8")
            code, text = _quiet(replay.main, [str(run), str(out), "--repo", str(ROOT),
                                              "--config", str(cfg_file)])
            self.assertEqual(code, 0)
            self.assertIn("replayed 1 clips", text)
            same, report = _quiet(compare_runs.compare, run, out)
        self.assertTrue(same, report)
        self.assertIn("IDENTICAL", report)

    def test_the_output_folder_must_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.csv").touch()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                replay.main([tmp, tmp])

    def _records(self, tmp: Path, frames, reliable) -> _records.ClipRecords:
        folder = tmp / "run"
        meta = {"source_fps_declared": FPS}
        marks = AuditLogger(folder, STEM + "-landmarks", fields=LANDMARK_FIELDS, metadata=meta)
        log = AuditLogger(folder, STEM, metadata=meta)
        for pf, rel in zip(frames, reliable):
            marks.log(landmark_row(pf))
            log.log_pose_frame(pf, extra={"reliable": "1" if rel else "0"})
        marks.close()
        log.close()
        return _records.find_runs(folder)[0]["A99-S1"]

    def test_reliability_follows_the_record(self) -> None:
        # Core visibility logged as 0.500 is reliable by the replay's own
        # reading; the recorded run said it was not (0.4996 before rounding).
        frames = _frames(3)
        frames[1].visibility[[11, 12, 23, 24]] = 0.5
        with tempfile.TemporaryDirectory() as tmp:
            rec = self._records(Path(tmp), frames, [True, False, True])
            pkg = replay.load_package(ROOT)
            stats = replay.ReplayStats()
            rebuilt = list(replay.rebuild_frames(pkg, rec, 0.5, 0.0, stats))
        self.assertEqual(stats.visibility_nudged, 1)
        self.assertLess(rebuilt[1].core_visibility, 0.5)
        self.assertEqual(rebuilt[0].core_visibility, 1.0)

    def test_tracker_reacquisition_is_rebuilt(self) -> None:
        # Core visibility low for 0.5 s on detected frames: PoseFrontend
        # re-seeds the tracker, and the NEXT detected frame is flagged.
        frames = _frames(30)
        for pf in frames[5:25]:
            pf.visibility[[11, 12, 23, 24]] = 0.3
        with tempfile.TemporaryDirectory() as tmp:
            rec = self._records(Path(tmp), frames, [pf.visibility[11] >= 0.5 for pf in frames])
            pkg = replay.load_package(ROOT)
            stats = replay.ReplayStats()
            rebuilt = list(replay.rebuild_frames(pkg, rec, 0.5, 0.5, stats))
        flagged = [pf.frame_index for pf in rebuilt if pf.reacquired]
        self.assertEqual(flagged, [21])      # low since 5 (0.167 s), 0.5 s later: 20
        self.assertEqual(stats.reacquired, 1)

    def test_package_from_elsewhere_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                replay.load_package(tmp)
        sys.path.remove(str(Path(tmp).resolve()))

    def test_event_rows_follow_the_record_schema(self) -> None:
        ev = TriggerEvent(frame_index=3, timestamp=0.1, t_deg=2.0, v_tps=-2.0,
                          formulation="score", trigger_source=A_FALLBACK, verdict="stage3_confirmed",
                          severity="severe", max_immobility_s=1.234, reason="why")
        row = replay.event_row(0, ev)
        self.assertEqual(list(row), EVENT_FIELDS)
        self.assertEqual((row["timestamp_s"], row["T_deg"], row["V_tps"], row["max_immobility_s"],
                          row["p_outside_fraction"], row["H_ratio_EXP"]),
                         ("0.100", "2.00", "-2.000", "1.23", "", ""))


class TestCompareRuns(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ref = Path(cls.tmp.name) / "run"
        _write_run(cls.ref, _frames(), _cfg())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _copy(self, name: str) -> Path:
        dst = Path(self.tmp.name) / name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(self.ref, dst)
        return dst

    def _check(self, b: Path, cmap=None, ignore=None) -> tuple[bool, str]:
        return _quiet(compare_runs.compare, self.ref, b, cmap, ignore)

    def test_identical_copies(self) -> None:
        same, text = self._check(self._copy("same"))
        self.assertTrue(same, text)

    def test_a_changed_frame_value_is_found(self) -> None:
        b = self._copy("frame")
        rec = _records.find_runs(b)[0]["A99-S1"]

        def edit(header, rows):
            rows[60]["T_deg"] = "12.345"
            return header, rows
        _rewrite(rec.frames, edit)
        same, text = self._check(b)
        self.assertFalse(same)
        self.assertIn("frame_index 60 T_deg", text)
        self.assertIn("frames differ 1", text)
        same, text = self._check(b, ignore={"T_deg"})
        self.assertTrue(same, text)

    def test_a_changed_verdict_changes_the_label(self) -> None:
        b = self._copy("verdict")
        rec = _records.find_runs(b)[0]["A99-S1"]

        def edit(header, rows):
            rows[0]["verdict"], rows[0]["severity"] = "stage3_confirmed", "severe"
            return header, rows
        _rewrite(rec.events, edit)
        same, text = self._check(b)
        self.assertFalse(same)
        self.assertIn("-> NotRecovered", text)
        self.assertIn("labels differ 1", text)

    def test_a_missing_clip_is_a_difference(self) -> None:
        b = self._copy("missing")
        for f in b.iterdir():
            f.unlink()
        same, text = self._check(b)
        self.assertFalse(same)
        self.assertIn("only in A 1", text)

    def test_renames_and_translations_through_the_map(self) -> None:
        b = self._copy("renamed")
        rec = _records.find_runs(b)[0]["A99-S1"]

        def edit(header, rows):
            for row in rows:
                row["reason"] = "EN: " + row.pop("motivo")
            return [("reason" if h == "motivo" else h) for h in header], rows
        _rewrite(rec.events, edit)
        same, text = self._check(b)
        self.assertFalse(same)
        self.assertIn("columns only in A: motivo", text)
        self.assertIn("columns only in B: reason", text)
        patterns = compare_runs.ColumnMap(columns={"motivo": "reason"},
                                          patterns={"reason": [(compare_runs.re.compile("^"),
                                                                "EN: ")]})
        same, text = self._check(b, patterns)
        self.assertTrue(same, text)
        with open(rec.events, newline="", encoding="utf-8") as fh:
            reason = next(csv.DictReader(fh))["reason"]
        values = compare_runs.ColumnMap(columns={"motivo": "reason"},
                                        values={"reason": {reason[4:]: reason}})
        same, text = self._check(b, values)
        self.assertTrue(same, text)

    def test_the_map_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(json.dumps({"columns": {"a": "b"}, "values": {"b": {"x": "y"}},
                                        "patterns": {"b": [["^z(.)$", "w\\1"]]}}))
            cmap = compare_runs.ColumnMap.load(path)
        header, rows = cmap.apply(["a"], [{"a": "x"}, {"a": "z1"}])
        self.assertEqual((header, rows), (["b"], [{"b": "y"}, {"b": "w1"}]))

    def test_landmarks_only_with_the_option(self) -> None:
        b = self._copy("marks")
        rec = _records.find_runs(b)[0]["A99-S1"]

        def edit(header, rows):
            rows[10]["lm00_x"] = "1.23"
            return header, rows
        _rewrite(rec.landmarks, edit)
        same, text = self._check(b)
        self.assertTrue(same, text)
        same, text = _quiet(compare_runs.compare, self.ref, b, landmarks=True)
        self.assertFalse(same)
        self.assertIn("frame_index 10 lm00_x", text)
        self.assertIn("landmarks differ 1", text)

    def test_a_landmark_record_missing_on_one_side(self) -> None:
        b = self._copy("nomarks")
        _records.find_runs(b)[0]["A99-S1"].landmarks.unlink()
        same, text = _quiet(compare_runs.compare, self.ref, b, landmarks=True)
        self.assertFalse(same)
        self.assertIn("no landmark record in B", text)
        # Missing on both sides (two replays) is nothing to compare.
        same, text = _quiet(compare_runs.compare, b, b, landmarks=True)
        self.assertTrue(same, text)

    def test_a_configuration_change_is_a_warning(self) -> None:
        b = self._copy("config")
        meta_path = _records.find_runs(b)[0]["A99-S1"].meta
        meta = json.loads(meta_path.read_text())
        before = meta["config"]["stage1"]["trigger_score"]
        meta["config"]["stage1"]["trigger_score"] = 9.9
        meta["config"].setdefault("logging", {})["output_dir"] = "logs/elsewhere"
        meta_path.write_text(json.dumps(meta))
        same, text = self._check(b)
        self.assertTrue(same, text)                          # records identical: verdict holds
        self.assertIn(f"stage1.trigger_score: {before} -> 9.9 (1 clips)", text)
        self.assertNotIn("output_dir", text.split("configuration differs")[1])
        same, text = self._check(self._copy("config-same"))
        self.assertIn("configuration: the same in every clip", text)

    def test_exit_status(self) -> None:
        code, _ = _quiet(compare_runs.main, [str(self.ref), str(self._copy("exit"))])
        self.assertEqual(code, 0)


class TestMetrics(unittest.TestCase):

    def test_headline_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            _write_run(run, _frames(), _cfg())
            _write_run(run, _frames(60, collapse=False), _cfg(), stem="B98-S2-NoFall")
            split = Path(tmp) / "split.yaml"
            split.write_text("test:\n  - A99\ntrain:\n  - B98\n", encoding="utf-8")
            code, text = _quiet(metrics.main, [str(run), str(run), "--partition", str(split)])
        self.assertEqual(code, 0)
        self.assertIn("2 clips", text)
        self.assertIn("events 1 | by condition: A 1", text)
        self.assertIn("specificity (label) 1/1", text)
        self.assertIn("label agreement: 2/2", text)
        self.assertIn("| train 1/1 |", text)


class TestProvenance(unittest.TestCase):

    def setUp(self) -> None:
        audit_log.git_provenance.cache_clear()

    def tearDown(self) -> None:
        audit_log.git_provenance.cache_clear()

    def test_every_sidecar_says_which_code_wrote_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip")
            logger.log({"frame_index": 0})
            logger.close()
            meta = json.loads(logger.path.with_suffix(".meta.json").read_text())
        self.assertEqual(meta["code_version"], __version__)
        self.assertIn("git_commit", meta)
        self.assertIn("git_dirty", meta)

    def test_without_git_it_says_unknown(self) -> None:
        with mock.patch.object(audit_log.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(audit_log.git_provenance(),
                             {"git_commit": None, "git_dirty": None})

    @unittest.skipIf(shutil.which("git") is None, "git not installed")
    def test_commit_and_dirty_state_of_a_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()

            def git(*args):
                return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                                       *args], cwd=repo, capture_output=True, text=True,
                                      check=True).stdout.strip()
            git("init", "-q")
            (repo / "f.txt").write_text("a")
            git("add", "f.txt")
            git("commit", "-q", "-m", "x")
            head = git("rev-parse", "HEAD")
            with mock.patch.object(audit_log, "REPO_ROOT", repo):
                self.assertEqual(audit_log.git_provenance(),
                                 {"git_commit": head, "git_dirty": False})
                audit_log.git_provenance.cache_clear()
                (repo / "f.txt").write_text("b")
                self.assertEqual(audit_log.git_provenance()["git_dirty"], True)
                audit_log.git_provenance.cache_clear()
                # A folder inside someone else's checkout is not a checkout of its own.
                (repo / "sub").mkdir()
                with mock.patch.object(audit_log, "REPO_ROOT", repo / "sub"):
                    self.assertIsNone(audit_log.git_provenance()["git_commit"])

    @unittest.skipIf(shutil.which("git") is None, "git not installed")
    def test_new_code_not_yet_added_is_dirty(self) -> None:
        # B4 splits state_machine into new modules: a run made before `git add`
        # must not claim to be the commit. New notes or a .DS_Store must not
        # make it dirty either.
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()

            def git(*args):
                subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                               cwd=repo, capture_output=True, text=True, check=True)
            git("init", "-q")
            (repo / "pef_fall_detector").mkdir()
            (repo / "pef_fall_detector" / "a.py").write_text("x = 1\n")
            git("add", ".")
            git("commit", "-q", "-m", "x")

            def dirty():
                audit_log.git_provenance.cache_clear()
                return audit_log.git_provenance()["git_dirty"]
            with mock.patch.object(audit_log, "REPO_ROOT", repo):
                (repo / "notas.md").write_text("n")
                (repo / "pef_fall_detector" / ".DS_Store").write_text("")
                self.assertFalse(dirty())
                (repo / "pef_fall_detector" / "trigger.py").write_text("y = 2\n")
                self.assertTrue(dirty())

    def test_porcelain_parsing(self) -> None:
        differs = audit_log._code_differs
        self.assertFalse(differs(""))
        self.assertTrue(differs(" M notas/x.md\0"))          # any tracked change counts
        self.assertFalse(differs("?? tools/replay.py\0?? notas/n.md\0"))
        self.assertTrue(differs("?? pef_fall_detector/stage3.py\0"))
        self.assertTrue(differs("?? config.yaml\0"))
        self.assertFalse(differs("?? pef_fall_detector/.DS_Store\0"))
        self.assertTrue(differs("R  new.md\0old.md\0"))          # a rename is a change


if __name__ == "__main__":
    unittest.main()
