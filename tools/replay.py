"""Replay recorded runs through the detector code, without MediaPipe.

Every frame is rebuilt from the run's 33-landmark record exactly as
``pose_frontend.PoseFrontend.process`` builds it: Step-0 anchors, core
visibility, world landmarks, and the tracker re-acquisition flag of phase 4.
The frames then go through ``FramePipeline.analyze`` and ``finalise`` as
``main.py --headless`` runs them, and the records are written with the
package's own writers. A replay folder therefore has the same files and
columns as a real run, and every tool that reads runs reads replays.

What it is for: checking that a code change leaves every decision
unchanged. Replay the reference landmarks with the old code and with the new
code, then compare the two folders with ``compare_runs.py``. Both replays read
the same landmarks, so identical behaviour gives identical records. A 128-clip
replay takes seconds; a MediaPipe run takes about twenty minutes.

What it is not: a copy of the original run. The landmark record is rounded
(0.01 px, 0.001 visibility, 0.1 mm), so a replay reproduces the run it came
from only to that precision. Measured on the phase-8.4 cloud run: all 128
labels and every event's frame identical, one verdict of an abandoned event
and a few last decimals different. Compare replays with replays.

Two facts are taken from the recorded run rather than recomputed:

* the timestamps, ``frame_index / fps`` with the fps the run declared, which
  is how ``sources.VideoFileSource`` makes them;
* whether each frame was reliable. Visibility is logged to 3 decimals, so a
  core visibility within rounding of the threshold could flip the decision;
  the replay nudges it to agree with the recorded ``reliable`` flag, and
  counts how often it had to.

Usage::

    python tools/replay.py LOGS OUT [--repo PATH] [--config PATH] [--clips A01-S1,B12-S2]

``--repo`` chooses whose code replays the frames (default: this checkout), so
the same tool replays a reference commit and the working tree.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _records import ClipRecords, find_runs  # noqa: E402

TOOLS_REPO = Path(__file__).resolve().parents[1]


def load_package(repo: str | Path) -> SimpleNamespace:
    """Import the detector from ``repo``: the code that replays the frames.

    Refuses to go on if the package was already imported from somewhere else,
    because then the replay would silently run different code than asked.
    """
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    mods = {name: importlib.import_module(f"pef_fall_detector.{name}")
            for name in ("audit_log", "config", "pipeline", "pose_frontend")}
    loaded = Path(mods["pipeline"].__file__).resolve().parents[1]
    if loaded != repo:
        raise RuntimeError(f"pef_fall_detector is already imported from {loaded}, not {repo}")
    return SimpleNamespace(**mods)


@dataclass
class ReplayStats:
    """What the rebuild had to do, per clip."""

    frames: int = 0
    reacquired: int = 0
    visibility_nudged: int = 0


def rebuild_frames(pkg: SimpleNamespace, records: ClipRecords, visibility_threshold: float,
                   reacquire_after_s: float, stats: ReplayStats):
    """Yield the clip's PoseFrames as ``PoseFrontend.process`` produced them."""
    pf_mod = pkg.pose_frontend
    meta = json.loads(records.meta.read_text(encoding="utf-8"))
    fps = float(meta["source_fps_declared"])
    with open(records.frames, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(records.landmarks, newline="", encoding="utf-8") as fh:
        marks = list(csv.DictReader(fh))
    if len(rows) != len(marks):
        raise ValueError(f"{records.stem}: {len(rows)} frames but {len(marks)} landmark rows")
    core_idx = list(pf_mod.CORE_LANDMARKS)
    # PoseFrontend._watch_tracker's state: when core visibility went low, and
    # whether the tracker was re-seeded (flagged on the next detected frame).
    low_since: float | None = None
    reseeded = False
    for row, mark in zip(rows, marks):
        index = int(mark["frame_index"])
        if index != int(row["frame_index"]):
            raise ValueError(f"{records.stem}: frame {index} out of step with the record")
        timestamp = index / fps
        size = (int(row["frame_width"]), int(row["frame_height"]))
        stats.frames += 1
        if mark["detected"] != "1" or not mark.get("lm00_x"):
            low_since = None
            yield pf_mod.PoseFrame(frame_index=index, timestamp=timestamp, detected=False,
                                   frame_size=size)
            continue
        landmarks = np.array([[float(mark[f"lm{i:02d}_{k}"]) for k in "xyz"]
                              for i in range(33)])
        visibility = np.array([float(mark[f"lm{i:02d}_v"]) for i in range(33)])
        mid_hip, mid_shoulder, torso = pf_mod.compute_step0(landmarks)
        core = float(visibility[core_idx].min())
        reliable = row.get("reliable") == "1"
        if (core >= visibility_threshold) != reliable:
            core = visibility_threshold + (1e-6 if reliable else -1e-6)
            stats.visibility_nudged += 1
        world = np.empty((0, 3))
        world_torso = None
        if mark.get("w00_x"):
            world = np.array([[float(mark[f"w{i:02d}_{k}"]) for k in "xyz"]
                              for i in range(33)])
            w_hip = (world[pf_mod.LEFT_HIP] + world[pf_mod.RIGHT_HIP]) / 2.0
            w_shoulder = (world[pf_mod.LEFT_SHOULDER] + world[pf_mod.RIGHT_SHOULDER]) / 2.0
            world_torso = float(np.linalg.norm(w_shoulder - w_hip))
        reacquired, reseeded = reseeded, False
        if reacquire_after_s <= 0.0 or core >= visibility_threshold:
            low_since = None
        elif low_since is None:
            low_since = timestamp
        elif timestamp - low_since >= reacquire_after_s - 1e-9:
            reseeded = True
            low_since = None
        stats.reacquired += int(reacquired)
        yield pf_mod.PoseFrame(
            frame_index=index, timestamp=timestamp, detected=True, frame_size=size,
            landmarks=landmarks, visibility=visibility, mid_hip=mid_hip,
            mid_shoulder=mid_shoulder, torso_length=torso, core_visibility=core,
            reacquired=reacquired, world_landmarks=world, world_torso_length=world_torso)


def event_row(index: int, ev) -> dict:
    """One row of the events record, formatted as ``main.py`` writes it."""
    def fmt(value: float, spec: str) -> str:
        return "" if math.isnan(value) else format(value, spec)
    return {
        "event_index": index,
        "frame_index": ev.frame_index,
        "timestamp_s": f"{ev.timestamp:.3f}",
        "T_deg": f"{ev.t_deg:.2f}",
        "V_tps": f"{ev.v_tps:.3f}",
        "formulation": ev.formulation,
        "trigger_source": getattr(ev, "trigger_source", ""),
        "verdict": ev.verdict,
        "severity": ev.severity,
        "max_immobility_s": fmt(ev.max_immobility_s, ".2f"),
        "p_outside_fraction": fmt(ev.p_outside_fraction, ".3f"),
        "p_samples": ev.p_samples,
        "H_ratio_EXP": fmt(ev.h_ratio, ".3f"),
        "motivo": ev.reason,
    }


def replay_clip(pkg: SimpleNamespace, cfg, records: ClipRecords, out: Path) -> tuple:
    """Replay one clip into ``out``; returns (events, stats)."""
    stats = ReplayStats()
    pipeline = pkg.pipeline.FramePipeline(cfg, labelling=True)
    recorded = json.loads(records.meta.read_text(encoding="utf-8"))
    metadata = {"source": records.stem, "source_kind": "replay",
                "replay_of": str(records.frames.resolve()),
                "source_fps_declared": recorded["source_fps_declared"],
                "config": cfg.as_dict()}
    frames_log = pkg.audit_log.AuditLogger(out, records.stem, metadata=metadata)
    frames = rebuild_frames(pkg, records, float(cfg.pose.visibility_threshold),
                            float(cfg.pose.as_dict().get("reacquire_after_s", 0.0)), stats)
    last = None
    for pf in frames:
        result = pipeline.analyze(pf)
        frames_log.log_pose_frame(result.pose, extra=result.csv_extra())
        last = pf.timestamp
    if last is not None:
        pipeline.finalise(last)
    frames_log.close()
    fields = pkg.audit_log.EVENT_FIELDS
    events_log = pkg.audit_log.AuditLogger(
        out, records.stem + "-events", fields=fields,
        metadata={**metadata, "record_kind": "stage1_events"})
    for i, ev in enumerate(pipeline.machine.events):
        events_log.log({k: v for k, v in event_row(i, ev).items() if k in fields})
    events_log.close()
    return pipeline.machine.events, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", help="run folder with per-frame, landmark and meta records")
    ap.add_argument("out", help="folder for the replayed records (must be empty or new)")
    ap.add_argument("--repo", default=str(TOOLS_REPO),
                    help="checkout whose code replays the frames (default: this one)")
    ap.add_argument("--config", default=None, help="config file (default: REPO/config.yaml)")
    ap.add_argument("--clips", default=None, help="comma-separated clip ids, e.g. A01-S1,B12-S2")
    ap.add_argument("--quiet", action="store_true", help="print only the totals")
    args = ap.parse_args(argv)

    pkg = load_package(args.repo)
    cfg = pkg.config.load_config(args.config or str(Path(args.repo) / "config.yaml"))
    # Absolute before it reaches AuditLogger, which reads a relative folder
    # against the repository root, not against where the tool was launched.
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} is not empty: replays are compared file by file, so they never mix")
    runs, extra = find_runs(args.logs)
    if args.clips:
        wanted = set(args.clips.split(","))
        runs = {c: r for c, r in runs.items() if c in wanted}
    missing = sorted(c for c, r in runs.items() if r.landmarks is None)
    if missing:
        print(f"skipped, no landmark record: {', '.join(missing)}")
    start = time.perf_counter()
    totals = ReplayStats()
    n = 0
    for clip, records in sorted(runs.items()):
        if records.landmarks is None:
            continue
        events, stats = replay_clip(pkg, cfg, records, out)
        n += 1
        totals.frames += stats.frames
        totals.reacquired += stats.reacquired
        totals.visibility_nudged += stats.visibility_nudged
        if not args.quiet:
            evs = ", ".join(f"{e.verdict}/{e.severity or '-'}" for e in events) or "no events"
            print(f"{clip:8s} {stats.frames:4d} frames | {evs}")
    if extra:
        print(f"extra runs ignored (earliest kept): {sum(extra.values())} "
              f"({', '.join(sorted(extra))})")
    print(f"replayed {n} clips, {totals.frames} frames in {time.perf_counter() - start:.1f} s "
          f"| re-acquisitions {totals.reacquired} | visibility nudged {totals.visibility_nudged} "
          f"| code {args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
