"""PEF-Fall-Detector — command-line entry point.

Two ways to run the same core pipeline:

GUI (PEF-Lab visualization workbench)::

    python main.py                          # open the lab, pick a source there
    python main.py --video path/to/clip.mp4 # open the lab preloading a clip
    python main.py --camera 0               # open the lab on a live camera

Headless (no GUI — batch processing, CI, Raspberry Pi)::

    python main.py --video clip.mp4 --headless
    python main.py --video clip.mp4 --headless --expected-seconds 20
    python main.py --folder ../recordings --mode research --output-dir logs/run --quiet

Headless mode processes every frame and prints a summary including the
Step-0 sanity checks (torso-length stability) and the FPS audit (declared vs.
effective rate, see implementation-plan §5.6 — pass --expected-seconds with
the 'Video seconds' value used when the clip was recorded in PEF-Video-Tool).

What a run records depends on ``logging.mode`` (or ``--mode``): research
writes the per-frame record, the events and the 33 landmarks of every frame;
deployment writes nothing (§3.6). ``--folder`` runs every clip of a folder,
as PEF-Lab's queue does, and closes with the same labels file and summary.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pef_fall_detector.alerts import console_sink
from pef_fall_detector.audit_log import AuditLogger
from pef_fall_detector.config import (
    LOGGING_MODES,
    PARTITION_PATH,
    Config,
    load_config,
    research_mode,
)
from pef_fall_detector.dataset import LABEL_FIELDS, label_row, video_files
from pef_fall_detector.evaluation import pass_summary
from pef_fall_detector.pipeline import FramePipeline
from pef_fall_detector.quantities import trunk_band
from pef_fall_detector.records import RunRecords
from pef_fall_detector.sources import VideoFileSource


def run_config(args: argparse.Namespace) -> Config:
    """``--config`` with ``--mode`` and ``--output-dir`` applied on top.

    The overrides go into the configuration itself, so every record's
    ``.meta.json`` states the mode and the folder the run actually used.
    ``--output-dir`` is taken from the current directory, as any command-line
    path; ``logging.output_dir`` in the file is taken from the repository root.
    """
    cfg = load_config(args.config)
    # Absent options are None: tests build the namespace by hand.
    mode = getattr(args, "mode", None)
    output_dir = getattr(args, "output_dir", None)
    if mode is None and output_dir is None:
        return cfg
    data = copy.deepcopy(cfg.as_dict())
    if mode is not None:
        data["logging"]["mode"] = mode
    if output_dir is not None:
        data["logging"]["output_dir"] = str(Path(output_dir).resolve())
    return Config(data, source="the configuration with the command-line options")


@dataclass
class ClipRun:
    """What one headless run of one clip produced."""

    video: str
    events: list
    records: RunRecords
    frames: int = 0
    detected_frames: int = 0
    reliable_frames: int = 0
    duration_s: float = 0.0
    alerts: int = 0
    fps_report: dict = field(default_factory=dict)
    torso_lengths: list = field(default_factory=list)
    world_torsos: list = field(default_factory=list)
    trunk_angles: list = field(default_factory=list)
    velocities: list = field(default_factory=list)
    state_counts: dict = field(default_factory=dict)


def run_clip(cfg: Config, video: str, alerts_to_console: bool = True,
             expected_seconds: float | None = None) -> ClipRun:
    """Process one video file sequentially; returns what the run produced."""
    source = VideoFileSource(video)
    # A recorded clip is labelled from how the episode ENDED (§3.3), the
    # same choice gui/lab_window.py makes for any non-live source. Without
    # it this run resolved greedily, as the live §3.6 device must, and an
    # event still open when the clip ran out — subject on the floor — was
    # never judged. Measured on A05-S1, same clip and same config:
    # stage2_inconclusive here, stage3_confirmed/severe in PEF-Lab. Every
    # NotRecovered clip depends on this, so a headless run that skips it
    # does not reproduce the paper's labels at all.
    pipeline = FramePipeline(cfg, labelling=not source.is_live)
    # The §3.5 alert, on the only transport that exists before Phase 7.
    pipeline.source_name = Path(video).name
    if alerts_to_console:
        pipeline.alerts.add_sink(console_sink)
    # The records travel with the facts needed to interpret them later: which
    # source, at what frame rate, and the exact thresholds in force. This is
    # what lets a calibration cite the configuration that produced it.
    records = RunRecords(
        cfg.logging.output_dir, Path(video).stem,
        metadata={
            "source": str(Path(video).resolve()),
            "source_kind": "video",
            "source_fps_declared": source.fps,
            "frame_count": source.frame_count,
            "config": cfg.as_dict(),
        },
        research=research_mode(cfg))
    run = ClipRun(video=str(video), events=pipeline.machine.events, records=records,
                  duration_s=(source.frame_count / source.fps if source.fps else 0.0))

    # Statistics are collected from RELIABLE frames only (person detected AND
    # core landmarks visible). Unreliable frames still go to the record — data
    # is never discarded — but their geometry is MediaPipe extrapolating
    # anatomy outside the image, and summarizing it would report noise as
    # signal (see pipeline.FramePipeline docstring).
    last_timestamp: float | None = None
    try:
        while True:
            ok, frame, timestamp = source.read()
            if not ok:
                break
            last_timestamp = timestamp
            result = pipeline.process(frame, run.frames, timestamp)
            records.frame(result)
            if result.resolved_event is not None:
                records.event_settled(pipeline.machine.events, result.resolved_event)

            if result.pose.detected:
                run.detected_frames += 1
            if result.reliable:
                run.reliable_frames += 1
                run.torso_lengths.append(result.pose.torso_length)
                t_deg = result.quantities.get("T_deg", float("nan"))
                if not math.isnan(t_deg):
                    run.trunk_angles.append(t_deg)
                v = result.quantities.get("V_tps", float("nan"))
                if not math.isnan(v):
                    run.velocities.append(v)
                w = result.quantities.get("world_torso_len_m")
                if w is not None and not math.isnan(w):
                    run.world_torsos.append(w)
                if result.state is not None:
                    run.state_counts[result.state] = run.state_counts.get(result.state, 0) + 1
            run.frames += 1

        # Close any event still being judged BEFORE the records are closed, as
        # gui/lab_window.py does at end of file: a clip that ends with the
        # subject on the floor has an open event, and that event is precisely
        # the answer the label needs.
        if last_timestamp is not None:
            pipeline.finalise(last_timestamp)
        run.fps_report = source.fps_report(expected_seconds)
    finally:
        # Released and closed even when a frame raises: an open capture
        # holds the file, and a record left open loses its buffered rows.
        pipeline.close()
        source.release()
        # Events that no verdict reached (a detection loss abandoned them) are
        # written here, in their place.
        records.close(pipeline.machine.events)
    run.alerts = len(pipeline.alerts.dispatched)
    return run


def event_lines(ev) -> list[str]:
    """How one event reads on the console."""
    lines = [f"  t={ev.timestamp:6.2f}s frame {ev.frame_index:5d}  "
             f"T={ev.t_deg:6.1f} deg  V={ev.v_tps:+6.2f} torso/s  "
             f"-> {ev.verdict}"
             + (f" [{ev.severity}]" if ev.severity else "")
             + ("" if math.isnan(ev.p_outside_fraction)
                else f" (COM outside {100*ev.p_outside_fraction:.0f}% "
                     f"of {ev.p_samples} frames)")]
    if ev.reason:
        lines.append(f"            reason: {ev.reason}")
    return lines


def print_summary(cfg: Config, run: ClipRun) -> None:
    """The full summary of one clip. All statistics cover reliable frames only."""
    print(f"Processed : {run.frames} frames from {run.video}")
    print(f"Detected  : {run.detected_frames} frames with a person "
          f"({100.0 * run.detected_frames / max(1, run.frames):.1f}%)")
    print(f"Reliable  : {run.reliable_frames} of {run.detected_frames} detected frames "
          f"(core visibility >= {cfg.pose.visibility_threshold}) — "
          f"stats below use only these")
    if run.torso_lengths:
        mean = statistics.fmean(run.torso_lengths)
        stdev = statistics.pstdev(run.torso_lengths)
        print(f"Torso px  : mean {mean:.1f}px, stdev {stdev:.1f}px "
              f"(CV {100.0 * stdev / mean:.1f}%) — grows/shrinks with distance; "
              f"expect smooth changes, not jumps")
    if run.world_torsos:
        w_mean = statistics.fmean(run.world_torsos)
        w_stdev = statistics.pstdev(run.world_torsos)
        print(f"Torso m   : mean {w_mean:.3f}m, CV {100.0 * w_stdev / w_mean:.1f}% "
              f"— metric estimate; should stay ~constant at any distance")
    if run.trunk_angles:
        bands: dict[str, int] = {}
        for angle in run.trunk_angles:
            band = trunk_band(angle)
            bands[band] = bands.get(band, 0) + 1
        spread = " ".join(f"{name}={count}" for name, count in sorted(bands.items()))
        print(f"Quantity T: min {min(run.trunk_angles):.1f} deg, "
              f"median {statistics.median(run.trunk_angles):.1f} deg, "
              f"max {max(run.trunk_angles):.1f} deg")
        print(f"T bands   : {spread}  (§3.4 reference table)")
        if run.velocities:
            print(f"Quantity V: min {min(run.velocities):+.2f}, "
                  f"median {statistics.median(run.velocities):+.2f}, "
                  f"max {max(run.velocities):+.2f} torso/s "
                  f"(negative = downward)")
        if run.state_counts:
            states = " ".join(
                f"{name}={count}"
                for name, count in sorted(run.state_counts.items(), key=lambda kv: -kv[1])
            )
            print(f"States    : {states}  (display labels, not decisions)")
    elif run.reliable_frames == 0 and run.detected_frames > 0:
        print("Quantity T: no reliable frames — subject's core landmarks were "
              "not visible (e.g. face-only footage); record full-body clips")
    report = run.fps_report
    print(f"FPS audit : declared {report['declared_fps']:.2f}, "
          f"implied duration {report['implied_duration_s']:.2f}s", end="")
    if "effective_fps" in report:
        ratio = report['fps_mismatch_ratio']
        verdict = 'OK' if 0.9 <= ratio <= 1.1 else 'MISMATCH: use corrected dt'
        print(f", effective {report['effective_fps']:.2f} (ratio {ratio:.2f} — {verdict})")
    else:
        print()
    print(f"Stage 1   : {len(run.events)} event(s) "
          f"[{cfg.stage1.trigger_formulation}, hold {cfg.stage1.trigger_hold_s}s] "
          f"— judged by Stages 2 (geometry) and 3 (immobility/recovery)")
    for ev in run.events[:10]:
        for line in event_lines(ev):
            print(line)
    print(f"Alerts    : {run.alerts} dispatched "
          f"(verdicts that alert: {', '.join(cfg.alerts.dispatch_verdicts)})")
    records = run.records
    if records.path is None:
        print("Records   : none — deployment mode writes nothing to disk (§3.6); "
              "use --mode research to record")
        return
    print(f"Audit CSV : {records.path}")
    if records.events_written:
        print(f"Events    : {records.events_path}")
    print(f"Landmarks : {records.landmarks_path}")
    print(f"Run meta  : {records.path.with_suffix('.meta.json').name} "
          f"(source, fps and the full configuration that produced this record)")


def quiet_line(run: ClipRun, row: dict) -> str:
    """One line per clip: what was detected, against the truth in the name."""
    truth = row["true_class"]
    mark = "" if not truth else ("OK" if row["correct"] == 1 else "X")
    verdicts = ", ".join(f"{e.verdict}/{e.severity or '-'}" for e in run.events)
    return (f"{row['detected_class']:<20} truth {truth or '?':<20} {mark:<2} | "
            f"{verdicts or 'no events'}")


def run_headless(args: argparse.Namespace) -> int:
    """One video file: process it and print its summary (or one line, --quiet)."""
    cfg = run_config(args)
    quiet = bool(getattr(args, "quiet", False))
    run = run_clip(cfg, args.video, alerts_to_console=not quiet,
                   expected_seconds=args.expected_seconds)
    if quiet:
        row = label_row(args.video, run.events, run.frames, run.reliable_frames,
                        run.duration_s, datetime.now())
        print(f"{Path(args.video).stem}  {quiet_line(run, row)}")
    else:
        print_summary(cfg, run)
    return 0


def _run_clips(cfg: Config, clips: list, labels: AuditLogger | None, quiet: bool) -> list[dict]:
    """Run every clip of a folder; one labels row per clip, printed as it goes."""
    rows = []
    for n, video in enumerate(clips, 1):
        run = run_clip(cfg, str(video), alerts_to_console=not quiet)
        row = label_row(video, run.events, run.frames, run.reliable_frames,
                        run.duration_s, datetime.now())
        rows.append(row)
        if labels is not None:
            labels.log(row)
        head = f"[{n}/{len(clips)}] {video.stem}"
        if quiet:
            print(f"{head}  {quiet_line(run, row)}")
        else:
            print(f"\n{head}")
            for ev in run.events:
                for line in event_lines(ev):
                    print(line)
            if not run.events:
                print("  (no events)")
            print(f"  -> {quiet_line(run, row)}")
    return rows


def run_folder(args: argparse.Namespace) -> int:
    """Every clip of a folder, as PEF-Lab's queue runs them.

    One labels row per clip (``dataset.label_row``, the same row PEF-Lab
    writes) and, at the end, the same summary: the confusion matrix and the
    train/test split. In research mode the rows also go to
    ``dataset-labels-<stamp>.csv`` next to the records.
    """
    cfg = run_config(args)
    quiet = bool(getattr(args, "quiet", False))
    clips = video_files(args.folder)
    if not clips:
        print(f"No video files in {args.folder}")
        return 1
    labels = None
    if research_mode(cfg):
        labels = AuditLogger(
            cfg.logging.output_dir, "dataset-labels", fields=LABEL_FIELDS,
            metadata={"record_kind": "clip_labels", "clips": len(clips),
                      "config": cfg.as_dict()},
            flush_each_row=True)
    try:
        rows = _run_clips(cfg, clips, labels, quiet)
    finally:
        if labels is not None:
            labels.close()
    print("\n" + pass_summary(rows, PARTITION_PATH))
    if labels is not None and labels.rows_written:
        print(f"\n-> {labels.path}")
    elif labels is None:
        print("\nRecords: none — deployment mode writes nothing to disk (§3.6)")
    return 0


def main(argv: list[str] | None = None, *, headless_only: bool = False) -> int:
    """Parse the command line and dispatch to the GUI or the headless run.

    Returns the process exit code: 0 on success, which matters for the batch
    pipelines of §3.7 that chain runs together.

    ``headless_only``: the ``pef-run`` command (see :func:`run`). A video is
    then processed headless without asking for ``--headless``, and a command
    line that would open PEF-Lab is refused instead: a box without a screen
    (the Raspberry Pi) has no PySide6 to open it with.
    """
    parser = argparse.ArgumentParser(
        prog="pef-run" if headless_only else None,
        description="PEF-Fall-Detector" + (" (headless)" if headless_only else ""))
    # A run has exactly one source. Declaring them mutually exclusive makes
    # argparse reject the combination with a clear message and document it in
    # --help, instead of silently preferring one and leaving the operator
    # wondering why the camera never opened.
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="path to a recorded clip (.mp4)")
    source.add_argument("--camera", type=int, default=None, help="live camera index")
    source.add_argument("--folder", help="run every clip of a folder headless, "
                                         "as PEF-Lab's queue does")
    parser.add_argument("--headless", action="store_true",
                        help="process without GUI (requires --video)")
    parser.add_argument("--config", default=None, help="alternative config.yaml")
    parser.add_argument("--mode", choices=LOGGING_MODES, default=None,
                        help="research records everything, deployment nothing "
                             "(default: logging.mode in the config)")
    parser.add_argument("--output-dir", default=None,
                        help="folder for the records (default: logging.output_dir)")
    parser.add_argument("--quiet", action="store_true",
                        help="headless: one line per clip instead of the full summary")
    parser.add_argument("--expected-seconds", type=float, default=None,
                        help="known wall-clock duration of the clip, for the FPS audit")
    args = parser.parse_args(argv)
    if headless_only:
        if not (args.video or args.folder):
            parser.error("needs --video FILE or --folder DIR (the camera opens in PEF-Lab)")
        args.headless = True
    # The package reports what goes wrong around a run (a failing alert
    # transport) through ``logging``; the summaries and the alerts themselves
    # are this program's output and stay on stdout.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.folder:
        return run_folder(args)
    if args.headless:
        if not args.video:
            parser.error("--headless requires --video")
        return run_headless(args)

    # GUI mode (imported lazily so headless boxes never need PySide6).
    from pef_fall_detector.gui.lab_window import run_gui

    cfg = run_config(args)
    run_gui(cfg, video=args.video, camera=args.camera)
    return 0


def run() -> int:
    """The ``pef-run`` command: ``main`` without the GUI."""
    return main(headless_only=True)


if __name__ == "__main__":
    sys.exit(main())
