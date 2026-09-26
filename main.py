"""PEF-Fall-Detector — command-line entry point.

Two ways to run the same core pipeline:

GUI (PEF-Lab visualization workbench)::

    python main.py                          # open the lab, pick a source there
    python main.py --video path/to/clip.mp4 # open the lab preloading a clip
    python main.py --camera 0               # open the lab on a live camera

Headless (no GUI — batch processing, CI, Raspberry Pi)::

    python main.py --video clip.mp4 --headless
    python main.py --video clip.mp4 --headless --expected-seconds 20

Headless mode processes every frame, writes the per-frame audit CSV, and
prints a summary including the Step-0 sanity checks (torso-length stability)
and the FPS audit (declared vs. effective rate, see implementation-plan
§5.6 — pass --expected-seconds with the 'Video seconds' value used when the
clip was recorded in PEF-Video-Tool).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

from pef_fall_detector.alerts import console_sink
from pef_fall_detector.audit_log import (
    EVENT_FIELDS,
    LANDMARK_FIELDS,
    AuditLogger,
    landmark_row,
)
from pef_fall_detector.config import load_config
from pef_fall_detector.pipeline import FramePipeline
from pef_fall_detector.quantities import trunk_band
from pef_fall_detector.sources import VideoFileSource


def run_headless(args: argparse.Namespace) -> int:
    """Process a video file sequentially and print a verification summary."""
    cfg = load_config(args.config)
    source = VideoFileSource(args.video)
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
    pipeline.source_name = Path(args.video).name
    pipeline.alerts.add_sink(console_sink)
    # The record travels with the facts needed to interpret it later: which
    # source, at what frame rate, and the exact thresholds in force. This is
    # what lets a calibration cite the configuration that produced it.
    logger = AuditLogger(
        cfg.logging.output_dir,
        Path(args.video).stem,
        metadata={
            "source": str(Path(args.video).resolve()),
            "source_kind": "video",
            "source_fps_declared": source.fps,
            "frame_count": source.frame_count,
            "config": cfg.as_dict(),
        },
    )

    # Statistics are collected from RELIABLE frames only (person detected AND
    # core landmarks visible). Unreliable frames still go to the CSV — data
    # is never discarded — but their geometry is MediaPipe extrapolating
    # anatomy outside the image, and summarizing it would report noise as
    # signal (see pipeline.FramePipeline docstring).
    torso_lengths: list[float] = []
    world_torsos: list[float] = []
    trunk_angles: list[float] = []
    velocities: list[float] = []
    state_counts: dict[str, int] = {}
    # Stage-1 events go to their own record: an alarm is reconstructed from a
    # handful of events, not by scanning thousands of per-frame rows.
    events = AuditLogger(
        cfg.logging.output_dir, Path(args.video).stem + "-events",
        fields=EVENT_FIELDS,
        metadata={"source": str(Path(args.video).resolve()),
                  "record_kind": "stage1_events",
                  "config": cfg.as_dict()},
    )
    # EXPERIMENTAL, research mode only: the full skeleton of every frame, in
    # its own file (see audit_log.LANDMARK_FIELDS). Off unless the config asks
    # for it — §3.6 promises a deployed device stores no landmark coordinates.
    landmarks = None
    if cfg.logging.as_dict().get("save_landmarks", False):
        landmarks = AuditLogger(
            cfg.logging.output_dir, Path(args.video).stem + "-landmarks",
            fields=LANDMARK_FIELDS,
            metadata={"source": str(Path(args.video).resolve()),
                      "record_kind": "landmarks",
                      "config": cfg.as_dict()},
        )
    detected_frames = 0
    reliable_frames = 0
    index = 0
    last_timestamp: float | None = None
    while True:
        ok, frame, timestamp = source.read()
        if not ok:
            break
        last_timestamp = timestamp
        result = pipeline.process(frame, index, timestamp)
        logger.log_pose_frame(result.pose, extra=result.csv_extra())
        if landmarks is not None:
            landmarks.log(landmark_row(result.pose))

        if result.pose.detected:
            detected_frames += 1
        if result.reliable:
            reliable_frames += 1
            torso_lengths.append(result.pose.torso_length)
            t_deg = result.quantities.get("T_deg", float("nan"))
            if not math.isnan(t_deg):
                trunk_angles.append(t_deg)
            v = result.quantities.get("V_tps", float("nan"))
            if not math.isnan(v):
                velocities.append(v)
            w = result.quantities.get("world_torso_len_m")
            if w is not None and not math.isnan(w):
                world_torsos.append(w)
            if result.state is not None:
                state_counts[result.state] = state_counts.get(result.state, 0) + 1
        index += 1

    # Close any event still being judged BEFORE the record is written, as
    # gui/lab_window.py does at end of file: a clip that ends with the
    # subject on the floor has an open event, and that event is precisely
    # the answer the label needs.
    if last_timestamp is not None:
        pipeline.finalise(last_timestamp)
    pipeline.close()
    source.release()
    logger.close()
    if landmarks is not None:
        landmarks.close()
    # Written only now, not as each event fired: Stage 2's verdict lands
    # several frames after the event is raised, so a row streamed at firing
    # time would record 'stage1_only' for every event regardless of outcome.
    for i, ev in enumerate(pipeline.machine.events):
        events.log({
            "event_index": i,
            "frame_index": ev.frame_index,
            "timestamp_s": f"{ev.timestamp:.3f}",
            "T_deg": f"{ev.t_deg:.2f}",
            "V_tps": f"{ev.v_tps:.3f}",
            "formulation": ev.formulation,
            "trigger_source": ev.trigger_source,
            "verdict": ev.verdict,
            "severity": ev.severity,
            "max_immobility_s": ("" if math.isnan(ev.max_immobility_s)
                                 else f"{ev.max_immobility_s:.2f}"),
            "p_outside_fraction": ("" if math.isnan(ev.p_outside_fraction)
                                   else f"{ev.p_outside_fraction:.3f}"),
            "p_samples": ev.p_samples,
            "H_ratio_EXP": "" if math.isnan(ev.h_ratio) else f"{ev.h_ratio:.3f}",
            "motivo": ev.reason,
        })
    events.close()

    # ---- summary ------------------------------------------------------------
    # All statistics below cover reliable frames only.
    print(f"Processed : {index} frames from {args.video}")
    print(f"Detected  : {detected_frames} frames with a person "
          f"({100.0 * detected_frames / max(1, index):.1f}%)")
    print(f"Reliable  : {reliable_frames} of {detected_frames} detected frames "
          f"(core visibility >= {cfg.pose.visibility_threshold}) — "
          f"stats below use only these")
    if torso_lengths:
        mean = statistics.fmean(torso_lengths)
        stdev = statistics.pstdev(torso_lengths)
        print(f"Torso px  : mean {mean:.1f}px, stdev {stdev:.1f}px "
              f"(CV {100.0 * stdev / mean:.1f}%) — grows/shrinks with distance; "
              f"expect smooth changes, not jumps")
    if world_torsos:
        w_mean = statistics.fmean(world_torsos)
        w_stdev = statistics.pstdev(world_torsos)
        print(f"Torso m   : mean {w_mean:.3f}m, CV {100.0 * w_stdev / w_mean:.1f}% "
              f"— metric estimate; should stay ~constant at any distance")
    if trunk_angles:
        bands: dict[str, int] = {}
        for angle in trunk_angles:
            band = trunk_band(angle)
            bands[band] = bands.get(band, 0) + 1
        spread = " ".join(
            f"{name}={count}" for name, count in sorted(bands.items())
        )
        print(f"Quantity T: min {min(trunk_angles):.1f} deg, "
              f"median {statistics.median(trunk_angles):.1f} deg, "
              f"max {max(trunk_angles):.1f} deg")
        print(f"T bands   : {spread}  (§3.4 reference table)")
        if velocities:
            print(f"Quantity V: min {min(velocities):+.2f}, "
                  f"median {statistics.median(velocities):+.2f}, "
                  f"max {max(velocities):+.2f} torso/s "
                  f"(negative = downward)")
        if state_counts:
            states = " ".join(
                f"{name}={count}"
                for name, count in sorted(state_counts.items(), key=lambda kv: -kv[1])
            )
            print(f"States    : {states}  (display labels, not decisions)")
    elif reliable_frames == 0 and detected_frames > 0:
        print("Quantity T: no reliable frames — subject's core landmarks were "
              "not visible (e.g. face-only footage); record full-body clips")
    report = source.fps_report(args.expected_seconds)
    print(f"FPS audit : declared {report['declared_fps']:.2f}, "
          f"implied duration {report['implied_duration_s']:.2f}s", end="")
    if "effective_fps" in report:
        print(f", effective {report['effective_fps']:.2f} "
              f"(ratio {report['fps_mismatch_ratio']:.2f} — "
              f"{'OK' if 0.9 <= report['fps_mismatch_ratio'] <= 1.1 else 'MISMATCH: use corrected dt'})")
    else:
        print()
    n_alerts = len(pipeline.alerts.dispatched)
    n_events = len(pipeline.machine.events)
    print(f"Stage 1   : {n_events} event(s) "
          f"[{cfg.stage1.trigger_formulation}, hold {cfg.stage1.trigger_hold_s}s] "
          f"— judged by Stages 2 (geometry) and 3 (immobility/recovery)")
    for ev in pipeline.machine.events[:10]:
        print(f"  t={ev.timestamp:6.2f}s frame {ev.frame_index:5d}  "
              f"T={ev.t_deg:6.1f} deg  V={ev.v_tps:+6.2f} torso/s  "
              f"-> {ev.verdict}"
              + (f" [{ev.severity}]" if ev.severity else "")
              + ("" if math.isnan(ev.p_outside_fraction)
                 else f" (COM outside {100*ev.p_outside_fraction:.0f}% "
                      f"of {ev.p_samples} frames)"))
        if ev.reason:
            print(f"            motivo: {ev.reason}")
    print(f"Alerts    : {n_alerts} dispatched "
          f"(verdicts that alert: {', '.join(cfg.alerts.dispatch_verdicts)})")
    print(f"Audit CSV : {logger.path}")
    if n_events:
        print(f"Events    : {events.path}")
    if landmarks is not None:
        print(f"Landmarks : {landmarks.path}")
    print(f"Run meta  : {logger.path.with_suffix('.meta.json').name} "
          f"(source, fps and the full configuration that produced this record)")
    return 0


def main() -> int:
    """Parse the command line and dispatch to the GUI or the headless run.

    Returns the process exit code: 0 on success, which matters for the batch
    pipelines of §3.7 that chain runs together.
    """
    parser = argparse.ArgumentParser(description="PEF-Fall-Detector (Phase 1)")
    # A run has exactly one source. Declaring them mutually exclusive makes
    # argparse reject the combination with a clear message and document it in
    # --help, instead of silently preferring one and leaving the operator
    # wondering why the camera never opened.
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="path to a recorded clip (.mp4)")
    source.add_argument("--camera", type=int, default=None, help="live camera index")
    parser.add_argument("--headless", action="store_true",
                        help="process without GUI (requires --video)")
    parser.add_argument("--config", default=None, help="alternative config.yaml")
    parser.add_argument("--expected-seconds", type=float, default=None,
                        help="known wall-clock duration of the clip, for the FPS audit")
    args = parser.parse_args()

    if args.headless:
        if not args.video:
            parser.error("--headless requires --video")
        return run_headless(args)

    # GUI mode (imported lazily so headless boxes never need PySide6).
    from pef_fall_detector.gui.lab_window import run_gui

    cfg = load_config(args.config)
    run_gui(cfg, video=args.video, camera=args.camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())
