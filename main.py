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

from pef_fall_detector.audit_log import AuditLogger
from pef_fall_detector.config import load_config
from pef_fall_detector.pipeline import FramePipeline
from pef_fall_detector.quantities import trunk_band
from pef_fall_detector.sources import VideoFileSource


def run_headless(args: argparse.Namespace) -> int:
    """Process a video file sequentially and print a verification summary."""
    cfg = load_config(args.config)
    source = VideoFileSource(args.video)
    pipeline = FramePipeline(cfg)
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
    detected_frames = 0
    reliable_frames = 0
    index = 0
    while True:
        ok, frame, timestamp = source.read()
        if not ok:
            break
        result = pipeline.process(frame, index, timestamp)
        logger.log_pose_frame(result.pose, extra=result.csv_extra())

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

    pipeline.close()
    source.release()
    logger.close()

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
    print(f"Audit CSV : {logger.path}")
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
