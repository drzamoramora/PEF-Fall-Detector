"""Turn a run's binary landmarks (``landmarks/npz``) into text, or check the two agree.

A schema-2 run keeps the 33 landmarks twice, exact both times: as numpy
arrays (``landmarks/npz/<clip>.npz``, for code and machine learning) and as
text (``landmarks/csv/<clip>.csv``, for a spreadsheet). The text can always
be regenerated from the arrays; this tool does it, for a run whose CSVs were
deleted to save space, or into another folder.

``--check`` writes nothing: it rebuilds each clip's CSV in memory and says
whether it equals the one in the run, character for character. It is the
proof that the two formats hold the same numbers.

Usage::

    python tools/export.py RUN [--out DIR] [--clips A01-S1,B12-S2] [--check]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _records import find_runs  # noqa: E402

TOOLS_REPO = Path(__file__).resolve().parents[1]


def _package():
    if str(TOOLS_REPO) not in sys.path:
        sys.path.insert(0, str(TOOLS_REPO))
    from pef_fall_detector import audit_log, landmark_arrays
    return audit_log, landmark_arrays


def landmark_csv(npz_path: Path) -> str:
    """The text record of one clip, rebuilt from its arrays."""
    audit_log, arrays = _package()
    data = arrays.load(npz_path)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=audit_log.LANDMARK_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    for n in range(len(data["frame_index"])):
        w, h = (int(v) for v in data["frame_size"][n])
        detected = bool(data["detected"][n])
        pf = SimpleNamespace(frame_index=int(data["frame_index"][n]),
                             timestamp=float(data["timestamp_s"][n]),
                             detected=detected, frame_size=(w, h),
                             landmarks=np.empty((0, 3)), visibility=np.empty(0),
                             world_landmarks=np.empty((0, 3)))
        if detected:
            pf.landmarks = arrays.pixel_landmarks(data["image"][n], w, h)
            pf.visibility = data["visibility"][n].astype(np.float64)
            world = data["world"][n]
            if not np.isnan(world).all():
                pf.world_landmarks = world.astype(np.float64)
        writer.writerow(audit_log.landmark_row(pf))
    return out.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a schema-2 run folder")
    ap.add_argument("--out", default=None,
                    help="where the CSVs go (default: the run's landmarks/csv, "
                         "only for clips that lack one)")
    ap.add_argument("--clips", default=None, help="comma-separated clip ids")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; say whether each CSV equals its npz")
    args = ap.parse_args(argv)
    runs, _ = find_runs(args.run)
    if args.clips:
        wanted = set(args.clips.split(","))
        runs = {c: r for c, r in runs.items() if c in wanted}
    with_npz = {c: r for c, r in sorted(runs.items()) if r.landmarks_npz is not None}
    if not with_npz:
        print(f"no landmarks/npz in {args.run} (a schema-1 run has none)")
        return 1
    same = differ = written = 0
    for clip, rec in with_npz.items():
        text = landmark_csv(rec.landmarks_npz)
        if args.check:
            if rec.landmarks is None:
                print(f"{clip}: no CSV to check")
                differ += 1
            elif rec.landmarks.read_bytes().decode("utf-8") == text:
                same += 1
            else:
                print(f"{clip}: the CSV differs from the npz")
                differ += 1
            continue
        if args.out:
            target = Path(args.out) / f"{rec.stem}.csv"
        elif rec.landmarks is None:
            target = Path(args.run) / "landmarks" / "csv" / f"{rec.stem}.csv"
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"{target} exists: not overwritten")
            continue
        target.write_text(text, encoding="utf-8", newline="")
        written += 1
    if args.check:
        print(f"landmarks: {same}/{len(with_npz)} CSVs equal their npz"
              + (f" | {differ} differ" if differ else "")
              + (" -> IDENTICAL" if not differ else " -> DIFFERENT"))
        return 0 if not differ else 1
    print(f"wrote {written} CSV(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
