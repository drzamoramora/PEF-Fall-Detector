"""Compare two runs clip by clip: labels, events and every per-frame value.

The check behind every cleanup block: a change meant to leave the detector's
behaviour alone must leave its records alone. Two run folders (real runs or
replays, see ``replay.py``) are compared on

* the clip label, from the events as ``classification.classify_clip`` reads them;
* every event row, every column;
* every per-frame row, every column, cell by cell as text. The records are
  already formatted, so equal text means equal value at the logged precision;
* with ``--landmarks``, the 33-point skeleton records the same way. Use it on
  two real runs: the skeleton is MediaPipe's output, the input of everything
  else. Two replays of the same run share it by construction.

The configuration each run recorded in its sidecars is compared too, as a
warning that does not change the verdict: a block that reorganises
``config.yaml`` (B2) should show no value changing, and a run made with a
different file shows up here before anyone reads its numbers.
``logging.output_dir`` is left out, since every run writes to its own folder.

A block that renames columns or translates texts on purpose (B3, B5) passes
a map that says what each old name or text became; any other difference is
reported.

Usage::

    python tools/compare_runs.py A B [--landmarks] [--map FILE.json] [--ignore COL,...]
                                 [--show N]

Exit status 0 when the runs are identical (after the map), 1 otherwise.

Map file (every part optional; values and patterns are keyed by the NEW
column name and applied to run A only)::

    {"columns":  {"motivo": "reason"},
     "values":   {"reason": {"sin eventos": "no events"}},
     "patterns": {"reason": [["^A final (\\\\S+)", "final A \\\\1"]]}}
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _records import find_runs  # noqa: E402

TOOLS_REPO = Path(__file__).resolve().parents[1]


@dataclass
class ColumnMap:
    """What run A's names and texts became in run B."""

    columns: dict[str, str] = field(default_factory=dict)
    values: dict[str, dict[str, str]] = field(default_factory=dict)
    patterns: dict[str, list[tuple[re.Pattern, str]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> "ColumnMap":
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(columns=dict(raw.get("columns", {})),
                   values={k: dict(v) for k, v in raw.get("values", {}).items()},
                   patterns={k: [(re.compile(p), r) for p, r in v]
                             for k, v in raw.get("patterns", {}).items()})

    def apply(self, header: list[str], rows: list[dict]) -> tuple[list[str], list[dict]]:
        header = [self.columns.get(c, c) for c in header]
        out = []
        for row in rows:
            new = {self.columns.get(k, k): v for k, v in row.items()}
            for col, table in self.values.items():
                if col in new and new[col] in table:
                    new[col] = table[new[col]]
            for col, rules in self.patterns.items():
                if col in new and new[col]:
                    for pattern, repl in rules:
                        new[col] = pattern.sub(repl, new[col])
            out.append(new)
        return header, out


def read_table(path: Path | None) -> tuple[list[str], list[dict]]:
    """Header and rows of a CSV record; an absent record is an empty table."""
    if path is None:
        return [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


@dataclass
class TableDiff:
    """How two tables differ. Empty (falsy) when they do not."""

    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    rows: tuple[int, int] | None = None
    cells: int = 0
    examples: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.only_a or self.only_b or self.rows or self.cells)

    def describe(self) -> str:
        parts = []
        if self.only_a:
            parts.append(f"columns only in A: {', '.join(self.only_a)}")
        if self.only_b:
            parts.append(f"columns only in B: {', '.join(self.only_b)}")
        if self.rows:
            parts.append(f"rows {self.rows[0]} != {self.rows[1]}")
        if self.cells:
            parts.append(f"{self.cells} cells differ, e.g. " + "; ".join(self.examples))
        return " | ".join(parts)


def compare_tables(a: tuple[list[str], list[dict]], b: tuple[list[str], list[dict]],
                   ignore: set[str], key: str, show: int = 3) -> TableDiff:
    """Cell-by-cell comparison of two tables, in row order."""
    (a_hdr, a_rows), (b_hdr, b_rows) = a, b
    diff = TableDiff(only_a=[c for c in a_hdr if c not in b_hdr and c not in ignore],
                     only_b=[c for c in b_hdr if c not in a_hdr and c not in ignore])
    if len(a_rows) != len(b_rows):
        diff.rows = (len(a_rows), len(b_rows))
    common = [c for c in a_hdr if c in b_hdr and c not in ignore]
    for ra, rb in zip(a_rows, b_rows):
        for col in common:
            va, vb = ra.get(col) or "", rb.get(col) or ""
            if va != vb:
                diff.cells += 1
                if len(diff.examples) < show:
                    diff.examples.append(f"{key} {ra.get(key, '?')} {col} {va!r} != {vb!r}")
    return diff


#: Configuration keys expected to differ between runs, left out of the warning.
CONFIG_EXPECTED = {"logging.output_dir"}


def _flatten(tree: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def recorded_config(meta: Path) -> dict[str, object] | None:
    """The configuration a run recorded in a sidecar, flattened; None if absent."""
    if not meta.exists():
        return None
    cfg = json.loads(meta.read_text(encoding="utf-8")).get("config")
    return _flatten(cfg) if isinstance(cfg, dict) else None


def config_differences(a: dict[str, object], b: dict[str, object],
                       skip: set[str] = CONFIG_EXPECTED) -> list[tuple[str, str, str]]:
    """(key, value in A, value in B) for every configuration value that differs."""
    def show(tree: dict[str, object], key: str) -> str:
        return json.dumps(tree[key]) if key in tree else "(absent)"
    return [(k, show(a, k), show(b, k))
            for k in sorted(set(a) | set(b)) if k not in skip and a.get(k) != b.get(k)]


class _Event:
    def __init__(self, row: dict) -> None:
        self.verdict = row.get("verdict", "")
        self.severity = row.get("severity", "")
        self.timestamp = float(row.get("timestamp_s") or "nan")


def clip_label(classify_clip, rows: list[dict]) -> str:
    label = classify_clip([_Event(r) for r in rows])
    return label[0] if isinstance(label, tuple) else label


def compare(a_dir: str | Path, b_dir: str | Path, cmap: ColumnMap | None = None,
            ignore: set[str] | None = None, repo: str | Path = TOOLS_REPO,
            show: int = 3, out=None, landmarks: bool = False) -> bool:
    """Print the comparison of two run folders; True when they are identical.

    ``landmarks``: compare the 33-point records too. A record present in one
    run and missing in the other is a difference; missing in both (replays
    write none) is not. The recorded configurations are always compared, as a
    warning only.
    """
    out = out or sys.stdout      # looked up now, so a redirected stdout is honoured
    cmap = cmap or ColumnMap()
    ignore = ignore or set()
    if str(Path(repo).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(repo).resolve()))
    classify_clip = importlib.import_module("pef_fall_detector.classification").classify_clip
    a_runs, a_extra = find_runs(a_dir)
    b_runs, b_extra = find_runs(b_dir)
    both = sorted(set(a_runs) & set(b_runs))
    only_a = sorted(set(a_runs) - set(b_runs))
    only_b = sorted(set(b_runs) - set(a_runs))
    print(f"clips: A {len(a_runs)} | B {len(b_runs)} | in both {len(both)}"
          f" | only in A {len(only_a)} | only in B {len(only_b)}", file=out)
    for tag, extra in (("A", a_extra), ("B", b_extra)):
        if extra:
            print(f"extra runs in {tag} ignored (earliest kept): {', '.join(sorted(extra))}",
                  file=out)
    n_label = n_events = n_frames = n_marks = differing = 0
    config_diffs: dict[tuple[str, str, str], int] = {}
    for clip in both:
        ra, rb = a_runs[clip], b_runs[clip]
        ev_a = cmap.apply(*read_table(ra.events))
        ev_b = read_table(rb.events)
        fr_a = cmap.apply(*read_table(ra.frames))
        fr_b = read_table(rb.frames)
        lines = []
        la, lb = clip_label(classify_clip, ev_a[1]), clip_label(classify_clip, ev_b[1])
        if la != lb:
            n_label += 1
            lines.append(f"label {la} -> {lb}")
        d_ev = compare_tables(ev_a, ev_b, ignore, "event_index", show)
        if d_ev:
            n_events += 1
            lines.append(f"events: {d_ev.describe()}")
        d_fr = compare_tables(fr_a, fr_b, ignore, "frame_index", show)
        if d_fr:
            n_frames += 1
            lines.append(f"frames: {d_fr.describe()}")
        if landmarks and (ra.landmarks is not None or rb.landmarks is not None):
            # Neither side has one (two replays): nothing to compare, not a difference.
            if ra.landmarks is None or rb.landmarks is None:
                n_marks += 1
                side = "A" if ra.landmarks is None else "B"
                lines.append(f"landmarks: no landmark record in {side}")
            else:
                d_lm = compare_tables(cmap.apply(*read_table(ra.landmarks)),
                                      read_table(rb.landmarks), ignore, "frame_index", show)
                if d_lm:
                    n_marks += 1
                    lines.append(f"landmarks: {d_lm.describe()}")
        cfg_a, cfg_b = recorded_config(ra.meta), recorded_config(rb.meta)
        if cfg_a is not None and cfg_b is not None:
            for diff in config_differences(cfg_a, cfg_b):
                config_diffs[diff] = config_diffs.get(diff, 0) + 1
        if lines:
            differing += 1
            print(f"{clip}: " + "\n    ".join(lines), file=out)
    if config_diffs:
        print("configuration differs (a warning, not part of the verdict):", file=out)
        for (key, va, vb), n in sorted(config_diffs.items()):
            print(f"    {key}: {va} -> {vb} ({n} clips)", file=out)
    else:
        print("configuration: the same in every clip"
              f" (left out: {', '.join(sorted(CONFIG_EXPECTED))})", file=out)
    same = len(both) - differing
    identical = not (only_a or only_b or n_label or n_events or n_frames or n_marks)
    marks = f" | landmarks differ {n_marks}" if landmarks else ""
    print(f"identical clips: {same}/{len(both)} | labels differ {n_label}"
          f" | events differ {n_events} | frames differ {n_frames}{marks}"
          f" -> {'IDENTICAL' if identical else 'DIFFERENT'}", file=out)
    return identical


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", help="reference run folder")
    ap.add_argument("b", help="run folder to check against it")
    ap.add_argument("--landmarks", action="store_true",
                    help="compare the 33-point records too (for two real runs)")
    ap.add_argument("--map", default=None, help="JSON map of renamed columns and texts")
    ap.add_argument("--ignore", default="", help="comma-separated columns to leave out")
    ap.add_argument("--repo", default=str(TOOLS_REPO),
                    help="checkout whose classification labels the clips (default: this one)")
    ap.add_argument("--show", type=int, default=3, help="examples per differing table")
    args = ap.parse_args(argv)
    ok = compare(args.a, args.b, ColumnMap.load(args.map),
                 {c for c in args.ignore.split(",") if c}, args.repo, args.show,
                 landmarks=args.landmarks)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
