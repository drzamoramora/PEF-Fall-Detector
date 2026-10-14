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
reported. Without ``--map``, a run A written before 0.2.0.dev5 (the B5
translation) compared with a run B written after it gets
``tools/maps/b5.json`` on its own, and the tool says so.

The two record layouts (``_records``) compare with each other:

* columns that only the newer schema has (the frame hash...) cannot be
  compared with a run that never wrote them; they are listed once, not
  counted as a difference. A column the older run has and
  the newer one lost is a difference;
* schema-1 landmarks were rounded (0.01 px, 0.001 visibility, 0.1 mm), schema
  2 keeps them exact: across the two, a landmark matches when it is within
  the old rounding;
* the timings and load columns (``_records.VOLATILE_COLUMNS``) measure the
  computer, not the scene, and are never compared. Neither are the landmark
  presence columns (``lm00_p`` ...) that the first runs of 0.3.0.dev0 wrote
  and the code no longer does: MediaPipe left them at 0.

Usage::

    python tools/compare_runs.py A B [--landmarks] [--map FILE.json | --no-map]
                                 [--ignore COL,...] [--show N]

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
from _records import (  # noqa: E402
    SCHEMA_LOOSE,
    VOLATILE_COLUMNS,
    ClipRecords,
    find_runs,
    version_key,
)

TOOLS_REPO = Path(__file__).resolve().parents[1]
MAPS = TOOLS_REPO / "tools" / "maps"
#: The first version whose records are in English (B5); older runs need b5.json.
ENGLISH_RECORDS = (0, 2, 0, 5)


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
    new: list[str] = field(default_factory=list)   # only in the newer schema: not a difference

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
                   ignore: set[str], key: str, show: int = 3, newer: str | None = None,
                   tolerance=None) -> TableDiff:
    """Cell-by-cell comparison of two tables, in row order.

    ``newer``: "a" or "b" when the two runs use different schemas; the columns
    only that side has are listed in ``new`` instead of counted. ``tolerance``:
    column -> largest numeric difference still equal (None: exact text).
    """
    (a_hdr, a_rows), (b_hdr, b_rows) = a, b
    only_a = [c for c in a_hdr if c not in b_hdr and c not in ignore]
    only_b = [c for c in b_hdr if c not in a_hdr and c not in ignore]
    diff = TableDiff(only_a=[] if newer == "a" else only_a,
                     only_b=[] if newer == "b" else only_b,
                     new=only_a if newer == "a" else only_b if newer == "b" else [])
    if len(a_rows) != len(b_rows):
        diff.rows = (len(a_rows), len(b_rows))
    common = [c for c in a_hdr if c in b_hdr and c not in ignore]
    tol = {c: tolerance(c) for c in common} if tolerance else {}
    # Row counts are compared above; the cells, over the rows both have.
    for ra, rb in zip(a_rows, b_rows, strict=False):
        for col in common:
            va, vb = ra.get(col) or "", rb.get(col) or ""
            if va != vb and not _within(va, vb, tol.get(col)):
                diff.cells += 1
                if len(diff.examples) < show:
                    diff.examples.append(f"{key} {ra.get(key, '?')} {col} {va!r} != {vb!r}")
    return diff


def _within(va: str, vb: str, tol: float | None) -> bool:
    """Two numeric cells that differ by no more than ``tol``."""
    if tol is None or not va or not vb:
        return False
    try:
        return abs(float(va) - float(vb)) <= tol
    except ValueError:
        return False


_IMAGE_MARK = re.compile(r"^lm\d\d_[xyz]$")
_VISIBILITY = re.compile(r"^lm\d\d_v$")
_WORLD_MARK = re.compile(r"^w\d\d_[xyz]$")


def rounding_of_schema1(column: str) -> float | None:
    """Half the last digit a schema-1 landmark record kept (plus float slack)."""
    if _IMAGE_MARK.match(column):
        return 0.005 + 1e-9
    if _VISIBILITY.match(column):
        return 0.0005 + 1e-9
    if _WORLD_MARK.match(column):
        return 0.00005 + 1e-9
    return None


#: Presence columns of the first 0.3.0.dev0 runs: always 0, then dropped.
PRESENCE_COLUMNS = frozenset(f"lm{i:02d}_p" for i in range(33))

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


def recorded_config(rec: ClipRecords) -> dict[str, object] | None:
    """The configuration a run recorded for a clip, flattened; None if absent."""
    cfg = rec.metadata().get("config")
    return _flatten(cfg) if isinstance(cfg, dict) else None


def _code_version(runs: dict[str, ClipRecords]) -> tuple[int, int, int, int] | None:
    if not runs:
        return None
    version = str(next(iter(runs.values())).metadata().get("code_version", ""))
    return version_key(version) if version else None


def default_map(a_runs: dict[str, ClipRecords],
                b_runs: dict[str, ClipRecords]) -> Path | None:
    """The map run A needs to read like run B, judged by their code versions.

    Only across the B5 translation: A written before it, B after. Two runs on
    the same side of it need none.
    """
    va, vb = _code_version(a_runs), _code_version(b_runs)
    if va is not None and vb is not None and va < ENGLISH_RECORDS <= vb:
        return MAPS / "b5.json"
    return None


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
    new_columns: set[str] = set()
    ignore = set(ignore) | VOLATILE_COLUMNS | PRESENCE_COLUMNS
    for clip in both:
        ra, rb = a_runs[clip], b_runs[clip]
        newer = None if ra.schema == rb.schema else ("a" if ra.schema > rb.schema else "b")
        rounded = ra.schema == SCHEMA_LOOSE or rb.schema == SCHEMA_LOOSE
        ev_a = cmap.apply(*read_table(ra.events))
        ev_b = read_table(rb.events)
        fr_a = cmap.apply(*read_table(ra.frames))
        fr_b = read_table(rb.frames)
        lines = []
        la, lb = clip_label(classify_clip, ev_a[1]), clip_label(classify_clip, ev_b[1])
        if la != lb:
            n_label += 1
            lines.append(f"label {la} -> {lb}")
        d_ev = compare_tables(ev_a, ev_b, ignore, "event_index", show, newer)
        new_columns.update(d_ev.new)
        if d_ev:
            n_events += 1
            lines.append(f"events: {d_ev.describe()}")
        d_fr = compare_tables(fr_a, fr_b, ignore, "frame_index", show, newer)
        new_columns.update(d_fr.new)
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
                # Rounded on one side only: equal within that rounding. Rounded on
                # both (two schema-1 runs): the same text, as always.
                d_lm = compare_tables(cmap.apply(*read_table(ra.landmarks)),
                                      read_table(rb.landmarks), ignore, "frame_index", show,
                                      newer, rounding_of_schema1 if rounded and newer else None)
                new_columns.update(d_lm.new)
                if d_lm:
                    n_marks += 1
                    lines.append(f"landmarks: {d_lm.describe()}")
        cfg_a, cfg_b = recorded_config(ra), recorded_config(rb)
        if cfg_a is not None and cfg_b is not None:
            for diff in config_differences(cfg_a, cfg_b):
                config_diffs[diff] = config_diffs.get(diff, 0) + 1
        if lines:
            differing += 1
            print(f"{clip}: " + "\n    ".join(lines), file=out)
    if new_columns:
        print("only in the newer schema, not compared: "
              + ", ".join(_summarise(sorted(new_columns))), file=out)
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


def _summarise(columns: list[str]) -> list[str]:
    """``lm00_p … lm32_p`` instead of 33 names."""
    marks = [c for c in columns if re.match(r"^lm\d\d_p$", c)]
    rest = [c for c in columns if c not in marks]
    return rest + ([f"{marks[0]} … {marks[-1]} ({len(marks)})"] if len(marks) > 2 else marks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", help="reference run folder")
    ap.add_argument("b", help="run folder to check against it")
    ap.add_argument("--landmarks", action="store_true",
                    help="compare the 33-point records too (for two real runs)")
    ap.add_argument("--map", default=None, help="JSON map of renamed columns and texts "
                    "(default: tools/maps/b5.json when only run A predates 0.2.0.dev5)")
    ap.add_argument("--no-map", action="store_true", help="never apply a map")
    ap.add_argument("--ignore", default="", help="comma-separated columns to leave out")
    ap.add_argument("--repo", default=str(TOOLS_REPO),
                    help="checkout whose classification labels the clips (default: this one)")
    ap.add_argument("--show", type=int, default=3, help="examples per differing table")
    args = ap.parse_args(argv)
    map_path = args.map
    if map_path is None and not args.no_map:
        map_path = default_map(find_runs(args.a)[0], find_runs(args.b)[0])
        if map_path is not None:
            print(f"map: {map_path.relative_to(TOOLS_REPO)} (run A predates "
                  f"0.2.0.dev5 and run B does not; --no-map to compare as is)")
    ok = compare(args.a, args.b, ColumnMap.load(None if args.no_map else map_path),
                 {c for c in args.ignore.split(",") if c}, args.repo, args.show,
                 landmarks=args.landmarks)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
