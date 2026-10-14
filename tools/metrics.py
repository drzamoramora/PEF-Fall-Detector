"""Headline numbers of a run, recomputed from its records alone.

Works the same on a real run and on a replay (``replay.py``):

* accuracy over the four classes, total and per partition (train/test by
  actor, ``config/partition.yaml``);
* falls detected (any fall class), the falls missed, the false-positive labels;
* alarms: an alarm is an event whose verdict the run's configuration
  dispatches (``alerts.dispatch_verdicts``, stage3_confirmed). Counted per
  fall, per Recovered fall (a subject who got up raises none) and per NoFall;
* events per Stage-1 condition (``trigger_source``: score, H or A), when the
  record has that column;
* the clips it gets wrong.

With a second folder: the label agreement between the two runs, and the clips
whose label differs.

Then, with code that has it (0.3.0.dev0 on), the full table of
``pef_fall_detector.run_metrics``: every metric by partition, the reading by
alarm, and the per-frame cost when the run recorded it. ``--write`` saves it
as ``metrics.csv`` and ``metrics.txt`` into a schema-2 run folder,
recomputed from its records.

Usage::

    python tools/metrics.py LOGS [LOGS_B] [--partition FILE] [--repo PATH] [--write]
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _records import CLIPS_DIR, FRAMES_FILE, SCHEMA_RUN_FOLDER, find_runs, schema_of  # noqa: E402

TOOLS_REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARTITION = TOOLS_REPO / "config" / "partition.yaml"


class _Event:
    def __init__(self, row: dict) -> None:
        self.verdict = row.get("verdict", "")
        self.severity = row.get("severity", "")
        self.timestamp = float(row.get("timestamp_s") or "nan")


def load_run(folder: str | Path, pkg) -> dict[str, dict]:
    """Per clip: truth, predicted label, event rows, whether it raised an alarm."""
    runs, _extra = find_runs(folder)
    out = {}
    for clip, rec in runs.items():
        rows = []
        if rec.events is not None:
            with open(rec.events, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        cfg = rec.metadata().get("config", {})
        dispatch = tuple(cfg.get("alerts", {}).get("dispatch_verdicts", ("stage3_confirmed",)))
        label = pkg.classification.classify_clip([_Event(r) for r in rows])
        meta = rec.metadata()
        fps = float(meta.get("source_fps_declared") or 0.0)
        frames = int(meta.get("frame_count") or meta.get("rows") or 0)
        out[clip] = {
            "stem": rec.stem,
            "duration_s": frames / fps if fps else 0.0,
            "truth": pkg.dataset.truth_from_name(rec.stem),
            "pred": label[0] if isinstance(label, tuple) else label,
            "events": rows,
            "alarm": any(r.get("verdict") in dispatch for r in rows),
            "actor": pkg.evaluation.actor_of(rec.stem),
        }
    return out


def report(name: str, clips: dict[str, dict], split: dict[str, str], falls,
           recovered_class: str = "Recovered") -> None:
    """Print the headline numbers of one run."""
    def ok(group):
        return sum(v["pred"] == v["truth"] for v in group)

    total = list(clips.values())
    train = [v for v in total if split.get(v["actor"]) == "train"]
    test = [v for v in total if split.get(v["actor"]) == "test"]
    unassigned = sorted(c for c, v in clips.items() if v["actor"] not in split)
    fall = {c: v for c, v in clips.items() if v["truth"] in falls}
    nofall = {c: v for c, v in clips.items() if v["truth"] not in falls}
    print(f"== {name}: {len(total)} clips")
    print(f"accuracy {ok(total)}/{len(total)} ({100 * ok(total) / max(len(total), 1):.1f} %)"
          f" | train {ok(train)}/{len(train)} | test {ok(test)}/{len(test)}"
          + (f" | not in the partition: {', '.join(unassigned)}" if unassigned else ""))
    detected = sum(v["pred"] in falls for v in fall.values())
    missed = sorted(c for c, v in fall.items() if v["pred"] not in falls)
    fp = sorted(c for c, v in nofall.items() if v["pred"] in falls)
    print(f"falls detected {detected}/{len(fall)} | missed: {', '.join(missed) or '-'}")
    print(f"specificity (label) {len(nofall) - len(fp)}/{len(nofall)}"
          f" | false positives: {', '.join(fp) or '-'}")
    recovered = [v for v in fall.values() if v["truth"] == recovered_class]
    alarm_nf = sorted(c for c, v in nofall.items() if v["alarm"])
    print(f"alarms: falls {sum(v['alarm'] for v in fall.values())}/{len(fall)}"
          f" | Recovered {sum(v['alarm'] for v in recovered)}/{len(recovered)}"
          f" | NoFall {len(alarm_nf)} ({', '.join(alarm_nf) or '-'})"
          f" -> specificity (alarm) {len(nofall) - len(alarm_nf)}/{len(nofall)}")
    events = [r for v in total for r in v["events"]]
    sources = Counter(r.get("trigger_source") or "?" for r in events)
    by_source = (" | by condition: " + ", ".join(f"{k} {n}" for k, n in sorted(sources.items()))
                 if any("trigger_source" in r for r in events) else "")
    print(f"events {len(events)}{by_source}")
    wrong = sorted((c, v["truth"], v["pred"]) for c, v in clips.items() if v["pred"] != v["truth"])
    print(f"errors ({len(wrong)}):")
    for clip, truth, pred in wrong:
        print(f"    {clip:8s} truth {truth:18s} detected {pred}")


def label_rows(clips: dict[str, dict]) -> list[dict]:
    """What dataset-labels.csv would say, rebuilt from the records."""
    return [{"video": v["stem"], "true_class": v["truth"], "detected_class": v["pred"],
             "duration_s": f"{v['duration_s']:.2f}", "events": len(v["events"]),
             "alarm": int(v["alarm"]),
             "trigger_sources": ";".join(r.get("trigger_source") or "?" for r in v["events"])}
            for v in clips.values()]


def full_table(folder: str | Path, clips: dict[str, dict], split: dict[str, str],
               write: bool = False) -> None:
    """The run_metrics table of a run, printed (and saved with ``write``)."""
    try:
        run_metrics = importlib.import_module("pef_fall_detector.run_metrics")
    except ModuleNotFoundError:
        return                                  # code before 0.3.0.dev0
    rows = label_rows(clips)
    perf = {}
    if schema_of(folder) == SCHEMA_RUN_FOLDER:
        perf = run_metrics.performance(sorted(Path(folder).glob(f"{CLIPS_DIR}/*/{FRAMES_FILE}")))
    table = run_metrics.compute(rows, split, perf)
    text = run_metrics.format_text(table, rows, f"{Path(folder).name} (from its records)")
    print("\n" + text)
    if write:
        if schema_of(folder) != SCHEMA_RUN_FOLDER:
            print("--write: only into a schema-2 run folder")
            return
        csv_path, txt_path = run_metrics.write(folder, table, text)
        print(f"-> {csv_path}\n-> {txt_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", help="run folder")
    ap.add_argument("logs_b", nargs="?", default=None, help="second run folder to compare labels")
    ap.add_argument("--partition", default=str(DEFAULT_PARTITION), help="train/test by actor")
    ap.add_argument("--repo", default=str(TOOLS_REPO),
                    help="checkout whose classification labels the clips (default: this one)")
    ap.add_argument("--write", action="store_true",
                    help="save metrics.csv and metrics.txt into the (schema-2) run folder")
    args = ap.parse_args(argv)
    repo = str(Path(args.repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    pkg = argparse.Namespace(**{m: importlib.import_module(f"pef_fall_detector.{m}")
                                for m in ("classification", "dataset", "evaluation")})
    split = pkg.evaluation.load_split(args.partition)
    falls = pkg.evaluation.FALL_CLASSES
    a = load_run(args.logs, pkg)
    rec = pkg.classification.RECOVERED
    report(args.logs, a, split, falls, rec)
    full_table(args.logs, a, split, write=args.write)
    if args.logs_b:
        b = load_run(args.logs_b, pkg)
        report(args.logs_b, b, split, falls, rec)
        both = sorted(set(a) & set(b))
        diff = [c for c in both if a[c]["pred"] != b[c]["pred"]]
        print(f"\nlabel agreement: {len(both) - len(diff)}/{len(both)}"
              + (f" | differ: {', '.join(diff)}" if diff else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
