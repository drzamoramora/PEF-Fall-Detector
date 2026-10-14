"""Every number of a labelled run, as a table (``metrics.csv``) and as text (``metrics.txt``).

Records schema 2 (0.3.0.dev0). Until then the numbers of a run were split
between the summary printed at the end of a pass (``evaluation.pass_summary``:
the confusion matrix and accuracy, sensitivity and specificity by partition)
and ``tools/metrics.py`` (alarms, events by trigger), and several of the
numbers compared across platforms (balanced accuracy, the F1 averaged over
the four classes, the reading by alarm) were computed by hand each time.
Here they are all computed once, from the rows of ``dataset-labels.csv``,
with the functions of ``evaluation`` (one code for the screen, the terminal
and the paper), and saved with the run.

THE TABLE
---------
One row per number, so a spreadsheet can filter and pivot it and two runs can
be joined on (split, metric, class)::

    split  metric            class  value  unit   num  den  ci_low  ci_high
    all    accuracy_4class          85.9   %      110  128  78.9    90.9
    test   class_accuracy    PartiallyRecovered  50.0  %  4  8

``split`` is ``all``, ``train``, ``test`` (the partition by actor of
``config/partition.yaml``, as recorded in ``run.json``) and ``unassigned``
when a clip's actor is in neither side. Intervals are 95 % Wilson.

TWO READINGS OF "DETECTED"
--------------------------
*By label*: the class the run gave the clip. *By alarm*: whether an event
ended in a verdict that dispatches an alert (``alerts.dispatch_verdicts``,
the ``alarm`` column). A Recovered fall is by design detected by label and
silent by alarm; a carer only sees the second.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .classification import NO_FALL, NOT_RECOVERED, PARTIALLY_RECOVERED, RECOVERED
from .dataset import format_confusion
from .evaluation import FALL_CLASSES, actor_of, binary_metrics, fp_per_hour, wilson

METRICS_CSV = "metrics.csv"
METRICS_TXT = "metrics.txt"
METRIC_FIELDS = ["split", "metric", "class", "value", "unit", "num", "den",
                 "ci_low", "ci_high"]

#: The four classes a clip is scored on.
FOUR_CLASSES = (NO_FALL, RECOVERED, PARTIALLY_RECOVERED, NOT_RECOVERED)
#: The falls that need help: the subject did not get up on their own.
NEED_HELP = (PARTIALLY_RECOVERED, NOT_RECOVERED)
SPLITS = ("all", "train", "test", "unassigned")


def _detected(row) -> str:
    return row.get("reviewed_class") or row.get("detected_class") or ""


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else float("nan")


class _Table:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, split: str, metric: str, value: float, unit: str, cls: str = "",
            num: int | None = None, den: int | None = None, ci: bool = False) -> None:
        row = {"split": split, "metric": metric, "class": cls, "unit": unit,
               "value": _fmt(value, unit), "num": "" if num is None else num,
               "den": "" if den is None else den, "ci_low": "", "ci_high": ""}
        if ci and num is not None and den:
            lo, hi = wilson(num, den)
            row["ci_low"], row["ci_high"] = f"{100 * lo:.1f}", f"{100 * hi:.1f}"
        self.rows.append(row)

    def ratio(self, split: str, metric: str, num: int, den: int, cls: str = "",
              ci: bool = False) -> None:
        self.add(split, metric, _pct(num, den), "%", cls, num, den, ci)


def _fmt(value: float, unit: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if unit == "count":
        return str(int(value))
    return f"{value:.1f}" if unit in ("%", "/h", "fps", "MB") else f"{value:.2f}"


def _split_of(row, partition: dict[str, str]) -> str:
    if not partition:
        return "unassigned"
    return partition.get(actor_of(row.get("video", "")), "unassigned")


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def _block(table: _Table, split: str, rows: list[dict]) -> None:
    """Every metric of one partition."""
    scored = [r for r in rows if r.get("true_class")]
    n = len(scored)
    table.add(split, "clips", n, "count")
    if not n:
        return
    hits = sum(_detected(r) == r["true_class"] for r in scored)
    table.ratio(split, "accuracy_4class", hits, n, ci=True)
    falls = [r for r in scored if r["true_class"] in FALL_CLASSES]
    table.ratio(split, "falls_exact_class", sum(_detected(r) == r["true_class"] for r in falls),
                len(falls))
    f1s = []
    for cls in FOUR_CLASSES:
        truth = [r for r in scored if r["true_class"] == cls]
        if not truth:
            continue
        tp = sum(_detected(r) == cls for r in truth)
        fp = sum(_detected(r) == cls and r["true_class"] != cls for r in scored)
        table.ratio(split, "class_accuracy", tp, len(truth), cls)
        f1 = _f1(tp, fp, len(truth) - tp)
        table.add(split, "class_f1", 100 * f1, "%", cls)
        f1s.append(f1)
    table.add(split, "macro_f1_4class", 100 * statistics.fmean(f1s) if f1s else math.nan, "%")

    b = binary_metrics(scored)
    tp, fp, tn, fn = b["tp"], b["fp"], b["tn"], b["fn"]
    table.ratio(split, "sensitivity", tp, tp + fn, ci=True)
    table.ratio(split, "specificity_label", tn, tn + fp, ci=True)
    table.ratio(split, "precision_label", tp, tp + fp)
    table.add(split, "f1_label", 100 * b["f1"], "%")
    sens, spec = b["sensitivity"], b["specificity"]
    table.add(split, "balanced_accuracy", 100 * (sens + spec) / 2, "%")
    for key, value in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
        table.add(split, key, value, "count")
    table.add(split, "false_pos_per_hour_label", fp_per_hour(scored), "/h")
    table.add(split, "undetermined", sum(_detected(r) == "Undetermined" for r in scored),
              "count")

    # The reading by alarm, when the rows say which clips raised one.
    if any(str(r.get("alarm", "")) != "" for r in scored):
        def alarmed(group):
            return sum(str(r.get("alarm")) == "1" for r in group)
        help_falls = [r for r in scored if r["true_class"] in NEED_HELP]
        table.ratio(split, "help_falls_with_alarm", alarmed(help_falls), len(help_falls))
        for cls in (PARTIALLY_RECOVERED, NOT_RECOVERED, RECOVERED):
            group = [r for r in scored if r["true_class"] == cls]
            if group:
                table.ratio(split, "class_with_alarm", alarmed(group), len(group), cls)
        table.ratio(split, "falls_with_alarm", alarmed(falls), len(falls))
        nofall = [r for r in scored if r["true_class"] == NO_FALL]
        false_alarms = alarmed(nofall)
        table.ratio(split, "specificity_alarm", len(nofall) - false_alarms, len(nofall),
                    ci=True)
        all_alarms = alarmed(scored)
        table.ratio(split, "alarm_precision", all_alarms - false_alarms, all_alarms)
        seconds = sum(float(r.get("duration_s") or 0.0) for r in nofall)
        table.add(split, "false_pos_per_hour_alarm",
                  false_alarms / (seconds / 3600.0) if seconds else math.nan, "/h")

    sources = Counter(s for r in scored for s in str(r.get("trigger_sources") or "").split(";")
                      if s)
    table.add(split, "events", sum(int(r.get("events") or 0) for r in scored), "count")
    for source, count in sorted(sources.items()):
        table.add(split, "events_by_trigger", count, "count", source)
    for truth in FOUR_CLASSES:
        for det in (*FOUR_CLASSES, "Undetermined"):
            count = sum(r["true_class"] == truth and _detected(r) == det for r in scored)
            if count:
                table.add(split, "confusion", count, "count", f"{truth}->{det}")


def performance(frame_files) -> dict[str, float]:
    """Per-frame cost over a run's frame records (schema 2 columns)."""
    cols = {"decode_ms": [], "pose_ms": [], "analysis_ms": []}
    total, cpu, rss = [], [], []
    for path in frame_files:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                parts = []
                for key, bucket in cols.items():
                    if row.get(key):
                        bucket.append(float(row[key]))
                        parts.append(float(row[key]))
                if parts:
                    total.append(sum(parts))
                if row.get("cpu_pct"):
                    cpu.append(float(row["cpu_pct"]))
                if row.get("peak_rss_mb"):
                    rss.append(float(row["peak_rss_mb"]))
    if not total:
        return {}
    out = {"frames_measured": float(len(total)),
           "frame_ms_median": statistics.median(total), "frame_ms_p95": _p95(total)}
    med = out["frame_ms_median"]
    out["fps_median"] = 1000.0 / med if med > 0 else math.nan
    for key, bucket in cols.items():
        if bucket:
            out[f"{key[:-3]}_ms_median"] = statistics.median(bucket)
            out[f"{key[:-3]}_ms_p95"] = _p95(bucket)
    if cpu:
        out["cpu_pct_mean"] = statistics.fmean(cpu)
    if rss:
        out["peak_rss_mb"] = max(rss)
    return out


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


_PERF_UNITS = {"frames_measured": "count", "fps_median": "fps", "cpu_pct_mean": "%",
               "peak_rss_mb": "MB"}


def compute(rows, partition: dict[str, str] | None = None,
            perf: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """The metrics table of a run, from its label rows."""
    partition = partition or {}
    table = _Table()
    groups: dict[str, list] = {s: [] for s in SPLITS}
    for row in rows:
        groups["all"].append(row)
        groups[_split_of(row, partition)].append(row)
    for split in SPLITS:
        if split == "all" or groups[split]:
            if split == "unassigned" and not partition:
                continue
            _block(table, split, groups[split])
    for key, value in (perf or {}).items():
        table.add("all", key, value, _PERF_UNITS.get(key, "ms"))
    return table.rows


# --------------------------------------------------------------------- text --
_LINES = (
    ("Classification (4 classes)", None),
    ("accuracy_4class", "Accuracy, 4 classes"),
    ("falls_exact_class", "Falls with the exact class"),
    ("macro_f1_4class", "F1 averaged over the 4 classes"),
    ("class_accuracy", "Accuracy: {cls}"),
    ("undetermined", "Undetermined"),
    ("Fall or not, by label", None),
    ("sensitivity", "Falls detected (sensitivity)"),
    ("specificity_label", "Specificity (label)"),
    ("precision_label", "Precision"),
    ("f1_label", "F1"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("tp·fp·tn·fn", "TP · FP · TN · FN"),
    ("false_pos_per_hour_label", "False positives per hour (label)"),
    ("Fall or not, by alarm", None),
    ("help_falls_with_alarm", "Falls that need help, with alarm"),
    ("class_with_alarm", "{cls} with alarm"),
    ("falls_with_alarm", "All falls with alarm"),
    ("specificity_alarm", "Specificity (alarm)"),
    ("alarm_precision", "Alarm precision"),
    ("false_pos_per_hour_alarm", "False positives per hour (alarm)"),
    ("Events", None),
    ("events", "Events"),
    ("events_by_trigger", "Raised by {cls}"),
)


def _cell(row: dict | None) -> str:
    if row is None or row["value"] == "":
        return "—"
    unit = row["unit"]
    text = row["value"] + ({"%": " %", "/h": " /h"}.get(unit, ""))
    if row["num"] != "" and row["den"] != "":
        text += f" ({row['num']}/{row['den']})"
    if row["ci_low"]:
        text += f" [{row['ci_low']}–{row['ci_high']}]"
    return text


def format_text(metric_rows, label_rows=None, title: str = "") -> str:
    """The table as a report a person reads: one column per partition."""
    splits = [s for s in SPLITS if any(r["split"] == s for r in metric_rows)]
    index = {(r["split"], r["metric"], r["class"]): r for r in metric_rows}
    clips = {s: index.get((s, "clips", "")) for s in splits}
    head = ["", *[f"{s.upper()} ({clips[s]['value'] if clips[s] else 0})" for s in splits]]
    body: list[list[str]] = []
    for metric, label in _LINES:
        if label is None:
            body.append([f"-- {metric} --"] + [""] * len(splits))
            continue
        if metric == "tp·fp·tn·fn":
            cells = []
            for s in splits:
                vals = [index.get((s, k, "")) for k in ("tp", "fp", "tn", "fn")]
                cells.append(" · ".join(v["value"] for v in vals) if all(vals) else "—")
            body.append([label, *cells])
            continue
        classes = sorted({r["class"] for r in metric_rows if r["metric"] == metric})
        for cls in classes:
            if not any(index.get((s, metric, cls)) for s in splits):
                continue
            body.append([label.format(cls=cls), *[_cell(index.get((s, metric, cls)))
                                                  for s in splits]])
    widths = [max(len(r[i]) for r in [head, *body]) for i in range(len(head))]
    lines = [title] if title else []
    if label_rows is not None:
        lines += [format_confusion(label_rows), ""]
    lines.append("  ".join(c.ljust(w) for c, w in zip(head, widths, strict=True)).rstrip())
    for r in body:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())
    perf = [r for r in metric_rows if r["split"] == "all" and r["metric"] in _PERF_ORDER]
    if perf:
        lines += ["", "-- Performance (per frame) --"]
        for r in sorted(perf, key=lambda r: _PERF_ORDER.index(r["metric"])):
            unit = {"count": "", "%": " %"}.get(r["unit"], f" {r['unit']}")
            lines.append(f"{_PERF_NAMES[r['metric']]:<28}{r['value']}{unit}")
    return "\n".join(lines) + "\n"


_PERF_NAMES = {
    "fps_median": "Frames per second (median)",
    "frame_ms_median": "Frame time, median",
    "frame_ms_p95": "Frame time, p95",
    "decode_ms_median": "  decode, median", "decode_ms_p95": "  decode, p95",
    "pose_ms_median": "  MediaPipe, median", "pose_ms_p95": "  MediaPipe, p95",
    "analysis_ms_median": "  analysis, median", "analysis_ms_p95": "  analysis, p95",
    "cpu_pct_mean": "CPU (100 = one core), mean",
    "peak_rss_mb": "Peak memory",
    "frames_measured": "Frames measured",
}
_PERF_ORDER = list(_PERF_NAMES)


def write(folder: str | Path, metric_rows, text: str, suffix: str = "") -> tuple[Path, Path]:
    """``metrics<suffix>.csv`` and ``.txt`` in a run folder (replaced: they are derived)."""
    folder = Path(folder)
    csv_path = folder / f"metrics{suffix}.csv"
    txt_path = folder / f"metrics{suffix}.txt"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metric_rows)
    txt_path.write_text(text, encoding="utf-8")
    return csv_path, txt_path


def for_run(run, label_rows, reviewed: bool = False) -> str:
    """Compute and save the metrics of a run folder; returns the text report.

    ``run``: a ``run_folder.RunFolder``. The partition is the one the run
    recorded; the performance comes from its frame records.
    """
    from .run_folder import CLIPS_DIR, FRAMES_FILE
    perf = performance(sorted((run.path / CLIPS_DIR).glob(f"*/{FRAMES_FILE}")))
    rows = compute(label_rows, run.info.get("partition") or {}, perf)
    title = f"{run.path.name}" + ("  (with the human review)" if reviewed else "")
    text = format_text(rows, label_rows, title)
    write(run.path, rows, text, "-reviewed" if reviewed else "")
    return text
