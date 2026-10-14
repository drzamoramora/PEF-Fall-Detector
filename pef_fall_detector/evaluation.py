"""The metrics of §3.7, and the partition that makes them mean something.

Qt-free and with no dependency outside the standard library: the GUI, the
headless runner and the tests all consume this same code, so a number
reported in Section 4 and one seen on screen cannot differ.

WHY ACCURACY IS NOT ENOUGH
--------------------------
§3.7 asks for *"accuracy, sensitivity on the fall class, specificity, precision,
and F1"* plus *"the false-positive rate per hour of non-fall footage"*. That is
not bureaucracy: accuracy over four classes mixes two different clinical
questions and shows neither.

A system that confuses severities but catches every fall **works** — someone
comes to help. One that gets severities right and misses one fall in five
**does not**, however good its accuracy. Both can give the same number, and
only the binary sensitivity tells them apart.

THE UNDETERMINED CASE
---------------------
An event that fired but that no stage could judge counts as NEGATIVE in the
binary metrics, because that is what the deployed system does:
`dispatch_verdicts` only dispatches `stage3_confirmed`, so an undetermined
event raises no alarm and nobody comes. Counting it as positive would measure
an intention instead of a consequence.

But it is reported separately and always visible. A system that cannot judge
and one that judges there was no fall are different failure modes, with
different causes and fixes, and collapsing them hides exactly the information
needed to fix them.

CONFIDENCE INTERVALS
--------------------
The test partition has 40 clips: **one clip is worth 2.5 points**, and a 90 %
carries an interval of roughly ±9. Reporting the bare point invites reading a
4-point difference as an improvement when it is noise. The Wilson interval is
used, which does not break near 0 % or 100 % as the normal approximation does.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from .classification import (
    NO_FALL,
    NOT_RECOVERED,
    PARTIALLY_RECOVERED,
    RECOVERED,
    UNDETERMINED,
)

#: The three classes that are a fall. `UNDETERMINED` is left out on purpose:
#: see the module docstring.
FALL_CLASSES = frozenset({RECOVERED, PARTIALLY_RECOVERED, NOT_RECOVERED})

#: Naming typos that do not change whose clip it is. `A3-S2` belongs to A13:
#: that actor has S1, S3 and S4, and this one completes the four everyone has.
_ACTOR_ALIASES = {"A3": "A13"}


def actor_of(video: str) -> str:
    """The actor a clip belongs to, from its file name."""
    stem = Path(video).stem
    head = stem.split("-", 1)[0].strip()
    return _ACTOR_ALIASES.get(head, head)


def load_split(path) -> dict[str, str]:
    """Read `config/partition.yaml` without depending on a YAML parser.

    The file is deliberately simple — two lists of strings — and reading it
    with a regular expression avoids adding PyYAML to an edge deployment for
    a twenty-line file.
    """
    side, out = None, {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r"^(test|train)\s*:", line):
            side = line.split(":", 1)[0].strip()
            continue
        item = re.match(r"^\s*-\s*(\S+)", line)
        if item and side:
            out[item.group(1)] = side
    return out


def split_rows(rows, split: dict[str, str]) -> dict[str, list]:
    """Split label rows into train / test / unassigned.

    A clip whose actor is not in the partition goes to ``unassigned`` and is
    NOT given a side by default. An unknown actor is new material or a
    misspelt name, and silently assigning it to one side would contaminate
    exactly what the partition protects.
    """
    out = {"train": [], "test": [], "unassigned": []}
    for row in rows:
        out[split.get(actor_of(row.get("video", "")), "unassigned")].append(row)
    return out


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval, as fractions.

    Not the normal approximation: with 40 clips and a proportion near 1, the
    normal one gives limits above 100 %, which besides being impossible hide
    how asymmetric the real interval is at that end.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = hits / total
    d = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _detected(row) -> str:
    """The label that counts: the human-reviewed one if it exists."""
    return row.get("reviewed_class") or row.get("detected_class") or ""


def binary_metrics(rows) -> dict:
    """Fall / no fall: the clinical question, separate from severity."""
    tp = fp = tn = fn = 0
    for row in rows:
        truth = row.get("true_class") or ""
        if not truth:
            continue
        actual = truth in FALL_CLASSES
        predicted = _detected(row) in FALL_CLASSES
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    def ratio(num, den):
        return num / den if den else float("nan")

    sens = ratio(tp, tp + fn)
    prec = ratio(tp, tp + fp)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens,
        "specificity": ratio(tn, tn + fp),
        "precision": prec,
        "f1": (2 * prec * sens / (prec + sens)
               if not math.isnan(prec) and not math.isnan(sens) and (prec + sens) > 0
               else float("nan")),
        "sens_ci": wilson(tp, tp + fn),
        "spec_ci": wilson(tn, tn + fp),
    }


def per_class(rows) -> dict[str, tuple[int, int]]:
    """Correct and total per true class. Where the real problem lives."""
    out: dict[str, list[int]] = {}
    for row in rows:
        truth = row.get("true_class") or ""
        if not truth:
            continue
        cell = out.setdefault(truth, [0, 0])
        cell[1] += 1
        cell[0] += int(_detected(row) == truth)
    return {k: (v[0], v[1]) for k, v in out.items()}


def fp_per_hour(rows) -> float:
    """False positives per hour of NON-fall footage (§3.7).

    The metric that decides whether the system can be lived with at home: one
    false alarm an hour empties a carer's trust within a week, however good
    the per-clip specificity. The denominator is only the non-fall clips,
    because an alarm on a real fall is not a false alarm.
    """
    seconds = 0.0
    false_alarms = 0
    for row in rows:
        if (row.get("true_class") or "") != NO_FALL:
            continue
        try:
            seconds += float(row.get("duration_s") or 0.0)
        except ValueError:
            pass
        false_alarms += int(_detected(row) in FALL_CLASSES)
    return false_alarms / (seconds / 3600.0) if seconds > 0 else float("nan")


def undetermined_count(rows) -> int:
    return sum(1 for row in rows if _detected(row) == UNDETERMINED)


def report(rows, title: str = "") -> str:
    """All of §3.7 over a set of rows, as text."""
    scored = [r for r in rows if r.get("true_class")]
    if not scored:
        return f"{title}: no clips with a readable truth"
    hits = sum(1 for r in scored if _detected(r) == r["true_class"])
    lo, hi = wilson(hits, len(scored))
    b = binary_metrics(scored)
    lines = [
        f"{title}  ({len(scored)} clips)",
        f"  4-class accuracy  : {hits}/{len(scored)} = {100 * hits / len(scored):.1f}% "
        f"[95% CI {100 * lo:.1f}–{100 * hi:.1f}]",
        f"  binary fall/no    : sens {100 * b['sensitivity']:.1f}% "
        f"[{100 * b['sens_ci'][0]:.0f}–{100 * b['sens_ci'][1]:.0f}]   "
        f"spec {100 * b['specificity']:.1f}% "
        f"[{100 * b['spec_ci'][0]:.0f}–{100 * b['spec_ci'][1]:.0f}]   "
        f"prec {100 * b['precision']:.1f}%   F1 {100 * b['f1']:.1f}%",
        f"                      TP {b['tp']}  FP {b['fp']}  TN {b['tn']}  FN {b['fn']}",
        f"  false pos/hour    : {fp_per_hour(scored):.1f}",
        f"  undetermined      : {undetermined_count(scored)}",
    ]
    for name, (ok, total) in sorted(per_class(scored).items()):
        lines.append(f"  {name:<20}: {ok}/{total} = {100 * ok / total:.0f}%")
    return "\n".join(lines)


def pass_summary(rows, split_path) -> str:
    """The text that closes a labelling pass, in PEF-Lab and in the terminal.

    The confusion matrix of every clip, then train and test SEPARATELY: one
    number over the whole set does not say whether the adjustments
    generalise, and that is the number a reviewer will look at twice. The
    split is read only if its file exists.
    """
    from .dataset import format_confusion
    parts = [format_confusion(rows)]
    if Path(split_path).exists():
        groups = split_rows(rows, load_split(split_path))
        for side, title in (("train", "CALIBRATION"), ("test", "TEST"),
                            ("unassigned", "UNASSIGNED")):
            if groups[side]:
                parts.append(report(groups[side], title))
    return "\n\n".join(parts)
