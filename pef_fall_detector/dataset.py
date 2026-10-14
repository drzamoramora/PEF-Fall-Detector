"""Reading the dataset's own labels, and writing back what the system saw.

PEF-FallDB carries its ground truth in the file name — ``A01-S1-Recovered.mp4``
— so a labelling pass can score itself without a separate annotation file.
That is convenient and fragile in equal measure, which is why the parsing
lives here, in one function, with its assumptions written down instead of
spread across the caller.

WHAT IS NORMALISED, AND WHAT IS NOT
-----------------------------------
Only ``Recovery`` -> ``Recovered``. That one is not a guess: across the 128
clips there are exactly six actors per fall class, and the ``Recovery`` actor
is what makes the third class reach six — the counts land on 24/24/24 only
under that reading.

Everything else is left exactly as it is. Two clips are numbered ``S14``
where their actor is missing an ``S4``, and one is ``A3-S2`` where its actor
is missing an ``S2``; both are almost certainly typos, and neither touches
the CLASS, which is the only thing this module claims to read. Silently
rewriting file names to match a theory is how a dataset acquires errors that
nobody can find later.

A name whose last token is not a known class yields ``""`` — unknown truth,
which scores as neither right nor wrong. Never a guess, and never
``NoFall``: "I cannot read this name" and "this clip has no fall" are
different statements.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from .classification import (
    CLASSES,
    NO_FALL,
    NOT_RECOVERED,
    PARTIALLY_RECOVERED,
    RECOVERED,
    classify_clip,
)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")

#: Spellings seen in the material, lower-cased, mapped to the canonical name.
_ALIASES = {
    "nofall": NO_FALL,
    "recovered": RECOVERED,
    "recovery": RECOVERED,          # see the module docstring: counts, not taste
    "notrecovered": NOT_RECOVERED,
    "partiallyrecovered": PARTIALLY_RECOVERED,
}

#: Column schema of ``dataset-labels.csv`` (``dataset-labels-<stamp>.csv``
#: before records schema 2): one row per clip.
#: Before 0.2.0.dev3 the columns were in Spanish (fecha, hora, duracion_s,
#: frames_confiables_pct, clase_detectada, clase_verdad, clase_revisada,
#: acierto, eventos, t_disparo_s, motivo).
LABEL_FIELDS = [
    "video",
    "date",
    "time",
    "duration_s",
    "frames",
    "reliable_frames_pct",
    "detected_class",
    "true_class",
    "reviewed_class",        # what a human says, when they disagree
    "correct",
    "events",
    "first_trigger_s",
    "verdict",
    "severity",
    "I_max_s",
    "reason",
    # Records schema 2 (0.3.0.dev0), appended so older readers keep working:
    "alarm",                 # 1 when an event ended in a verdict that dispatches an alert
    "trigger_sources",       # what raised each event, in order: "score;H"
    "video_sha256",          # the clip by content (run_folder.file_sha256)
]


def label_row(path, events, frames: int, reliable_frames: int, duration_s: float,
              when: datetime, dispatch=("stage3_confirmed",), video_sha256: str = "") -> dict:
    """The labels-file row of one clip, from the events its run left.

    Shared by PEF-Lab's queue and ``main.py --folder``, so both write the
    same row for the same run. The class is the one ``classify_clip`` reads
    from the events; the truth is the one the file name carries. ``dispatch``:
    the verdicts that raise an alarm (``alerts.dispatch_verdicts``), for the
    alarm column.
    """
    detected, reason = classify_clip(events)
    truth = truth_from_name(path)
    worst = events[-1] if events else None
    return {
        "video": Path(path).name,
        "date": when.strftime("%Y-%m-%d"),
        "time": when.strftime("%H:%M:%S"),
        "duration_s": f"{duration_s:.2f}",
        "frames": frames,
        "reliable_frames_pct": (f"{100.0 * reliable_frames / frames:.1f}"
                                if frames else ""),
        "detected_class": detected,
        "true_class": truth,
        "reviewed_class": "",
        "correct": ("" if not truth else int(truth == detected)),
        "events": len(events),
        "first_trigger_s": f"{events[0].timestamp:.2f}" if events else "",
        "verdict": worst.verdict if worst else "",
        "severity": worst.severity if worst else "",
        "I_max_s": (f"{worst.max_immobility_s:.2f}"
                    if worst and not math.isnan(worst.max_immobility_s)
                    else ""),
        "reason": (f"{reason} — {worst.reason}" if worst and worst.reason
                   else reason),
        "alarm": int(any(ev.verdict in tuple(dispatch) for ev in events)),
        "trigger_sources": ";".join(getattr(ev, "trigger_source", "") or "?"
                                    for ev in events),
        "video_sha256": video_sha256,
    }


def clip_id(path) -> str:
    """``A01-S1-Recovered.mp4`` -> ``A01-S1``: the clip, without its class."""
    return "-".join(Path(path).stem.split("-")[:2])


def truth_from_name(path) -> str:
    """The clip's annotated class, or ``""`` when the name does not say.

    Reads the LAST hyphen-separated token, because the middle of the name is
    not stable across the material: the same clip appears as
    ``A13-S1-Recovery`` in one generation and ``A13-S1-Kitchen-Recovered`` in
    another. Position-based parsing would read "Kitchen" as a class.
    """
    stem = Path(path).stem
    token = stem.rsplit("-", 1)[-1].strip().lower()
    return _ALIASES.get(token, "")


def video_files(folder) -> list[Path]:
    """Every clip in a folder, sorted, non-recursive.

    Non-recursive on purpose: PEF-FallDB is flat, and quietly walking
    subdirectories would let a stray copy of the material join a run without
    anybody noticing it in the count.
    """
    root = Path(folder)
    return sorted(p for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def confusion(rows) -> tuple[dict, int, int]:
    """Confusion matrix over rows that carry a readable truth.

    Returns ``(matrix, scored, correct)`` where ``matrix[truth][detected]``
    is a count. Rows with no readable truth are excluded from ``scored``
    rather than counted as failures — an unlabelled clip says nothing about
    the detector.
    """
    labels = list(CLASSES)
    matrix = {t: {d: 0 for d in labels} for t in labels}
    scored = correct = 0
    for row in rows:
        truth = row.get("true_class") or ""
        detected = row.get("reviewed_class") or row.get("detected_class") or ""
        if truth not in matrix or detected not in labels:
            continue
        matrix[truth][detected] += 1
        scored += 1
        correct += int(truth == detected)
    return matrix, scored, correct


def format_confusion(rows) -> str:
    """The matrix as text, for a status bar, a console or a note file."""
    matrix, scored, correct = confusion(rows)
    labels = [c for c in CLASSES
              if any(matrix[t][c] for t in matrix) or any(matrix[c].values())]
    if not labels:
        return "no clips with a readable truth"
    width = max(len(c) for c in labels) + 2
    head = "truth \\ detected".ljust(22) + "".join(c[:width - 1].rjust(width)
                                                     for c in labels)
    lines = [head]
    for truth in labels:
        cells = "".join(str(matrix[truth][d]).rjust(width) for d in labels)
        lines.append(truth.ljust(22) + cells)
    pct = (100.0 * correct / scored) if scored else 0.0
    lines.append(f"correct: {correct}/{scored} ({pct:.1f}%)")
    return "\n".join(lines)
