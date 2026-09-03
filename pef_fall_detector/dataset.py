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

from pathlib import Path

from .classification import (
    CLASSES,
    NO_FALL,
    NOT_RECOVERED,
    PARTIALLY_RECOVERED,
    RECOVERED,
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

#: Column schema of ``dataset-labels-<stamp>.csv``: one row per clip.
LABEL_FIELDS = [
    "video",
    "fecha",
    "hora",
    "duracion_s",
    "frames",
    "frames_confiables_pct",
    "clase_detectada",
    "clase_verdad",
    "clase_revisada",        # what a human says, when they disagree
    "acierto",
    "eventos",
    "t_disparo_s",
    "verdict",
    "severity",
    "I_max_s",
    "motivo",
]


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
        truth = row.get("clase_verdad") or ""
        detected = row.get("clase_revisada") or row.get("clase_detectada") or ""
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
        return "sin clips con verdad legible"
    width = max(len(c) for c in labels) + 2
    head = "verdad \\ detectada".ljust(22) + "".join(c[:width - 1].rjust(width)
                                                     for c in labels)
    lines = [head]
    for truth in labels:
        cells = "".join(str(matrix[truth][d]).rjust(width) for d in labels)
        lines.append(truth.ljust(22) + cells)
    pct = (100.0 * correct / scored) if scored else 0.0
    lines.append(f"aciertos: {correct}/{scored} ({pct:.1f}%)")
    return "\n".join(lines)
