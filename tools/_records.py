"""Finding the records of a run: one per-frame CSV per clip, plus its companions.

A run folder (``logs/...``) holds, per clip, the records ``main.py`` or
PEF-Lab wrote for it::

    A01-S1-Recovered-20260925-192230.csv            per-frame record
    A01-S1-Recovered-20260925-192230.meta.json      its sidecar (fps, config...)
    A01-S1-Recovered-events-20260925-192230.csv     Stage-1 events (absent: none)
    A01-S1-Recovered-landmarks-20260925-192230.csv  33-point skeleton (research mode)

Each record takes its timestamp when its logger is created, one after the
other, so the events and landmarks of a clip can carry a stamp a second later
than its per-frame record. They are matched to the latest per-frame record of
the same clip that is not newer than they are, not by an identical stamp.

A folder can hold the same clip twice (a batch restarted from the top). The
earliest run of each clip is used, and the extra runs are counted so a tool
can say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_RECORD = re.compile(
    r"^(?P<stem>.+?)(?P<kind>-events|-landmarks)?-(?P<stamp>\d{8}-\d{6}(?:-\d+)?)\.csv$")


@dataclass
class ClipRecords:
    """The records of one clip in one run folder."""

    clip: str                 # "A01-S1": actor and scene
    stem: str                 # "A01-S1-Recovered": the video's name
    stamp: str
    frames: Path
    events: Path | None = None
    landmarks: Path | None = None

    @property
    def meta(self) -> Path:
        """Sidecar of the per-frame record (fps, configuration, code version)."""
        return self.frames.with_suffix(".meta.json")


def clip_id(stem: str) -> str:
    """``A01-S1-Recovered`` -> ``A01-S1``: the clip, without its class."""
    return "-".join(stem.split("-")[:2])


def find_runs(folder: str | Path) -> tuple[dict[str, ClipRecords], dict[str, int]]:
    """Every clip in a run folder, and how many extra runs of each were ignored."""
    frames: dict[str, list[tuple[str, str, Path]]] = {}
    companions: list[tuple[str, str, str, Path]] = []
    for path in sorted(Path(folder).glob("*.csv")):
        if path.name.startswith("dataset-labels"):
            continue
        m = _RECORD.match(path.name)
        if not m:
            continue
        stem, kind, stamp = m.group("stem"), m.group("kind"), m.group("stamp")
        if kind is None:
            frames.setdefault(clip_id(stem), []).append((stamp, stem, path))
        else:
            companions.append((clip_id(stem), kind, stamp, path))

    runs: dict[str, list[ClipRecords]] = {
        clip: [ClipRecords(clip, stem, stamp, path) for stamp, stem, path in sorted(found)]
        for clip, found in frames.items()
    }
    for clip, kind, stamp, path in companions:
        owners = [r for r in runs.get(clip, []) if r.stamp <= stamp]
        if not owners:
            continue
        owner = owners[-1]
        if kind == "-events":
            owner.events = path
        else:
            owner.landmarks = path
    first = {clip: found[0] for clip, found in runs.items()}
    extra = {clip: len(found) - 1 for clip, found in runs.items() if len(found) > 1}
    return first, extra
