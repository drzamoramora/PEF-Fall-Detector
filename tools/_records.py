"""Finding the records of a run, in either of the two layouts.

Schema 2 (from 0.3.0.dev0): one folder per run, named
``<date>_<time>-<platform>-<dataset>``::

    run.json                              the run, once: code, platform,
                                          libraries, dataset, configuration
    dataset-labels.csv                    one row per clip (labelled runs)
    clips/A01-S1-Recovered/clip.json      the clip: source, fps, frame count
    clips/A01-S1-Recovered/frames.csv     per-frame record
    clips/A01-S1-Recovered/events.csv     Stage-1 events (absent: none)
    landmarks/npz/A01-S1-Recovered.npz    33-point skeleton, exact (binary)
    landmarks/csv/A01-S1-Recovered.csv    the same, exact, as text

Schema 1 (up to 0.2.0.dev9, ``logs-historicos/``): every record of every
clip loose in one folder, each with a timestamp and its own sidecar::

    A01-S1-Recovered-20260925-192230.csv            per-frame record
    A01-S1-Recovered-20260925-192230.meta.json      its sidecar (fps, config...)
    A01-S1-Recovered-events-20260925-192230.csv     Stage-1 events (absent: none)
    A01-S1-Recovered-landmarks-20260925-192230.csv  33-point skeleton (research mode)

In schema 1 each record takes its timestamp when its logger is created, one
after the other, so the events and landmarks of a clip can carry a stamp a
second later than its per-frame record. They are matched to the latest
per-frame record of the same clip that is not newer than they are, not by an
identical stamp. A schema-1 folder can hold the same clip twice (a batch
restarted from the top): the earliest run of each clip is used, and the extra
runs are counted so a tool can say so. A schema-2 run cannot: its folder is
never written twice.

Every tool reads runs through :func:`find_runs` and
:meth:`ClipRecords.metadata`, so it reads both layouts the same way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The two record layouts. A schema-1 folder has no ``run.json``.
SCHEMA_LOOSE = 1
SCHEMA_RUN_FOLDER = 2

RUN_FILE = "run.json"
LABELS_FILE = "dataset-labels.csv"
CLIPS_DIR = "clips"
CLIP_FILE = "clip.json"
FRAMES_FILE = "frames.csv"
EVENTS_FILE = "events.csv"
LANDMARKS_CSV_DIR = Path("landmarks") / "csv"
LANDMARKS_NPZ_DIR = Path("landmarks") / "npz"

#: Per-frame columns that measure the machine, not the scene: wall-clock
#: timings and load. Two runs of the same clip on the same computer never
#: agree on them, so no comparison reads them.
VOLATILE_COLUMNS = frozenset({"decode_ms", "pose_ms", "analysis_ms", "cpu_pct",
                              "peak_rss_mb"})

_RECORD = re.compile(
    r"^(?P<stem>.+?)(?P<kind>-events|-landmarks)?-(?P<stamp>\d{8}-\d{6}(?:-\d+)?)\.csv$")


@dataclass
class ClipRecords:
    """The records of one clip in one run folder."""

    clip: str                 # "A01-S1": actor and scene
    stem: str                 # "A01-S1-Recovered": the video's name
    stamp: str                # schema 1: the record's timestamp; schema 2: ""
    frames: Path
    events: Path | None = None
    landmarks: Path | None = None
    landmarks_npz: Path | None = None
    schema: int = SCHEMA_LOOSE
    run_dir: Path | None = None

    @property
    def meta(self) -> Path:
        """The clip's own metadata file: the sidecar (schema 1) or ``clip.json``."""
        if self.schema == SCHEMA_RUN_FOLDER:
            return self.frames.parent / CLIP_FILE
        return self.frames.with_suffix(".meta.json")

    def metadata(self) -> dict[str, Any]:
        """What the run recorded about this clip: source, fps, configuration, code.

        Schema 1 keeps it all in the clip's sidecar. Schema 2 splits it into
        ``run.json`` (once per run) and ``clip.json``; they are merged here,
        the clip's own facts taking precedence. Empty when nothing was recorded.
        """
        out: dict[str, Any] = {}
        if self.schema == SCHEMA_RUN_FOLDER and self.run_dir is not None:
            out.update(read_json(self.run_dir / RUN_FILE))
        out.update(read_json(self.meta))
        return out


def read_json(path: Path) -> dict[str, Any]:
    """A JSON object from disk; empty if the file is absent."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def clip_id(stem: str) -> str:
    """``A01-S1-Recovered`` -> ``A01-S1``: the clip, without its class."""
    return "-".join(stem.split("-")[:2])


def schema_of(folder: str | Path) -> int:
    """Which layout a run folder uses."""
    return SCHEMA_RUN_FOLDER if (Path(folder) / RUN_FILE).exists() else SCHEMA_LOOSE


def run_info(folder: str | Path) -> dict[str, Any]:
    """A schema-2 run's ``run.json``; empty for a schema-1 folder."""
    return read_json(Path(folder) / RUN_FILE)


def find_runs(folder: str | Path) -> tuple[dict[str, ClipRecords], dict[str, int]]:
    """Every clip in a run folder, and how many extra runs of each were ignored."""
    if schema_of(folder) == SCHEMA_RUN_FOLDER:
        return _find_run_folder(Path(folder)), {}
    return _find_loose(Path(folder))


def _find_run_folder(run: Path) -> dict[str, ClipRecords]:
    found: dict[str, ClipRecords] = {}
    for clip_dir in sorted((run / CLIPS_DIR).glob("*/")):
        frames = clip_dir / FRAMES_FILE
        if not frames.exists():
            continue
        stem = clip_dir.name
        events = clip_dir / EVENTS_FILE
        marks = run / LANDMARKS_CSV_DIR / f"{stem}.csv"
        npz = run / LANDMARKS_NPZ_DIR / f"{stem}.npz"
        found[clip_id(stem)] = ClipRecords(
            clip_id(stem), stem, "", frames,
            events=events if events.exists() else None,
            landmarks=marks if marks.exists() else None,
            landmarks_npz=npz if npz.exists() else None,
            schema=SCHEMA_RUN_FOLDER, run_dir=run)
    return found


def _find_loose(folder: Path) -> tuple[dict[str, ClipRecords], dict[str, int]]:
    frames: dict[str, list[tuple[str, str, Path]]] = {}
    companions: list[tuple[str, str, str, Path]] = []
    for path in sorted(folder.glob("*.csv")):
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


_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:\.dev(\d+))?")


def version_key(version: str) -> tuple[int, int, int, int]:
    """``"0.2.0.dev4"`` -> ``(0, 2, 0, 4)``; a release sorts after its dev versions."""
    m = _VERSION.match(version or "")
    if not m:
        return (0, 0, 0, 0)
    major, minor, patch, dev = m.groups()
    return (int(major), int(minor), int(patch), int(dev) if dev is not None else 10**9)
