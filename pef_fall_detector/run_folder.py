"""One folder per run: where a run's records live, and what it says about itself.

Records schema 2 (0.3.0.dev0). Every run that records anything (research
mode, PEF-Lab, a replay) gets its own folder under ``logging.output_dir``,
named after when, where and on what it ran::

    logs/2026-10-14_0043-macos-arm64-recordings/
        run.json                              the run, once (see below)
        dataset-labels.csv                    one row per clip (labelled runs)
        metrics.csv, metrics.txt              the run's numbers (labelled runs)
        clips/A01-S1-Recovered/clip.json      the clip: source, fps, its SHA-256
        clips/A01-S1-Recovered/frames.csv     per-frame record
        clips/A01-S1-Recovered/events.csv     Stage-1 events
        landmarks/npz/A01-S1-Recovered.npz    the 33 landmarks, exact, binary
        landmarks/csv/A01-S1-Recovered.csv    the same, exact, as text

WHY A FOLDER PER RUN
--------------------
Schema 1 wrote every record of every run into one flat folder. A batch
restarted from the top left the same clip twice side by side (``cli-a84``
has nine such clips), the configuration was repeated in every one of the
384 sidecars of a batch, and nothing in the files said on which computer
they had been produced. Here a run folder is created once, never reused
(:meth:`RunFolder.new` picks a free name), and the facts that are the same
for the whole run are written once, in ``run.json``.

WHAT ``run.json`` CARRIES
-------------------------
* the code: version, git commit and whether the checkout had changes;
* the platform: operating system, architecture, CPU model, Python;
* the libraries: the exact versions of MediaPipe, OpenCV, numpy, protobuf,
  and the video decoder OpenCV was built with. These are what differ
  between the Mac and the x86 cloud, and what the cross-platform diagnosis
  has to be able to read off a run;
* the dataset by CONTENT: its name, where it was seen (informative only:
  a path is valid on one computer) and a SHA-256 over the clips' own
  SHA-256s. Two runs with the same dataset hash processed the same bytes,
  whatever the paths or the file names;
* the configuration and the train/test partition in force;
* the column schemas of every record.

It is written when the run starts (a run that crashes still says what it
was) and rewritten when it ends.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import PARTITION_PATH, REPO_ROOT

SCHEMA_VERSION = 2

RUN_FILE = "run.json"
LABELS_FILE = "dataset-labels.csv"
REVIEWED_LABELS_FILE = "dataset-labels-reviewed.csv"
CLIPS_DIR = "clips"
CLIP_FILE = "clip.json"
FRAMES_FILE = "frames.csv"
EVENTS_FILE = "events.csv"
LANDMARKS_CSV_DIR = Path("landmarks") / "csv"
LANDMARKS_NPZ_DIR = Path("landmarks") / "npz"

#: Kinds of run.
DATASET = "dataset"      # every clip of a folder (main.py --folder, PEF-Lab's queue)
VIDEO = "video"          # one recorded clip
CAMERA = "camera"        # a live camera
REPLAY = "replay"        # tools/replay.py: recorded landmarks through the code again

#: Distributions whose version decides the landmarks or reads the video.
LIBRARIES = ("mediapipe", "opencv-contrib-python", "opencv-python", "numpy", "protobuf",
             "PyYAML", "PySide6", "pyqtgraph")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ----------------------------------------------------------------- platform --
def platform_slug() -> str:
    """``macos-arm64``, ``linux-x86_64``, ``linux-aarch64``: the name part of a run."""
    system = {"darwin": "macos"}.get(platform.system().lower(), platform.system().lower())
    machine = platform.machine().lower() or "unknown"
    return f"{system or 'unknown'}-{machine}"


@functools.lru_cache(maxsize=1)
def cpu_model() -> str:
    """The processor's marketing name, as the operating system reports it."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5, check=True)
            if out.stdout.strip():
                return out.stdout.strip()
        elif system == "Linux":
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            # x86: "model name"; Raspberry Pi: "Model" (the board) and no model name.
            for key in ("model name", "Model", "Hardware"):
                m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE)
                if m:
                    return m.group(1).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or ""


@functools.lru_cache(maxsize=1)
def platform_info() -> dict[str, Any]:
    """Where the run ran."""
    info: dict[str, Any] = {
        "slug": platform_slug(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    if platform.system() == "Darwin":
        info["macos"] = platform.mac_ver()[0]
    elif platform.system() == "Linux":
        try:
            info["os"] = platform.freedesktop_os_release().get("PRETTY_NAME", "")
        except OSError:
            pass
    return info


@functools.lru_cache(maxsize=1)
def library_versions() -> dict[str, Any]:
    """Exact versions of what decides the landmarks, plus OpenCV's video decoder."""
    out: dict[str, Any] = {}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    decoder = opencv_video_build()
    if decoder:
        out["opencv_video_io"] = decoder
    return out


def opencv_video_build() -> dict[str, str]:
    """The video I/O part of ``cv2.getBuildInformation()``: FFmpeg and its libraries.

    The first suspect when two computers disagree on frame 0 is the decoder:
    the OpenCV wheel bundles its own FFmpeg, built per platform.
    """
    try:
        import cv2
        text = cv2.getBuildInformation()
    except Exception:       # noqa: BLE001 - no OpenCV, or a broken one: nothing to say
        return {}
    out: dict[str, str] = {"opencv": getattr(cv2, "__version__", "")}
    in_video = False
    for line in text.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped.startswith("Video I/O:"):
            in_video = True
            continue
        if not in_video or not stripped:
            continue
        if indent <= 2:           # the next section header: the video part is over
            break
        key, _, value = stripped.partition(":")
        if key and value.strip():
            out[key.strip()] = value.strip()
    return out


# ------------------------------------------------------------------- hashes --
def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes: what a clip IS, whatever it is called."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def dataset_sha256(clip_hashes) -> str:
    """One hash for a set of clips, from their contents only.

    Sorted, so the order the clips ran in does not matter; names left out,
    so a renamed copy of the same videos gets the same hash.
    """
    digest = hashlib.sha256()
    for h in sorted(h for h in clip_hashes if h):
        digest.update(h.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------- names --
def safe_name(text: str) -> str:
    """A file-system-safe, space-free version of a name."""
    return _SAFE.sub("-", text).strip("-") or "run"


def run_name(started: datetime, dataset: str, slug: str | None = None) -> str:
    """``2026-10-14_0043-macos-arm64-recordings``."""
    return f"{started:%Y-%m-%d_%H%M}-{slug or platform_slug()}-{safe_name(dataset)}"


def _partition() -> dict[str, str]:
    if not PARTITION_PATH.exists():
        return {}
    from .evaluation import load_split
    return load_split(PARTITION_PATH)


def _schemas() -> dict[str, Any]:
    from .audit_log import EVENT_FIELDS, FRAME_FIELDS, LANDMARK_FIELDS
    from .dataset import LABEL_FIELDS
    from .landmark_arrays import NPZ_LAYOUT
    return {"frames": FRAME_FIELDS, "events": EVENT_FIELDS, "landmarks_csv": LANDMARK_FIELDS,
            "landmarks_npz": NPZ_LAYOUT, "labels": LABEL_FIELDS}


def _resolve(output_dir: str | Path) -> Path:
    out = Path(output_dir).expanduser()
    return out if out.is_absolute() else REPO_ROOT / out


class RunFolder:
    """The folder of one run, and its ``run.json``.

    Use :meth:`new` to start a run: it picks a name nobody has used. The
    constructor opens a folder by its exact path (a replay into a folder of
    its choosing, ``--resume``, the tests) and creates it when it is missing.
    Either way, a clip's records are never written twice: :meth:`clip_dir`
    refuses a clip that already has a folder.
    """

    def __init__(self, path: str | Path, *, kind: str = VIDEO, dataset: str = "",
                 seen_at: str = "", config: dict | None = None, command: str | None = None,
                 extra: dict | None = None) -> None:
        self.path = Path(path)
        existing = self.read_info(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        if existing:
            self.info = existing
            return
        now = datetime.now()
        from .audit_log import git_provenance
        self.info = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.path.name,
            "kind": kind,
            "command": command if command is not None else " ".join(sys.argv),
            "started_at": now.isoformat(timespec="seconds"),
            "finished_at": None,
            "status": "running",
            "dataset": {"name": dataset or self.path.name, "seen_at": seen_at,
                        "clips": 0, "sha256": ""},
            "code_version": __version__,
            **git_provenance(),
            "platform": dict(platform_info()),
            "libraries": dict(library_versions()),
            "config": config or {},
            "partition": _partition(),
            "schemas": _schemas(),
        }
        if extra:
            self.info.update(extra)
        self.write_info()

    @classmethod
    def new(cls, output_dir: str | Path, dataset: str, *, kind: str, seen_at: str = "",
            config: dict | None = None, command: str | None = None,
            started: datetime | None = None, extra: dict | None = None) -> RunFolder:
        """A new run folder under ``output_dir``, never one that exists."""
        parent = _resolve(output_dir)
        base = run_name(started or datetime.now(), dataset)
        path, n = parent / base, 2
        while path.exists():
            path, n = parent / f"{base}-{n}", n + 1
        return cls(path, kind=kind, dataset=dataset, seen_at=seen_at, config=config,
                   command=command, extra=extra)

    # ------------------------------------------------------------- run.json --
    @staticmethod
    def read_info(path: str | Path) -> dict[str, Any]:
        file = Path(path) / RUN_FILE
        if not file.exists():
            return {}
        data = json.loads(file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def write_info(self) -> None:
        (self.path / RUN_FILE).write_text(
            json.dumps(self.info, indent=2, default=str), encoding="utf-8")

    @property
    def config(self) -> dict[str, Any]:
        return self.info.get("config") or {}

    # ---------------------------------------------------------------- places --
    @property
    def labels_path(self) -> Path:
        return self.path / LABELS_FILE

    @property
    def reviewed_labels_path(self) -> Path:
        return self.path / REVIEWED_LABELS_FILE

    def clip_path(self, stem: str) -> Path:
        return self.path / CLIPS_DIR / safe_name(stem)

    def clip_dir(self, stem: str) -> Path:
        """Create the folder of a clip's records; refuses one that exists."""
        path = self.clip_path(stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
        return path

    def landmarks_csv_path(self, stem: str) -> Path:
        return self.path / LANDMARKS_CSV_DIR / f"{safe_name(stem)}.csv"

    def landmarks_npz_path(self, stem: str) -> Path:
        return self.path / LANDMARKS_NPZ_DIR / f"{safe_name(stem)}.npz"

    def complete_clips(self) -> set[str]:
        """Stems whose records were closed after a full pass (``clip.json`` says so)."""
        done = set()
        for clip_json in (self.path / CLIPS_DIR).glob(f"*/{CLIP_FILE}"):
            try:
                if json.loads(clip_json.read_text(encoding="utf-8")).get("complete"):
                    done.add(clip_json.parent.name)
            except (OSError, ValueError):
                continue
        return done

    def clip_hashes(self) -> list[str]:
        out = []
        for clip_json in sorted((self.path / CLIPS_DIR).glob(f"*/{CLIP_FILE}")):
            try:
                h = json.loads(clip_json.read_text(encoding="utf-8")).get("video_sha256")
            except (OSError, ValueError):
                continue
            if h:
                out.append(h)
        return out

    # --------------------------------------------------------------- closing --
    def close(self, status: str = "finished", **extra: Any) -> None:
        """Rewrite ``run.json`` with how the run ended and the dataset's hash."""
        hashes = self.clip_hashes()
        dataset = dict(self.info.get("dataset") or {})
        dataset["clips"] = len(list((self.path / CLIPS_DIR).glob(f"*/{CLIP_FILE}")))
        dataset["sha256"] = dataset_sha256(hashes) if hashes else ""
        self.info["dataset"] = dataset
        self.info["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.info["status"] = status
        self.info.update(extra)
        self.write_info()
