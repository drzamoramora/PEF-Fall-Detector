"""Per-frame audit record (the auditability promise of §3.5).

Every processed frame appends one CSV row. The paper claims that any alert
can be reconstructed by "reading off" the physical quantities from the
corresponding frames — this file is the artifact that makes that claim
true. It is also the raw material for calibration plots and for the figures
and tables of Section 4.

Three properties make the record trustworthy, and each is enforced here:

**Nothing is lost silently.** A value written under a column name that is
not in the schema raises immediately rather than disappearing. With the
schema growing one phase at a time, a mistyped quantity name would
otherwise produce a CSV that looks complete and is not.

**It can be read without external context.** A CSV alone cannot say which
thresholds produced it, at what frame rate it was captured, or with which
version of the code — and those are precisely the facts a calibration has
to cite. A ``.meta.json`` sidecar carries them, so any record can be traced
back to the exact configuration that generated it, and to the code: the
release (``code_version``) and the git commit (``git_commit``, ``git_dirty``).

**It lands where it is expected.** A relative output folder is resolved
against the repository root, not against whatever directory the program was
launched from.

The file itself is created on the first row, not on construction: opening a
source and closing it without recording anything leaves no trace.
"""

from __future__ import annotations

import csv
import functools
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import REPO_ROOT


@functools.lru_cache(maxsize=1)
def git_provenance() -> dict[str, Any]:
    """The git commit the code runs from, and whether the code differs from it.

    ``code_version`` says which release a record came from; this says which
    exact code, including whether it was uncommitted work (``git_dirty``).
    Both are None when the code is not a git checkout of its own (a copied
    package, a machine without git): a record says "unknown" rather than
    borrow the commit of some enclosing repository. Read once per process,
    since the code already loaded cannot change while it runs. What counts
    as "dirty" is :func:`_code_differs`.
    """
    def git(*args: str) -> str:
        # --no-optional-locks: reading the state must never take the index
        # lock that the author's own git command may be holding.
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout

    unknown = {"git_commit": None, "git_dirty": None}
    try:
        if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != REPO_ROOT.resolve():
            return unknown
        commit = git("rev-parse", "HEAD").strip()
        # Not stripped: in porcelain output a leading space is part of the
        # status code of the first entry.
        dirty = _code_differs(git("status", "--porcelain", "-z", "--untracked-files=all"))
    except (OSError, subprocess.SubprocessError):
        return unknown
    return {"git_commit": commit, "git_dirty": dirty}


def _code_differs(porcelain: str) -> bool:
    """Whether ``git status --porcelain -z`` says the running code is not HEAD.

    Any change to a tracked file counts. An untracked file counts only when it
    is code the detector runs: a ``.py`` file inside the package (imported the
    moment something names it, committed or not; B4 splits ``state_machine``
    into new modules), or an untracked ``main.py`` / ``config.yaml``. New
    notes, tools, tests or a Finder ``.DS_Store`` do not change what the
    detector decides, so they leave a record clean.
    """
    for entry in porcelain.split("\0"):
        if not entry:
            continue
        status, path = entry[:2], entry[3:]
        if status != "??":
            # Any tracked change, a rename included, answers at once; so the
            # old path that -z lists after a rename is never read as an entry.
            return True
        if (path.startswith("pef_fall_detector/") and path.endswith(".py")) \
                or path in ("main.py", "config.yaml"):
            return True
    return False


#: Phase-1 schema — pose front-end and Step-0 normalization.
PHASE1_FIELDS = [
    "frame_index",
    "timestamp_s",
    "detected",
    "core_visibility",
    "torso_length_px",
    "mid_hip_x_px",
    "mid_hip_y_px",
    "mid_shoulder_x_px",
    "mid_shoulder_y_px",
    "frame_width",
    "frame_height",
]

#: Suffix marking a column that is **not defined in the paper**.
#:
#: The record mixes two kinds of measurement, and a reader cannot tell them
#: apart from the numbers alone: quantities the paper defines and commits to
#: (§3.4), and signals we added to evaluate whether they earn a place there.
#: Publishing a table where both look alike invites a reviewer — or a
#: co-author — to read an exploratory column as an established result.
#: The suffix makes the distinction travel with the data itself, so it
#: survives being opened in a spreadsheet with no README in sight.
EXPERIMENTAL_SUFFIX = "_EXP"

#: Internal quantity names that are recorded under an ``_EXP`` column.
#:
#: Deliberately NOT everything we invented. ``centroid_*`` is absent because
#: §3.4 defines the centroid — we only smooth it (the smoothing itself is
#: paper divergence P3, a text fix, not a new quantity). ``reliable`` and
#: ``core_visibility`` are absent because they are record hygiene, not
#: measurements under evaluation; marking them would blur what the suffix
#: means. When one of these graduates into the paper, delete its entry here
#: and the column loses the suffix — that one-line edit is the whole
#: migration, which is why the mapping lives at the record boundary instead
#: of in the names the code uses internally.
EXPERIMENTAL_QUANTITIES = frozenset({
    "Vh_tps",             # horizontal velocity — §3.4 defines V as vertical only
    "world_torso_len_m",  # metric torso from pose_world_landmarks (3D diagnostic)
    "extension_ratio",    # hip-to-ankle extent / torso (posture feature, 2.4)
    "state",              # person-state display label (person_state.py)
    "I_displacement",     # how far the subject has drifted from the stillness
                          # anchor. Not a §3.4 quantity: it is the internal
                          # term I is thresholded on, recorded because ε
                          # cannot be calibrated without seeing it.
    "H_ratio",            # Quantity H (not in the paper): body height vs. the
                          # subject's own calibrated standing baseline.
                          # Proposed to cover two gaps measured in this
                          # project's footage — P unmeasurable when feet are
                          # occluded, T falsely low when a fall's rotation
                          # axis points at the camera. See quantities.h.
    "H_raw",              # H's numerator/denominator ratio BEFORE baselining
                          # (vertical spine extent / shoulder width). Recorded
                          # so a calibration pass can see the raw signal
                          # independently of how the baseline behaved.
    "H_baseline",         # H's personal baseline, as currently calibrated.
                          # NaN until the subject has been seen confidently
                          # standing; recorded to see the baseline converge.
    "reach_proximity",    # Quantity R (not in §3.4): smallest wrist-to-ankle
                          # distance, torso lengths. Proposed as the signal
                          # that can tell a deliberate bend (tying a shoe)
                          # apart from a fall — T/V/P/I/H all read posture,
                          # this is the only one that reads the hands. Logged
                          # only; not wired into any §3.5 stage yet.
    "A_head",             # Quantity A (phase 8, not in the paper, in tension
                          # with §3.2): head height above the ankles along
                          # gravity, from the 3D world landmarks, over the
                          # same subject's standing height. ~1 standing, ~0
                          # on the floor. NaN until calibrated. Read by
                          # Stage 3 (8.2/8.3) and Stage 1 (8.4).
    "A_hip",              # the same for the hips (sitting on a sofa vs on
                          # the floor). Logged only.
})


#: Per-frame skeleton record, as text (EXPERIMENTAL, research mode only).
#:
#: Every one of the 33 MediaPipe landmarks, in image pixels (x, y, and the
#: z MediaPipe estimates on the same pixel scale) with its visibility, plus
#: the 33 metric world landmarks (meters, hip-centred) of the same inference.
#: The decision pipeline reads none of this; it exists so that
#: a candidate measurement can be evaluated on every clip, on every platform,
#: without re-running MediaPipe. Written to its own file, never into the frame
#: record, and only in research mode (``records.RunRecords``): §3.6 promises
#: that the deployed device stores no landmark coordinates.
#:
#: Exact since records schema 2 (0.3.0.dev0): every value is written with
#: ``repr``, the shortest text that reads back as the same float, so the CSV
#: and the binary ``landmarks/npz`` hold the same numbers. Schema 1 rounded
#: them (0.01 px, 0.001, 0.1 mm). MediaPipe's presence score is not recorded:
#: the ``mp.solutions.pose`` API pinned here leaves it at 0 for every landmark
#: (measured on the 128 clips, 14/10/2026), so it carried no information.
LANDMARK_FIELDS: list[str] = (
    ["frame_index", "timestamp_s", "detected"]
    + [f"lm{i:02d}_{c}" for i in range(33) for c in ("x", "y", "z", "v")]
    + [f"w{i:02d}_{c}" for i in range(33) for c in ("x", "y", "z")]
)


def _exact(value) -> str:
    """The shortest text that reads back as the same float; blank for NaN."""
    value = float(value)
    return "" if value != value else repr(value)


def landmark_row(pf) -> dict[str, Any]:
    """The :data:`LANDMARK_FIELDS` row for one PoseFrame (blank if undetected)."""
    row: dict[str, Any] = {
        "frame_index": pf.frame_index,
        "timestamp_s": f"{pf.timestamp:.4f}",
        "detected": int(pf.detected),
    }
    if not pf.detected or len(pf.landmarks) != 33:
        return row
    for i in range(33):
        x, y, z = pf.landmarks[i][:3]
        row[f"lm{i:02d}_x"] = _exact(x)
        row[f"lm{i:02d}_y"] = _exact(y)
        row[f"lm{i:02d}_z"] = _exact(z)
        row[f"lm{i:02d}_v"] = _exact(pf.visibility[i])
    if len(pf.world_landmarks) == 33:
        for i in range(33):
            wx, wy, wz = pf.world_landmarks[i][:3]
            row[f"w{i:02d}_x"] = _exact(wx)
            row[f"w{i:02d}_y"] = _exact(wy)
            row[f"w{i:02d}_z"] = _exact(wz)
    return row


def csv_column(name: str) -> str:
    """The column an internal quantity is recorded under.

    The single point where the ``_EXP`` convention is applied. Code upstream
    keeps its own vocabulary; only the record carries the mark.
    """
    return name + EXPERIMENTAL_SUFFIX if name in EXPERIMENTAL_QUANTITIES else name


#: Phase-2 schema — adds the physics-informed quantities of §3.4 as they are
#: implemented. Columns are appended, never reordered, so CSVs from earlier
#: runs stay readable by the same tooling.
#:
#: Columns ending in ``_EXP`` are experimental (see ``EXPERIMENTAL_SUFFIX``):
#: signals the paper does not define. Most are diagnostic only; two
#: (``H_ratio_EXP``, indirectly ``H_raw_EXP``/``H_baseline_EXP``) ARE wired
#: into Stage1Trigger/Stage3Evaluator as OR'd fallbacks, gated by the
#: ``experimental.*`` thresholds in config.yaml — see the ``quantities``
#: package docstring. The mark is about the QUANTITY'S STATUS in the paper, not
#: about whether today's config happens to consume it.
PHASE2_FIELDS = PHASE1_FIELDS + [
    "T_deg",              # Quantity T: trunk inclination angle (§3.4)
    csv_column("world_torso_len_m"),
    "centroid_x_px",      # smoothed centroid (EMA), the point behind V
    "centroid_y_px",
    "V_tps",              # Quantity V: vertical centroid velocity, torso/s
                          # (negative = downward; NaN while window fills)
    csv_column("Vh_tps"),
    csv_column("extension_ratio"),
    csv_column("state"),
    "trigger_score",      # what Stage 1 compares against trigger_score (§3.5's
                          # "instantaneous combination of the two quantities"):
                          # T/threshold_T + V/threshold_V over the peak window.
                          # Empty when the formulation is not "score" or V is
                          # not downward (the trigger does not evaluate it then).
    "P_offset",           # Quantity P: signed COM-to-support offset, torso
                          # lengths (negative = inside, positive = outside)
    "P_support_width",    # width of the support polygon, torso lengths —
                          # §3.4's "sudden contraction" signature
    "I_still_s",          # Quantity I: seconds the subject has been still
    "still_fraction",     # share of Stage-3 frames with I >= persistent_still_s
                          # so far (phase 3, §3.5 "if the immobility persists");
                          # empty outside Stage 3.
    csv_column("I_displacement"),
    csv_column("H_ratio"),
    csv_column("H_raw"),
    csv_column("H_baseline"),
    csv_column("reach_proximity"),
    csv_column("A_head"),
    csv_column("A_hip"),
    "stage",              # §3.5 funnel position: MONITORING/CONFIRMING/COOLDOWN
    "stage1_fired",       # 1 on the exact frame the kinematic trigger fired
    "reliable",           # 1 = detection + core visibility above threshold;
                          # statistics/calibration must filter on this
]


#: Records schema 2 (0.3.0.dev0): what a frame's record adds to PHASE2_FIELDS.
#:
#: ``frame_sha1``: the first 16 hex digits of the SHA-1 of the decoded image
#: MediaPipe received. Two computers that disagree on a frame's landmarks
#: either decoded different images (the hashes differ: the video decoder) or
#: ran the same image through the model differently (the hashes agree: the
#: inference). Without it the two causes cannot be told apart.
PROVENANCE_FIELDS = ["frame_sha1"]

#: The computer's side of a frame: milliseconds spent reading it from the
#: source, in MediaPipe and in our own analysis, and once per second the CPU
#: use (100 = one core busy) and the peak memory of the process. They measure
#: the machine, not the scene: two runs never agree on them, and the
#: comparison tools never read them.
PERFORMANCE_FIELDS = ["decode_ms", "pose_ms", "analysis_ms", "cpu_pct", "peak_rss_mb"]

#: The frame record of a schema-2 run (``clips/<clip>/frames.csv``).
FRAME_FIELDS = PHASE2_FIELDS + PROVENANCE_FIELDS + PERFORMANCE_FIELDS


#: Event-record schema (§3.5). One row per Stage-1 firing, as opposed to the
#: per-frame CSV: an alarm is reconstructed from a handful of events, not by
#: scanning thousands of frames. It carries the quantity values that caused
#: the firing and the formulation that was in force, so a record stays
#: interpretable after the configuration changes.
EVENT_FIELDS = [
    "event_index",
    "frame_index",
    "timestamp_s",
    "T_deg",
    "V_tps",
    "formulation",
    "trigger_source",       # which Stage-1 condition raised the event: the
                            # formulation's name, or an EXPERIMENTAL fallback,
                            # "H" (H+V) or "A" (A drop, phase 8.4).
    "verdict",
    "severity",             # §3.3 tag: mild / moderate / severe. Defined by
                            # RECOVERY, not by impact. Empty if Stage 3 never ran.
    "max_immobility_s",     # Stage-3 evidence: the longest stillness reached
    "p_outside_fraction",   # Stage-2 evidence: share of measurable frames in
    "p_samples",            # the window with the COM outside, and how many
                            # frames that share was computed from. A verdict
                            # without its evidence can only be believed.
    csv_column("H_ratio"),  # Quantity H (EXPERIMENTAL) at the firing frame —
                            # recorded even when it was NOT what fired, so a
                            # reviewer can tell whether Stage1Trigger's H+V
                            # fallback or T/V's own formulation was responsible.
    "reason",               # the rule that decided the verdict, with the numbers
                            # it read (TriggerEvent.reason). §3.5: "a downstream
                            # reviewer can reconstruct the decision".
]


class AuditLogger:
    """Streams per-frame rows to a CSV, with a metadata sidecar.

    Args:
        output_dir: directory for the records. A relative path is resolved
            against the repository root (see module docstring).
        source_name: identifier of the video or camera, used in the filename.
        fields: column schema; defaults to :data:`PHASE2_FIELDS`.
        metadata: facts about the run — source, frame rate, frame size, and
            ideally a snapshot of the configuration — written to
            ``<record>.meta.json`` when the first row is recorded. The
            schema, the code version, the git commit and a timestamp are
            added automatically.
        flush_each_row: push every row to the operating system as it is
            written, instead of letting Python's buffer fill first. Off by
            default because a per-frame record writes tens of rows a second
            and the buffer is exactly the right tool for that. **On for the
            event record**, where the argument reverses: events are rare, so
            the buffer may hold the only copy of a detection for minutes, and
            §3.6 deploys this on a device that can lose power without
            warning. Measured on this project's own material: a session that
            ended without a clean close left a 0-byte event file that had, in
            fact, recorded a confirmed fall. A flush is one ``write`` syscall
            — not an ``fsync`` — so the cost on a handful of rows is nil.
        filename: an exact file name inside ``output_dir``, instead of
            ``<source>-<stamp>.csv`` (records schema 2, ``run_folder``). An
            existing file of that name is refused, never overwritten.
        sidecar: write the ``.meta.json`` (schema 1). Schema 2 keeps those
            facts once per run in ``run.json`` and once per clip in
            ``clip.json``, so its records go without.
        append: with ``filename``, add rows to the file if it exists (a
            resumed run's labels), provided its columns are the same.
    """

    def __init__(
        self,
        output_dir: str | Path,
        source_name: str,
        fields: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        flush_each_row: bool = False,
        filename: str | None = None,
        sidecar: bool = True,
        append: bool = False,
    ) -> None:
        self.fields = fields or PHASE2_FIELDS
        self._metadata = dict(metadata or {})
        self._flush_each_row = bool(flush_each_row)
        out = Path(output_dir).expanduser()
        if not out.is_absolute():
            out = REPO_ROOT / out
        self._output_dir = out
        self._exact_name = filename is not None
        self._sidecar = bool(sidecar)
        self._append = bool(append) and filename is not None
        if filename is not None:
            self.path = out / filename
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name)
            self.path = out / f"{safe}-{stamp}.csv"
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._closed = False
        self.rows_written = 0

    # ------------------------------------------------------------------ core
    def _open(self) -> None:
        """Create the directory and the file. Called on the first row only.

        Names carry the time to the second, so two records of the same
        source opened within one second used to share a name — and ``"w"``
        mode silently truncated the first. Re-analysing a short clip is
        enough to do it. The later record takes a numbered name instead;
        nothing already on disk is ever overwritten.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if self._append and self.path.exists():
            # A resumed run adds its rows to the file it started; only a file
            # with the very same columns, so a row never lands under the
            # wrong header.
            with open(self.path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])
            if header != self.fields:
                raise ValueError(f"cannot append to {self.path}: its columns differ")
            self._fh = open(self.path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fields,
                                          extrasaction="raise")
            return
        if self._exact_name and self.path.exists():
            # A schema-2 record lives in its clip's own folder under a fixed
            # name: finding one there means a run is being written twice.
            raise FileExistsError(f"record already exists: {self.path}")
        if self.path.exists():
            stem, n = self.path.stem, 2
            while (candidate := self.path.with_name(f"{stem}-{n}.csv")).exists():
                n += 1
            self.path = candidate
        self._fh = open(self.path, "x", newline="", encoding="utf-8")
        # extrasaction="raise": an unknown column name is a bug, and a silent
        # drop would produce a record that looks complete but is not.
        self._writer = csv.DictWriter(
            self._fh, fieldnames=self.fields, extrasaction="raise"
        )
        self._writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        """Append one frame's data. Unknown keys raise; missing ones are blank."""
        if self._closed:
            # Not a defensive nicety: ``_open`` opens in "w" mode, so a write
            # after close would silently TRUNCATE a finished record and start
            # it over — losing a completed session to what looks like a
            # harmless late call. Failing loudly is the only safe answer.
            raise RuntimeError(f"record already closed: {self.path}")
        if self._writer is None:
            self._open()
        self._writer.writerow(row)
        self.rows_written += 1
        if self._flush_each_row:
            self._fh.flush()

    def log_pose_frame(self, pf, extra: dict[str, Any] | None = None) -> None:
        """Build and append the row for a PoseFrame.

        Args:
            pf: the frame's pose data (front-end output).
            extra: quantities computed downstream of the front-end
                (T, V, P, I, person state...), merged into the row.
        """
        row: dict[str, Any] = {
            "frame_index": pf.frame_index,
            "timestamp_s": f"{pf.timestamp:.4f}",
            "detected": int(pf.detected),
            "frame_width": pf.frame_size[0],
            "frame_height": pf.frame_size[1],
        }
        if pf.detected:
            row.update(
                core_visibility=f"{pf.core_visibility:.3f}",
                torso_length_px=f"{pf.torso_length:.2f}",
                mid_hip_x_px=f"{pf.mid_hip[0]:.2f}",
                mid_hip_y_px=f"{pf.mid_hip[1]:.2f}",
                mid_shoulder_x_px=f"{pf.mid_shoulder[0]:.2f}",
                mid_shoulder_y_px=f"{pf.mid_shoulder[1]:.2f}",
            )
        if extra:
            row.update(extra)
        self.log(row)

    # --------------------------------------------------------------- closing
    def _write_metadata(self) -> None:
        payload = dict(self._metadata)
        payload.update(
            schema=self.fields,
            rows=self.rows_written,
            code_version=__version__,
            **git_provenance(),
            written_at=datetime.now().isoformat(timespec="seconds"),
        )
        side = self.path.with_suffix(".meta.json")
        side.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def close(self) -> None:
        """Flush the record and write its metadata. A no-op if nothing was logged."""
        self._closed = True
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None
        self._writer = None
        if self._sidecar:
            self._write_metadata()

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
