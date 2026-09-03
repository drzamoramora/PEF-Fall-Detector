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
back to the exact configuration that generated it.

**It lands where it is expected.** A relative output folder is resolved
against the repository root, not against whatever directory the program was
launched from.

The file itself is created on the first row, not on construction: opening a
source and closing it without recording anything leaves no trace.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import REPO_ROOT

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
})


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
#: they are diagnostic or display signals, and **no stage of §3.5 consumes
#: them**. Any of them entering a decision requires editing the draft first,
#: with the evidence that justified it.
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
    "P_offset",           # Quantity P: signed COM-to-support offset, torso
                          # lengths (negative = inside, positive = outside)
    "P_support_width",    # width of the support polygon, torso lengths —
                          # §3.4's "sudden contraction" signature
    "I_still_s",          # Quantity I: seconds the subject has been still
    csv_column("I_displacement"),
    "stage",              # §3.5 funnel position: MONITORING/CONFIRMING/COOLDOWN
    "stage1_fired",       # 1 on the exact frame the kinematic trigger fired
    "reliable",           # 1 = detection + core visibility above threshold;
                          # statistics/calibration must filter on this
]


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
    "verdict",
    "severity",             # §3.3 tag: mild / moderate / severe. Defined by
                            # RECOVERY, not by impact. Empty if Stage 3 never ran.
    "max_immobility_s",     # Stage-3 evidence: the longest stillness reached
    "p_outside_fraction",   # Stage-2 evidence: share of measurable frames in
    "p_samples",            # the window with the COM outside, and how many
                            # frames that share was computed from. A verdict
                            # without its evidence can only be believed.
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
            schema, the code version and a timestamp are added automatically.
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
    """

    def __init__(
        self,
        output_dir: str | Path,
        source_name: str,
        fields: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        flush_each_row: bool = False,
    ) -> None:
        self.fields = fields or PHASE2_FIELDS
        self._metadata = dict(metadata or {})
        self._flush_each_row = bool(flush_each_row)
        out = Path(output_dir).expanduser()
        if not out.is_absolute():
            out = REPO_ROOT / out
        self._output_dir = out
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name)
        self.path = out / f"{safe}-{stamp}.csv"
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._closed = False
        self.rows_written = 0

    # ------------------------------------------------------------------ core
    def _open(self) -> None:
        """Create the directory and the file. Called on the first row only."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
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
        self._write_metadata()

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
