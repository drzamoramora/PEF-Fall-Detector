# PEF-Fall-Detector

Reference implementation for the paper **"Beyond Black-Box AI: A
Physics-Informed Edge Framework for Explainable Markerless Fall Detection
Using MediaPipe Landmarks"** (ULACIT).

The system derives four named physical quantities from MediaPipe's
33-landmark skeleton — trunk inclination **T**, vertical centroid velocity
**V**, center-of-mass projection vs. support polygon **P**, and post-fall
immobility **I** (§3.4) — and confirms falls through a three-stage,
fully auditable decision logic (§3.5). No neural network is trained on the
inference path.

## Repository layout

```
main.py                      CLI entry point (GUI and headless modes)
config.yaml                  ALL tunable thresholds/windows (§3.5 calibration surface)
pef_fall_detector/
    sources.py               frame sources: video file / live camera
    pose_frontend.py         MediaPipe wrapper + Step-0 normalization (§3.2)
    audit_log.py             per-frame audit CSV (§3.5 auditability)
    overlay.py               skeleton/HUD drawing (OpenCV, GUI-independent)
    gui/lab_window.py        PEF-Lab: PySide6 visualization workbench
tests/                       unit tests with synthetic skeletons
notas/                       development record, in Spanish (see below)
logs/                        generated at runtime: per-frame audit CSVs (git-ignored)
```

`notas/` is versioned on purpose. It holds the phase plan (`FASES.md`), the
code-review record (`REVISION-CODIGO.md`), the findings and test catalogue
(`HALLAZGOS-PROPUESTAS-PRUEBAS.md`) and `verificar_mediciones.py`, which
regenerates every figure those documents cite. They are the evidence behind
the method claims: which findings were raised, what was measured, and how to
reproduce each number. Written in Spanish — the working language of the
project — while the code and its docstrings are in English, the paper's
language.

Design rule: the GUI is a **viewer** over a source-agnostic core. All
computation lives in importable, Qt-free modules so the same pipeline runs
in PEF-Lab, in headless batch evaluation, and on the Raspberry Pi (§3.6).

## Prerequisites

- Python 3.11 (same as PEF-Video-Tool)
- A webcam (for live mode) and/or recorded `.mp4` clips
- No internet connection is needed at runtime: the pinned MediaPipe version
  bundles its pose models inside the package (see `requirements.txt`)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**PEF-Lab (GUI):**

```bash
python main.py                                  # pick a source in the window
python main.py --video path/to/recording.mp4    # preload a clip
python main.py --camera 0                       # live camera
```

Video mode supports Play/Pause, single-frame stepping (◀/▶) and scrubbing.
Every session writes a per-frame audit CSV under `logs/`.

**Headless (batch / verification):**

```bash
python main.py --video clip.mp4 --headless --expected-seconds 20
```

Prints detection rate, torso-length stability (Step-0 sanity check) and an
FPS audit comparing the file's declared FPS against the effective capture
rate (`--expected-seconds` = the "Video seconds" value used when recording
the clip in PEF-Video-Tool).

**Tests:**

```bash
python -m unittest discover tests
```

## Outputs

Every run (GUI or headless) streams one CSV to `logs/`, named
`<source>-<datetime>.csv`, with **one row per processed frame**, plus a
`<source>-<datetime>.meta.json` sidecar carrying the source, its frame
rate, the code version and a snapshot of the full configuration — the facts
a record needs to be interpretable on its own.

This file is the system's **audit trail** — the artifact behind the paper's
claim (§3.5) that any alert can be reconstructed by reading the physical
quantities off the corresponding frames — and the raw material for the
calibration plots and the results in Section 4.

### Columns

A column ending in **`_EXP` is experimental**: a signal we record to find
out whether it deserves a place in the paper. Nothing in §3.4 defines it and
**no stage of §3.5 consumes it**. Any `_EXP` column entering a decision
requires editing the draft first, with the evidence that justified it. The
suffix travels with the data so the distinction survives a spreadsheet
opened without this README.

Pose front-end and Step-0 (§3.2):

| Column                                   | Meaning                                        |
| ---------------------------------------- | ---------------------------------------------- |
| `frame_index`                            | 0-based index within the source                |
| `timestamp_s`                            | file time for videos, wall clock for cameras   |
| `detected`                               | 1 if MediaPipe found a person                  |
| `core_visibility`                        | lowest visibility among shoulders and hips     |
| `torso_length_px`                        | mid-hip↔mid-shoulder distance, the Step-0 unit |
| `mid_hip_x_px`, `mid_hip_y_px`           | hip anchor (mean of landmarks 23, 24)          |
| `mid_shoulder_x_px`, `mid_shoulder_y_px` | shoulder anchor (mean of 11, 12)               |
| `frame_width`, `frame_height`            | source resolution, for reading pixels back     |

Physics-informed quantities (§3.4) and record hygiene:

| Column                           | Meaning                                                                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `T_deg`                          | **Quantity T** — trunk inclination vs. vertical. 0° upright, 90° horizontal, >90° shoulders below hips                                        |
| `centroid_x_px`, `centroid_y_px` | the §3.4 centroid, EMA-smoothed before differentiation                                                                                        |
| `V_tps`                          | **Quantity V** — vertical centroid velocity in torso-lengths/second. **Negative = downward.** Empty while the window fills                    |
| `reliable`                       | 1 = detected **and** core visibility above threshold. Statistics and calibration must filter on this; unreliable rows are kept, never deleted |

Experimental — under evaluation, not in the paper:

| Column                  | Meaning                                                                                         | Why we record it                                                                                                                                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Vh_tps_EXP`            | horizontal centroid velocity, positive = rightward                                              | §3.4 defines V as vertical only. Walking is sustained horizontal motion; trip-type and lateral falls carry a horizontal component. If it separates cases the vertical cannot, adding it to the paper becomes an evidence-backed proposal |
| `world_torso_len_m_EXP` | torso length in **metres**, from MediaPipe's metric 3D landmarks                                | Second output of the same inference, so it is free. Should stay ~constant at any camera distance, which makes it a better Step-0 stability check than the pixel torso. The decision pipeline stays 2D monocular (§3.2)                   |
| `extension_ratio_EXP`   | signed hip→ankle vertical extent ÷ torso                                                        | The feature that separates crouching from lying — image centroid height cannot. Empty when the ankles are occluded: an occluded posture must read as _unknown_, never as an invented one. Negative means ankles above hips               |
| `state_EXP`             | `STANDING`, `WALKING`, `LEANING`, `CROUCHING`, `SITTING`, `LYING`, `TRANSITION`, `NOT_DETECTED` | Display label for the HUD and for reading a record against what the subject was doing. Thresholds live in `config.yaml` under `state_display`, kept separate from the §3.5 calibration surface                                           |

## Configuration

Every tunable parameter (thresholds, window lengths, MediaPipe settings)
lives in [`config.yaml`](config.yaml), grouped by pipeline stage and
commented, including the phase in which each parameter becomes active.
Calibration never requires touching code; the YAML file _is_ the
calibration surface reported in the paper. An alternative file can be
passed with `--config`.

## Troubleshooting

- **macOS asks for camera permission (or the preview stays black):** the
  first time you run `--camera 0`, macOS prompts you to grant camera access
  to your terminal app. Approve it in _System Settings → Privacy & Security
  → Camera_, then run again.
- **`Could not open camera index 0`:** try `--camera 1` (external webcams
  often enumerate after the built-in one), and close other apps using the
  camera.
- **Moved or renamed the project folder?** Recreate the virtual environment
  (`venv/` stores absolute paths and breaks when moved).

## Related

- [PEF-Video-Tool](https://github.com/drzamoramora/PEF-Video-Tool) —
  companion recorder used to capture the test clips and the PEF-FallDB
  dataset that this detector consumes.

## Implementation status

| Phase | Scope                                                   | Status |
| ----- | ------------------------------------------------------- | ------ |
| 0     | Repo foundations, config, CLI                           | ✅     |
| 1     | Pose front-end (§3.2), Step-0, audit CSV, PEF-Lab shell | ✅     |
| 2     | Quantities T and V + calibration curves (§3.4)          | —      |
| 3     | Stage-1 kinematic trigger + state machine (§3.5)        | —      |
| 4     | Quantity P + Stage-2 geometric verification             | —      |
| 5     | Quantity I + Stage-3 immobility/severity                | —      |
| 6     | Batch evaluation harness for PEF-FallDB (§3.7)          | —      |
| 7     | Real-time polish, alerts (§3.6), Raspberry Pi port      | —      |

## Citation

The accompanying paper is under preparation; a citation entry will be added
upon publication. Until then, please do not redistribute the draft
materials associated with this repository.

## License

Released under the [MIT License](LICENSE).
