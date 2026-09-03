# PEF-Fall-Detector

Reference implementation for the paper **"Beyond Black-Box AI: A
Physics-Informed Edge Framework for Explainable Markerless Fall Detection
Using MediaPipe Landmarks"** (ULACIT).

The system derives four named physical quantities from MediaPipe's
33-landmark skeleton — trunk inclination **T**, vertical centroid velocity
**V**, center-of-mass projection vs. support polygon **P**, and post-fall
immobility **I** (§3.4) — and confirms falls through a three-stage,
auditable decision logic (§3.5) that ends in a severity-tagged alert (§3.3).
**No neural network is trained on the inference path.**

## Status: implemented, not yet validated

The distinction matters enough to lead with, because a repository that looks
finished is easy to mistake for one that has been shown to work.

**Implemented and unit-tested.** All four quantities, all three stages, the
severity tags and the alert seam. 174 tests on synthetic skeletons, every
batch of work put through mutation testing.

**Measured on real footage.** Almost nothing. The only empirical result the
project holds is for Stage 1: over 13 real falls, the four readings of
§3.5's trigger sentence caught 13, 12, 13 and 8 falls respectively (score,
sequential, V-alone, simultaneous — `python notas/replay_disparador.py`).
The specificity side of that table rests on four non-fall sessions and is
worth nothing until curated ADL footage exists.
**Quantities P and I, and Stages 2 and 3, have never been measured on a real
fall** — the recordings that produced the Stage-1 numbers predate those
quantities existing.

A synthetic test proves the code does what its author intended. It does not
prove that what its author intended detects falls. Section 4 of the paper is
empty, and it will stay empty until the system runs over a labelled corpus.

## Where the code and the draft disagree

The implementation currently departs from the paper draft in nine
places, tracked in `notas/`. They are of two kinds, and only the first kind
is urgent:

**The code contradicts the text.** None left. Three were closed: the `score`
trigger follows §3.5's sentence literally (a combination of both quantities,
on one frame, against a threshold, with V negative on that same frame), and
Quantity I now measures its radius around the running **mean** of the
stillness episode, as §3.4's "around their average position" asks.

**The code does things the text never mentions.** Hysteresis and cooldown,
the Stage-2 evaluation window, the `inconclusive` verdict when the feet
cannot be seen, the optional nose in Quantity I, the sign convention of P,
and the use of leg extension to separate a `mild` recovery from a `moderate`
one.

Every one of them has a measured or physical justification recorded next to
the code that implements it. None of them is a silent drift. But until the
draft is updated, **the paper and this repository describe different
systems**, and the paper is the one that has to move.

## Repository layout

```
main.py                      CLI entry point (GUI and headless modes)
config.yaml                  ALL tunable thresholds/windows (the calibration surface)
pef_fall_detector/
    sources.py               frame sources: video file / live camera
    pose_frontend.py         MediaPipe wrapper + Step-0 normalization (§3.2)
    quantities.py            the four physics-informed quantities (§3.4)
    person_state.py          posture labels for display and for reading records
    state_machine.py         the three-stage confirmation funnel (§3.5)
    alerts.py                alert construction + dispatch seam (§3.5/§3.6)
    audit_log.py             per-frame and per-event records
    pipeline.py              per-frame orchestration, shared by every entry point
    overlay.py               skeleton/HUD drawing (OpenCV, GUI-independent)
    gui/lab_window.py        PEF-Lab: PySide6 visualization workbench
tests/                       unit tests with synthetic skeletons
notas/                       development record, in Spanish (see below)
logs/                        generated at runtime: records (git-ignored)
```

Design rule: the GUI is a **viewer** over a source-agnostic core. All
computation lives in importable, Qt-free modules, so the same pipeline runs
in PEF-Lab, in headless batch processing and on the Raspberry Pi (§3.6).

`notas/` is versioned on purpose. It holds the phase plan (`FASES.md`), the
code-review record (`REVISION-CODIGO.md`), the findings and test catalogue
(`HALLAZGOS-PROPUESTAS-PRUEBAS.md`), and two scripts that regenerate the
figures those documents cite — `verificar_mediciones.py` and
`replay_disparador.py`. They are the evidence behind the method claims:
which findings were raised, what was measured, and how to reproduce each
number. Written in Spanish, the working language of the project; the code
and its docstrings are in English, the paper's language.

## Prerequisites

- Python 3.11
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

The window shows the skeleton with the trunk vector, the centroid, the
support polygon and the COM's drop line — green when it falls inside the
feet, red when outside — over four live curves (T, V, P, I) with the current
thresholds drawn on them. A firing trigger drops a red marker on all four at
once. A confirmed fall raises a colour-coded banner.

Video mode supports Play/Pause, single-frame stepping (◀/▶) and scrubbing.
Frames reached by scrubbing are deliberately **not** recorded: the CSV is
the chronicle of a continuous observation, not of what someone looked at.

**Headless (batch / verification):**

```bash
python main.py --video clip.mp4 --headless --expected-seconds 20
```

Prints the detection rate, torso-length stability (the Step-0 sanity check),
an FPS audit, every Stage-1 event with its verdict and severity, and how
many alerts were dispatched.

**Tests:**

```bash
python -m unittest discover tests
```

## Outputs

A run produces two records under `logs/`, plus a `.meta.json` sidecar
carrying the source, its frame rate, the code version and a snapshot of the
full configuration — the facts a record needs to be interpretable years
later, without which a calibration cannot cite what produced it.

- `<source>-<datetime>.csv` — **one row per processed frame**: every quantity,
  the funnel state, and whether the trigger fired.
- `<source>-events-<datetime>.csv` — **one row per Stage-1 event**: the T and
  V that caused it, the verdict of each stage that judged it, the evidence
  behind those verdicts, and the severity tag. An alert is reconstructed
  from a handful of rows here rather than by scanning thousands of frames.

These are **research instruments**, not part of the deployed system. §3.6
requires that the device on the wall persist nothing — no frames, no
landmarks, no derived values — so the edge profile runs with recording off,
and the §4 hardware measurements must be taken that way too.

### Columns

A column ending in **`_EXP` is experimental**: a signal recorded to find out
whether it deserves a place in the paper. Nothing in §3.4 defines it. The
suffix travels with the data so the distinction survives a spreadsheet
opened without this README.

The rule was that no `_EXP` signal may enter a decision before the draft
says so. **Stage 3 currently breaks it**, using `extension_ratio_EXP` to
tell a subject who stood up from one who only sat up — with the trunk angle
alone those are indistinguishable, and that is exactly the `mild`/`moderate`
boundary. It is switchable off (`stage3.severity_uses_leg_extension`) and it
is item 12 of the pending draft changes.

Pose front-end and Step-0 (§3.2):

| Column | Meaning |
|---|---|
| `frame_index` | 0-based index within the source |
| `timestamp_s` | file time for videos, wall clock for cameras |
| `detected` | 1 if MediaPipe found a person |
| `core_visibility` | lowest visibility among shoulders and hips |
| `torso_length_px` | mid-hip↔mid-shoulder distance, the Step-0 unit |
| `mid_hip_x_px`, `mid_hip_y_px` | hip anchor (mean of landmarks 23, 24) |
| `mid_shoulder_x_px`, `mid_shoulder_y_px` | shoulder anchor (mean of 11, 12) |
| `frame_width`, `frame_height` | source resolution, for reading pixels back |

Physics-informed quantities (§3.4), the funnel, and record hygiene:

| Column | Meaning |
|---|---|
| `T_deg` | **Quantity T** — trunk inclination vs. vertical. 0° upright, 90° horizontal, >90° shoulders below hips. NaN below 0.001 px of trunk |
| `centroid_x_px`, `centroid_y_px` | the §3.4 centroid, EMA-smoothed before differentiation |
| `V_tps` | **Quantity V** — vertical centroid velocity in torso-lengths/second. **Negative = downward.** Empty while the window fills |
| `P_offset` | **Quantity P** — signed horizontal distance from the hip-weighted centre of mass to the support polygon, in torso lengths. **Negative = inside** (the magnitude is the margin to the nearest edge), **positive = outside**, the §3.4 fall signature. Empty when the feet are not visible enough to define a support |
| `P_support_width` | Horizontal extent of the support polygon, in torso lengths — §3.4's second signature, "a sudden contraction... from a full two-foot footprint to a heel-only contact pair". The polygon is built only from foot landmarks **in bipedal contact** (inside the frame, and within `stage2.contact_band_torso` of the lowest such point), so a lifted foot now contracts it as §3.4 describes. Recorded, not yet consumed by any stage |
| `I_still_s` | **Quantity I** — seconds the subject has been essentially still: how long every visible head-and-torso landmark has stayed within `stage3.epsilon` torso lengths of where the stillness episode began. Resets on movement past that radius, and on any loss of the subject |
| `stage` | position in the §3.5 funnel: `MONITORING`, `CONFIRMING` (Stage 2 judging the geometry), `OBSERVING` (Stage 3 waiting on immobility or recovery), `COOLDOWN` |
| `stage1_fired` | 1 on the exact frame the kinematic trigger fired; the event itself goes to the event record |
| `reliable` | 1 = detected **and** core visibility above threshold. Statistics and calibration must filter on this; unreliable rows are kept, never deleted |

Experimental — under evaluation, not in the paper:

| Column | Meaning | Why it is recorded |
|---|---|---|
| `Vh_tps_EXP` | horizontal centroid velocity, positive = rightward | §3.4 defines V as vertical only. Walking is sustained horizontal motion; trip-type and lateral falls carry a horizontal component. If it separates cases the vertical cannot, adding it to the paper becomes an evidence-backed proposal |
| `world_torso_len_m_EXP` | torso length in **metres**, from MediaPipe's metric 3D landmarks | Second output of the same inference, so it is free. Should stay ~constant at any camera distance, which makes it a better Step-0 stability check than the pixel torso. The decision pipeline stays 2D monocular (§3.2) |
| `extension_ratio_EXP` | signed hip→ankle vertical extent ÷ torso | Separates crouching from lying, which image centroid height cannot. Empty when the ankles are occluded: an unobserved posture must read as *unknown*, never as an invented one. Negative means the ankles sit above the hips |
| `state_EXP` | `STANDING`, `WALKING`, `LEANING`, `CROUCHING`, `SITTING`, `LYING`, `TRANSITION`, `NOT_DETECTED` | Posture label for the HUD, for reading a record against what the subject was doing, and for the alert's prelude. Thresholds live under `state_display`, kept off the §3.5 calibration surface |
| `I_displacement_EXP` | how far the worst head-or-torso landmark sits from the stillness anchor, in torso lengths | The internal term `I_still_s` is thresholded on. Not a §3.4 quantity, but `stage3.epsilon` cannot be calibrated without seeing how close each episode ran to breaking |

### Event verdicts

| Verdict | Meaning |
|---|---|
| `stage2_rejected` | the COM stayed inside the support — a controlled sit-down. The funnel ends here, deliberately: a subject who sat down also lies still afterwards, and Stage 3 would otherwise rescue the false positive Stage 2 just killed |
| `stage2_inconclusive` | the feet were not measurable. Passed to Stage 3 rather than dropped: treating "I could not look" as "no fall" would turn the sensor's most common failure into silence |
| `stage3_confirmed` | immobility reached the confirmation threshold W, or the subject recovered only partially |
| `stage3_nullified` | the subject got up and held it — §3.5's "the alarm is nullified" |
| `stage3_unresolved` | the observation window expired without either outcome |
| `stage1_only` | no later stage judged it (Stage 2 disabled, or the event abandoned mid-window by a discontinuity) |

Confirmed and nullified events carry a **severity** tag per §3.3, defined by
recovery rather than by impact: `mild` (recovered alone), `moderate`
(partially recovered), `severe` (remained down).

## Alerts

A confirmed fall produces an alert, not just a log line. `alerts.py` builds
it and hands it to whatever sinks are attached; the decision logic never
knows what a sink is. Today the headless runner attaches a console sink and
PEF-Lab an on-screen banner. Phase 7 adds the §3.6 transports — local
buzzer, SMS, e-mail — by attaching more sinks, changing nothing about how
falls are decided.

The message carries the evidence, because §3.5's explainability claim is
worth least if it is dropped at the moment someone has to act on it:

```
FALL [SEVERE] | source=clip.mp4 | t=12.47s frame=374 | subject was WALKING before
  | T=82.4 deg, V=-4.10 torso/s, COM outside 75% of window, still 6.2s
```

Which verdicts alert is set by `alerts.dispatch_verdicts`. The default is
`stage3_confirmed` alone: a nullified event is the alarm being called off,
and events no stage could resolve are recorded but not dispatched. One
failing sink never silences the others — a dead SMS gateway must not swallow
the local buzzer.

## Configuration

Every tunable parameter lives in [`config.yaml`](config.yaml), grouped by
stage and commented with the evidence behind its value where one exists.
Calibration never requires touching code; the YAML file *is* the calibration
surface reported in the paper. An alternative file can be passed with
`--config`.

**All current values are provisional.** Nothing here has been calibrated
against a dataset; the numbers are physically plausible starting points, and
several carry a `PROVISIONAL` note saying so. Windows and hold times are
expressed in **seconds, never in frames** — the same "5 frames" spanned
0.08 s in one of this project's clips and 0.40 s in another, and a threshold
tuned on 30 fps footage would miss falls on a slower edge device.

Two settings exist as ablations required by §3.7 rather than as options:
`stage1.trigger_formulation` (`score` / `simultaneous` / `sequential` /
`v_only`) and
`stage1.ema_time_constant_s: 0`, which disables the smoothing that §3.4
defines as part of Quantity V.

## Troubleshooting

- **macOS asks for camera permission (or the preview stays black):** the
  first time you run `--camera 0`, macOS prompts you to grant camera access
  to your terminal app. Approve it in *System Settings → Privacy & Security
  → Camera*, then run again.
- **`Could not open camera index 0`:** try `--camera 1` (external webcams
  often enumerate after the built-in one), and close other apps using the
  camera.
- **All quantities empty, or `Reliable: 0 of N`:** the subject's shoulders and
  hips were not visible enough. Face-only or waist-up footage cannot produce
  Step-0 anchors; record full-body clips with the ankles inside the frame.
- **Moved or renamed the project folder?** Recreate the virtual environment
  (`venv/` stores absolute paths and breaks when moved).

## Related

- [PEF-Video-Tool](https://github.com/drzamoramora/PEF-Video-Tool) —
  companion recorder used to capture test clips and the PEF-FallDB dataset
  this detector consumes.

## Implementation status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo foundations, config, CLI | ✅ |
| 1 | Pose front-end (§3.2), Step-0, records, PEF-Lab | ✅ |
| 2 | Quantities T and V, live curves (§3.4) | ✅ |
| 3 | Stage-1 kinematic trigger + state machine (§3.5) | ✅ |
| 4a / 4b | Quantity P + Stage-2 geometric verification | ✅ |
| 5a / 5b | Quantity I + Stage-3 immobility, recovery, severity | ✅ |
| — | Threshold calibration against a dataset | ⏸ blocked on data |
| — | Subject-continuity guard (multi-person robustness, §3.3) | ⬜ |
| 6 | Batch evaluation harness and baselines (§3.7) | ⬜ |
| 7 | Alert transports and Raspberry Pi port (§3.6) | ⬜ |

## Citation

The accompanying paper is under preparation; a citation entry will be added
upon publication. Until then, please do not redistribute the draft materials
associated with this repository.

## License

Released under the [MIT License](LICENSE).
