# PEF-Fall-Detector

Reference implementation for the paper **"Beyond Black-Box AI: A
Physics-Informed Edge Framework for Explainable Markerless Fall Detection
Using MediaPipe Landmarks"** (ULACIT).

From MediaPipe's 33-landmark skeleton the detector derives named physical
quantities — trunk inclination **T**, vertical centroid velocity **V**,
centre of mass against the support polygon **P**, post-fall immobility **I**
(§3.4), plus the experimental head height against gravity **A** and body
height ratio **H** — and confirms a fall through a three-stage funnel (§3.5)
that ends in a severity tag (§3.3) and, when warranted, an alert. **No neural
network is trained on the inference path.** Every event is recorded with the
rule that decided it and the numbers it read, so any decision can be
reconstructed from its record.

## Status

Development figures on PEF-FallDB (128 clips: 72 falls in three severity
classes, 56 activities of daily living), configuration of phase 8.4:

| | cloud x86 | Mac ARM |
|---|---|---|
| accuracy, 4 classes | 115/128 (89.8 %) | 110/128 (85.9 %) |
| train / test (split by actor) | 81/88 · 34/40 | 80/88 · 30/40 |
| falls detected | 70/72 | 66/72 |
| specificity, label / alarm | 49/56 · 52/56 | 49/56 · 52/56 |

These are not the paper's results: the thresholds were adjusted while
looking at all 128 clips. The paper's figures come from the frozen
configuration, reported on the test split, on the Raspberry Pi 4B (§3.6,
§3.7). The two columns differ because MediaPipe gives slightly different
landmarks on each platform; the detector code decides identically on both
(see *Reproducibility*).

## Install

Python 3.11 (3.12 also works). From a checkout, in a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # exact versions of the reference Mac
pip install -e ".[gui,dev]" --no-deps
```

On a box without a screen (the Raspberry Pi) only the detector is needed:

```bash
pip install -e .                    # mediapipe, OpenCV, numpy, PyYAML
```

`pyproject.toml` declares the direct dependencies in three groups: the
detector, `gui` (PEF-Lab: PySide6, pyqtgraph) and `dev` (ruff).
`requirements.txt` locks every package of the reference environment.
The install is editable on purpose: the detector reads `config.yaml` and
`config/partition.yaml` from the repository.

## Run

**PEF-Lab**, the visual workbench (skeleton, live curves of every quantity,
the funnel's state, a queue that runs a whole folder and scores it):

```bash
pef-lab                              # or: python main.py
pef-lab --video clip.mp4
pef-lab --camera 0
```

**Headless**, one clip or a folder:

```bash
pef-run --video clip.mp4             # or: python main.py --video clip.mp4 --headless
pef-run --folder ../recordings --mode research --output-dir logs/run-1 --quiet
```

Options: `--config FILE`, `--mode research|deployment`, `--output-dir DIR`,
`--quiet` (one line per clip), `--expected-seconds S` (FPS audit).

**Deployment mode writes nothing to disk** (§3.6), and it is the default in
`config.yaml`. Research mode writes, per clip:

- `<clip>-<stamp>.csv`: one row per frame, every quantity and the funnel's state;
- `<clip>-events-<stamp>.csv`: one row per Stage-1 event, its verdict,
  severity and the reason, in words;
- `<clip>-landmarks-<stamp>.csv`: the 33 landmarks of every frame;
- a `.meta.json` beside each one: source, frame rate, the full configuration,
  the code version and the git commit it ran from;
- with `--folder`, `dataset-labels-<stamp>.csv`: the label of every clip.

Columns ending in `_EXP` are experimental: recorded to find out whether they
deserve a place in the paper.

## Verify

```bash
python -m unittest discover -s tests    # PEF-Lab's tests skip without PySide6
python -m ruff check .
```

A change meant to leave the decisions alone is checked without MediaPipe:
replay the recorded landmarks with the old and the new code, and compare.

```bash
python tools/replay.py logs/run-1 /tmp/old --repo ../old-checkout
python tools/replay.py logs/run-1 /tmp/new
python tools/compare_runs.py /tmp/old /tmp/new      # labels, events, every value
python tools/metrics.py logs/run-1                  # the headline numbers
```

A real run is compared the same way, with `--landmarks` to include the
skeleton. Runs written before version 0.2.0.dev5 used other column names and
texts: add `--map tools/maps/b5.json`.

## Layout

```
main.py                     command line: PEF-Lab and headless runs
config.yaml                 every tunable value, validated on load
config/partition.yaml       train/test split by actor (§3.7)
pef_fall_detector/
    sources.py              video file / live camera
    pose_frontend.py        MediaPipe and Step-0 normalisation (§3.2)
    quantities/             T, V, P, I (§3.4); A, H, R and leg extension [EXP]
    state_machine/          trigger, Stage 2, Stage 3, the funnel (§3.5)
    ablations.py            trigger formulations kept for the §3.7 ablations
    pipeline.py             the per-frame orchestration every entry point shares
    alerts.py               alert construction and dispatch (§3.5, §3.6)
    records.py, audit_log.py  the records of a run
    classification.py, dataset.py, evaluation.py   clip labels and §3.7 metrics
    gui/                    PEF-Lab (PySide6)
tests/                      unit and end-to-end tests
tools/                      replay, compare and metrics for recorded runs
notas/                      the development record, in Spanish (notas/README.md)
```

All computation lives in Qt-free modules: the same pipeline runs in PEF-Lab,
headless and on the Raspberry Pi.

## Reproducibility

Versions are pinned exactly. MediaPipe's landmarks change with the MediaPipe,
OpenCV and numpy versions, and with them some decisions. They also change
with the platform: on x86 and on ARM the landmarks of the same clip differ by
about 2 px, and 7 of 128 clip labels differ (phase 8.4). Each platform is
deterministic — the same run repeated gives the same bytes — and the
detector code decides identically on both: replaying one platform's landmarks
on the other reproduces its decisions exactly.

## Related

- [PEF-Video-Tool](https://github.com/drzamoramora/PEF-Video-Tool): the
  recorder used to capture PEF-FallDB.

## Citation

The paper is in preparation; a citation will be added on publication.

## License

[MIT](LICENSE).
