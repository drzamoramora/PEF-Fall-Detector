# Legacy snapshot — before the repository cleanup

**Branch:** `legacy-pre-cleanup` · **Tag:** `v0.1.0-legacy` · **Base commit:** `2cdd99b` (branch `82-90`, 25 Sep 2026)

This branch freezes the code exactly as it was before the September 2026
cleanup. It is kept for reference and reproducibility: every result reported
up to this point can be regenerated from here. No development happens on this
branch.

## What this snapshot contains

- The full detection funnel as used for the 82→90 work:
  - Stage 1: score trigger (T/45 + V/−1.5 ≥ 2.6) over a 0.8 s peak window, with the experimental H fallback;
  - Stage 2: support-polygon check (P);
  - Stage 3 in labelling mode: final posture, persistent immobility (phase 3), hip get-up rule (phase 6a);
  - tracker re-acquisition and no cooldown after a Stage 2 rejection (phase 4);
  - Quantity A (3D head height against gravity): severity bands (phase 8.2) and the "reached the floor" check with a 0.31 cut and abstention when A was not calibrated at the trigger (phase 8.3).
- PEF-Lab GUI with the redesigned layout (English UI).
- 562 unit tests (1 skipped).

## Results of this snapshot

PEF-FallDB, 128 clips, labelling mode, same code and `config.yaml` on both platforms.

| | cloud (x86) | Mac (ARM, GUI) |
|---|---|---|
| accuracy (4 classes) | 114/128 (89.1 %) | 109/128 (85.2 %) |
| train / test (partition by actor) | 80/88 · 34/40 | 79/88 · 30/40 |
| falls detected (any class) | 69/72 | 64/72 |
| specificity, clip label | 49/56 | 49/56 |
| specificity, dispatched alarm | 52/56 | 52/56 |

The two platforms differ because MediaPipe's inference differs between x86 and
ARM (see `notas/POSIBLES-CAMBIOS-DEL-DRAFT.md`, point C8). Official numbers
for the paper come from the target device (Raspberry Pi 4B).

## Improvements made in the new code

The cleanup is behaviour-preserving: every block must reproduce the results
above exactly (labels, events and per-frame values), verified on the 128 clips.
Status is updated once, when the cleanup is finished.

| Block | Improvement | Status |
|---|---|---|
| B0 | Verification base: analysis scripts in `tools/`, a 128-clip comparator against this snapshot, git commit and version recorded in every run, CHANGELOG | planned |
| B1 | Remove obsolete code: provisional trigger band, descent rejection; alternative trigger formulations moved to an ablations module (`v_only` kept) | planned |
| B2 | Configuration: `config.yaml` rewritten in English and reordered, key validation, a single source of defaults, `config/partition.yaml`, research/deployment mode (landmarks off by default) | planned |
| B3 | CLI and records: `main.py --folder`, `--output-dir`, `--landmarks/--no-landmarks`, `--quiet`; one record writer shared with the GUI; English CSV column names | planned |
| B4 | Core structure: `state_machine.py`, `quantities.py` (paper vs experimental) and `pipeline.py` split into focused modules | planned |
| B5 | Code language and documentation: everything in English, history moved to `notas/`, event reasons in English, H documented | planned |
| B6 | GUI split into panels; faster GUI tests | planned |
| B7 | Tests: package with shared helpers, behaviour-based English names, end-to-end tests with the real configuration | planned |
| B8 | Style and best practices: ruff (line length 100), queues, NaN checks, resource handling, logging | planned |
| B9 | Packaging and documents: `pyproject.toml`, core/GUI/dev requirements with a lock file, `.gitignore`, archived notes with an index, README rewritten | planned |

## Known differences from the new code

- CSV column names and event reasons (`motivo`) are in Spanish here.
- Some docstrings and comments are in Spanish.
- The per-frame CSV is always written (no deployment mode).

## How to use it

```
git checkout legacy-pre-cleanup      # or: git checkout v0.1.0-legacy
python -m unittest discover -s tests
python main.py                       # PEF-Lab GUI
```
