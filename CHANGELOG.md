# Changelog

Versions of PEF-Fall-Detector. Every record's sidecar (`*.meta.json`)
carries `code_version`, and since 0.2.0.dev0 the git commit it ran from
(`git_commit`, `git_dirty`), so any result can be traced to an entry here.

## 0.2.0 — repository cleanup (in progress)

The cleanup changes how the code is organised, never what it decides. Each
block must reproduce the phase-8.4 records exactly (labels, every event,
every per-frame value), checked with `tools/replay.py` and
`tools/compare_runs.py` on the recorded landmarks of both platforms (cloud
x86 and Mac ARM). One branch per block (`cleanup-b0` … `cleanup-b9`), each
starting from the previous one, and one development version per block.

### 0.2.0.dev1 — B1: dead code out (branch `cleanup-b1`)

- Removed the provisional trigger band (`stage1.trigger_score_provisional`,
  the `provisional` event flag, its promotion and the
  `provisional_unconfirmed` verdict) and the controlled-descent rejection
  of Stage 2 (`DescentTracker`, `stage2.descent_*`), with their tests
  (`test_zona_gris.py`, `test_descent.py`). Both were switched off in
  `config.yaml`. Why they were switched off, with the measurements, is kept
  in `notas/DESCARTADOS.md`.
- `simultaneous` and `sequential` moved to `pef_fall_detector/ablations.py`.
  They are still chosen with `stage1.trigger_formulation`, for the §3.7
  ablations; `score` and `v_only` stay in `state_machine.py`.
- Verified: replays of the phase-8.4 landmarks with 0.2.0.dev0 and with this
  version are identical on both platforms (128/128 clips). The only
  configuration difference is the five keys removed.

### 0.2.0.dev0 — B0: verification base (branch `cleanup-b0`)

- `tools/replay.py`: replays a recorded run from its 33-landmark record
  through any checkout of the code, without MediaPipe, and writes records in
  the same form as a real run. 128 clips in 10–40 s instead of ~20 min.
- `tools/compare_runs.py`: compares two runs clip by clip (label, every
  event, every per-frame value). A map declares the columns and texts a
  block renames on purpose; every other difference is reported.
- `tools/metrics.py`: the headline numbers of a run (accuracy per
  partition, falls detected, specificity by label and by alarm, events per
  Stage-1 condition, the clips it gets wrong).
- Every record sidecar carries `git_commit` and `git_dirty` next to
  `code_version`. `git_dirty` is also true when a new `.py` file inside the
  package is not yet committed (it runs as soon as it is imported); new
  notes, tools or tests leave a record clean.
- `compare_runs.py --landmarks` compares the 33-point records too (for two
  real runs), and every comparison warns about any configuration value that
  differs between the runs (`logging.output_dir` excepted).
- Verified: replays of the phase-8.4 landmarks with the 0.1.1 code and with
  this version are identical on both platforms (128/128 clips), and a real
  Mac run with this version (`logs/cli-b0`) is identical to the phase-8.4
  run (`logs/cli-a84`) in every per-frame, event and landmark record.

## 0.1.1-legacy — phase 8.4 (commit `38c8df7`; tag `v0.1.1-legacy` = that code + `LEGACY.md`)

- Quantity A as an additional Stage-1 trigger: the head falls from ≥ 0.7
  to ≤ 0.3 of its standing height within 1.5 s while V < −1.5 on the same
  frame.
- Events record which Stage-1 condition raised them (`trigger_source`).
- PEF-FallDB: 115/128 in the cloud, 110/128 on the Mac.
- Its records still say `code_version` 0.1.0: the version string was not
  bumped for this phase. The commit identifies it.

## 0.1.0-legacy — phase 8.3 (commit `2cdd99b`; tag `v0.1.0-legacy` = that code + `LEGACY.md`)

- Quantity A in Stage 3: severity bands (8.2) and the "reached the floor"
  check (8.3).
- PEF-FallDB: 114/128 in the cloud, 109/128 on the Mac.
