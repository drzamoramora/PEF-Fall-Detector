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

### 0.2.0.dev9 — B9: package and documentation (branch `cleanup-b9`)

- `pyproject.toml`: the package, its direct dependencies in three groups —
  the detector (mediapipe, OpenCV, numpy, PyYAML), `gui` (PySide6,
  pyqtgraph) and `dev` (ruff) — and two commands: `pef-lab` (PEF-Lab, same
  options as `python main.py`) and `pef-run` (headless: `--video` without
  `--headless`, or `--folder`; refuses anything that would open the GUI).
  Python 3.11 or 3.12. Installed editable from a checkout, because the
  detector reads `config.yaml` and `config/partition.yaml` from the
  repository. ruff's settings moved here from `ruff.toml` (target 3.11).
- `requirements.txt` is the lock of the reference Mac (`pip freeze`); its
  header no longer quotes the 104/128 of an old phase.
- `.gitignore`: macOS `.DS_Store`.
- `notas/`: the notes of closed phases and the three scripts that regenerate
  their figures moved to `notas/archivo/`; `notas/README.md` is the index.
  The scripts use the modules from before the cleanup: they run on tag
  `v0.1.1-legacy`.
- README rewritten: what the detector is, the phase-8.4 figures and why they
  are not the paper's, install, run, verify, layout, reproducibility.
- Verified: no detector code changed; replays of the phase-8.4 landmarks are
  identical to 0.2.0.dev8 on both platforms (128/128 clips). The package
  builds; in a clean Python 3.11 environment on Linux ARM (aarch64, the
  Raspberry Pi's architecture) every pinned dependency resolves to a
  published wheel and both commands install, pointing at `main:main` and
  `main:run`.

### 0.2.0.dev8 — B8: code hygiene (branch `cleanup-b8`)

- **ruff**, lines of 100 (`ruff.toml`; B9 moves it into `pyproject.toml`
  and declares ruff as a development dependency). Rules: pycodestyle (E, W),
  pyflakes (F), import order (I) and bugbear (B). `python -m ruff check .`
  passes. Not applied: `ruff format`, which would rewrite every file's
  layout, hand-aligned comments included, for no change in behaviour.
- Sliding windows are `collections.deque` (`popleft` instead of
  `list.pop(0)`): Stage 3's final-posture, A and hip windows, the trunk
  guard's reference, and the immobility episode (`maxlen`, which drops the
  oldest sample by itself). The medians read from them are untouched: they
  still take the upper middle value.
- NaN is tested with `math.isnan`, never with `x == x` / `x != x` (four
  places: the label row, the queue panel, the trunk guard, F1).
- `OSError`, not its old alias `IOError` (a source that cannot be opened).
- `try`/`finally`: a headless run releases the video, closes MediaPipe and
  closes its records even when a frame raises; the folder run closes its
  labels file; the replay tool closes its records.
- `logging`: a failing alert transport is logged as a warning, besides being
  recorded in `AlertDispatcher.failures`. `main.py` configures logging.
  What a run prints for the user (summaries, the console alert, the pass
  summary of PEF-Lab's queue) stays on stdout: it is the program's output,
  not a diagnostic.
- `zip(..., strict=True)` where both sides must match (and `strict=False`
  where the shorter one is meant to win); `raise ... from None` for the GUI
  tests' skip.
- Tests: a run that fails mid-clip still releases its source and closes its
  records; an unmeasured I max leaves the labels row empty; an unmeasured
  trunk never becomes the trunk guard's reference.
- Verified: replays of the phase-8.4 landmarks with 0.2.0.dev7 and with this
  version are identical on both platforms (128/128 clips; cloud 115/128,
  Mac 110/128). Mutation check of the changes: 12 deliberate breaks; 10
  caught, 2 equivalent (the `maxlen` in the constructor, overwritten before
  any sample arrives, and the NaN guard of F1, whose `> 0` already fails for
  NaN).

### 0.2.0.dev7 — B7: tests (branch `cleanup-b7`)

- **End to end with the shipped configuration.** Every other test switches
  on one behaviour over `TEST_CFG`, which differs from `config.yaml` in 17
  keys (sequential trigger at 2.2, no peak window, H and A off, no
  persistence, no get-up check…), so the combination that is actually run
  was checked only by the replay tool, outside the suite. New
  `tests/test_end_to_end.py` replays the 33-landmark records of twelve real
  clips through `config.yaml` as it is and checks every decision: each
  clip's label and every event's frame, trigger condition, verdict,
  severity and reason. The clips cover the three trigger conditions (score,
  H, A), every Stage-3 ending, a Stage-2 rejection, a detection loss on the
  floor, an abandoned event, a known false positive (B12-S4) and a clip
  with no event. `tools/e2e_fixture.py` builds the data from a
  research-mode run. The records are the volunteers' poses, so
  `tests/data/` is ignored by git; the test skips where the data is
  missing. Built from the Mac's phase-8.4 run (`logs/cli-a84`), and
  checked on x86 and ARM.
- `tests/` is a package; what several modules share (`TEST_CFG`, the
  synthetic poses and clips, the headless-run mixin, the Stage-1 helpers,
  the 3D world poses) lives in `tests/support.py`. Before, test modules
  imported it from one another (44 imports), and importing a test class
  made unittest run it twice: 6 tests ran twice and 1 was skipped on every
  run. They now run once (662 → 655 for the same tests, mapped one by one).
- Everything in English: test names (33), class names (6), docstrings, comments
  and failure messages. Phase-named files renamed by what they test:
  `test_fase4_cooldown` → `test_cooldown_after_rejection`, `test_fase6` →
  `test_getup_in_progress`, `test_fase8` → `test_quantity_a_stage3`,
  `test_fase84` → `test_quantity_a_trigger`, `test_fases_2_3` →
  `test_loss_and_persistence`, `test_motivo` → `test_reasons`. Only names,
  comments and messages changed (checked token by token, comments and
  docstrings aside), but the GUI queue tests, which now keep the pass
  summary off the console. One stale claim corrected: an unjudged event is
  NoFall by default (D12), not UNDETERMINED.
- `TEST_CFG` says why it is not `config.yaml`.
- Verified: 659 tests; replays unchanged (no detector code changed).
  Mutation check of the shipped combination: 6 deliberate breaks, all
  caught; the end-to-end test catches 4 of them, and is the only test that
  catches dropping the peak window from the shipped configuration besides
  the configuration wiring check.

### 0.2.0.dev6 — B6: PEF-Lab split (branch `cleanup-b6`)

- `gui/lab_window.py` (1,792 lines, one class of 66 methods) keeps the flow:
  sources, playback, the per-frame loop, frozen review and the queue pass.
  What it shows moved beside it:
  - `gui/plots.py`: `PlotPanel`, the curves of both tabs and the timeline
    under the video, with their data, cursor and marks (they share one time
    axis and are cleared together);
  - `gui/queue_panel.py`: `QueuePanel`, the Queue tab's table, buttons and
    score line, and `metrics_text`;
  - `gui/widgets.py`: the video view, the View tab (`ViewPanel`) and the
    small building blocks (cards, captions, time format).
- Not split further on purpose: sources, playback, loop, frozen review and
  queue flow share 10–19 attributes each with the others; spreading them
  over files (mixins) would scatter one state, not separate it.
- The last two Spanish words B5 missed (`destino`, "distintos") are gone.
- Verified: ten screenshots of the window in fixed states (empty, mid-pass,
  frozen, both curve tabs, View tab, queue loaded, queue finished, review
  arrows, frame steps) are identical pixel for pixel to 0.2.0.dev5; the
  records PEF-Lab writes for a clip and a two-clip queue are identical byte
  for byte; replays are identical (the detector was not touched). Mutation
  check: 10 deliberate breaks of the new seams, all caught; 9 needed new
  tests, because the window tests called its methods directly and never
  pressed a button, the curves switch or a live source. New
  `tests/test_gui_panels.py`, `TestWindowWiring` and `TestTheButtonsAreWired`.
  662 tests.

### 0.2.0.dev5 — B5: language (branch `cleanup-b5`)

- The code is in English: comments, docstrings, the texts it writes and
  prints, and the last Spanish names (`resuelto` → `settled`, PEF-Lab's
  `nombre` → `name`). The comments that told a phase's history (what was
  measured, on which clip, with which numbers) moved verbatim, with their
  B4 line, to `notas/CODIGO-HISTORIA.md`; a short English version stays in
  the code and points there. `notas/` stays in Spanish.
- Event and label reasons in English, same numbers and same format, e.g.
  `A final 0.12 (T final 95 deg) <= 0.26: termino en el suelo` →
  `… <= 0.26: ended on the floor`, `quieto` → `still`, `no llego al suelo` →
  `never reached the floor`, `abandonado por discontinuidad` → `abandoned at
  a discontinuity`, `sin eventos` → `no events`. `tools/maps/b5.json`
  translates every old reason, to compare runs from before this version
  (`compare_runs.py OLD NEW --map tools/maps/b5.json`). It carries B3's
  column rename too, so it also reads runs from before B3.
- Reports in English: the confusion matrix (`truth \ detected`,
  `correct: n/m`), the §3.7 report (`4-class accuracy`, `binary fall/no`,
  `95% CI`, `false pos/hour`, `undetermined`) and the pass summary titles
  (`CALIBRATION`, `TEST`, `UNASSIGNED`). `evaluation.binary_metrics` keys:
  `sensibilidad`, `especificidad`, `sens_ic`, `espec_ic` → `sensitivity`,
  `specificity`, `sens_ci`, `spec_ci`; `split_rows`' `sin_asignar` →
  `unassigned`.
- Quantity H documented as it is: its thresholds ship on, were never swept
  on PEF-FallDB, and it decides — it raised 25 of the 90 Stage-1 events of
  phase 8.4 in the cloud and 21 of 90 on the Mac (`quantities` package
  docstring, `pipeline.py`, `config.yaml`). The package docstring said it
  must not enter a decision before evidence says so; it now says that rule
  is not met today (`notas/POSIBLES-CAMBIOS-DEL-DRAFT.md` C19).
- Comments that had become wrong were corrected: `A_head` is read by Stage 3
  (8.2/8.3) and Stage 1 (8.4), not "logged only"; references to
  `quantities.py` and `_quantity_h`, gone since B4, point to the modules that
  replaced them.
- Tests: only those that read a translated text were changed (the reasons,
  the reports, the metric keys). Test names and comments are B7. New: the B5
  map against one old reason of every shape; every §3.7 ratio on a set
  where no two counts are equal; the pass summary's three blocks.
- Verified: replays of the phase-8.4 landmarks with this version are
  identical to 8.4 and to B4 on both platforms (128/128 clips, through the
  map); without the map, only the `reason` column differs. Mutation check:
  8 deliberate breaks, 7 caught; the eighth (swapping two map rules) is
  equivalent, the two rules cannot match the same text. Two needed new
  tests: a specificity computed over the wrong denominator and a pass
  summary repeating the test block — both untested before this block too.
  638 tests.

### 0.2.0.dev4 — B4: structure (branch `cleanup-b4`)

- `state_machine.py` (1,647 lines) became the package `state_machine/`:
  `events.py` (formulations, stages, verdicts, the event record),
  `trigger.py` (Stage 1 and its peak window), `stage2.py`, `stage3.py`,
  `machine.py` (the funnel) and `build.py` (the funnel assembled from the
  configuration, which used to live inside the pipeline).
- `quantities.py` (1,033 lines) became the package `quantities/`, one module
  per quantity: `t`, `v`, `p`, `i` (§3.4, listed in `PAPER`) and `h`, `r`,
  `a`, `extension` (experimental, listed in `EXPERIMENTAL`), plus `filters`.
  Each quantity's per-frame measurement moved into its module
  (`p.measure`, `i.measure`, `h.measure`, `r.measure`, `a.measure`,
  `t.TrunkGuard`, `extension.LegExtension`).
- `pipeline.py` went from 1,053 to 558 lines: only the per-frame walk is
  left. `FrameResult` and `Timings` moved to `results.py`.
- Both packages re-export every name, so `from pef_fall_detector.state_machine
  import X` and `from pef_fall_detector.quantities import X` read as before;
  no import had to change. The code was moved, not rewritten: the order of
  every operation is the same.
- Verified: replays of the phase-8.4 landmarks with this version are
  identical to 8.4 on both platforms (128/128 clips, landmark records
  included), and the B2 wiring test still finds every configuration key in
  the object that reads it. Mutation check of the cut points: 7 deliberate
  breaks, all caught (three needed new tests, for behaviour that was
  untested before the move too). 634 tests.

### 0.2.0.dev3 — B3: terminal and records (branch `cleanup-b3`)

- One writer for the records of a run, `pef_fall_detector/records.py`
  (`RunRecords`, `event_row`), used by `main.py`, PEF-Lab and
  `tools/replay.py`. Before, each kept its own copy of the event row, and they
  did not record the same thing: PEF-Lab left out the events a detection loss
  abandoned and never wrote the 33 landmarks. Events are written in order,
  each one as soon as its verdict lands; abandoned ones are written in their
  place.
- `logging.mode`: research records the per-frame record, the events and the
  33 landmarks; **deployment records nothing** (§3.6: the deployed detector
  persists no frames, landmark coordinates or derived quantity values;
  `notas/POSIBLES-CAMBIOS-DEL-DRAFT.md` C9). In 0.2.0.dev2 deployment still
  wrote the per-frame record and the events. PEF-Lab, the laboratory, always
  records; `logging.mode` governs `main.py`, which is what runs on the Pi.
- `main.py`: `--folder` runs every clip of a folder in one process, as
  PEF-Lab's queue does, and closes with the same labels file
  (`dataset.label_row`) and the same summary (`evaluation.pass_summary`);
  `--mode research|deployment`, `--output-dir` (from the current directory)
  and `--quiet` (one line per clip). The overrides go into the configuration,
  so every `.meta.json` states the mode and folder actually used. The
  128-clip run is now `python main.py --folder ../recordings --mode research
  --output-dir logs/<name> --quiet`.
- Record columns in English. Events: `motivo` → `reason`. Labels file:
  `fecha`, `hora`, `duracion_s`, `frames_confiables_pct`, `clase_detectada`,
  `clase_verdad`, `clase_revisada`, `acierto`, `eventos`, `t_disparo_s`,
  `motivo` → `date`, `time`, `duration_s`, `reliable_frames_pct`,
  `detected_class`, `true_class`, `reviewed_class`, `correct`, `events`,
  `first_trigger_s`, `reason`. The reason texts stay in Spanish until B5.
  `tools/maps/b3.json` maps the old event column, to compare runs from
  before this version (`compare_runs.py OLD NEW --map tools/maps/b3.json`).
- `tools/replay.py` writes the landmarks too, so a replay has every record a
  real run has, and still replays code from before this version.
- Verified: replays of the phase-8.4 landmarks with this version are
  identical to 8.4 on both platforms (128/128 clips, through the map), and
  the replay rewrites the landmark records byte for byte. Mutation check of
  the new code: 10 deliberate breaks, 9 caught; the tenth (main.py writing
  every event at close instead of as it settles) leaves the same file and
  only loses the crash safety. 625 tests.

### 0.2.0.dev2 — B2: configuration (branch `cleanup-b2`)

- `config.yaml` rewritten in English and in the order of the §3.5 funnel
  (pose, logging, stage1, stage2, stage3, quantity_a, experimental, alerts,
  state_display). Same keys (but one, below), same values and types, one line
  per key: what it is, its unit and why. The measurement histories that were
  in its comments moved, verbatim and key by key, to
  `notas/CONFIG-HISTORIA.md`. Three comments that had become wrong were
  corrected: four `state_display` keys feed Stage 3, the Quantity-H
  thresholds are on, and a relative `output_dir` is taken from the
  repository root.
- The configuration is checked when it is loaded (`config.SCHEMA`): a
  missing key, an unknown key or a value of the wrong kind (a flag that is
  not true/false, a number given as text, true, NaN or infinity, a whole
  number with decimals, a choice outside its list) stops the program with a
  message that names every problem. A configuration from an earlier version
  is refused, and a renamed key says what replaced it.
- No fallback values in the code: 21 reads had one in the pipeline and
  `main.py` (e.g. `qa.get("band_floor", 0.26)`), plus PEF-Lab's A plot and
  `tools/replay.py`. Seven of those fallbacks disagreed with `config.yaml`
  (`trigger_from_a`, `severity_from_a`, `reach_check`, `getup_rise_torsos`,
  `cooldown_after_rejection`, `reacquire_after_s`, `save_landmarks`). They
  never decided a run, because `config.yaml` declared every key; they did
  decide the tests that built incomplete configurations.
- `logging.save_landmarks` became `logging.mode`: `research` also records
  the 33 landmarks of every frame, `deployment` does not (§3.6).
  `config.yaml` ships in `deployment`, so **a plain run no longer records
  landmarks**; dataset runs must use `research`.
- The train/test partition moved from `notas/particion.yaml` to
  `config/partition.yaml` (`config.PARTITION_PATH`), in English: `fecha`,
  `regla` and `nota` became `date`, `rule` and `note`. Both lists are
  unchanged.
- Tests: `TEST_CFG` declares every key, with the values the code used to
  fall back to. Every pipeline the suite builds gets the same parameters as
  before (compared object by object in the 117 tests that build one). The
  tests of "a configuration without the key keeps the old behaviour" now
  also check that the key is required. New `tests/test_config.py`, which
  also changes every key of `config.yaml` in turn and checks that the change
  reaches what reads it. Checked by mutation: 37 deliberate breaks of this
  block's code, all caught by the suite. 610 tests.
- Verified: replays of the phase-8.4 landmarks with this version are
  identical to 8.4 on both platforms (128/128 clips; cloud 115/128, Mac
  110/128). The only configuration differences are `logging.save_landmarks`
  → `logging.mode` and the five keys B1 removed.

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
