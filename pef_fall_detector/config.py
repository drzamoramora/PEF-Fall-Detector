"""Configuration: loading and validation.

Every tunable value of the detector lives in ``config.yaml`` (repository
root), and only there. §3.5 states that thresholds and window lengths are
"tunable at deployment time" and are reported in Section 4 together with the
calibration protocol, so the file is part of the scientific record: every run
stores a copy of it in its ``.meta.json``.

The code keeps no fallback values. A fallback is a second copy of a value,
and a second copy can disagree with the first without anyone noticing.
Instead, :class:`Config` checks the file against :data:`SCHEMA` before
anything reads it. A missing key, a key this version does not read (a typo,
or a key that was renamed or removed) and a value of the wrong type all stop
the program with a message that names every offending key.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .state_machine import FORMULATIONS

#: Repository root. Used to anchor every relative path in the configuration,
#: so that where the program was launched from never changes where its files
#: land — a run started from another directory must not scatter its records.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

#: The train/test partition of PEF-FallDB, by actor. Read by the evaluation
#: reports (PEF-Lab's queue summary, tools/metrics.py).
PARTITION_PATH = REPO_ROOT / "config" / "partition.yaml"

#: ``logging.mode``. Research records every run (the per-frame record, the
#: events and the 33 landmarks of every frame; records.RunRecords). Deployment
#: records nothing: §3.6 says the deployed detector persists no frames,
#: landmark coordinates or derived quantity values. PEF-Lab, the laboratory,
#: always records.
RESEARCH = "research"
DEPLOYMENT = "deployment"
LOGGING_MODES = (RESEARCH, DEPLOYMENT)

#: Kinds of value. A tuple instead of a kind is a choice among its values.
NUMBER = "number"          # int or float, finite, never true/false
INTEGER = "integer"        # a whole number, never true/false
FLAG = "flag"              # true or false
TEXT = "text"              # a string
TEXT_LIST = "text list"    # a list of strings

#: Every key the detector reads, by section, in the order of config.yaml.
#: What each one means is documented there.
SCHEMA: dict[str, dict[str, Any]] = {
    "pose": {
        "model_complexity": (0, 1, 2),
        "min_detection_confidence": NUMBER,
        "min_tracking_confidence": NUMBER,
        "smooth_landmarks": FLAG,
        "visibility_threshold": NUMBER,
        "reacquire_after_s": NUMBER,
    },
    "logging": {
        "output_dir": TEXT,
        "mode": LOGGING_MODES,
    },
    "stage1": {
        "threshold_T_deg": NUMBER,
        "threshold_V": NUMBER,
        "velocity_window_s": NUMBER,
        "ema_time_constant_s": NUMBER,
        "history_max_gap_s": NUMBER,
        "trigger_formulation": FORMULATIONS,
        "trigger_score": NUMBER,
        "min_trunk_ratio": NUMBER,
        "trunk_reference_window_s": NUMBER,
        "trigger_hold_s": NUMBER,
        "confirm_window_s": NUMBER,
        "trigger_peak_window_s": NUMBER,
    },
    "stage2": {
        "com_hip_weight": NUMBER,
        "com_eval_window_s": NUMBER,
        "com_outside_fraction": NUMBER,
        "com_min_samples": INTEGER,
        "min_foot_visibility": NUMBER,
        "contact_band_torso": NUMBER,
    },
    "stage3": {
        "epsilon": NUMBER,
        "threshold_W_seconds": NUMBER,
        "observation_window_seconds": NUMBER,
        "upright_T_deg": NUMBER,
        "recovery_hold_seconds": NUMBER,
        "severity_uses_leg_extension": FLAG,
        "cooldown_seconds": NUMBER,
        "cooldown_after_rejection": FLAG,
        "persistent_still_s": NUMBER,
        "persistent_immobility_fraction": NUMBER,
        "getup_rise_torsos": NUMBER,
        "getup_window_s": NUMBER,
    },
    "quantity_a": {
        "standing_T_deg": NUMBER,
        "standing_knee_deg": NUMBER,
        "min_standing_s": NUMBER,
        "max_samples": INTEGER,
        "severity_from_a": FLAG,
        "band_floor": NUMBER,
        "band_standing": NUMBER,
        "reach_check": FLAG,
        "floor_reached": NUMBER,
        "hip_raised": NUMBER,
        "reach_before_s": NUMBER,
        "reach_after_s": NUMBER,
        "trigger_from_a": FLAG,
        "trigger_high": NUMBER,
        "trigger_low": NUMBER,
        "trigger_window_s": NUMBER,
    },
    "experimental": {
        "height_baseline_time_constant_s": NUMBER,
        "trigger_H_ratio": NUMBER,
        "trigger_H_erect": NUMBER,
        "trigger_H_sequence_window_s": NUMBER,
        "recovery_H_ratio": NUMBER,
    },
    "alerts": {
        "dispatch_verdicts": TEXT_LIST,
    },
    "state_display": {
        "transition_v_tps": NUMBER,
        "lying_T_deg": NUMBER,
        "leaning_T_deg": NUMBER,
        "standing_extension": NUMBER,
        "crouch_extension": NUMBER,
        "walking_vh_tps": NUMBER,
        "max_extension_ratio": NUMBER,
        "ratio_time_constant_s": NUMBER,
    },
}

#: Keys that were renamed: an old configuration is told what replaced them.
REPLACED = {
    "logging.save_landmarks": "logging.mode: research for true, deployment for false",
}


class ConfigError(ValueError):
    """A configuration with missing keys, unknown keys or wrong values."""

    def __init__(self, source: str, problems: list[str]) -> None:
        self.problems = list(problems)
        listed = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{source} is not a valid configuration:\n{listed}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _kind_problem(value: Any, kind: Any) -> str | None:
    """Why ``value`` is not of ``kind``; None when it is."""
    if isinstance(kind, tuple):
        # Compared with their type: true is not 1, and 1.0 is not 1.
        if any(type(value) is type(choice) and value == choice for choice in kind):
            return None
        return f"must be one of {', '.join(str(c) for c in kind)}, got {value!r}"
    if kind == NUMBER:
        if _is_number(value) and math.isfinite(value):
            return None
        return f"must be a finite number, got {value!r}"
    if kind == INTEGER:
        if isinstance(value, int) and not isinstance(value, bool):
            return None
        return f"must be a whole number, got {value!r}"
    if kind == FLAG:
        if isinstance(value, bool):
            return None
        return f"must be true or false, got {value!r}"
    if kind == TEXT:
        if isinstance(value, str):
            return None
        return f"must be text, got {value!r}"
    if kind == TEXT_LIST:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return None
        return f"must be a list of text, got {value!r}"
    raise AssertionError(f"unknown kind {kind!r} in SCHEMA")


def validate(data: Any) -> list[str]:
    """Every problem of ``data`` against :data:`SCHEMA`; empty when there is none."""
    if not isinstance(data, dict):
        return [f"must be a mapping of sections, got {type(data).__name__}"]
    problems: list[str] = []
    for section, keys in SCHEMA.items():
        if section not in data:
            problems.append(f"missing section '{section}'")
            continue
        values = data[section]
        if not isinstance(values, dict):
            problems.append(f"'{section}' must be a section of keys, got {values!r}")
            continue
        for key, kind in keys.items():
            if key not in values:
                problems.append(f"missing key '{section}.{key}'")
                continue
            why = _kind_problem(values[key], kind)
            if why:
                problems.append(f"'{section}.{key}' {why}")
        for key in values:
            if key not in keys:
                dotted = f"{section}.{key}"
                hint = (f"replaced by {REPLACED[dotted]}" if dotted in REPLACED
                        else "a typo, or a key this version no longer reads: see CHANGELOG.md")
                problems.append(f"unknown key '{dotted}' ({hint})")
    for section in data:
        if section not in SCHEMA:
            problems.append(f"unknown section '{section}'")
    return problems


class Config:
    """Read-only, attribute-style view of a validated configuration.

    ``Config(data)`` checks ``data`` against :data:`SCHEMA` and raises
    :class:`ConfigError`, naming every problem, before anything can read it.
    Sections are views of the same, already checked mapping, so
    ``cfg.stage1.threshold_T_deg`` reads ``stage1: {threshold_T_deg: ...}``.
    """

    def __init__(self, data: dict[str, Any], source: str = "the configuration") -> None:
        problems = validate(data)
        if problems:
            raise ConfigError(source, problems)
        self._data = data
        self._path = ""

    @classmethod
    def _section(cls, data: dict[str, Any], path: str) -> "Config":
        view = cls.__new__(cls)
        view._data = data
        view._path = path
        return view

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        where = f"{self._path}.{name}" if self._path else name
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{where}' is not a configuration key (SCHEMA in config.py)") from exc
        if isinstance(value, dict):
            return Config._section(value, where)
        return value

    def as_dict(self) -> dict[str, Any]:
        """Return the raw underlying mapping (e.g. for logging a snapshot)."""
        return self._data


def research_mode(cfg: Config) -> bool:
    """Whether runs of the detector record anything (research) or not (deployment)."""
    return cfg.logging.mode == RESEARCH


def load_config(path: str | Path | None = None) -> Config:
    """Load and check ``config.yaml`` (or an alternative file).

    Args:
        path: Optional explicit path. Defaults to the repository's
            ``config.yaml`` so every entry point shares one source of truth.

    Raises:
        ConfigError: the file is empty, or does not match :data:`SCHEMA`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(str(cfg_path), ["the file is empty or is not a mapping of sections"])
    return Config(data, source=str(cfg_path))
