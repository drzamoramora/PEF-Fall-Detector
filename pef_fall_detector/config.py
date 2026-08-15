"""Configuration loading.

All tunable parameters of the pipeline live in ``config.yaml`` (repository
root). Centralizing them is a deliberate design decision: §3.5 of the paper
states that thresholds and window lengths are "tunable at deployment time"
and are reported in Section 4 together with the calibration protocol, so the
configuration file *is* part of the scientific record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Repository root. Used to anchor every relative path in the configuration,
#: so that where the program was launched from never changes where its files
#: land — a run started from another directory must not scatter its records.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class Config:
    """Read-only, attribute-style view over the YAML configuration.

    Nested mappings are exposed recursively, so ``cfg.stage1.threshold_T_deg``
    reads ``stage1: {threshold_T_deg: ...}`` from the YAML file.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"Missing configuration key '{name}'. Check config.yaml."
            ) from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def as_dict(self) -> dict[str, Any]:
        """Return the raw underlying mapping (e.g. for logging a snapshot)."""
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    """Load ``config.yaml`` (or an alternative file) into a :class:`Config`.

    Args:
        path: Optional explicit path. Defaults to the repository's
            ``config.yaml`` so every entry point shares one source of truth.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {cfg_path} is empty or malformed.")
    return Config(data)
