"""The commands that work on runs already recorded: ``pef-compare``, ``pef-metrics``,
``pef-replay`` and ``pef-export``.

They are the tools of ``tools/`` (compare_runs.py, metrics.py, replay.py,
export.py), installed as commands so they read like ``pef-run`` and need no
path to a script. The tools stay plain scripts in ``tools/`` because they
also run against OTHER checkouts (``--repo``: replay a run with the code of a
reference commit); these entry points only find them in this checkout, which
an editable install (``pip install -e .``) always has.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from .config import REPO_ROOT

TOOLS = REPO_ROOT / "tools"


def tool(name: str) -> ModuleType:
    """Import ``tools/<name>.py``."""
    if not TOOLS.is_dir():
        raise SystemExit(f"the tools folder is missing ({TOOLS}): "
                         f"install from a checkout, with pip install -e .")
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    return importlib.import_module(name)


def compare_runs(reference: str, run: str) -> bool:
    """Compare two runs as ``pef-compare --landmarks`` does; True when identical."""
    module = tool("compare_runs")
    a_runs, _ = module.find_runs(reference)
    b_runs, _ = module.find_runs(run)
    path = module.default_map(a_runs, b_runs)
    if path is not None:
        print(f"map: {path.relative_to(REPO_ROOT)} (the reference predates 0.2.0.dev5)")
    return module.compare(reference, run, module.ColumnMap.load(path), landmarks=True)


def _run(name: str) -> int:
    return tool(name).main(sys.argv[1:])


def compare_main() -> int:
    """``pef-compare A B [--landmarks] ...``: tools/compare_runs.py."""
    return _run("compare_runs")


def metrics_main() -> int:
    """``pef-metrics RUN [RUN_B] ...``: tools/metrics.py."""
    return _run("metrics")


def replay_main() -> int:
    """``pef-replay RUN OUT ...``: tools/replay.py."""
    return _run("replay")


def export_main() -> int:
    """``pef-export RUN ...``: tools/export.py."""
    return _run("export")
