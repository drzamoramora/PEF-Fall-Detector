"""Build the data of tests/test_end_to_end.py from a recorded run.

Copies the per-frame and 33-landmark records of a few clips into
``tests/data/e2e/`` and replays them with this checkout and ``config.yaml``;
what the replay decides (each clip's label and every event, with its reason)
is written to ``expected.json``. The end-to-end test replays the same records
and must get exactly that.

The records are the volunteers' poses, so ``tests/data/`` is ignored by git:
the data stays on the machine that built it, and the test skips where it is
missing.

Rebuild it only when a change of behaviour is intended (a new rule, a new
config.yaml value), and say so in the CHANGELOG: rebuilding is accepting the
new decisions as the reference.

Usage::

    python tools/e2e_fixture.py LOGS [--clips A01-S1,B05-S1,...] [--out tests/data/e2e]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _records import find_runs  # noqa: E402
from replay import TOOLS_REPO, load_package, replay_clip  # noqa: E402

#: One clip per branch of the shipped funnel, chosen on the Mac's phase-8.4
#: run: each trigger condition (score, H, A), every Stage-3 ending (A bands,
#: T bands, legs, persistence, get-up in progress, never reached the floor),
#: a Stage-2 rejection, a detection loss on the floor, an abandoned event, a
#: known false positive and a clip with no event at all.
DEFAULT_CLIPS = ("A01-S1", "A05-S2", "A12-S2", "A06-S14", "A18-S3", "A17-S2",
                 "A03-S1", "B05-S1", "A17-S3", "B08-S3", "B12-S4", "B01-S1")
DEFAULT_OUT = TOOLS_REPO / "tests" / "data" / "e2e"


def config_digest(cfg_dict: dict) -> str:
    """A stable fingerprint of a configuration's values."""
    blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def decisions(pkg, cfg, records) -> dict:
    """What the detector decides on one clip: its label and every event."""
    from pef_fall_detector.classification import classify_clip
    with tempfile.TemporaryDirectory() as tmp:
        events, _stats = replay_clip(pkg, cfg, records, Path(tmp))
    return {
        "label": classify_clip(events)[0],
        "events": [{"frame_index": e.frame_index, "timestamp": round(e.timestamp, 4),
                    "trigger_source": e.trigger_source, "verdict": e.verdict,
                    "severity": e.severity, "reason": e.reason} for e in events],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", help="a research-mode run folder (frames, landmarks, meta)")
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS), help="comma-separated clip ids")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="where the data goes (replaced)")
    args = ap.parse_args(argv)

    pkg = load_package(TOOLS_REPO)
    cfg = pkg.config.load_config()
    runs, _extra = find_runs(args.logs)
    wanted = args.clips.split(",")
    missing = [c for c in wanted if c not in runs or runs[c].landmarks is None]
    if missing:
        ap.error(f"no frames or landmark record for: {', '.join(missing)}")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for clip in wanted:
        rec = runs[clip]
        for path in (rec.frames, rec.meta, rec.landmarks, rec.landmarks.with_suffix(".meta.json")):
            if path.exists():
                shutil.copy2(path, out / path.name)
    copied, _ = find_runs(out)
    expected = {
        "built_from": str(Path(args.logs).resolve()),
        "built_on": platform.platform(),
        "code_version": _version(),
        "config_digest": config_digest(cfg.as_dict()),
        "clips": {clip: decisions(pkg, cfg, copied[clip]) for clip in wanted},
    }
    (out / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
    for clip, d in expected["clips"].items():
        evs = ", ".join(f"{e['trigger_source']}:{e['verdict']}/{e['severity'] or '-'}"
                        for e in d["events"]) or "no events"
        print(f"{clip:9s} {d['label']:20s} {evs}")
    print(f"-> {out}")
    return 0


def _version() -> str:
    from pef_fall_detector import __version__
    return __version__


if __name__ == "__main__":
    sys.exit(main())
