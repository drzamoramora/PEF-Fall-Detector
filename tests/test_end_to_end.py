"""The shipped detector, end to end: config.yaml as it is, on recorded clips.

Every other test switches on one behaviour over TEST_CFG's plain baseline
(see tests/support.py). This one runs the combination that is actually
shipped — score trigger with its peak window, H, A in Stage 1 and Stage 3,
persistence, the get-up check, re-acquisition — through the whole funnel, on
the 33-landmark records of a dozen real clips, and checks every decision:
each clip's label and every event, with its trigger condition, verdict,
severity, frame and reason.

The records are the volunteers' poses, so they are not in the repository:
``tests/data/`` is ignored by git and the test skips where it is missing.
Build it with ``python tools/e2e_fixture.py LOGS`` from a research-mode run.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import e2e_fixture  # noqa: E402
from _records import find_runs  # noqa: E402
from replay import load_package  # noqa: E402

DATA = REPO / "tests" / "data" / "e2e"


class TestTheShippedDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        expected = DATA / "expected.json"
        if not expected.exists():
            raise unittest.SkipTest(f"no end-to-end data in {DATA}: build it with "
                                    f"python tools/e2e_fixture.py LOGS")
        cls.expected = json.loads(expected.read_text(encoding="utf-8"))
        cls.pkg = load_package(REPO)
        cls.cfg = cls.pkg.config.load_config()
        cls.runs, _ = find_runs(DATA)

    def test_config_yaml_is_the_one_the_data_was_built_with(self) -> None:
        # If this fails, config.yaml changed. Intended? Rebuild the data
        # (tools/e2e_fixture.py) and say so in the CHANGELOG.
        self.assertEqual(e2e_fixture.config_digest(self.cfg.as_dict()),
                         self.expected["config_digest"])

    def test_every_clip_is_decided_as_recorded(self) -> None:
        for clip, want in self.expected["clips"].items():
            with self.subTest(clip=clip):
                got = e2e_fixture.decisions(self.pkg, self.cfg, self.runs[clip])
                self.assertEqual(got["label"], want["label"])
                self.assertEqual(got["events"], want["events"])

    def test_the_data_covers_every_trigger_condition(self) -> None:
        # The point of the selection: the shipped combination, not one path.
        sources = {e["trigger_source"] for d in self.expected["clips"].values()
                   for e in d["events"]}
        self.assertEqual(sources, {"score", "H", "A"})

    def test_the_replay_writes_nothing_into_the_data(self) -> None:
        before = sorted(p.name for p in DATA.iterdir())
        with tempfile.TemporaryDirectory():
            e2e_fixture.decisions(self.pkg, self.cfg, next(iter(self.runs.values())))
        self.assertEqual(sorted(p.name for p in DATA.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
