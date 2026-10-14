"""The configuration is checked before anything reads it (0.2.0.dev2).

config.yaml is the only place a value lives: the code keeps no fallbacks, so
a missing key, a key this version does not read or a value of the wrong type
must stop the program, naming the key, instead of running on something else.
"""

from __future__ import annotations

import copy
import math
import re
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pef_fall_detector.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PARTITION_PATH,
    SCHEMA,
    Config,
    ConfigError,
    load_config,
    research_mode,
    validate,
)
from pef_fall_detector.evaluation import load_split  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.state_machine import FORMULATIONS  # noqa: E402
from tests.support import TEST_CFG  # noqa: E402


def _problems(section: str, key: str, value) -> list[str]:
    cfg = copy.deepcopy(TEST_CFG)
    cfg[section][key] = value
    return validate(cfg)


class TestShippedConfig(unittest.TestCase):

    def test_it_loads(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.logging.mode, "deployment")

    def test_it_has_every_key_in_the_schema_order(self) -> None:
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(list(data), list(SCHEMA))
        for section, keys in SCHEMA.items():
            self.assertEqual(list(data[section]), list(keys), section)

    def test_the_test_config_is_complete(self) -> None:
        self.assertEqual(validate(TEST_CFG), [])
        for section, keys in SCHEMA.items():
            self.assertEqual(set(TEST_CFG[section]), set(keys), section)

    def test_the_formulations_come_from_the_trigger(self) -> None:
        self.assertEqual(SCHEMA["stage1"]["trigger_formulation"], FORMULATIONS)


class TestRefused(unittest.TestCase):
    """Every problem is named, all of them at once."""

    def test_a_missing_key(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        del cfg["stage3"]["getup_window_s"]
        with self.assertRaises(ConfigError) as ctx:
            Config(cfg)
        self.assertEqual(ctx.exception.problems, ["missing key 'stage3.getup_window_s'"])
        self.assertIn("stage3.getup_window_s", str(ctx.exception))

    def test_a_missing_section(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        del cfg["experimental"]
        self.assertEqual(validate(cfg), ["missing section 'experimental'"])

    def test_an_unknown_key_and_section(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        cfg["stage1"]["trigger_scor"] = 2.6
        cfg["stage4"] = {}
        problems = validate(cfg)
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith("unknown key 'stage1.trigger_scor'"))
        self.assertEqual(problems[1], "unknown section 'stage4'")

    def test_the_renamed_key_says_what_replaced_it(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        del cfg["logging"]["mode"]
        cfg["logging"]["save_landmarks"] = True
        problems = validate(cfg)
        self.assertEqual(problems[0], "missing key 'logging.mode'")
        self.assertIn("unknown key 'logging.save_landmarks' (replaced by logging.mode",
                      problems[1])

    def test_all_problems_at_once(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        del cfg["pose"]["reacquire_after_s"]
        cfg["stage2"]["com_min_samples"] = 3.0
        cfg["alerts"]["dispatch_verdicts"] = "stage3_confirmed"
        self.assertEqual(len(validate(cfg)), 3)

    def test_a_section_that_is_not_a_mapping(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        cfg["stage1"] = None
        self.assertEqual(validate(cfg), ["'stage1' must be a section of keys, got None"])

    def test_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, re.escape(str(path))):
                load_config(path)

    def test_the_file_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.yaml"
            cfg = copy.deepcopy(TEST_CFG)
            del cfg["quantity_a"]
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
        self.assertTrue(str(ctx.exception).startswith(f"{path} is not a valid configuration"))
        self.assertEqual(ctx.exception.problems, ["missing section 'quantity_a'"])


class TestTypes(unittest.TestCase):
    """A value of the wrong kind is refused, not converted."""

    def test_numbers(self) -> None:
        for good in (0.5, 1, -1.5):
            self.assertEqual(_problems("stage1", "trigger_hold_s", good), [], good)
        for bad in (True, "0.1", None, math.nan, math.inf, [0.1]):
            problems = _problems("stage1", "trigger_hold_s", bad)
            self.assertEqual(len(problems), 1, bad)
            self.assertIn("'stage1.trigger_hold_s' must be a finite number", problems[0])

    def test_whole_numbers(self) -> None:
        self.assertEqual(_problems("stage2", "com_min_samples", 3), [])
        for bad in (3.0, True, "3"):
            self.assertIn("must be a whole number",
                          _problems("stage2", "com_min_samples", bad)[0], bad)

    def test_flags(self) -> None:
        self.assertEqual(_problems("quantity_a", "reach_check", False), [])
        for bad in (1, 0, "true", None):
            self.assertIn("must be true or false",
                          _problems("quantity_a", "reach_check", bad)[0], bad)

    def test_text_and_text_lists(self) -> None:
        self.assertIn("must be text", _problems("logging", "output_dir", 5)[0])
        self.assertEqual(_problems("alerts", "dispatch_verdicts", []), [])
        for bad in ("stage3_confirmed", [3], None):
            self.assertIn("must be a list of text",
                          _problems("alerts", "dispatch_verdicts", bad)[0], bad)

    def test_choices(self) -> None:
        for good in FORMULATIONS:
            self.assertEqual(_problems("stage1", "trigger_formulation", good), [])
        self.assertIn("must be one of score",
                      _problems("stage1", "trigger_formulation", "Score")[0])
        self.assertIn("must be one of research, deployment",
                      _problems("logging", "mode", "dev")[0])
        # true == 1 and 1.0 == 1 in Python; neither is model 1.
        for bad in (True, 1.0, 3):
            self.assertIn("must be one of 0, 1, 2",
                          _problems("pose", "model_complexity", bad)[0], bad)


class TestReading(unittest.TestCase):

    def test_sections_are_views_of_the_checked_mapping(self) -> None:
        cfg = Config(copy.deepcopy(TEST_CFG))
        self.assertEqual(cfg.stage1.threshold_T_deg, 45.0)
        self.assertIs(cfg.stage1.as_dict(), cfg.as_dict()["stage1"])

    def test_a_key_outside_the_schema_is_an_error_naming_it(self) -> None:
        cfg = Config(copy.deepcopy(TEST_CFG))
        with self.assertRaisesRegex(AttributeError, r"'stage1\.nope' is not a configuration key"):
            _ = cfg.stage1.nope

    def test_the_logging_modes(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        cfg["logging"]["mode"] = "research"
        self.assertTrue(research_mode(Config(cfg)))
        cfg["logging"]["mode"] = "deployment"
        self.assertFalse(research_mode(Config(cfg)))

    def test_the_code_reads_no_fallback_values(self) -> None:
        # The pattern every fallback had before 0.2.0.dev2:
        # cfg.stage3.as_dict().get("getup_window_s", 2.0).
        pattern = re.compile(r"as_dict\(\)\s*\.get\(")
        sources = [REPO / "main.py", *sorted((REPO / "pef_fall_detector").rglob("*.py")),
                   *sorted((REPO / "tools").glob("*.py"))]
        found = [f"{p.relative_to(REPO)}:{n}" for p in sources
                 for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if pattern.search(line)]
        self.assertEqual(found, [])


def _state(obj, depth: int = 0, seen: set | None = None) -> dict:
    """The plain attributes of ``obj`` and of the package objects it holds."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 3:
        return {}
    seen.add(id(obj))
    plain = (bool, int, float, str, type(None))
    out: dict = {}
    for name, value in vars(obj).items():
        if isinstance(value, plain):
            out[name] = repr(value)
        elif isinstance(value, (tuple, list)) and all(isinstance(v, plain) for v in value):
            out[name] = repr(tuple(value))
        elif isinstance(value, dict) and all(isinstance(v, plain) for v in value.values()):
            out[name] = repr(sorted(value.items()))
        elif isinstance(value, deque):
            out[name] = f"deque(maxlen={value.maxlen})"
        elif (type(value).__module__.startswith("pef_fall_detector")
              and hasattr(value, "__dict__") and not isinstance(value, Config)):
            out[name] = _state(value, depth + 1, seen)
    return out


def _other(section: str, key: str, value):
    """Another valid value for ``section.key``."""
    kind = SCHEMA[section][key]
    if isinstance(kind, tuple):
        return next(choice for choice in kind if choice != value)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return round(value * 1.1, 6) if value else 0.05
    if isinstance(value, list):
        return value + ["stage3_nullified"]
    return value + "-other"


class TestWiring(unittest.TestCase):
    """Every key of config.yaml reaches what reads it: no value lives in the code.

    Each key is changed on its own, and what is built from the configuration
    must change with it. A value copied into the code, even as the same
    number, fails here.
    """

    #: Read by PoseFrontend when the first frame arrives (the second test).
    FRONTEND = ("model_complexity", "min_detection_confidence", "min_tracking_confidence",
                "smooth_landmarks", "visibility_threshold", "reacquire_after_s")
    #: Read by the writers, main.py and PEF-Lab, not by the pipeline
    #: (tests/test_landmarks_log.py runs both).
    WRITERS = (("logging", "output_dir"), ("logging", "mode"))

    def test_every_key_reaches_the_pipeline(self) -> None:
        base = load_config().as_dict()
        reference = _state(FramePipeline(Config(copy.deepcopy(base)), labelling=True))
        # The comparison means something only if the same values build the same state.
        self.assertEqual(_state(FramePipeline(Config(copy.deepcopy(base)), labelling=True)),
                         reference)
        missed = []
        for section, keys in SCHEMA.items():
            for key in keys:
                if (section, key) in self.WRITERS or (
                        section == "pose" and key != "visibility_threshold"):
                    continue
                cfg = copy.deepcopy(base)
                cfg[section][key] = _other(section, key, cfg[section][key])
                if _state(FramePipeline(Config(cfg), labelling=True)) == reference:
                    missed.append(f"{section}.{key}")
        self.assertEqual(missed, [])

    def test_every_pose_key_reaches_the_frontend(self) -> None:
        def frontend_kwargs(cfg: dict) -> dict:
            pipe = FramePipeline(Config(cfg))
            with mock.patch("pef_fall_detector.pipeline.PoseFrontend") as fake:
                fake.return_value.process.side_effect = RuntimeError("stop")
                with self.assertRaises(RuntimeError):
                    pipe.process(None, 0, 0.0)
            return fake.call_args.kwargs

        base = load_config().as_dict()
        reference = frontend_kwargs(copy.deepcopy(base))
        self.assertEqual(frontend_kwargs(copy.deepcopy(base)), reference)
        for key in self.FRONTEND:
            cfg = copy.deepcopy(base)
            cfg["pose"][key] = _other("pose", key, cfg["pose"][key])
            self.assertNotEqual(frontend_kwargs(cfg), reference, f"pose.{key}")


class TestPartition(unittest.TestCase):

    def test_it_lives_in_config(self) -> None:
        self.assertEqual(PARTITION_PATH, REPO / "config" / "partition.yaml")
        split = load_split(PARTITION_PATH)
        self.assertEqual(len(split), 32)
        self.assertEqual(sum(side == "test" for side in split.values()), 10)


if __name__ == "__main__":
    unittest.main()
