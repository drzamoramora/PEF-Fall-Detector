"""The layout of B4 (0.2.0.dev4): the funnel and the quantities as packages.

Each piece lives in its own module, but every name is still read from where
it always was, and the quantities are split into the ones §3.4 defines and
the experimental ones.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pef_fall_detector import quantities, state_machine  # noqa: E402
from pef_fall_detector.audit_log import EXPERIMENTAL_QUANTITIES  # noqa: E402
from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from pef_fall_detector.state_machine.build import build_machine  # noqa: E402
from tests.support import TEST_CFG, make_pose  # noqa: E402


class TestPackages(unittest.TestCase):

    def test_every_exported_name_exists(self) -> None:
        for pkg in (state_machine, quantities):
            missing = [n for n in pkg.__all__ if not hasattr(pkg, n)]
            self.assertEqual(missing, [], pkg.__name__)

    def test_each_class_lives_in_its_module(self) -> None:
        homes = {"Stage1Trigger": "trigger", "PeakWindow": "trigger",
                 "Stage2Evaluator": "stage2", "Stage3Evaluator": "stage3",
                 "FallStateMachine": "machine", "TriggerEvent": "events", "Stage": "events"}
        for name, module in homes.items():
            self.assertEqual(getattr(state_machine, name).__module__,
                             f"pef_fall_detector.state_machine.{module}", name)


class TestQuantityLists(unittest.TestCase):

    def test_the_four_of_the_paper(self) -> None:
        self.assertEqual(sorted(quantities.PAPER), ["I", "P", "T", "V"])

    def test_no_quantity_is_in_both_lists(self) -> None:
        self.assertEqual(set(quantities.PAPER) & set(quantities.EXPERIMENTAL), set())

    def test_the_experimental_columns_are_marked(self) -> None:
        # What the experimental modules measure is recorded with _EXP.
        for column in ("H_ratio", "reach_proximity", "A_head", "A_hip", "extension_ratio"):
            self.assertIn(column, EXPERIMENTAL_QUANTITIES)
        for column in ("T_deg", "V_tps", "P_offset", "I_still_s"):
            self.assertNotIn(column, EXPERIMENTAL_QUANTITIES)


class TestBuild(unittest.TestCase):

    def test_the_pipeline_uses_the_built_machine(self) -> None:
        cfg = Config(copy.deepcopy(TEST_CFG))
        built = build_machine(cfg, labelling=True)
        pipe = FramePipeline(cfg, labelling=True)
        self.assertEqual(vars(built.trigger).keys(), vars(pipe.machine.trigger).keys())
        self.assertTrue(built.stage3.defer_to_end)
        self.assertFalse(build_machine(cfg, labelling=False).stage3.defer_to_end)


class TestMovedMeasurements(unittest.TestCase):
    """What the per-frame measurements kept when they left the pipeline."""

    def test_the_trunk_guard_forgets_its_reference_when_cleared(self) -> None:
        from pef_fall_detector.quantities import TrunkGuard
        guard = TrunkGuard(min_ratio=0.15, window_s=1.0)
        for k in range(5):
            self.assertTrue(guard.plausible(100.0, k * 0.1))
        self.assertFalse(guard.plausible(5.0, 0.5))      # collapsed against the reference
        guard.clear()
        self.assertTrue(guard.plausible(5.0, 0.6))       # no reference after a break

    def test_an_unmeasured_trunk_never_becomes_the_reference(self) -> None:
        from pef_fall_detector.quantities import TrunkGuard
        guard = TrunkGuard(min_ratio=0.15, window_s=1.0)
        for k in range(3):
            self.assertTrue(guard.plausible(100.0, k * 0.1))
        for k in range(3, 6):
            self.assertTrue(guard.plausible(float("nan"), k * 0.1))   # nothing to judge
        # Had the NaNs entered the history, the reference would be NaN and a
        # normal trunk would read as implausible.
        self.assertTrue(guard.plausible(100.0, 0.6))

    def test_the_leg_extension_restarts_its_smoothing_on_reset(self) -> None:
        from pef_fall_detector.quantities import LegExtension
        ext = LegExtension(max_ratio=3.0, time_constant_s=0.5)
        first = ext.update(make_pose(0, 0.0, ankle_y=700.0), 0.0, 0.5)
        blended = ext.update(make_pose(1, 0.1, ankle_y=560.0), 0.1, 0.5)
        ext.reset()
        fresh = ext.update(make_pose(2, 0.2, ankle_y=560.0), 0.1, 0.5)
        # After a reset the next value is the measurement itself, as for a
        # fresh filter: nothing of the old posture is carried over.
        alone = LegExtension(max_ratio=3.0, time_constant_s=0.5).update(
            make_pose(2, 0.2, ankle_y=560.0), 0.1, 0.5)
        self.assertEqual(fresh, alone)
        self.assertNotEqual(blended, alone)
        self.assertGreater(first, alone)

    def test_immobility_ignores_a_hidden_landmark(self) -> None:
        from pef_fall_detector.quantities import ImmobilityTimer
        from pef_fall_detector.quantities import i as quantity_i
        timer = ImmobilityTimer(0.05)
        quantity_i.measure(timer, make_pose(0, 0.0, nose_visible=False), 0.0, 0.5)
        moved = make_pose(1, 1.0, nose_visible=False)
        moved.landmarks[0, 0] += 300.0                   # the hidden nose jumps
        still_s, _ = quantity_i.measure(timer, moved, 1.0, 0.5)
        self.assertGreater(still_s, 0.5)


class TestGuiLayout(unittest.TestCase):
    """B6 (0.2.0.dev6): the window is the flow; what it shows lives beside it."""

    def _source(self, name: str) -> str:
        return (REPO / "pef_fall_detector" / "gui" / name).read_text(encoding="utf-8")

    def test_the_window_draws_no_plot_and_builds_no_table(self) -> None:
        window = self._source("lab_window.py")
        for marker in ("pyqtgraph", "pg.", "PlotWidget", "QTableWidget", "QCheckBox"):
            self.assertNotIn(marker, window, marker)

    def test_the_panels_never_read_a_decision(self) -> None:
        # Looks only: no panel builds a pipeline, a machine or a record.
        for name in ("plots.py", "queue_panel.py", "widgets.py"):
            source = self._source(name)
            for marker in ("FramePipeline", "build_machine", "RunRecords", "AuditLogger"):
                self.assertNotIn(marker, source, f"{name}: {marker}")



if __name__ == "__main__":
    unittest.main()
