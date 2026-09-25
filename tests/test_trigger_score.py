"""El puntaje del disparador, cuadro a cuadro, es lo que la Etapa 1 compara.

Las líneas punteadas de threshold_T y threshold_V en PEF-Lab no dicen nada del
disparo cuando la formulación es "score": lo que se compara es
T/threshold_T + V/threshold_V sobre la ventana de pico. Estas pruebas fijan
que el valor expuesto sea exactamente ese.
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.audit_log import PHASE2_FIELDS  # noqa: E402
from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.pipeline import FramePipeline  # noqa: E402
from tests.test_main_headless import _toppling_pose  # noqa: E402
from tests.test_pipeline import TEST_CFG, make_pose  # noqa: E402

DT = 1.0 / 30.0


def _fall():
    frames = [make_pose(i, i * DT) for i in range(30)]           # 1 s standing
    frames += [_toppling_pose(i, i * DT, angle_deg=(i - 30) * 4.5,
                              hip_y=500.0 + (i - 30) * 12.0) for i in range(30, 50)]
    frames += [_toppling_pose(i, i * DT, angle_deg=90.0, hip_y=740.0)
               for i in range(50, 90)]
    return frames


def _run(formulation: str):
    cfg = copy.deepcopy(TEST_CFG)
    cfg["stage1"]["trigger_formulation"] = formulation
    cfg["stage1"]["trigger_peak_window_s"] = 0.8
    pipe = FramePipeline(Config(cfg))
    return pipe, [pipe.analyze(pf) for pf in _fall()]


class TestTriggerScore(unittest.TestCase):

    def test_it_fires_exactly_where_the_score_says(self) -> None:
        pipe, results = _run("score")
        thr = float(TEST_CFG["stage1"]["trigger_score"])
        fired = [i for i, r in enumerate(results) if r.event is not None]
        self.assertEqual(len(fired), 1, "la Etapa 1 no disparo una vez")
        f = fired[0]
        self.assertGreaterEqual(results[f].quantities["trigger_score"], thr)
        # The hold: the score was already at/over the threshold for the
        # frames just before firing, and never before that stretch began.
        over = [i for i, r in enumerate(results[:f])
                if not math.isnan(r.quantities["trigger_score"])
                and r.quantities["trigger_score"] >= thr]
        self.assertTrue(over, "el puntaje no estuvo sobre el umbral antes del disparo")
        self.assertEqual(over, list(range(over[0], f)))

    def test_standing_scores_far_below(self) -> None:
        _, results = _run("score")
        early = [r.quantities["trigger_score"] for r in results[5:30]]
        self.assertTrue(all(math.isnan(s) or s < 1.0 for s in early))

    def test_other_formulations_have_no_score(self) -> None:
        _, results = _run("sequential")
        self.assertTrue(all(math.isnan(r.quantities["trigger_score"]) for r in results))

    def test_an_unreliable_frame_has_no_score(self) -> None:
        cfg = copy.deepcopy(TEST_CFG)
        cfg["stage1"]["trigger_formulation"] = "score"
        pipe = FramePipeline(Config(cfg))
        for pf in _fall()[:40]:
            pipe.analyze(pf)
        low = replace(make_pose(40, 40 * DT), core_visibility=0.1)
        self.assertTrue(math.isnan(pipe.analyze(low).quantities["trigger_score"]))

    def test_it_is_a_column_of_the_frame_record(self) -> None:
        self.assertIn("trigger_score", PHASE2_FIELDS)


if __name__ == "__main__":
    unittest.main()
