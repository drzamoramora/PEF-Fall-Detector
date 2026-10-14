"""metrics.csv / metrics.txt (records schema 2, 0.3.0.dev0).

Checked against the Mac's phase-8.4 run, the numbers compared by hand across
platforms until now: the confusion matrix below is that run's, every clip
10.43 s, an alarm on every clip labelled PartiallyRecovered or NotRecovered.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector import run_metrics  # noqa: E402

MATRIX = {   # truth -> {detected: clips}
    "NoFall": {"NoFall": 49, "Recovered": 3, "NotRecovered": 4},
    "Recovered": {"NoFall": 3, "Recovered": 21},
    "PartiallyRecovered": {"NoFall": 1, "Recovered": 1, "PartiallyRecovered": 20,
                           "NotRecovered": 2},
    "NotRecovered": {"NoFall": 2, "PartiallyRecovered": 2, "NotRecovered": 20},
}


def _rows() -> list[dict]:
    rows, n = [], 0
    for truth, detected in MATRIX.items():
        for det, count in detected.items():
            for _ in range(count):
                n += 1
                actor = "B01" if truth == "NoFall" else "A01"
                rows.append({"video": f"{actor}-S{n}-{truth}.mp4", "true_class": truth,
                             "detected_class": det, "reviewed_class": "",
                             "duration_s": "10.43", "events": 1,
                             "alarm": int(det in ("PartiallyRecovered", "NotRecovered")),
                             "trigger_sources": "score"})
    return rows


class TestTheTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = run_metrics.compute(_rows())
        cls.index = {(r["split"], r["metric"], r["class"]): r for r in cls.rows}

    def value(self, metric: str, cls: str = "", split: str = "all") -> str:
        return self.index[(split, metric, cls)]["value"]

    def test_classification(self) -> None:
        acc = self.index[("all", "accuracy_4class", "")]
        self.assertEqual((acc["value"], acc["num"], acc["den"]), ("85.9", 110, 128))
        self.assertEqual((acc["ci_low"], acc["ci_high"]), ("78.9", "90.9"))
        self.assertEqual(self.value("falls_exact_class"), "84.7")
        self.assertEqual(self.value("macro_f1_4class"), "85.2")
        self.assertEqual(self.value("class_accuracy", "PartiallyRecovered"), "83.3")

    def test_fall_or_not_by_label(self) -> None:
        sens = self.index[("all", "sensitivity", "")]
        self.assertEqual((sens["value"], sens["num"], sens["den"]), ("91.7", 66, 72))
        self.assertEqual((sens["ci_low"], sens["ci_high"]), ("83.0", "96.1"))
        self.assertEqual(self.value("specificity_label"), "87.5")
        self.assertEqual(self.value("precision_label"), "90.4")
        self.assertEqual(self.value("f1_label"), "91.0")
        self.assertEqual(self.value("balanced_accuracy"), "89.6")
        self.assertEqual([self.value(k) for k in ("tp", "fp", "tn", "fn")],
                         ["66", "7", "49", "6"])
        self.assertEqual(self.value("false_pos_per_hour_label"), "43.1")

    def test_fall_or_not_by_alarm(self) -> None:
        self.assertEqual(self.value("help_falls_with_alarm"), "91.7")
        self.assertEqual(self.value("class_with_alarm", "Recovered"), "0.0")
        self.assertEqual(self.value("falls_with_alarm"), "61.1")
        spec = self.index[("all", "specificity_alarm", "")]
        self.assertEqual((spec["value"], spec["ci_low"], spec["ci_high"]),
                         ("92.9", "83.0", "97.2"))
        self.assertEqual(self.value("alarm_precision"), "91.7")
        self.assertEqual(self.value("false_pos_per_hour_alarm"), "24.7")

    def test_confusion_and_events(self) -> None:
        self.assertEqual(self.value("confusion", "NoFall->NotRecovered"), "4")
        self.assertEqual(self.value("events"), "128")
        self.assertEqual(self.value("events_by_trigger", "score"), "128")

    def test_partitions(self) -> None:
        rows = run_metrics.compute(_rows(), {"A01": "train", "B01": "test"})
        splits = {r["split"] for r in rows}
        self.assertEqual(splits, {"all", "train", "test"})
        clips = {r["split"]: r["value"] for r in rows if r["metric"] == "clips"}
        self.assertEqual(clips, {"all": "128", "train": "72", "test": "56"})

    def test_text_and_files(self) -> None:
        text = run_metrics.format_text(self.rows, _rows(), "run")
        self.assertIn("correct: 110/128", text)
        self.assertIn("85.9 % (110/128) [78.9–90.9]", text)
        self.assertIn("66 · 7 · 49 · 6", text)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, txt_path = run_metrics.write(tmp, self.rows, text)
            self.assertEqual(csv_path.read_text().splitlines()[0],
                             ",".join(run_metrics.METRIC_FIELDS))
            self.assertEqual(txt_path.read_text(), text)

    def test_performance_from_frame_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.csv"
            path.write_text("frame_index,decode_ms,pose_ms,analysis_ms,cpu_pct,peak_rss_mb\n"
                            + "".join(f"{i},1.00,{20 + i}.00,1.00,,\n" for i in range(20))
                            + "20,1.00,30.00,1.00,150.0,512.0\n", encoding="utf-8")
            perf = run_metrics.performance([path])
        self.assertEqual(perf["frames_measured"], 21)
        self.assertEqual(perf["pose_ms_median"], 30.0)
        self.assertAlmostEqual(perf["fps_median"], 1000 / 32)
        self.assertEqual(perf["peak_rss_mb"], 512.0)
        rows = run_metrics.compute(_rows(), perf=perf)
        text = run_metrics.format_text(rows)
        self.assertIn("Frames per second (median)", text)


if __name__ == "__main__":
    unittest.main()
