"""Tests for the audit record (Lote C: M1, M4, M8 and the run metadata).

The CSV is the project's evidence: the artifact behind the paper's claim
that any alert can be reconstructed from named physical quantities, and the
raw material for every calibration plot. These tests pin the properties that
make it trustworthy — that it never loses a value silently, that it can be
interpreted without external context, that it lands where it is expected,
and that it is not created before there is anything to record.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pef_fall_detector import __version__
from pef_fall_detector.audit_log import (
    EXPERIMENTAL_QUANTITIES,
    EXPERIMENTAL_SUFFIX,
    PHASE2_FIELDS,
    AuditLogger,
    csv_column,
)
from pef_fall_detector.config import REPO_ROOT


class TestLazyCreation(unittest.TestCase):
    """M8: opening a source must not litter the folder with empty files."""

    def test_no_file_until_the_first_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip")
            self.assertFalse(
                logger.path.exists(),
                "el archivo se creó antes de tener una sola fila",
            )
            logger.close()
            self.assertFalse(logger.path.exists())

    def test_file_appears_with_the_first_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip")
            logger.log({"frame_index": 0, "timestamp_s": "0.0"})
            logger.close()
            self.assertTrue(logger.path.exists())
            with open(logger.path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)


class TestSchemaValidation(unittest.TestCase):
    """M1: a mistyped column name must fail loudly, not vanish."""

    def test_unknown_column_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip")
            with self.assertRaises(ValueError):
                logger.log({"frame_index": 0, "V_pts": "-1.2"})  # typo for V_tps
            logger.close()

    def test_known_columns_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip")
            logger.log({name: "" for name in PHASE2_FIELDS})
            logger.close()
            self.assertTrue(logger.path.exists())


class TestExperimentalMarking(unittest.TestCase):
    """The ``_EXP`` convention: what the paper defines vs. what we are testing.

    Without a mark in the record itself, a column that exists to be evaluated
    is indistinguishable from one the paper commits to. These tests pin both
    directions of the rule, because either failure is bad in its own way: an
    unmarked experimental column overstates a result, and a marked paper
    quantity understates one.
    """

    def test_every_experimental_quantity_is_marked(self) -> None:
        for name in EXPERIMENTAL_QUANTITIES:
            column = csv_column(name)
            self.assertTrue(column.endswith(EXPERIMENTAL_SUFFIX),
                            f"{name} es experimental pero su columna no lo dice")
            self.assertIn(column, PHASE2_FIELDS,
                          f"{column} no está en el esquema")

    def test_paper_quantities_are_not_marked(self) -> None:
        # T and V are §3.4. The centroid is too — we only smooth it, which is
        # a divergence to fix in the draft (P3), not a new quantity.
        for name in ("T_deg", "V_tps", "centroid_x_px", "centroid_y_px"):
            self.assertEqual(csv_column(name), name)
            self.assertIn(name, PHASE2_FIELDS)

    def test_record_hygiene_columns_are_not_marked(self) -> None:
        # These are ours, but they are not measurements under evaluation.
        # Marking them would blur what the suffix means.
        for name in ("reliable", "core_visibility"):
            self.assertEqual(csv_column(name), name)

    def test_no_unmarked_column_shares_a_name_with_an_experimental_one(self) -> None:
        # Guards the rename: if a stale unsuffixed spelling survived anywhere
        # in the schema, the record would carry the same quantity twice.
        for name in EXPERIMENTAL_QUANTITIES:
            self.assertNotIn(name, PHASE2_FIELDS,
                             f"{name} quedó también sin sufijo en el esquema")


class TestOutputLocation(unittest.TestCase):
    """M4: a relative output folder must not depend on where you stood."""

    def test_relative_path_anchors_to_the_repository(self) -> None:
        logger = AuditLogger("logs", "clip")
        self.assertEqual(logger.path.parent, REPO_ROOT / "logs")
        logger.close()

    def test_absolute_path_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip")
            self.assertEqual(logger.path.parent, Path(tmp))
            logger.close()


class TestRunMetadata(unittest.TestCase):
    """The record must be interpretable without external context.

    Reading a CSV alone cannot tell you which thresholds produced it, at what
    frame rate, or with which version of the code — and those are exactly the
    facts the calibration protocol has to cite. A sidecar file carries them.
    """

    META = {"source": "clip.mp4", "source_fps": 25.0, "frame_size": [1280, 960]}

    def test_sidecar_is_written_next_to_the_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip", metadata=dict(self.META))
            logger.log({"frame_index": 0})
            logger.close()
            side = logger.path.with_suffix(".meta.json")
            self.assertTrue(side.exists(), "no se escribió el archivo de metadatos")
            meta = json.loads(side.read_text())
            self.assertEqual(meta["source_fps"], 25.0)
            self.assertEqual(meta["schema"], PHASE2_FIELDS)
            self.assertEqual(meta["rows"], 1)
            # Assert the VALUES, not merely that the keys exist: a record
            # stamped with an empty version is as untraceable as one with no
            # stamp at all.
            self.assertEqual(meta["code_version"], __version__)
            self.assertRegex(meta["written_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    def test_config_snapshot_travels_with_the_record(self) -> None:
        # This is what makes a calibration reproducible: the exact thresholds
        # that produced these numbers, stored beside them.
        with tempfile.TemporaryDirectory() as tmp:
            meta = dict(self.META, config={"stage1": {"threshold_V": -1.5}})
            logger = AuditLogger(tmp, "clip", metadata=meta)
            logger.log({"frame_index": 0})
            logger.close()
            stored = json.loads(logger.path.with_suffix(".meta.json").read_text())
            self.assertEqual(stored["config"]["stage1"]["threshold_V"], -1.5)

    def test_no_sidecar_when_nothing_was_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp, "clip", metadata=dict(self.META))
            logger.close()
            self.assertFalse(logger.path.with_suffix(".meta.json").exists())


if __name__ == "__main__":
    unittest.main()
