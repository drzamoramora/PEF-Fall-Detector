"""La memoria del análisis congelado (gui/review_cache.py), sin Qt."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pef_fall_detector.gui.review_cache import (  # noqa: E402
    PassRecorder,
    ReviewCache,
    config_key,
)


def _pass(n=5, path="/tmp/A01-S1-Recovered.mp4", key="k"):
    rec = PassRecorder(path, key, fps=30.0)
    for i in range(n):
        rec.record(i, f"resultado-{i}")
    return rec


class TestConfigKey(unittest.TestCase):

    def test_same_content_same_key_whatever_the_order(self) -> None:
        self.assertEqual(config_key({"a": 1, "b": {"c": 2}}),
                         config_key({"b": {"c": 2}, "a": 1}))

    def test_any_change_changes_the_key(self) -> None:
        self.assertNotEqual(config_key({"stage1": {"trigger_score": 2.6}}),
                            config_key({"stage1": {"trigger_score": 2.5}}))


class TestPassRecorder(unittest.TestCase):

    def test_a_continuous_pass_freezes_every_frame(self) -> None:
        a = _pass(5).finish(events=["ev"], record_path="log.csv")
        self.assertIsNotNone(a)
        self.assertEqual(a.frame_count, 5)
        self.assertEqual(a.frames[3], "resultado-3")
        self.assertEqual((a.events, a.record_path), (["ev"], "log.csv"))

    def test_a_jump_forward_invalidates_the_pass(self) -> None:
        rec = _pass(3)
        rec.record(7, "salto")
        self.assertFalse(rec.valid)
        self.assertIsNone(rec.finish(events=[]))

    def test_revisiting_a_frame_invalidates_the_pass(self) -> None:
        rec = _pass(3)
        rec.record(1, "otra vez")
        self.assertIsNone(rec.finish(events=[]))

    def test_a_pass_that_did_not_start_at_zero_is_not_frozen(self) -> None:
        rec = PassRecorder("/tmp/x.mp4", "k", 30.0)
        rec.record(4, "r")
        self.assertIsNone(rec.finish(events=[]))

    def test_an_explicit_invalidation_is_kept_with_its_first_reason(self) -> None:
        rec = _pass(3)
        rec.invalidate("salto durante la pasada")
        rec.invalidate("otro")
        self.assertEqual(rec.invalid_reason, "salto durante la pasada")
        self.assertIsNone(rec.finish(events=[]))

    def test_an_empty_pass_is_not_frozen(self) -> None:
        self.assertIsNone(PassRecorder("/tmp/x.mp4", "k", 30.0).finish(events=[]))

    def test_after_invalidation_nothing_more_is_kept(self) -> None:
        rec = _pass(2)
        rec.record(5, "salto")
        rec.record(2, "r2")
        self.assertIsNone(rec.finish(events=[]))


class TestReviewCache(unittest.TestCase):

    def test_hit_only_with_the_same_config(self) -> None:
        cache = ReviewCache()
        cache.store(_pass(key="k1").finish(events=[]))
        self.assertIsNotNone(cache.get("/tmp/A01-S1-Recovered.mp4", "k1"))
        self.assertIsNone(cache.get("/tmp/A01-S1-Recovered.mp4", "k2"))

    def test_paths_are_compared_resolved(self) -> None:
        cache = ReviewCache()
        cache.store(_pass(path="/tmp/../tmp/A01-S1-Recovered.mp4").finish(events=[]))
        self.assertIsNotNone(cache.get("/tmp/A01-S1-Recovered.mp4", "k"))

    def test_drop_forgets_one_clip(self) -> None:
        cache = ReviewCache()
        cache.store(_pass(path="/tmp/a.mp4").finish(events=[]))
        cache.store(_pass(path="/tmp/b.mp4").finish(events=[]))
        cache.drop("/tmp/a.mp4", "k")
        self.assertIsNone(cache.get("/tmp/a.mp4", "k"))
        self.assertIsNotNone(cache.get("/tmp/b.mp4", "k"))
        self.assertEqual(len(cache), 1)

    def test_a_new_pass_replaces_the_old_one(self) -> None:
        cache = ReviewCache()
        cache.store(_pass(3).finish(events=["viejo"]))
        cache.store(_pass(4).finish(events=["nuevo"]))
        got = cache.get("/tmp/A01-S1-Recovered.mp4", "k")
        self.assertEqual((got.events, got.frame_count), (["nuevo"], 4))


class TestRecordsNeverOverwrite(unittest.TestCase):
    """Two records of one source opened in the same second: both survive."""

    def test_the_second_takes_a_numbered_name(self) -> None:
        import tempfile
        from pef_fall_detector.audit_log import AuditLogger
        with tempfile.TemporaryDirectory() as out:
            first = AuditLogger(out, "A01-S1", fields=["x"])
            second = AuditLogger(out, "A01-S1", fields=["x"])
            first.log({"x": 1})
            second.log({"x": 2})
            first.close()
            second.close()
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(sorted(p.name for p in Path(out).glob("*.csv")),
                             sorted([first.path.name, second.path.name]))
            self.assertIn("1", first.path.read_text())


if __name__ == "__main__":
    unittest.main()
