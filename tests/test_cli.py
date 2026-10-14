"""The installed commands: ``pef-lab`` (main.main), ``pef-run`` (main.run) and the tools.

``pef-run`` is what a box without a screen runs, so it must never reach the
GUI import, and must not need ``--headless`` to process a video.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

import main


class TestPefRun(unittest.TestCase):

    def _main(self, argv, **kw):
        with mock.patch.object(main, "run_headless", return_value=0) as headless, \
                mock.patch.object(main, "run_folder", return_value=0) as folder:
            code = main.main(argv, **kw)
        return code, headless, folder

    def test_a_video_runs_headless_without_asking(self) -> None:
        code, headless, folder = self._main(["--video", "x.mp4"], headless_only=True)
        self.assertEqual(code, 0)
        headless.assert_called_once()
        self.assertTrue(headless.call_args.args[0].headless)
        folder.assert_not_called()

    def test_a_folder_runs_as_before(self) -> None:
        code, headless, folder = self._main(["--folder", "clips"], headless_only=True)
        self.assertEqual(code, 0)
        folder.assert_called_once()
        headless.assert_not_called()

    def test_nothing_that_would_open_the_gui_is_accepted(self) -> None:
        for argv in ([], ["--camera", "0"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                self._main(argv, headless_only=True)

    def test_python_main_py_still_needs_headless_for_a_video(self) -> None:
        # Without --headless, a video opens PEF-Lab: unchanged for main.py.
        lab = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pef_fall_detector.gui.lab_window": lab}):
            code, headless, _folder = self._main(["--video", "x.mp4"])
        self.assertEqual(code, 0)
        headless.assert_not_called()
        lab.run_gui.assert_called_once()

    def test_resume_needs_no_folder(self) -> None:
        code, headless, folder = self._main(["--resume", "logs/run"], headless_only=True)
        self.assertEqual(code, 0)
        folder.assert_called_once()
        self.assertEqual(folder.call_args.args[0].resume, "logs/run")

    def test_clips_and_compare_to_go_with_a_folder(self) -> None:
        for argv in (["--video", "x.mp4", "--clips", "A01-S1"],
                     ["--video", "x.mp4", "--compare-to", "logs/ref"],
                     ["--resume", "logs/run", "--video", "x.mp4"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                self._main(argv, headless_only=True)
        code, _h, folder = self._main(["--folder", "clips", "--clips", "A01-S1",
                                       "--compare-to", "logs/ref"], headless_only=True)
        self.assertEqual(code, 0)
        args = folder.call_args.args[0]
        self.assertEqual((args.clips, args.compare_to), ("A01-S1", "logs/ref"))

    def test_run_is_main_headless_only(self) -> None:
        with mock.patch.object(main, "main", return_value=0) as m:
            self.assertEqual(main.run(), 0)
        m.assert_called_once_with(headless_only=True)


class TestToolCommands(unittest.TestCase):
    """pef-compare, pef-metrics, pef-replay and pef-export run the tools/ scripts."""

    def test_every_entry_point_reaches_its_tool(self) -> None:
        from pef_fall_detector import cli
        for entry, name in ((cli.compare_main, "compare_runs"), (cli.metrics_main, "metrics"),
                            (cli.replay_main, "replay"), (cli.export_main, "export")):
            with self.subTest(tool=name):
                fake = mock.MagicMock()
                fake.main.return_value = 0
                with mock.patch.object(cli, "tool", return_value=fake) as tool, \
                        mock.patch("sys.argv", ["pef-x", "a", "b"]):
                    self.assertEqual(entry(), 0)
                tool.assert_called_once_with(name)
                fake.main.assert_called_once_with(["a", "b"])

    def test_the_tools_are_found(self) -> None:
        from pef_fall_detector import cli
        for name in ("compare_runs", "metrics", "replay", "export"):
            self.assertTrue(hasattr(cli.tool(name), "main"), name)

    def test_the_commands_are_declared(self) -> None:
        from pathlib import Path
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        for line in ('pef-compare = "pef_fall_detector.cli:compare_main"',
                     'pef-metrics = "pef_fall_detector.cli:metrics_main"',
                     'pef-replay = "pef_fall_detector.cli:replay_main"',
                     'pef-export = "pef_fall_detector.cli:export_main"'):
            self.assertIn(line, text)


if __name__ == "__main__":
    unittest.main()
