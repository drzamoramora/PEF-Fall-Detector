"""The two installed commands: ``pef-lab`` (main.main) and ``pef-run`` (main.run).

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

    def test_run_is_main_headless_only(self) -> None:
        with mock.patch.object(main, "main", return_value=0) as m:
            self.assertEqual(main.run(), 0)
        m.assert_called_once_with(headless_only=True)


if __name__ == "__main__":
    unittest.main()
