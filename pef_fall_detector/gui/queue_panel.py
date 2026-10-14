"""PEF-Lab's Queue tab: the table of clips, its buttons and the running score.

The panel only shows the queue. Running it — which clip opens next, when a
label is written, what is frozen — is the window's job (``LabWindow``'s
queue flow), because it is the same playback path as a single clip.
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..classification import CLASSES
from ..dataset import truth_from_name
from ..evaluation import binary_metrics


def metrics_text(rows) -> str:
    """The §3.7 metrics, with the same code as the report.

    Uses evaluation.binary_metrics: the number on screen and the one in
    the report cannot differ, which is what evaluation's docstring asks.
    """
    scored = [r for r in rows if r["true_class"]]
    if not scored:
        return ""
    hits = sum(1 for r in scored if r["true_class"] == r["detected_class"])
    falls = [r for r in scored if r["true_class"] != "NoFall"]
    fall_hits = sum(1 for r in falls if r["true_class"] == r["detected_class"])
    b = binary_metrics(scored)
    parts = [f"4-class {hits}/{len(scored)} ({100.0 * hits / len(scored):.0f}%)"]
    if falls:
        parts.append(f"exact falls {fall_hits}/{len(falls)}")
    for key, name in (("sensitivity", "sens"), ("specificity", "spec")):
        v = b.get(key, float("nan"))
        if not math.isnan(v):
            parts.append(f"{name} {100.0 * v:.0f}%")
    return "  ·  ".join(parts)


class QueuePanel(QWidget):
    """The dataset pass: a folder of clips, labelled one after another.

    It reuses the ordinary playback path rather than running its own
    loop, so what you watch during the pass is exactly what produced the
    label — the same overlay, the same curves, the same pipeline. A
    separate fast path would label clips nobody ever sees, and the first
    disagreement with the annotation would have no way to be examined.
    """

    def __init__(self, style: QStyle) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.setToolTip("Run the whole queue, labelling every clip")
        buttons.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setIcon(style.standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.setToolTip("Stop after the current clip, keeping every label so far")
        buttons.addWidget(self.stop_btn)
        self.save_btn = QPushButton("Save Reviews")
        self.save_btn.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        self.save_btn.setToolTip("Write a labels file with the reviewed column filled in")
        buttons.addWidget(self.save_btn)
        buttons.addStretch(1)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("danger")
        self.clear_btn.setToolTip("Empty the queue")
        buttons.addWidget(self.clear_btn)
        layout.addLayout(buttons)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        progress_row.addWidget(self.progress, stretch=1)
        layout.addLayout(progress_row)
        self.score = QLabel("")
        self.score.setWordWrap(True)
        self.score.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.score)

        self.hint = QLabel("No clips yet. Use «Add Folder…» at the top to load a "
                           "folder of recordings.")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Video", "Truth", "Detected", "", "Reviewed"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        # Double-click (connected by the window) replays one clip on its own,
        # outside the pass, so a disagreement can be inspected frame by frame
        # without re-running the whole folder.
        layout.addWidget(self.table, stretch=1)

        # Review: move between the clips already run.
        #
        # Double-click already opened a single clip, but reviewing a folder
        # is comparing: PEF-FallDB's situations are recorded with four
        # subjects, and what explains why three fail and one succeeds is
        # almost never inside one clip — it is in the difference between
        # them. Going back to the table each time breaks that comparison, so
        # the two buttons and their shortcuts live here.
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.setShortcut(QKeySequence("Alt+Left"))
        self.prev_btn.setToolTip("Alt+←   (or double-click a row)")
        nav.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setShortcut(QKeySequence("Alt+Right"))
        self.next_btn.setToolTip("Alt+→   (or double-click a row)")
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

        self.review_label = QLabel("")
        self.review_label.setObjectName("muted")
        self.review_label.setWordWrap(True)
        layout.addWidget(self.review_label)

    def add_clips(self, clips: list[Path]) -> None:
        """One row per clip, after the rows already there; progress over all."""
        start = self.table.rowCount()
        self.table.setRowCount(start + len(clips))
        for offset, path in enumerate(clips):
            row = start + offset
            truth = truth_from_name(path)
            self.table.setItem(row, 0, QTableWidgetItem(path.name))
            self.table.setItem(row, 1, QTableWidgetItem(truth or "?"))
            self.table.setItem(row, 2, QTableWidgetItem(""))
            self.table.setItem(row, 3, QTableWidgetItem(""))
            box = QComboBox()
            box.addItems(("",) + tuple(CLASSES))
            # The human column. A dataset needs both opinions: without the
            # reviewed one there is nothing to score the machine against
            # except a file name, and file names are where this material
            # already carries typos.
            self.table.setCellWidget(row, 4, box)
        self.progress.setRange(0, self.table.rowCount())
        self.progress.setValue(0)

    def show_result(self, row: int, detected: str, truth: str, score: str) -> None:
        self.table.setItem(row, 2, QTableWidgetItem(detected))
        mark = "" if not truth else ("OK" if truth == detected else "X")
        item = QTableWidgetItem(mark)
        if mark == "X":
            item.setForeground(Qt.red)
        self.table.setItem(row, 3, item)
        self.progress.setValue(row + 1)
        self.score.setText(score)

    def reviewed(self, row: int) -> str | None:
        """The human column of one row; None if the row has no box."""
        box = self.table.cellWidget(row, 4)
        return box.currentText() if box is not None else None

    def select(self, row: int) -> None:
        self.table.selectRow(row)
        item = self.table.item(row, 0)
        if item is not None:
            self.table.scrollToItem(item)

    def clear(self) -> None:
        self.table.setRowCount(0)
        self.progress.setRange(0, 0)
        self.score.setText("")
