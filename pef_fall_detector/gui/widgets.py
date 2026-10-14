"""PEF-Lab's small building blocks: cards, captions, the video view, the View tab.

Looks only, like :mod:`.theme`: nothing here reads a frame result or the
configuration, so nothing here can change what the detector measures or
decides.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import theme


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setStyleSheet(f"color: {theme.BORDER};")
    return line


def caption(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("cardTitle")
    return label


def card(title: str) -> tuple[QFrame, QVBoxLayout]:
    """A titled panel: rounded, one step lighter than the window."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 8, 12, 10)
    layout.setSpacing(6)
    layout.addWidget(caption(title))
    return frame, layout


def legend_row(colour: str, text: str) -> QWidget:
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    swatch = QLabel()
    swatch.setFixedSize(14, 14)
    swatch.setStyleSheet(f"background-color: {colour}; border: 1px solid {theme.BORDER}; "
                         f"border-radius: 3px;")
    h.addWidget(swatch)
    h.addWidget(QLabel(text), stretch=1)
    return row


class VideoView(QLabel):
    """The video frame, rescaled to whatever space the layout gives it.

    Its size policy is Ignored so the pixmap never drives the layout (a
    pixmap-sized label grows the window frame after frame); the explicit
    minimum still holds.
    """

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(426, 240)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._source: QPixmap | None = None

    def show_image(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        target = self.size() - QSize(4, 4)
        self.setPixmap(self._source.scaled(target, Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._rescale()


def fit_to_screen(window: QWidget) -> None:
    """Open at a size that fits the screen, leaving room for the menu bar and dock."""
    screen = QApplication.primaryScreen()
    if screen is None:
        window.resize(1280, 800)
        return
    geo = screen.availableGeometry()
    window.resize(min(1520, int(geo.width() * 0.94)), min(960, int(geo.height() * 0.92)))


class ViewPanel(QWidget):
    """The View tab: what is drawn on the video, the curves switch, the legend."""

    def __init__(self) -> None:
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setContentsMargins(12, 10, 12, 10)
        vl.setSpacing(4)
        vl.addWidget(caption("Draw on the video"))
        self.skeleton = QCheckBox("Skeleton")
        self.trunk = QCheckBox("Trunk vector")
        self.centroid = QCheckBox("Centroid")
        self.support = QCheckBox("Support polygon / COM")
        for chk in (self.skeleton, self.trunk, self.centroid, self.support):
            chk.setChecked(True)
            vl.addWidget(chk)
        vl.addSpacing(8)
        vl.addWidget(caption("Panels"))
        self.curves = QCheckBox("Curves and timeline")
        self.curves.setChecked(True)
        vl.addWidget(self.curves)
        vl.addSpacing(8)
        vl.addWidget(caption("Timeline colours"))
        for colour, text in ((theme.STAGE_COLOURS["MONITORING"], "Monitoring"),
                             (theme.STAGE_COLOURS["CONFIRMING"], "Stage 2 · geometry (P)"),
                             (theme.STAGE_COLOURS["OBSERVING"], "Stage 3 · observing"),
                             (theme.STAGE_COLOURS["COOLDOWN"], "Cooldown"),
                             ("#3c3c3c", "Detected, low visibility"),
                             ("#000000", "No detection"),
                             ("#ff1744", "Trigger fired (dashed mark)")):
            vl.addWidget(legend_row(colour, text))
        vl.addStretch(1)
