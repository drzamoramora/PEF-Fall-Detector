"""Tema visual de PEF-Lab: oscuro, sobrio, con un solo color de acento.

Solo aspecto. Ningun valor de aqui cambia lo que el detector mide o decide.
Los colores de etapa son los mismos de la linea de tiempo, para que la
etiqueta de la etapa y la franja de colores digan lo mismo.
"""

from __future__ import annotations

BG = "#111317"          # fondo de la ventana
PANEL = "#1a1d23"       # tarjetas y paneles
PANEL_2 = "#22262e"     # campos, tablas, botones secundarios
BORDER = "#2c313a"
TEXT = "#e6e8eb"
MUTED = "#98a2ad"
ACCENT = "#3d8bfd"
ACCENT_HOVER = "#5a9dfd"
DANGER = "#e5484d"
WARNING = "#f5a524"
SUCCESS = "#30a46c"
INFO = "#4fc3f7"

#: Fondo de las graficas (pyqtgraph), un punto mas claro que la ventana.
PLOT_BG = "#15181d"
PLOT_FG = "#aab2bd"

#: Color de la etiqueta de etapa; iguales a la linea de tiempo.
STAGE_COLOURS = {
    "MONITORING": "#546e7a",
    "CONFIRMING": "#ff9800",
    "OBSERVING": "#1e88e5",
    "COOLDOWN": "#8e24aa",
    "FIRED": "#ff1744",
    "FROZEN": "#0277bd",
}

STAGE_NAMES = {
    "MONITORING": "Monitoring",
    "CONFIRMING": "Stage 2 · geometry",
    "OBSERVING": "Stage 3 · observing",
    "COOLDOWN": "Cooldown",
}

QSS = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-size: 13px;
}}
QLabel {{ background: transparent; }}
QToolTip {{
    background-color: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER};
    padding: 4px;
}}

/* Tarjetas */
QFrame#card, QFrame#header, QFrame#playbar {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QLabel#cardTitle {{
    color: {MUTED}; font-size: 11px; font-weight: bold;
    letter-spacing: 1px;
}}
QLabel#appTitle {{ font-size: 18px; font-weight: bold; }}
QLabel#appSubtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#statusLine {{ color: {MUTED}; font-size: 12px; padding: 2px 4px; }}
QLabel#videoView {{
    background-color: #000000; color: {MUTED};
    border: 1px solid {BORDER}; border-radius: 8px;
}}
QLabel#timeLabel {{ font-family: Menlo, Consolas, monospace; color: {MUTED}; }}

/* Botones */
QPushButton {{
    background-color: {PANEL_2}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 12px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background-color: {BORDER}; }}
QPushButton:disabled {{ color: #5c6570; border-color: #23272e; background-color: #1a1d22; }}
QPushButton#primary {{
    background-color: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: bold;
}}
QPushButton#primary:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background-color: #2a3a52; border-color: #2a3a52; color: #8a97a8; }}
QPushButton#danger:hover {{ border-color: {DANGER}; color: {DANGER}; }}

/* Campos */
QSpinBox, QComboBox {{
    background-color: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 4px 6px; min-height: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL_2}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}
QCheckBox {{ spacing: 8px; padding: 3px 0; }}

/* Deslizador */
QSlider::groove:horizontal {{ height: 6px; background: {BORDER}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: white; width: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:disabled {{ background: #5c6570; }}

/* Pestanas */
QTabWidget::pane {{
    border: 1px solid {BORDER}; border-radius: 8px; top: -1px;
    background-color: {PANEL};
}}
QTabBar::tab {{
    background: transparent; color: {MUTED};
    padding: 7px 14px; margin-right: 2px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

/* Tabla */
QTableWidget {{
    background-color: {PANEL}; alternate-background-color: {PANEL_2};
    gridline-color: {BORDER}; border: 1px solid {BORDER}; border-radius: 6px;
    selection-background-color: #24466f;
}}
QHeaderView::section {{
    background-color: {PANEL_2}; color: {MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 5px; font-weight: bold;
}}

/* Barra de progreso */
QProgressBar {{
    background-color: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px;
    text-align: center; color: {TEXT}; min-height: 18px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 5px; }}

/* Divisores y desplazamiento */
QSplitter::handle {{ background-color: {BG}; }}
QSplitter::handle:hover {{ background-color: {BORDER}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


def chip_style(colour: str) -> str:
    """Una etiqueta tipo pastilla con fondo de color."""
    return (f"background-color: {colour}; color: white; font-weight: bold; "
            f"border-radius: 10px; padding: 3px 12px;")
