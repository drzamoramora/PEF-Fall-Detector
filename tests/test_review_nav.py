"""Moverse entre los clips de la cola después de la pasada.

POR QUÉ EXISTE. Las situaciones de PEF-FallDB se graban con los cuatro
sujetos, y lo que explica por qué tres clips fallan y uno acierta casi nunca
está dentro de un clip: está en la diferencia entre ellos. Revisar es
comparar, y comparar exige ir y venir.

Lo que se fija:

* que las flechas no funcionen durante la pasada, porque cargar otro clip a
  media medición relabelaría un clip contra un pipeline que vio la mitad de
  otro — el mismo accidente que el botón de abrir video ya tiene prohibido;
* que se detengan en los extremos en vez de dar la vuelta;
* que la revisión NO toque los resultados de la pasada, que son justamente
  contra lo que uno compara mientras mira.

Correr con:  python -m unittest discover tests
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# PEF-Lab corre donde hay Qt; el resto del proyecto —el pipeline, la
# evaluacion, el despliegue del §3.6— no lo necesita y a veces se prueba en
# maquinas que no lo tienen. Saltar es la respuesta correcta ahi: un fallo de
# importacion en un entorno sin GUI no dice nada sobre esta ventana, y deja
# la suite en rojo por una razon que no es un defecto.
try:
    from PySide6.QtWidgets import QApplication, QTableWidgetItem
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("PySide6 no esta instalado; PEF-Lab no se prueba aqui")

from pef_fall_detector.config import Config  # noqa: E402
from pef_fall_detector.gui.lab_window import LabWindow  # noqa: E402
from tests.test_pipeline import TEST_CFG  # noqa: E402

_app = QApplication.instance() or QApplication([])


class TestNavegacionDeRevision(unittest.TestCase):

    def setUp(self) -> None:
        self.win = LabWindow(Config(TEST_CFG))
        self.win._queue = [Path(f"/no/existe/A14-S{i}-NotRecovered.mp4")
                           for i in (1, 2, 3, 4)]
        self.win.queue_table.setRowCount(4)
        for r, p in enumerate(self.win._queue):
            self.win.queue_table.setItem(r, 0, QTableWidgetItem(p.name))
        self.abiertos: list[int] = []
        # La carga real abre el archivo; aquí sólo interesa a cuál apunta.
        self.win._open_for_review = self._espia(self.win._open_for_review)

    def _espia(self, real):
        def envuelto(row):
            self.abiertos.append(row)
            self.win._review_index = row
        return envuelto

    def tearDown(self) -> None:
        self.win.close()

    def test_sin_clip_abierto_la_flecha_derecha_entra_por_el_primero(self) -> None:
        # Llegar a una carpeta y no saber por dónde empezar es el caso normal,
        # no un error: la primera flecha tiene que hacer algo.
        self.win._review_step(+1)
        self.assertEqual(self.abiertos, [0])

    def test_sin_clip_abierto_la_flecha_izquierda_entra_por_el_ultimo(self) -> None:
        self.win._review_step(-1)
        self.assertEqual(self.abiertos, [3])

    def test_avanza_y_retrocede_de_a_uno(self) -> None:
        self.win._review_index = 1
        self.win._review_step(+1)
        self.win._review_step(+1)
        self.win._review_step(-1)
        self.assertEqual(self.abiertos, [2, 3, 2])

    def test_se_detiene_en_el_ultimo_en_vez_de_dar_la_vuelta(self) -> None:
        """Volver al primero sin avisar se confunde con "no pasó nada".

        Con cuatro sujetos por situación, el punto es comparar clips
        distintos; una vuelta silenciosa hace creer que la flecha no
        respondió y que hay que apretarla otra vez.
        """
        self.win._review_index = 3
        self.win._review_step(+1)
        self.assertEqual(self.abiertos, [])

    def test_se_detiene_en_el_primero(self) -> None:
        self.win._review_index = 0
        self.win._review_step(-1)
        self.assertEqual(self.abiertos, [])

    def test_durante_la_pasada_las_flechas_no_hacen_nada(self) -> None:
        # Cargar otro clip a media medición relabelaría un clip contra un
        # pipeline que vio la mitad de otro.
        self.win._queue_running = True
        self.win._review_index = 1
        self.win._review_step(+1)
        self.assertEqual(self.abiertos, [])

    def test_con_la_cola_vacia_no_hacen_nada(self) -> None:
        self.win._queue = []
        self.win._review_step(+1)
        self.assertEqual(self.abiertos, [])

    def test_vaciar_la_cola_suelta_el_ancla(self) -> None:
        # Si no, la siguiente carpeta arrancaría desde un índice de la
        # anterior, que puede ni existir.
        self.win._review_index = 2
        self.win._on_queue_clear()
        self.assertIsNone(self.win._review_index)


class TestLoQueLaRevisionNoDebeTocar(unittest.TestCase):

    def setUp(self) -> None:
        self.win = LabWindow(Config(TEST_CFG))
        self.win._queue = [Path("/no/existe/A14-S1-NotRecovered.mp4")]
        self.win.queue_table.setRowCount(1)
        self.win.queue_table.setItem(0, 0, QTableWidgetItem("A14-S1"))

    def tearDown(self) -> None:
        self.win.close()

    def test_los_resultados_de_la_pasada_sobreviven_a_la_revision(self) -> None:
        """Son contra lo que uno compara mientras mira el clip."""
        self.win._show_queue_result(0, "NoFall", "NotRecovered")
        self.win._review_index = 0
        self.win._update_review_label()
        self.assertEqual(self.win.queue_table.item(0, 2).text(), "NoFall")
        self.assertEqual(self.win.queue_table.item(0, 3).text(), "X")

    def test_el_contador_dice_donde_esta_y_de_cuantos(self) -> None:
        self.win._review_index = 0
        self.win._update_review_label()
        texto = self.win.review_label.text()
        self.assertIn("1 de 1", texto)
        self.assertIn("A14-S1", texto)

    def test_sin_clip_abierto_el_contador_explica_como_empezar(self) -> None:
        self.win._review_index = None
        self.win._update_review_label()
        self.assertIn("Alt", self.win.review_label.text())


if __name__ == "__main__":
    unittest.main()
