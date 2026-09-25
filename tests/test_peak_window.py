"""La ventana de pico de la Etapa 1.

POR QUÉ EXISTE ESTE ARCHIVO. El §3.5 pide que el evento pase a la Etapa 2
cuando *"an instantaneous combination of the two quantities exceeds a
configurable trigger threshold"*. Exigir el mismo cuadro es una condición
**adicional**: el §3.4 describe la firma temporal de cada cantidad por
separado —T llegando al rango de caída *"within a short time window"*, V muy
negativa *"within a short window"*— y no dice que coincidan.

Medido sobre PEF-FallDB no coinciden nunca: en las doce caídas que el
disparador perdió, el pico de T y el de V están separados entre 0.6 s y
3.1 s. Y es mecánico, no accidental — T es una posición y V una velocidad del
mismo cuerpo.

Estas pruebas fijan tres cosas, en orden de importancia:

* que con ``peak_window_s = 0`` el disparador se comporta **exactamente** como
  antes de que el parámetro existiera, porque de eso depende que la ablación
  del §3.7 sea un cambio de configuración y no un cambio de código;
* que la ventana recupera una caída desincronizada;
* que la ventana **no** inventa evidencia: no cruza una discontinuidad, no
  lee un NaN como cero, y olvida lo viejo.

Correr con:  python -m unittest discover tests
"""

from __future__ import annotations

import math
import unittest

from pef_fall_detector.state_machine import (
    SCORE,
    SEQUENTIAL,
    SIMULTANEOUS,
    PeakWindow,
    Stage1Trigger,
)

NAN = float("nan")


def drive(trigger, muestras, fps=30.0, start=0.0):
    """Alimenta (T, V) cuadro a cuadro; devuelve los tiempos en que disparó."""
    disparos = []
    t = start
    for t_deg, v_tps in muestras:
        if trigger.update(t, t_deg, v_tps):
            disparos.append(round(t, 4))
        t += 1.0 / fps
    return disparos


def trigger(**kw):
    base = dict(formulation=SCORE, threshold_t_deg=45.0, threshold_v_tps=-1.5,
                hold_s=0.0, threshold_score=2.2)
    base.update(kw)
    return Stage1Trigger(**base)


class TestLaVentanaEnSi(unittest.TestCase):

    def test_devuelve_el_maximo_de_T_y_el_minimo_de_V(self) -> None:
        w = PeakWindow(1.0)
        w.update(0.0, 10.0, -0.5)
        w.update(0.1, 90.0, -0.2)
        t, v = w.update(0.2, 20.0, -3.0)
        self.assertAlmostEqual(t, 90.0)
        self.assertAlmostEqual(v, -3.0)

    def test_olvida_lo_que_salio_de_la_ventana(self) -> None:
        # Un clip largo no puede acumular evidencia de hace diez segundos: la
        # caída que importa es la de este momento.
        w = PeakWindow(0.5)
        w.update(0.0, 170.0, -5.0)
        t, v = w.update(2.0, 10.0, -0.1)
        self.assertAlmostEqual(t, 10.0)
        self.assertAlmostEqual(v, -0.1)

    def test_un_NaN_no_se_lee_como_cero(self) -> None:
        # Leer NaN como 0.0 dejaría que una oclusión BAJE un máximo de T o
        # SUBA un mínimo de V, que es evidencia inventada al revés.
        w = PeakWindow(1.0)
        w.update(0.0, 150.0, -4.0)
        t, v = w.update(0.1, NAN, NAN)
        self.assertAlmostEqual(t, 150.0)
        self.assertAlmostEqual(v, -4.0)

    def test_sin_ningun_valor_medible_devuelve_NaN(self) -> None:
        # No es lo mismo "no hubo movimiento" que "no se pudo medir".
        w = PeakWindow(1.0)
        t, v = w.update(0.0, NAN, NAN)
        self.assertTrue(math.isnan(t))
        self.assertTrue(math.isnan(v))

    def test_la_ventana_avanza_con_el_reloj_no_con_los_cuadros(self) -> None:
        # La lección C5: "24 cuadros" son 0.8 s a 30 fps y 1.6 s a 15 fps.
        # Con NaN de por medio el conteo de cuadros medibles tampoco sirve.
        w = PeakWindow(0.5)
        w.update(0.0, 170.0, -5.0)
        for i in range(1, 20):
            t, v = w.update(i * 0.1, 10.0, -0.1)
        self.assertAlmostEqual(t, 10.0)

    def test_una_ventana_no_positiva_se_rechaza(self) -> None:
        for malo in (0.0, -1.0):
            with self.assertRaises(ValueError):
                PeakWindow(malo)


class TestCeroEsElComportamientoAnterior(unittest.TestCase):
    """Lo que hace posible la ablación del §3.7 sin tocar código."""

    ESCENAS = [
        [(10.0, -0.1)] * 30,                                   # quieto
        [(170.0, -0.05)] * 30,                                 # ya en el suelo
        [(80.0, -0.5)] * 5 + [(20.0, -3.0)] * 5,               # desincronizada
        [(100.0, -2.0)] * 10,                                  # caída franca
        [(NAN, NAN)] * 10 + [(90.0, -2.0)] * 10,               # con oclusión
    ]

    def test_cero_dispara_igual_que_sin_el_parametro(self) -> None:
        for i, escena in enumerate(self.ESCENAS):
            with self.subTest(escena=i):
                sin = drive(trigger(), escena)
                cero = drive(trigger(peak_window_s=0.0), escena)
                self.assertEqual(sin, cero)

    def test_cero_no_construye_la_ventana(self) -> None:
        # No es sólo que dé el mismo resultado: no debe pagar el costo ni
        # arrastrar estado que después haya que resetear.
        self.assertIsNone(trigger(peak_window_s=0.0)._peaks)
        self.assertIsNotNone(trigger(peak_window_s=0.8)._peaks)

    def test_una_ventana_negativa_se_rechaza(self) -> None:
        with self.assertRaises(ValueError):
            trigger(peak_window_s=-0.1)


class TestRecuperaLaCaidaDesincronizada(unittest.TestCase):

    # El perfil medido: V pica primero con el tronco aún casi vertical, T pica
    # medio segundo después con el cuerpo ya casi detenido, y en el medio no
    # hay nada notable. Ningún cuadro llega a 2.2 por sí solo:
    #   0.20 s  20/45 + 2.5/1.5 = 2.11   (V al máximo, tronco vertical)
    #   0.40 s  20/45 + 0.4/1.5 = 0.71   (la transición, nada extremo)
    #   0.17 s  85/45 + 0.2/1.5 = 2.09   (T al máximo, ya detenido)
    # y ninguna pareja parcial tampoco: puentear la transición con el pico de
    # V da 2.11, y puentearla con el de T da 2.16. Sólo llegar de un extremo
    # al otro, 85/45 + 2.5/1.5 = 3.56, cruza el umbral — que es exactamente lo
    # que hace de la longitud de la ventana un parámetro con contenido.
    # El pico de T se deja por debajo de 99 grados a propósito: por encima de
    # eso el término del tronco cruza 2.2 sin ayuda de V y la prueba dejaría
    # de medir la ventana.
    CAIDA = ([(20.0, -2.5)] * 6 + [(20.0, -0.4)] * 12 + [(85.0, -0.2)] * 6)

    def test_sin_ventana_no_dispara(self) -> None:
        self.assertEqual(drive(trigger(), self.CAIDA), [])

    def test_con_ventana_dispara(self) -> None:
        self.assertTrue(drive(trigger(peak_window_s=0.8), self.CAIDA))

    def test_una_ventana_mas_corta_que_la_separacion_no_alcanza(self) -> None:
        # La ventana es el parámetro, y tiene que MEDIR algo: si cualquier
        # valor sirviera, no habría nada que calibrar.
        self.assertEqual(drive(trigger(peak_window_s=0.1), self.CAIDA), [])

    def test_tambien_funciona_con_la_formulacion_simultaneous(self) -> None:
        # La ventana no es del 'score': ensancha el instante para cualquier
        # lectura de la misma frase del §3.5.
        sin = Stage1Trigger(formulation=SIMULTANEOUS, hold_s=0.0)
        con = Stage1Trigger(formulation=SIMULTANEOUS, hold_s=0.0,
                            peak_window_s=0.8)
        self.assertEqual(drive(sin, self.CAIDA), [])
        self.assertTrue(drive(con, self.CAIDA))


class TestNoInventaEvidencia(unittest.TestCase):

    def test_no_dispara_sobre_alguien_quieto(self) -> None:
        quieto = [(12.0, -0.08)] * 60
        self.assertEqual(drive(trigger(peak_window_s=0.8), quieto), [])

    def test_sin_ningun_V_negativo_no_dispara(self) -> None:
        # §3.5: el disparo ocurre "while V remains negative (downward)".
        # Con el tronco a 170 el término de T vale 3.8 por sí solo, así que
        # esta guarda es lo único que separa una caída de alguien que se está
        # LEVANTANDO del suelo.
        subiendo = [(170.0, 2.0)] * 30
        self.assertEqual(drive(trigger(peak_window_s=0.8), subiendo), [])

    def test_reset_impide_cruzar_una_discontinuidad(self) -> None:
        """El pico de T de antes de una oclusión no puede casarse con la V de después.

        Puede ser otro cuerpo, u otro punto de la línea de tiempo. Es la misma
        razón por la que ``sequential`` no deja sobrevivir un evento armado.
        """
        t = trigger(peak_window_s=0.8)
        drive(t, [(85.0, -0.1)] * 6)           # sólo T, no alcanza
        t.reset()                              # sujeto perdido
        self.assertEqual(drive(t, [(20.0, -2.6)] * 3, start=0.2), [])

    def test_sin_reset_si_se_combinarian(self) -> None:
        # El complemento del anterior: prueba que el reset es lo que impide
        # la combinación, y no que la escena no diera para disparar.
        t = trigger(peak_window_s=0.8)
        drive(t, [(85.0, -0.1)] * 6)
        self.assertTrue(drive(t, [(20.0, -2.6)] * 3, start=0.2))


if __name__ == "__main__":
    unittest.main()
