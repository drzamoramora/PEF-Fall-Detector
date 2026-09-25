"""El criterio de descenso controlado: la segunda evidencia de la Etapa 2.

POR QUÉ EXISTE ESTE ARCHIVO. El §3.5 encarga a la Etapa 2 separar una caída de
*"a controlled sit-down in which COM stays within the support polygon"*, y le
da P como única herramienta. Medido sobre PEF-FallDB, esa premisa es falsa: los
ADL que disparan ponen el centro de masa **fuera** del polígono de apoyo en el
100 % de los cuadros medibles, porque sentarse traslada el peso al mueble. La
geometría dice "caída" con razón.

Lo que sí separa es la magnitud del descenso, y el §3.4 ya lo describe así al
definir V. Estas pruebas fijan ese criterio y, sobre todo, fijan sus **límites**:
que no rechace un síncope por ser corto, ni una caída lenta por ser suave.

Correr con:  python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from pef_fall_detector.state_machine import (
    REJECTED,
    SCORE,
    DescentTracker,
    FallStateMachine,
    Stage1Trigger,
    Stage2Evaluator,
    Stage3Evaluator,
)

FLOOR = -1.0
DEPTH = -1.3          # más superficial que esto -> poco profundo
SUSTAINED = 0.40      # menos que esto bajo el piso -> breve


def feed(tracker, samples, start=0.0, fps=30.0):
    t = start
    for v in samples:
        tracker.update(t, v)
        t += 1.0 / fps
    return t


class TestMeasurement(unittest.TestCase):

    def test_the_deepest_value_is_the_most_negative(self) -> None:
        tracker = DescentTracker(FLOOR)
        feed(tracker, [-0.2, -3.1, -0.5])
        self.assertAlmostEqual(tracker.deepest, -3.1)

    def test_sustained_measures_the_longest_run_not_the_total(self) -> None:
        # Dos bajones separados no son un descenso sostenido. Sumarlos
        # convertiría dos tropiezos en una caída.
        tracker = DescentTracker(FLOOR)
        feed(tracker, [-2.0] * 6 + [0.0] * 6 + [-2.0] * 6)   # 0.2s, pausa, 0.2s
        self.assertLess(tracker.sustained_s, 0.30)

    def test_nan_frames_are_skipped_not_treated_as_zero(self) -> None:
        # Un cuadro sin V medible no interrumpe el descenso: un cero lo
        # cortaría en dos y haría corto lo que fue largo.
        tracker = DescentTracker(FLOOR)
        feed(tracker, [-2.0] * 9 + [float("nan")] + [-2.0] * 9)
        self.assertGreater(tracker.sustained_s, 0.50)

    def test_the_window_forgets_old_motion(self) -> None:
        # El descenso que importa es el de este evento, no el de hace diez
        # segundos: si no, un clip largo acumula evidencia de otro momento.
        tracker = DescentTracker(FLOOR, window_s=1.0)
        end = feed(tracker, [-3.0] * 30)
        feed(tracker, [0.0] * 45, start=end)
        self.assertEqual(tracker.deepest, 0.0)

    def test_an_empty_tracker_reports_no_descent(self) -> None:
        tracker = DescentTracker(FLOOR)
        self.assertEqual(tracker.deepest, 0.0)
        self.assertEqual(tracker.sustained_s, 0.0)


class TestTheDecision(unittest.TestCase):
    """Poco profundo Y breve. Las dos, nunca una."""

    def test_a_shallow_brief_descent_is_controlled(self) -> None:
        # El perfil medido de los falsos positivos: V_min -1.16 y -1.39,
        # tiempo bajo el piso 0.17 s y 0.20 s.
        tracker = DescentTracker(FLOOR)
        feed(tracker, [-1.2] * 5)
        self.assertTrue(tracker.is_controlled(DEPTH, SUSTAINED))

    def test_a_deep_but_brief_descent_is_not_controlled(self) -> None:
        """El síncope: caída vertical silenciosa, muy rápida.

        El §3.3 lo lista como uno de los cuatro arquetipos. Rechazarlo por
        durar poco sería perder exactamente la caída más peligrosa.
        """
        tracker = DescentTracker(FLOOR)
        feed(tracker, [-4.0] * 5)
        self.assertFalse(tracker.is_controlled(DEPTH, SUSTAINED))

    def test_a_shallow_but_long_descent_is_not_controlled(self) -> None:
        # Alguien que se desliza despacio por una pared hasta el suelo. Suave,
        # pero no controlado: dura.
        tracker = DescentTracker(FLOOR)
        feed(tracker, [-1.1] * 30)
        self.assertFalse(tracker.is_controlled(DEPTH, SUSTAINED))

    def test_a_deep_and_long_descent_is_a_fall(self) -> None:
        # El perfil medido de las caídas: V_min mediana -3.00, tiempo 0.77 s.
        tracker = DescentTracker(FLOOR)
        feed(tracker, [-3.0] * 24)
        self.assertFalse(tracker.is_controlled(DEPTH, SUSTAINED))

    def test_no_motion_at_all_reads_as_controlled(self) -> None:
        # Sin descenso no hay caída. Es el caso que protege contra que la
        # Etapa 2 confirme sobre geometría sin movimiento — alguien ya
        # acostado, que fue el falso positivo severo de la sesión en vivo.
        tracker = DescentTracker(FLOOR)
        feed(tracker, [0.0] * 30)
        self.assertTrue(tracker.is_controlled(DEPTH, SUSTAINED))


class TestItIsActuallyWiredIntoStage2(unittest.TestCase):
    """Que el criterio EXISTA no sirve si la Etapa 2 no lo consulta.

    Esta clase se escribió porque una mutación que desconectaba el rechazo por
    descenso —dejando la Etapa 2 exactamente como estaba antes— pasó toda la
    suite. Las pruebas de arriba miden el instrumento; estas miden que esté
    enchufado.
    """

    def machine(self, **kw):
        trigger = Stage1Trigger(formulation=SCORE, threshold_t_deg=45.0,
                                threshold_v_tps=-1.5, hold_s=0.1,
                                confirm_window_s=1.0, threshold_score=2.2)
        return FallStateMachine(
            trigger, Stage2Evaluator(window_s=0.33, min_samples=3),
            Stage3Evaluator(defer_to_end=True),
            descent=DescentTracker(FLOOR),
            min_depth_tps=DEPTH, min_sustained_s=SUSTAINED, **kw)

    def drive(self, machine, v_tps, descent_s=0.20, total_s=1.2,
              t_deg=70.0, p_offset=1.0):
        """Un descenso de ``descent_s``, luego quietud.

        La duración importa tanto como la profundidad, así que el perfil no
        puede ser una velocidad constante: mantener -1.2 durante todo el clip
        describe a alguien deslizándose por una pared, que NO es un descenso
        controlado. El perfil real de un ADL es un bajón corto y luego nada.
        """
        for i in range(int(total_s * 30)):
            t = i / 30.0
            v = v_tps if t < descent_s else -0.05
            machine.update(i, t, t_deg, v, p_offset, 0.0, 1.4)
        return machine.events

    def test_a_controlled_descent_is_rejected_even_with_P_outside(self) -> None:
        """El caso medido: el COM SÍ está fuera del apoyo, y aun así no es caída.

        Tres falsos positivos de calibración tenían p_samples=11 con el COM
        fuera en el 100 % de los cuadros. Sentarse en un sofá saca el centro de
        masa de sobre los pies porque el peso pasa al mueble. Sin este
        criterio, la Etapa 2 los confirma con toda la razón geométrica.
        """
        events = self.drive(self.machine(), v_tps=-1.2)   # poco profundo y breve
        self.assertTrue(events, "la Etapa 1 debia disparar")
        self.assertEqual(events[0].verdict, REJECTED)

    def test_a_real_descent_is_not_rejected(self) -> None:
        events = self.drive(self.machine(), v_tps=-3.0)   # profundo
        self.assertTrue(events)
        self.assertNotEqual(events[0].verdict, REJECTED)

    def test_without_a_tracker_nothing_changes(self) -> None:
        # El criterio es opcional: sin tracker, la Etapa 2 se comporta como
        # antes. Es lo que permite la ablación que pide el §3.7.
        trigger = Stage1Trigger(formulation=SCORE, threshold_t_deg=45.0,
                                threshold_v_tps=-1.5, hold_s=0.1,
                                confirm_window_s=1.0, threshold_score=2.2)
        machine = FallStateMachine(trigger,
                                   Stage2Evaluator(window_s=0.33, min_samples=3),
                                   Stage3Evaluator(defer_to_end=True))
        events = self.drive(machine, v_tps=-1.2)
        self.assertTrue(events)
        self.assertNotEqual(events[0].verdict, REJECTED)


class TestConstruction(unittest.TestCase):

    def test_a_positive_floor_is_rejected(self) -> None:
        # V negativa es hacia abajo. Un piso positivo mediría ascensos, y
        # fallaría en silencio dando siempre "no controlado".
        for bad in (0.0, 1.0):
            with self.assertRaises(ValueError):
                DescentTracker(bad)

    def test_a_non_positive_window_is_rejected(self) -> None:
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                DescentTracker(FLOOR, window_s=bad)

    def test_reset_forgets_everything(self) -> None:
        tracker = DescentTracker(FLOOR)
        feed(tracker, [-3.0] * 30)
        tracker.reset()
        self.assertEqual(tracker.deepest, 0.0)


if __name__ == "__main__":
    unittest.main()
