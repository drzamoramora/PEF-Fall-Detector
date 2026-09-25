"""La banda del disparador: eventos provisionales que deben ganarse la alarma.

POR QUÉ EXISTE. Medido sobre PEF-FallDB, la proporción de caídas crece de
forma monótona con el puntaje del disparador, sin inversiones:

    < 1.8      5.3 %        2.6 – 3.5   57.7 %
  1.8 – 2.2    7.1 %        > 3.5       90.9 %
  2.2 – 2.6   35.7 %

O sea que el puntaje se comporta como una confianza y no como un interruptor.
La banda 2.2–2.6 contiene 5 caídas y 9 ADL: admitirla entera como alarma
cuesta nueve falsos positivos, y descartarla entera pierde cinco caídas.

La salida no es elegir un corte mejor, es dejarla entrar al embudo con la
carga de la prueba más alta. El §3.5 ya describe tres etapas donde la
sospecha se confirma o se descarta, y el §3.4 ya condiciona la confirmación a
que I supere W. La banda sólo le exige a la evidencia más débil exactamente
esa prueba.

Lo que se fija aquí, en orden de importancia:

* que con la banda en 0 el disparador se comporte **exactamente** como antes
  —de eso depende que la ablación del §3.7 sea configuración y no código—;
* que un evento provisional que NO alcanza ``stage3_confirmed`` no produzca
  alarma, cualquiera sea la vía por la que resolvió;
* que un evento provisional que SÍ se confirma por inmovilidad conserve su
  severidad, porque si no la banda sería sólo un rechazo con otro nombre;
* que un evento pleno no quede afectado por nada de esto.

Correr con:  python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from pef_fall_detector.classification import NO_FALL, class_for_event
from pef_fall_detector.state_machine import (
    SCORE,
    FallStateMachine,
    Stage1Trigger,
    Stage2Evaluator,
    Stage3Evaluator,
)

PLENO = 2.6
BANDA = 2.2


def disparador(**kw):
    base = dict(formulation=SCORE, threshold_t_deg=45.0, threshold_v_tps=-1.5,
                hold_s=0.0, threshold_score=PLENO)
    base.update(kw)
    return Stage1Trigger(**base)


def puntaje(t_deg, v_tps):
    return t_deg / 45.0 + v_tps / -1.5


class TestElTramoDelDisparo(unittest.TestCase):

    def test_por_debajo_de_la_banda_no_dispara(self) -> None:
        t = disparador(threshold_score_provisional=BANDA)
        self.assertFalse(t.update(0.0, 50.0, -0.8))     # 1.64

    def test_dentro_de_la_banda_dispara_y_se_marca(self) -> None:
        t = disparador(threshold_score_provisional=BANDA)
        self.assertTrue(t.update(0.0, 80.0, -1.2))      # 2.58
        self.assertTrue(t.last_firing_provisional)

    def test_por_encima_dispara_sin_marca(self) -> None:
        t = disparador(threshold_score_provisional=BANDA)
        self.assertTrue(t.update(0.0, 120.0, -2.0))     # 4.00
        self.assertFalse(t.last_firing_provisional)

    def test_el_borde_superior_pertenece_al_tramo_pleno(self) -> None:
        # Exactamente 2.6 no es sospecha: es el umbral que el paper llama
        # "exceeds a configurable trigger threshold".
        t = disparador(threshold_score_provisional=BANDA)
        t_deg = PLENO * 45.0            # V = 0 no dispara, hace falta V < 0
        self.assertTrue(t.update(0.0, t_deg - 0.15, -0.1))
        self.assertFalse(t.last_firing_provisional)

    def test_el_sostenimiento_corre_sobre_el_borde_INFERIOR(self) -> None:
        """Una señal que pasa de gris a plena no reinicia su propio reloj.

        Con dos enganches separados, el cuadro en que el puntaje cruza de la
        banda al umbral pleno arrancaría un hold nuevo desde cero y el evento
        se perdería justo en el caso más fuerte. El enganche es uno solo.
        """
        t = disparador(threshold_score_provisional=BANDA, hold_s=0.2)
        for i in range(6):                         # 0.167 s en la banda
            self.assertFalse(t.update(i / 30.0, 80.0, -1.2))
        disparo = t.update(6 / 30.0, 120.0, -2.0)  # cruza a pleno a los 0.200 s
        self.assertTrue(disparo, "el hold acumulado en la banda debe contar")
        self.assertFalse(t.last_firing_provisional)


class TestCeroEsElComportamientoAnterior(unittest.TestCase):
    """La ablación del §3.7: un corte único, sin banda."""

    ESCENAS = [
        [(50.0, -0.8)] * 10,     # por debajo de todo
        [(80.0, -1.2)] * 10,     # lo que seria la banda
        [(120.0, -2.0)] * 10,    # pleno
        [(80.0, -1.2)] * 5 + [(120.0, -2.0)] * 5,
    ]

    def test_sin_banda_solo_dispara_el_tramo_pleno(self) -> None:
        for i, escena in enumerate(self.ESCENAS):
            with self.subTest(escena=i):
                sin = disparador()
                cero = disparador(threshold_score_provisional=0.0)
                a = [sin.update(j / 30.0, T, V) for j, (T, V) in enumerate(escena)]
                b = [cero.update(j / 30.0, T, V) for j, (T, V) in enumerate(escena)]
                self.assertEqual(a, b)

    def test_sin_banda_ningun_disparo_se_marca(self) -> None:
        t = disparador(threshold_score_provisional=0.0)
        t.update(0.0, 120.0, -2.0)
        self.assertFalse(t.last_firing_provisional)

    def test_una_banda_por_encima_del_umbral_pleno_se_rechaza(self) -> None:
        # Al reves la banda no existe: todo disparo seria provisional y el
        # umbral pleno no se alcanzaria nunca.
        with self.assertRaises(ValueError):
            disparador(threshold_score_provisional=PLENO + 0.1)

    def test_una_banda_negativa_se_rechaza(self) -> None:
        with self.assertRaises(ValueError):
            disparador(threshold_score_provisional=-0.1)

    def test_el_reset_suelta_la_marca(self) -> None:
        t = disparador(threshold_score_provisional=BANDA)
        t.update(0.0, 80.0, -1.2)
        t.reset()
        self.assertFalse(t.last_firing_provisional)


class TestLaReglaDeLaBanda(unittest.TestCase):
    """Un evento provisional sólo sobrevive si la inmovilidad lo confirma."""

    def maquina(self, provisional=BANDA):
        return FallStateMachine(
            disparador(threshold_score_provisional=provisional, hold_s=0.1),
            Stage2Evaluator(window_s=0.33, min_samples=3),
            Stage3Evaluator(defer_to_end=True),
        )

    def correr(self, m, T, V, segundos=8.0, p_offset=1.0, quieto=True):
        """Alimenta la maquina; devuelve el evento resuelto."""
        for i in range(int(segundos * 30)):
            t = i / 30.0
            # Tras el disparo el sujeto se queda quieto (o no), que es lo que
            # la Etapa 3 mide.
            mover = 0.0 if quieto else (i % 2) * 0.5
            m.update(i, t, T, V, p_offset, 6.0 if quieto else mover, 1.4)
        m.finalise(segundos)
        return m.events[0] if m.events else None

    def test_un_provisional_confirmado_por_inmovilidad_conserva_su_severidad(self) -> None:
        # Si la banda tambien tumbara a los confirmados seria un rechazo con
        # otro nombre, y no habria nada que ganar admitiendolos.
        # El sujeto se queda quieto 6 s, por encima de W = 5 s.
        ev = self.correr(self.maquina(), 80.0, -1.2)
        self.assertIsNotNone(ev)
        self.assertTrue(ev.provisional)
        self.assertEqual(ev.verdict, "stage3_confirmed")
        self.assertTrue(ev.severity)

    def test_un_provisional_sin_confirmar_no_produce_alarma(self) -> None:
        """El caso que motiva la regla: termina tumbado pero nunca estuvo inmovil.

        En modo etiquetado la Etapa 3 resuelve por la postura final, asi que
        este evento SI llega a ``stage3_confirmed``. La banda no mira el
        veredicto: mira I contra W, que es lo que el §3.4 condiciona.
        """
        m = self.maquina()
        ev = self.correr(m, 80.0, -1.2, quieto=False)
        self.assertIsNotNone(ev)
        self.assertTrue(ev.provisional)
        self.assertEqual(ev.verdict, FallStateMachine.PROVISIONAL_UNCONFIRMED)
        self.assertEqual(ev.severity, "", "sin confirmacion no hay severidad")

    def test_el_veredicto_sin_confirmar_se_lee_como_NoFall(self) -> None:
        # Es lo que el sistema desplegado hace: dispatch_verdicts solo
        # despacha stage3_confirmed, asi que nadie acude.
        self.assertEqual(
            class_for_event(FallStateMachine.PROVISIONAL_UNCONFIRMED, ""),
            NO_FALL)

    def test_un_evento_pleno_no_queda_afectado(self) -> None:
        m = self.maquina()
        ev = self.correr(m, 120.0, -2.0, quieto=False)
        self.assertIsNotNone(ev)
        self.assertFalse(ev.provisional)
        self.assertNotEqual(ev.verdict, FallStateMachine.PROVISIONAL_UNCONFIRMED)

    def test_un_evento_que_nace_debil_y_se_fortalece_deja_de_ser_debil(self) -> None:
        """La promocion: la marca depende de la evidencia, no del reloj.

        El enganche corre sobre el borde inferior, asi que el evento se
        levanta mientras el puntaje todavia sube. Medido en A04-S1: cruza 2.2
        en t=2.60, dispara en t=2.70, cruza 2.6 en t=2.73 y llega a 3.32. Sin
        promocion ese evento carga la exigencia de inmovilidad por trece
        centesimas de diferencia, y una caida franca termina saliendo NoFall.
        """
        m = self.maquina()
        # Arranca en la banda el tiempo justo para disparar, y despues sube.
        for i in range(6):
            m.update(i, i / 30.0, 80.0, -1.2, 1.0, 0.0, 1.4)
        self.assertTrue(m.events, "la Etapa 1 debia disparar en la banda")
        self.assertTrue(m.events[0].provisional)
        for i in range(6, 120):
            m.update(i, i / 30.0, 120.0, -2.0, 1.0, 0.0, 1.4)
        self.assertFalse(m.events[0].provisional,
                         "al cruzar el umbral pleno deja de ser provisional")

    def test_promueve_en_el_MISMO_cuadro_del_cruce(self) -> None:
        """El orden importa: promover antes de alimentar al disparador lee el
        puntaje del cuadro anterior.

        Escrito porque una mutacion que invertia ese orden paso toda la
        suite. Con el desfase la promocion sigue ocurriendo —un cuadro mas
        tarde— y ningun escenario largo lo nota; el unico caso donde muerde
        es el evento que resuelve en el mismo cuadro en que la evidencia se
        vuelve fuerte. Esta prueba mide el cuadro, no el desenlace.
        """
        m = self.maquina()
        for i in range(6):
            m.update(i, i / 30.0, 80.0, -1.2, 1.0, 0.0, 1.4)
        self.assertTrue(m.events[0].provisional)
        m.update(6, 6 / 30.0, 120.0, -2.0, 1.0, 0.0, 1.4)   # un solo cuadro fuerte
        self.assertFalse(m.events[0].provisional,
                         "la promocion debe ocurrir en el cuadro del cruce")

    def test_NO_promueve_cuando_el_puntaje_sube_solo_por_estar_tumbado(self) -> None:
        """La ventana de promocion termina con la Etapa 2.

        En OBSERVING el sujeto ya esta en el suelo: T ronda 180, que da 4.0
        de puntaje por si solo. Si la promocion siguiera activa ahi,
        cualquier evento provisional ascenderia a pleno por el mero hecho de
        estar acostado, y la banda no exigiria nada a nadie.
        """
        m = self.maquina()
        for i in range(6):                      # nace en la banda
            m.update(i, i / 30.0, 80.0, -1.2, 1.0, 0.0, 1.4)
        self.assertTrue(m.events[0].provisional)
        for i in range(6, 40):                  # la Etapa 2 cierra (ventana 0.33 s)
            m.update(i, i / 30.0, 80.0, -1.2, 1.0, 0.0, 1.4)
        # Ahora el cuerpo esta en el suelo: T alto, V apenas negativa.
        for i in range(40, 200):
            m.update(i, i / 30.0, 178.0, -0.05, 1.0, (i % 2) * 0.5, 1.4)
        self.assertTrue(m.events[0].provisional,
                        "estar tumbado no es evidencia de disparo fuerte")

    def test_promovido_no_carga_la_exigencia_de_inmovilidad(self) -> None:
        # El efecto que importa: un evento promovido resuelve como cualquier
        # otro, aunque el sujeto nunca se quede quieto.
        m = self.maquina()
        for i in range(6):
            m.update(i, i / 30.0, 80.0, -1.2, 1.0, 0.0, 1.4)
        for i in range(6, 240):
            m.update(i, i / 30.0, 120.0, -2.0, 1.0, (i % 2) * 0.5, 1.4)
        m.finalise(8.0)
        ev = m.events[0]
        self.assertFalse(ev.provisional)
        self.assertNotEqual(ev.verdict, FallStateMachine.PROVISIONAL_UNCONFIRMED)

    def test_uno_que_nunca_supera_el_umbral_sigue_cargandola(self) -> None:
        # El caso legitimo de la banda: A02-S3 tiene pico 2.55 y nunca
        # alcanza 2.6, asi que la exigencia se mantiene.
        m = self.maquina()
        for i in range(240):
            m.update(i, i / 30.0, 80.0, -1.2, 1.0, (i % 2) * 0.5, 1.4)
        m.finalise(8.0)
        ev = m.events[0]
        self.assertTrue(ev.provisional)
        self.assertEqual(ev.verdict, FallStateMachine.PROVISIONAL_UNCONFIRMED)

    def test_sin_banda_el_mismo_clip_debil_ni_siquiera_entra(self) -> None:
        m = self.maquina(provisional=0.0)
        ev = self.correr(m, 80.0, -1.2)
        self.assertIsNone(ev, "sin banda, 2.58 esta por debajo de 2.6")


if __name__ == "__main__":
    unittest.main()
