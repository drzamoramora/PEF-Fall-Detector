"""Las métricas del §3.7 y la partición.

Estas pruebas protegen números que van a la Sección 4 del paper. Un error aquí
no rompe nada visiblemente: produce una cifra plausible y equivocada, que es la
peor clase de error que puede tener un instrumento de medición.

Lo que se fija, y por qué:

* el binario caída/no-caída, porque accuracy sobre cuatro clases mezcla dos
  preguntas clínicas distintas y no deja ver ninguna;
* que el caso no determinado cuente como negativo —lo que el sistema
  desplegado hace— y se reporte aparte;
* el intervalo de Wilson, porque con 40 clips la aproximación normal produce
  límites por encima del 100 %;
* que un actor desconocido NO se reparta por omisión, porque asignarlo en
  silencio contamina justo lo que la partición protege.

Correr con:  python -m unittest discover tests
"""

from __future__ import annotations

import unittest
from pathlib import Path

from pef_fall_detector.classification import (
    NOT_RECOVERED,
    NO_FALL,
    PARTIALLY_RECOVERED,
    RECOVERED,
    UNDETERMINED,
)
from pef_fall_detector.evaluation import (
    actor_of,
    binary_metrics,
    fp_per_hour,
    load_split,
    per_class,
    report,
    split_rows,
    undetermined_count,
    wilson,
)

SPLIT = Path(__file__).resolve().parents[1] / "notas" / "particion.yaml"


def row(video, truth, detected, duracion="10.4", revisada=""):
    return {"video": video, "clase_verdad": truth, "clase_detectada": detected,
            "clase_revisada": revisada, "duracion_s": duracion}


class TestActorOf(unittest.TestCase):

    def test_the_actor_is_the_first_field(self) -> None:
        self.assertEqual(actor_of("A01-S1-Recovered.mp4"), "A01")
        self.assertEqual(actor_of("B14-S3-NoFall.mp4"), "B14")

    def test_the_A3_typo_resolves_to_its_real_actor(self) -> None:
        # A13 tiene S1, S3 y S4; con este completa los cuatro que tienen todos
        # los actores. Si se leyera como actor propio, quedaría un actor de un
        # solo clip fuera de la partición.
        self.assertEqual(actor_of("A3-S2-Recovery.mp4"), "A13")

    def test_a_full_path_works(self) -> None:
        self.assertEqual(actor_of("/x/y/A07-S2-Recovered.mp4"), "A07")


class TestSplit(unittest.TestCase):

    def test_the_shipped_split_covers_every_actor_exactly_once(self) -> None:
        split = load_split(SPLIT)
        self.assertEqual(len(split), 32, "deberian ser 32 actores")
        self.assertEqual(sum(1 for s in split.values() if s == "test"), 10)
        self.assertEqual(sum(1 for s in split.values() if s == "train"), 22)

    def test_the_split_is_stratified(self) -> None:
        # Dos actores de cada clase de caída y cuatro de NoFall en prueba. Una
        # partición que dejara una clase entera de un lado no mediría nada
        # sobre ella.
        split = load_split(SPLIT)
        test = {a for a, s in split.items() if s == "test"}
        self.assertEqual(test, {"A07", "A16", "A08", "A17", "A09", "A18",
                                "B03", "B06", "B09", "B12"})

    def test_an_unknown_actor_is_not_silently_assigned(self) -> None:
        # Material nuevo, o un nombre mal escrito. Mandarlo por omisión a un
        # lado contaminaría exactamente lo que la partición protege.
        groups = split_rows([row("Z99-S1-Recovered.mp4", RECOVERED, RECOVERED)],
                            load_split(SPLIT))
        self.assertEqual(len(groups["sin_asignar"]), 1)
        self.assertEqual(len(groups["train"]) + len(groups["test"]), 0)

    def test_rows_land_on_the_side_their_actor_belongs_to(self) -> None:
        groups = split_rows([row("A07-S1-Recovered.mp4", RECOVERED, RECOVERED),
                             row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED)],
                            load_split(SPLIT))
        self.assertEqual(len(groups["test"]), 1)
        self.assertEqual(len(groups["train"]), 1)

    def test_comments_are_not_read_as_entries(self) -> None:
        # El archivo lleva un comentario por actor (`- A07   # Recovered`) y
        # un bloque de comentarios arriba con guiones. Ninguno es un dato.
        self.assertNotIn("Recovered", load_split(SPLIT))
        self.assertNotIn("#", "".join(load_split(SPLIT)))


class TestBinaryMetrics(unittest.TestCase):

    def test_severity_confusions_do_not_hurt_the_binary(self) -> None:
        # El punto entero de reportar el binario aparte: un sistema que
        # confunde severidades pero atrapa las caídas sirve, y accuracy sobre
        # cuatro clases no lo deja ver.
        m = binary_metrics([
            row("A01-S1-Recovered.mp4", RECOVERED, NOT_RECOVERED),
            row("A02-S1-NotRecovered.mp4", NOT_RECOVERED, PARTIALLY_RECOVERED),
            row("B01-S1-NoFall.mp4", NO_FALL, NO_FALL),
        ])
        self.assertEqual(m["sensibilidad"], 1.0)
        self.assertEqual(m["especificidad"], 1.0)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (2, 0, 0, 1))

    def test_a_missed_fall_is_a_false_negative(self) -> None:
        m = binary_metrics([row("A01-S1-Recovered.mp4", RECOVERED, NO_FALL)])
        self.assertEqual((m["tp"], m["fn"]), (0, 1))
        self.assertEqual(m["sensibilidad"], 0.0)

    def test_undetermined_counts_as_no_alarm(self) -> None:
        # Es lo que el sistema desplegado hace: `dispatch_verdicts` sólo
        # despacha `stage3_confirmed`, así que un no-determinado no produce
        # alarma y nadie acude. Contarlo como positivo mediría una intención.
        m = binary_metrics([
            row("A01-S1-Recovered.mp4", RECOVERED, UNDETERMINED),
            row("B01-S1-NoFall.mp4", NO_FALL, UNDETERMINED),
        ])
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (0, 1, 0, 1))

    def test_a_human_review_overrides_the_machine(self) -> None:
        m = binary_metrics([row("A01-S1-Recovered.mp4", RECOVERED, NO_FALL,
                                revisada=RECOVERED)])
        self.assertEqual(m["tp"], 1)

    def test_f1_is_nan_rather_than_zero_when_undefined(self) -> None:
        # Sin ningún positivo predicho ni real, F1 no vale 0: no está definido.
        # Un 0 se leería como "el sistema falló", que es una afirmación
        # distinta de "no hay con qué medirlo".
        m = binary_metrics([row("B01-S1-NoFall.mp4", NO_FALL, NO_FALL)])
        self.assertNotEqual(m["f1"], m["f1"])


class TestWilson(unittest.TestCase):

    def test_perfect_score_does_not_exceed_one(self) -> None:
        # Lo que la aproximación normal hace mal: 40/40 le da un límite
        # superior por encima del 100 %.
        lo, hi = wilson(40, 40)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 1.0)

    def test_zero_does_not_go_below_zero(self) -> None:
        lo, hi = wilson(0, 40)
        self.assertGreaterEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_the_interval_narrows_with_more_clips(self) -> None:
        narrow = wilson(360, 400)
        wide = wilson(36, 40)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_forty_clips_at_ninety_percent_span_about_eighteen_points(self) -> None:
        # El número que justifica reportar el intervalo: sobre la partición de
        # prueba, 90 % y 81 % no son distinguibles.
        lo, hi = wilson(36, 40)
        self.assertGreater(hi - lo, 0.12)

    def test_an_empty_set_is_nan_not_zero(self) -> None:
        lo, hi = wilson(0, 0)
        self.assertNotEqual(lo, lo)
        self.assertNotEqual(hi, hi)


class TestFalsePositivesPerHour(unittest.TestCase):

    def test_only_non_fall_footage_is_in_the_denominator(self) -> None:
        # Una alarma sobre una caída real no es una alarma falsa, y su metraje
        # no diluye la tasa.
        rows = [row("B01-S1-NoFall.mp4", NO_FALL, RECOVERED, duracion="3600"),
                row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED, duracion="3600")]
        self.assertAlmostEqual(fp_per_hour(rows), 1.0, places=6)

    def test_no_footage_is_nan_not_zero(self) -> None:
        self.assertNotEqual(fp_per_hour([]), fp_per_hour([]))


class TestReport(unittest.TestCase):

    def test_the_report_carries_the_interval_and_the_undetermined_row(self) -> None:
        rows = [row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED),
                row("B01-S1-NoFall.mp4", NO_FALL, UNDETERMINED)]
        text = report(rows, "PRUEBA")
        self.assertIn("IC95", text)
        self.assertIn("no determinados   : 1", text)
        self.assertIn("PRUEBA", text)

    def test_per_class_counts(self) -> None:
        counts = per_class([row("A01-S1-Recovered.mp4", RECOVERED, RECOVERED),
                            row("A04-S1-Recovered.mp4", RECOVERED, NO_FALL)])
        self.assertEqual(counts[RECOVERED], (1, 2))

    def test_unlabelled_rows_are_excluded_not_failed(self) -> None:
        # Un clip sin verdad legible no dice nada sobre el detector; contarlo
        # como fallo deprimiría todas las cifras por una cantidad que depende
        # de los nombres de archivo.
        self.assertIn("sin clips", report([row("x.mp4", "", NO_FALL)], "X"))
        self.assertEqual(undetermined_count([row("x.mp4", "", UNDETERMINED)]), 1)


if __name__ == "__main__":
    unittest.main()
