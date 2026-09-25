"""Métricas del §3.7, y la partición que las hace significar algo.

Qt-free y sin dependencias fuera de la biblioteca estándar: la GUI, el runner
headless y las pruebas consumen esto mismo, de modo que un número reportado en
la Sección 4 y uno visto en pantalla no puedan diferir.

POR QUÉ NO BASTA ACCURACY
-------------------------
El §3.7 pide *"accuracy, sensitivity on the fall class, specificity, precision,
and F1"* más *"the false-positive rate per hour of non-fall footage"*. No es
burocracia: accuracy sobre cuatro clases mezcla dos preguntas clínicas
distintas y deja ver ninguna.

Un sistema que confunde severidades pero atrapa todas las caídas **sirve** —
alguien llega a auxiliar. Uno que acierta severidades y pierde una caída de
cada cinco **no sirve**, por buena que sea su accuracy. Las dos situaciones
pueden dar el mismo número y sólo la sensibilidad binaria las separa.

EL CASO NO DETERMINADO
----------------------
Un evento que disparó pero que ninguna etapa pudo juzgar cuenta como NEGATIVO
en el binario, porque es lo que el sistema desplegado hace: `dispatch_verdicts`
sólo despacha `stage3_confirmed`, así que un no-determinado no produce alarma y
nadie acude. Contarlo como positivo mediría una intención en vez de una
consecuencia.

Pero se reporta aparte y siempre visible. Un sistema que no puede juzgar y uno
que juzga que no hubo caída son modos de falla distintos, con causas y arreglos
distintos, y colapsarlos esconde exactamente la información que hace falta para
arreglarlos.

INTERVALOS DE CONFIANZA
-----------------------
Sobre la partición de prueba hay 40 clips: **un clip vale 2.5 puntos**, y un
90 % carga un intervalo de aproximadamente ±9. Reportar el punto pelado invita
a leer una diferencia de 4 puntos como una mejora cuando es ruido. Se usa el
intervalo de Wilson, que no se rompe cerca de 0 % ni de 100 % como sí lo hace
la aproximación normal.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from .classification import (
    NOT_RECOVERED,
    NO_FALL,
    PARTIALLY_RECOVERED,
    RECOVERED,
    UNDETERMINED,
)

#: Las tres clases que son una caída. `UNDETERMINED` no está aquí a propósito:
#: ver el docstring del módulo.
FALL_CLASSES = frozenset({RECOVERED, PARTIALLY_RECOVERED, NOT_RECOVERED})

#: Erratas de nombre que no cambian de quién es el clip. `A3-S2` es de A13: su
#: actor tiene S1, S3 y S4, y con este completa los cuatro que tienen todos.
_ACTOR_ALIASES = {"A3": "A13"}


def actor_of(video: str) -> str:
    """El actor dueño de un clip, a partir del nombre del archivo."""
    stem = Path(video).stem
    head = stem.split("-", 1)[0].strip()
    return _ACTOR_ALIASES.get(head, head)


def load_split(path) -> dict[str, str]:
    """Lee `particion.yaml` sin depender de un parser de YAML.

    El archivo es deliberadamente simple —dos listas de cadenas— y leerlo con
    una expresión regular evita agregar PyYAML a un despliegue de borde por un
    archivo de veinte líneas.
    """
    side, out = None, {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r"^(test|train)\s*:", line):
            side = line.split(":", 1)[0].strip()
            continue
        item = re.match(r"^\s*-\s*(\S+)", line)
        if item and side:
            out[item.group(1)] = side
    return out


def split_rows(rows, split: dict[str, str]) -> dict[str, list]:
    """Reparte filas de etiquetas en train / test / sin_asignar.

    Un clip cuyo actor no está en la partición va a ``sin_asignar`` y NO se
    reparte por omisión. Un actor desconocido es material nuevo o un nombre mal
    escrito, y asignarlo en silencio a un lado contaminaría justamente lo que
    la partición protege.
    """
    out = {"train": [], "test": [], "sin_asignar": []}
    for row in rows:
        out[split.get(actor_of(row.get("video", "")), "sin_asignar")].append(row)
    return out


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson al 95 %, como fracciones.

    No la aproximación normal: con 40 clips y una proporción cerca de 1, la
    normal produce límites por encima del 100 %, que además de imposibles
    esconden lo asimétrico que es el intervalo real en ese extremo.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = hits / total
    d = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _detected(row) -> str:
    """La etiqueta que vale: la revisada por un humano si existe."""
    return row.get("clase_revisada") or row.get("clase_detectada") or ""


def binary_metrics(rows) -> dict:
    """Caída / no-caída: la pregunta clínica, separada de la severidad."""
    tp = fp = tn = fn = 0
    for row in rows:
        truth = row.get("clase_verdad") or ""
        if not truth:
            continue
        actual = truth in FALL_CLASSES
        predicted = _detected(row) in FALL_CLASSES
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    def ratio(num, den):
        return num / den if den else float("nan")

    sens = ratio(tp, tp + fn)
    prec = ratio(tp, tp + fp)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensibilidad": sens,
        "especificidad": ratio(tn, tn + fp),
        "precision": prec,
        "f1": (2 * prec * sens / (prec + sens)
               if prec == prec and sens == sens and (prec + sens) > 0
               else float("nan")),
        "sens_ic": wilson(tp, tp + fn),
        "espec_ic": wilson(tn, tn + fp),
    }


def per_class(rows) -> dict[str, tuple[int, int]]:
    """Aciertos y total por clase verdadera. Donde vive el problema real."""
    out: dict[str, list[int]] = {}
    for row in rows:
        truth = row.get("clase_verdad") or ""
        if not truth:
            continue
        cell = out.setdefault(truth, [0, 0])
        cell[1] += 1
        cell[0] += int(_detected(row) == truth)
    return {k: (v[0], v[1]) for k, v in out.items()}


def fp_per_hour(rows) -> float:
    """Falsos positivos por hora de metraje NO-caída (§3.7).

    La métrica que decide si el sistema es vivible en una casa: una alarma
    falsa por hora vacía la confianza del cuidador en una semana, por buena que
    sea la especificidad por clip. El denominador son sólo los clips no-caída,
    porque una alarma sobre una caída real no es una alarma falsa.
    """
    seconds = 0.0
    false_alarms = 0
    for row in rows:
        if (row.get("clase_verdad") or "") != NO_FALL:
            continue
        try:
            seconds += float(row.get("duracion_s") or 0.0)
        except ValueError:
            pass
        false_alarms += int(_detected(row) in FALL_CLASSES)
    return false_alarms / (seconds / 3600.0) if seconds > 0 else float("nan")


def undetermined_count(rows) -> int:
    return sum(1 for row in rows if _detected(row) == UNDETERMINED)


def report(rows, title: str = "") -> str:
    """Todo el §3.7 sobre un conjunto de filas, como texto."""
    scored = [r for r in rows if r.get("clase_verdad")]
    if not scored:
        return f"{title}: sin clips con verdad legible"
    hits = sum(1 for r in scored if _detected(r) == r["clase_verdad"])
    lo, hi = wilson(hits, len(scored))
    b = binary_metrics(scored)
    lines = [
        f"{title}  ({len(scored)} clips)",
        f"  accuracy 4 clases : {hits}/{len(scored)} = {100 * hits / len(scored):.1f}% "
        f"[IC95 {100 * lo:.1f}–{100 * hi:.1f}]",
        f"  binario caida/no  : sens {100 * b['sensibilidad']:.1f}% "
        f"[{100 * b['sens_ic'][0]:.0f}–{100 * b['sens_ic'][1]:.0f}]   "
        f"espec {100 * b['especificidad']:.1f}% "
        f"[{100 * b['espec_ic'][0]:.0f}–{100 * b['espec_ic'][1]:.0f}]   "
        f"prec {100 * b['precision']:.1f}%   F1 {100 * b['f1']:.1f}%",
        f"                      TP {b['tp']}  FP {b['fp']}  TN {b['tn']}  FN {b['fn']}",
        f"  falsos pos/hora   : {fp_per_hour(scored):.1f}",
        f"  no determinados   : {undetermined_count(scored)}",
    ]
    for name, (ok, total) in sorted(per_class(scored).items()):
        lines.append(f"  {name:<20}: {ok}/{total} = {100 * ok / total:.0f}%")
    return "\n".join(lines)
