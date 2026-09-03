"""Re-mide las cuatro formulaciones de la Etapa 1 sobre los CSV ya grabados.

Por qué existe: las cifras que justifican `trigger_formulation: score`
en el `config.yaml` no deben vivir solo en un mensaje de chat. Este script
las regenera con un comando, **pasando los datos reales por el mismo código
que corre en producción** — no por una reimplementación del criterio, que es
como se cuelan las diferencias entre lo que se midió y lo que se envió.

    python notas/replay_disparador.py [carpeta_de_logs]

Clasificación de clips: cualquier CSV cuyo nombre empiece con `camera0` se
trata como NO-caída (son sesiones en vivo del autor); el resto, como caída
(vienen del dataset público). Es una convención frágil y deliberadamente
visible — cuando exista PEF-FallDB con su protocolo de anotación del §3.3,
la etiqueta verdadera sale de la ruta y esta heurística se va.

ALCANCE: 13 caídas y 4 sesiones no-caída es una muestra chica y desbalanceada.
La columna de sensibilidad es creíble; la de **especificidad no** — no hay
clips de ADL curados (caminar, sentarse, agacharse). No cite el número de
falsos positivos como resultado.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pef_fall_detector.state_machine import (  # noqa: E402
    SCORE,
    SEQUENTIAL,
    SIMULTANEOUS,
    V_ONLY,
    FallStateMachine,
    Stage1Trigger,
)

T_TH, V_TH, HOLD, WINDOW, COOLDOWN = 45.0, -1.5, 0.1, 1.0, 3.0
SCORE_TH = 2.2


def load(path: str) -> list[tuple[float, float, float]]:
    """Reliable frames only, as (timestamp, T, V)."""
    out = []
    for row in csv.DictReader(open(path)):
        if row.get("reliable") != "1":
            continue

        def value(name: str) -> float:
            raw = row.get(name)
            if raw in (None, "", "nan"):
                return float("nan")
            try:
                return float(raw)
            except ValueError:
                return float("nan")

        ts = value("timestamp_s")
        if ts == ts:                      # not NaN
            out.append((ts, value("T_deg"), value("V_tps")))
    return out


def fires(track, formulation: str) -> bool:
    """Whether Stage 1 raises any event over this clip."""
    machine = FallStateMachine(
        Stage1Trigger(formulation=formulation, threshold_t_deg=T_TH,
                      threshold_v_tps=V_TH, hold_s=HOLD,
                      confirm_window_s=WINDOW, threshold_score=SCORE_TH),
        cooldown_s=COOLDOWN,
    )
    return any(
        machine.update(i, ts, t_deg, v_tps) is not None
        for i, (ts, t_deg, v_tps) in enumerate(track)
    )


def main(folder: str) -> int:
    falls, adl = [], []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        track = load(path)
        if len(track) < 20:
            continue
        name = os.path.basename(path)
        (adl if name.startswith("camera0") else falls).append((name[:26], track))

    if not falls:
        print(f"No hay CSV utilizables en {folder}")
        return 1

    print(f"Clips: {len(falls)} caidas, {len(adl)} no-caida "
          f"(umbral T>{T_TH}, V<{V_TH}, hold {HOLD}s, score>={SCORE_TH})\n")
    print(f"{'formulacion':<14}{'caidas':>12}{'falsos+':>12}")
    for formulation in (SCORE, SIMULTANEOUS, SEQUENTIAL, V_ONLY):
        tp = sum(fires(t, formulation) for _, t in falls)
        fp = sum(fires(t, formulation) for _, t in adl)
        print(f"{formulation:<14}{tp:>8}/{len(falls):<3}{fp:>8}/{len(adl):<3}")

    print("\nCaidas perdidas por cada formulacion:")
    for formulation in (SCORE, SIMULTANEOUS, SEQUENTIAL, V_ONLY):
        missed = [n for n, t in falls if not fires(t, formulation)]
        print(f"  {formulation:<13} {missed or 'ninguna'}")

    print("\nRecordatorio: la columna de falsos positivos NO es evidencia de "
          "especificidad.\nFaltan clips de ADL curados (caminar, sentarse, "
          "agacharse).")
    return 0


if __name__ == "__main__":
    default = Path(__file__).resolve().parent.parent / "logs"
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(default)))
