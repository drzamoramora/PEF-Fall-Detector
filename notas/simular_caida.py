"""Caídas sintéticas de punta a punta por el embudo real del §3.5.

Por qué existe: hasta ahora el embudo completo (Etapa 1 -> 2 -> 3 + severidad)
sólo se había visto resolver sobre grabaciones donde NO hay una caída anotada,
y sobre pruebas unitarias que alimentan cada etapa por separado. Ninguna de las
dos cosas responde la pregunta que importa: **¿una caída entra por un extremo y
sale por el otro con el veredicto correcto?**

Este script construye cuerpos sintéticos — un tronco rígido que pivota sobre los
tobillos, con la anatomía en proporciones reales — y los pasa por las MISMAS
funciones que corren en producción: `trunk_inclination_deg`, `centroid`,
`ExponentialMovingAverage`, `VelocityEstimator`, `feet_in_contact`,
`support_polygon`, `com_support_offset`, `ImmobilityTimer` y `FallStateMachine`.
No hay reimplementación de ningún criterio; si el embudo cambia, este script
cambia de resultado solo.

Lo que es y lo que NO es
-----------------------
ES una prueba de que la cadena está conectada y de que los umbrales actuales
dejan pasar una caída limpia. NO es evidencia de exactitud: la cinemática es
un modelo, no una persona, y no hay ruido de MediaPipe, ni oclusión, ni
perspectiva. La sensibilidad y la especificidad reales salen del dataset
(§3.7), no de aquí. Un escenario ADL de control va incluido justamente para
que el script no pueda "aprobar" declarando caída a todo.

    python notas/simular_caida.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pef_fall_detector.quantities import (  # noqa: E402
    ExponentialMovingAverage,
    ImmobilityTimer,
    VelocityEstimator,
    centroid,
    com_support_offset,
    feet_in_contact,
    support_polygon,
    trunk_inclination_deg,
)
from pef_fall_detector.state_machine import (  # noqa: E402
    SCORE,
    FallStateMachine,
    Stage1Trigger,
    Stage2Evaluator,
    Stage3Evaluator,
)

# --- Configuración: los valores de config.yaml, no valores de conveniencia ---
T_TH, V_TH, SCORE_TH, HOLD, WINDOW = 45.0, -1.5, 2.2, 0.1, 1.0
TAU, VEL_WINDOW = 0.1, 0.167
HIP_W = 0.65
P_WINDOW, P_FRACTION, P_MIN = 0.33, 0.5, 3
CONTACT_BAND = 0.15
EPS, W_SECONDS, OBS_WINDOW = 0.05, 5.0, 30.0
UPRIGHT_T, RECOVERY_HOLD, STANDING_EXT = 30.0, 1.0, 1.1
COOLDOWN = 3.0

FRAME = (1280, 960)
TORSO = 200.0                    # hombro -> cadera, en píxeles
HIP_ANKLE = 1.5 * TORSO          # proporción anatómica: pierna 1.5x el tronco
ANKLE = np.array([440.0, 900.0])  # el pivote, con espacio a la derecha


def body(theta_deg: float, ankle: np.ndarray = ANKLE) -> dict:
    """Un cuerpo rígido inclinado `theta_deg` grados desde la vertical.

    El tronco pivota sobre los tobillos, que es cómo cae de verdad alguien que
    tropieza hacia adelante: los pies quedan donde estaban y el centro de masa
    sale del polígono de apoyo. Devuelve exactamente los puntos que el
    front-end de pose entrega al pipeline.
    """
    rad = math.radians(theta_deg)
    direction = np.array([math.sin(rad), -math.cos(rad)])
    hip = ankle + HIP_ANKLE * direction
    shoulder = ankle + (HIP_ANKLE + TORSO) * direction
    nose = ankle + (HIP_ANKLE + TORSO * 1.35) * direction
    # Seis puntos de pie alrededor del tobillo, apoyados en el suelo.
    feet = np.array([
        ankle + [-35.0, 0.0], ankle + [35.0, 0.0],       # tobillos
        ankle + [-45.0, 6.0], ankle + [25.0, 6.0],       # talones
        ankle + [-15.0, 11.0], ankle + [55.0, 11.0],     # puntas
    ])
    return {"hip": hip, "shoulder": shoulder, "nose": nose, "feet": feet}


def stand(_t: float) -> float:
    return 0.0


def topple(duration: float = 0.8, theta_max: float = 90.0):
    """Perfil angular de una caída: aceleración angular constante."""
    def profile(t: float) -> float:
        return theta_max * min(1.0, (t / duration)) ** 2
    return profile


def sit(duration: float = 2.0):
    """ADL de control: sentarse. El tronco apenas se inclina y baja despacio."""
    def profile(t: float) -> float:
        return 25.0 * min(1.0, t / duration)
    return profile


def scenario_frames(segments, fps: float = 30.0):
    """Concatena (duración_s, perfil_de_angulo, descenso_de_cadera_px)."""
    dt, clock, out = 1.0 / fps, 0.0, []
    for duration, profile, sink in segments:
        n = int(round(duration * fps))
        for i in range(n):
            local = i * dt
            theta = profile(local)
            b = body(theta)
            if sink:
                # Sentarse baja la cadera sin tumbar el tronco.
                drop = np.array([0.0, sink * min(1.0, local / max(duration, 1e-9))])
                b["hip"] = b["hip"] + drop
                b["shoulder"] = b["shoulder"] + drop
                b["nose"] = b["nose"] + drop
            out.append((clock + local, b))
        clock += n * dt
    return out


def run(frames, label: str, verbose: bool = False) -> dict:
    """Pasa una secuencia por el embudo completo y devuelve lo que salió."""
    ema_c = ExponentialMovingAverage(TAU)
    ema_t = ExponentialMovingAverage(TAU)
    velocity = VelocityEstimator(VEL_WINDOW)
    still = ImmobilityTimer(EPS)
    machine = FallStateMachine(
        Stage1Trigger(formulation=SCORE, threshold_t_deg=T_TH,
                      threshold_v_tps=V_TH, hold_s=HOLD,
                      confirm_window_s=WINDOW, threshold_score=SCORE_TH),
        Stage2Evaluator(window_s=P_WINDOW, outside_fraction=P_FRACTION,
                        min_samples=P_MIN),
        Stage3Evaluator(window_s=OBS_WINDOW, threshold_w_s=W_SECONDS,
                        upright_t_deg=UPRIGHT_T, recovery_hold_s=RECOVERY_HOLD,
                        standing_extension=STANDING_EXT),
        cooldown_s=COOLDOWN,
    )

    previous, peak_v, peak_score, resolved = None, 0.0, 0.0, []
    for index, (timestamp, b) in enumerate(frames):
        dt = 0.0 if previous is None else timestamp - previous
        previous = timestamp

        torso = float(np.linalg.norm(b["shoulder"] - b["hip"]))
        t_deg = trunk_inclination_deg(b["hip"], b["shoulder"])
        raw_com = centroid(b["hip"], b["shoulder"], HIP_W)
        com = ema_c.update(raw_com, dt)
        smooth_torso = float(ema_t.update(torso, dt))
        v_tps, _ = velocity.update(timestamp, com, smooth_torso)

        contact = feet_in_contact(b["feet"], FRAME, torso, CONTACT_BAND)
        p_offset = com_support_offset(raw_com, support_polygon(contact), torso)

        pts = np.stack([b["nose"], b["shoulder"], b["hip"]])
        i_still, _ = still.update(timestamp, pts, torso)
        extension = float(np.linalg.norm(b["hip"] - b["feet"][:2].mean(axis=0))) / torso

        if v_tps == v_tps:
            peak_v = min(peak_v, v_tps)
            if t_deg == t_deg and v_tps < 0.0:
                peak_score = max(peak_score, t_deg / T_TH + v_tps / V_TH)

        machine.update(index, timestamp, t_deg, v_tps, p_offset, i_still, extension)
        if machine.just_resolved is not None:
            event = machine.just_resolved
            resolved.append(event)
            if verbose:
                print(f"      resuelto en t={timestamp:.2f}s: "
                      f"{event.verdict}/{event.severity or '-'}")

    return {"label": label, "events": machine.events, "resolved": resolved,
            "peak_v": peak_v, "peak_score": peak_score}


def report(result: dict) -> None:
    print(f"--- {result['label']}")
    print(f"    V pico {result['peak_v']:.2f} torso/s | "
          f"puntaje pico {result['peak_score']:.2f} (umbral {SCORE_TH})")
    if not result["events"]:
        print("    Etapa 1: no disparó\n")
        return
    for event in result["events"]:
        print(f"    disparo t={event.timestamp:.2f}s  T={event.t_deg:.1f}deg  "
              f"V={event.v_tps:.2f}  ->  {event.verdict}"
              f"{'/' + event.severity if event.severity else ''}")
        print(f"      Etapa 2: {event.p_samples} muestras, "
              f"{event.p_outside_fraction:.0%} fuera del apoyo"
              if event.p_samples else "      Etapa 2: sin muestras (no concluyente)")
        if event.max_immobility_s == event.max_immobility_s:
            print(f"      Etapa 3: inmovilidad máxima {event.max_immobility_s:.2f}s "
                  f"(umbral W={W_SECONDS}s)")
    print()


def main() -> int:
    print("Caídas sintéticas por el embudo real (§3.5), formulación 'score'\n")

    severe = run(scenario_frames([
        (3.0, stand, 0.0),
        (0.8, topple(), 0.0),
        (12.0, lambda _t: 90.0, 0.0),          # queda en el suelo, inmóvil
    ]), "CAÍDA — permanece en el suelo (esperado: confirmada, severa)")

    recovered = run(scenario_frames([
        (3.0, stand, 0.0),
        (0.8, topple(), 0.0),
        (2.0, lambda _t: 90.0, 0.0),           # dos segundos abajo
        (1.0, lambda t: 90.0 * (1.0 - min(1.0, t / 1.0)), 0.0),   # se levanta
        (4.0, stand, 0.0),                     # y se mantiene de pie
    ]), "CAÍDA — se levanta sola (esperado: anulada o leve)")

    adl = run(scenario_frames([
        (3.0, stand, 0.0),
        (2.0, sit(), 260.0),                   # se sienta, despacio
        (8.0, lambda _t: 25.0, 260.0),
    ]), "ADL de control — sentarse (esperado: NO dispara)")

    for result in (severe, recovered, adl):
        report(result)

    # La lección C5, aplicada al embudo entero: el mismo evento medido a dos
    # velocidades de captura debe dar el mismo veredicto. Un umbral en frames
    # daría dos respuestas distintas aquí, y la Pi del §3.6 corre más lento.
    print("Invariancia a la tasa de captura (la misma caída a 10 y 30 fps):")
    for fps in (10.0, 30.0, 60.0):
        r = run(scenario_frames([(3.0, stand, 0.0), (0.8, topple(), 0.0),
                                 (12.0, lambda _t: 90.0, 0.0)], fps=fps), "")
        verdicts = [f"{e.verdict}/{e.severity or '-'}" for e in r["events"]]
        print(f"  {fps:5.0f} fps -> {verdicts or 'sin eventos'}")

    ok = (
        len(severe["events"]) == 1
        and severe["events"][0].verdict == "stage3_confirmed"
        and severe["events"][0].severity == "severe"
        and not adl["events"]
    )
    print("\n" + ("OK: la caída atraviesa el embudo y el ADL no dispara."
                  if ok else "FALLA: revise los veredictos de arriba."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
