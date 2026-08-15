"""Re-ejecuta las mediciones numéricas citadas en notas/REVISION-CODIGO.md.

Por qué existe este archivo: las mediciones originales se hicieron con
scripts ad-hoc que NO se guardaron. Un número que nadie puede volver a
producir no es evidencia, es una afirmación. Este script deja cada cifra
reproducible con un solo comando:

    python notas/verificar_mediciones.py

Cada bloque imprime el valor documentado y el valor medido ahora, para que
la comparación sea explícita en vez de quedar en la memoria de nadie.

ALCANCE — leer antes de comparar cifras:

Las mediciones de aquí son SINTÉTICAS: ruido gaussiano y una caída modelada
como medio coseno. Dos de las cifras del documento (la reducción de temblor
del EMA, ~63 %, y parte de la dispersión de C5) se midieron sobre METRAJE
REAL. Un número sintético no puede reproducir un número medido sobre video
real: el ruido de MediaPipe no es gaussiano y la caída real no es un coseno.

Por eso lo que este script demuestra es la DIRECCIÓN y el ORDEN DE MAGNITUD
de cada conclusión, no el decimal exacto. Cuando exista la mini-suite de
cuerpo completo, las cifras de metraje real deben re-medirse sobre ella
—corriendo el pipeline con `ema_time_constant_s: 0.0` y con 0.1— y el
documento debe citar esas, no estas.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pef_fall_detector.quantities import (  # noqa: E402
    ExponentialMovingAverage,
    VelocityEstimator,
)

SEP = "=" * 72


def titulo(n: str, doc: str) -> None:
    print(f"\n{SEP}\n{n}\n  documentado: {doc}\n{SEP}")


# ---------------------------------------------------------------------------
# M5 — ¿es alcanzable el desbordamiento del coseno?
# ---------------------------------------------------------------------------
def m5_recorte_coseno() -> None:
    titulo("M5 — recorte del coseno",
           "400 000 troncos casi verticales, 0 desbordamientos; "
           "fórmula general: 13 % de los pares paralelos desbordan")

    rng = np.random.default_rng(2024)
    vertical = np.array([0.0, -1.0])

    # (a) NUESTRA formulación: eje unitario, se divide por una sola norma.
    desbordes = 0
    n = 400_000
    for _ in range(n):
        escala = 10.0 ** rng.uniform(-6, 6)          # 12 órdenes de magnitud
        trunk = np.array([rng.normal(0.0, escala * 1e-9), -escala])
        cos_theta = float(np.dot(trunk, vertical) / np.linalg.norm(trunk))
        if abs(cos_theta) > 1.0:
            desbordes += 1
    print(f"  (a) formulación actual : {desbordes} desbordamientos en {n:,} casos")

    # (b) Fórmula GENERAL (la que escribe la guía y la que haría una versión
    #     3D): se normalizan ambos vectores. Pares paralelos, que es donde
    #     el coseno vale exactamente 1 y el redondeo puede pasarse.
    desbordes_g = 0
    m = 400_000
    for _ in range(m):
        escala = 10.0 ** rng.uniform(-6, 6)
        a = rng.normal(0.0, escala, 3)
        b = a * rng.uniform(0.5, 2.0)               # exactamente paralelo a `a`
        cos_g = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if cos_g > 1.0:
            desbordes_g += 1
    pct = 100.0 * desbordes_g / m
    print(f"  (b) fórmula general    : {desbordes_g:,} de {m:,} pares paralelos "
          f"= {pct:.1f} %  -> arccos daría NaN")


# ---------------------------------------------------------------------------
# EMA — ¿cuánto temblor quita realmente?
# ---------------------------------------------------------------------------
def ema_temblor() -> None:
    titulo("EMA — reducción del temblor de landmarks",
           "~63 % menos temblor, ~4 % de atenuación del pico")

    fps, dt = 30.0, 1.0 / 30.0
    tau = 0.1                    # config.yaml: ema_time_constant_s
    torso = 150.0                # px
    rng = np.random.default_rng(7)

    # Sujeto que camina despacio (movimiento REAL, de baja frecuencia) con
    # el temblor de MediaPipe encima (ruido, de alta frecuencia).
    n = 3000
    t = np.arange(n) * dt
    real_y = 400.0 + 6.0 * np.sin(2 * np.pi * 0.4 * t)     # 0.4 Hz
    ruido = rng.normal(0.0, 2.0, n)                         # ~2 px de temblor

    def serie_v(usar_ema: bool) -> np.ndarray:
        est = VelocityEstimator(window_seconds=0.167, max_gap_s=0.5)
        ema = ExponentialMovingAverage(tau if usar_ema else 0.0)
        out = []
        for i in range(n):
            c = np.array([320.0, real_y[i] + ruido[i]])
            c = np.asarray(ema.update(c, dt=dt), dtype=float)
            v, _ = est.update(t[i], c, torso)
            out.append(v)
        return np.array([x for x in out if not math.isnan(x)])

    v_sin, v_con = serie_v(False), serie_v(True)
    k = min(len(v_sin), len(v_con))
    v_sin, v_con = v_sin[-k:], v_con[-k:]

    # AISLAMIENTO DE ALTA FRECUENCIA: la dispersión total de V incluye el
    # movimiento real del sujeto. Restar muestras consecutivas cancela lo
    # lento y deja el temblor, que es lo único que el filtro debe quitar.
    # (Medir la dispersión total fue el error que casi hace quitar el EMA.)
    hf_sin = float(np.std(np.diff(v_sin)))
    hf_con = float(np.std(np.diff(v_con)))
    print(f"  temblor de V sin EMA   : {hf_sin:.4f} torso/s")
    print(f"  temblor de V con EMA   : {hf_con:.4f} torso/s")
    print(f"  reducción              : {100.0 * (1 - hf_con / hf_sin):.1f} %")

    # Atenuación del pico: una caída limpia, sin ruido, con y sin filtro.
    T, A = 0.3, 1.0
    def pico(usar_ema: bool) -> float:
        est = VelocityEstimator(window_seconds=0.167, max_gap_s=0.5)
        ema = ExponentialMovingAverage(tau if usar_ema else 0.0)
        peor = 0.0
        for i in range(int(2.0 * fps)):
            ti = i * dt
            caida = A * (1 - math.cos(math.pi * min(ti / T, 1.0))) / 2.0
            c = np.array([320.0, 300.0 + caida * torso])
            c = np.asarray(ema.update(c, dt=dt), dtype=float)
            v, _ = est.update(ti, c, torso)
            if not math.isnan(v):
                peor = min(peor, v)
        return peor

    p_sin, p_con = pico(False), pico(True)
    print(f"  pico sin EMA           : {p_sin:+.2f} torso/s")
    print(f"  pico con EMA           : {p_con:+.2f} torso/s")
    print(f"  atenuación del pico    : {100.0 * (1 - p_con / p_sin):.1f} %")


# ---------------------------------------------------------------------------
# M3 — qué filtrar en extension_ratio
# ---------------------------------------------------------------------------
def m3_politica_de_filtrado() -> None:
    titulo("M3 — política de suavizado en extension_ratio",
           "suavizar solo el denominador: 16 % · filtrar la salida: 80 %")

    dt, tau, n = 1.0 / 30.0, 0.1, 3000
    rng = np.random.default_rng(19)

    torso_real, extension_real = 150.0, 260.0        # sujeto quieto
    ruido_t = rng.normal(0.0, 3.0, n)
    ruido_e = rng.normal(0.0, 5.0, n)

    # (1) crudo: sin filtrar nada
    crudo = (extension_real + ruido_e) / (torso_real + ruido_t)

    # (2) mezcla (lo que hacía el código): torso filtrado, extensión cruda
    ema_t = ExponentialMovingAverage(tau)
    mezcla = np.array([
        (extension_real + ruido_e[i]) / float(ema_t.update(torso_real + ruido_t[i], dt=dt))
        for i in range(n)
    ])

    # (3) filtrar la salida (lo que hace ahora): razón cruda, resultado filtrado
    ema_r = ExponentialMovingAverage(tau)
    salida = np.array([float(ema_r.update(crudo[i], dt=dt)) for i in range(n)])

    base = float(np.std(crudo[100:]))
    for nombre, serie in (("mezcla (denominador)", mezcla), ("filtrar la salida", salida)):
        red = 100.0 * (1 - float(np.std(serie[100:])) / base)
        print(f"  {nombre:<22}: reduce el ruido {red:.0f} %")


# ---------------------------------------------------------------------------
# C5 — ventana en frames vs. ventana en segundos
# ---------------------------------------------------------------------------
def c5_ventana_tiempo_vs_frames() -> None:
    titulo("C5 — la misma caída física medida a distintos fps",
           "en frames: pico de −5.03 a −2.11 (dispersión 138 %) · "
           "en segundos: dispersión 5.7 %")

    T, A, torso = 0.3, 1.0, 150.0          # caída: 1 torso en 0.3 s

    def pico(fps: float, ventana_s: float | None, ventana_frames: int | None) -> float:
        dt = 1.0 / fps
        w = ventana_frames * dt if ventana_s is None else ventana_s
        est = VelocityEstimator(window_seconds=w, max_gap_s=1.0)
        peor = 0.0
        for i in range(int(2.0 * fps)):
            ti = i * dt
            caida = A * (1 - math.cos(math.pi * min(ti / T, 1.0))) / 2.0
            v, _ = est.update(ti, np.array([320.0, 300.0 + caida * torso]), torso)
            if not math.isnan(v):
                peor = min(peor, v)
        return peor

    tasas = [60.0, 30.0, 25.0, 15.0, 10.0]
    print("   fps |  ventana=5 frames  |  ventana=0.167 s")
    print("  -----+--------------------+------------------")
    en_frames, en_segundos = [], []
    for fps in tasas:
        a = pico(fps, None, 5)
        b = pico(fps, 0.167, None)
        en_frames.append(abs(a))
        en_segundos.append(abs(b))
        print(f"  {fps:5.0f}|      {a:+6.2f}        |     {b:+6.2f}")

    def dispersion(vals: list[float]) -> float:
        return 100.0 * (max(vals) - min(vals)) / min(vals)

    print(f"\n  dispersión en frames  : {dispersion(en_frames):.1f} %")
    print(f"  dispersión en segundos: {dispersion(en_segundos):.1f} %")


if __name__ == "__main__":
    m5_recorte_coseno()
    ema_temblor()
    m3_politica_de_filtrado()
    c5_ventana_tiempo_vs_frames()
    print(f"\n{SEP}\nFin. Compare cada 'medido ahora' con el 'documentado'.\n{SEP}")
