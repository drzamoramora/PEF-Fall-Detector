# ANALISIS-82-90.md — dónde estamos y la medida nueva A (25/09)

Rama `82-90` (sale de `main` = commit `432fdf9 update`, fases 0–6).
Medido en dos plataformas: nube x86 (`PEF-lm`) y Mac, corrida en terminal (`logs/cli-lm`).
El registro del esqueleto completo (`logging.save_landmarks`) no cambió ninguna
decisión: los 128 CSV de cuadros son idénticos a los de la fase 6.

## 1. Lo hecho

| | nube | Mac |
|---|---|---|
| fase 0 | 99/128 (77.3 %) | — |
| fases 1→6 | 106/128 (82.8 %) | 102/128 (79.7 %) |
| caídas exactas | 62/72 | 58/72 |
| sensibilidad / especificidad | 95.8 % / 78.6 % | 88.9 % / 78.6 % |

Cerrado con medición: fase 5, fase 7 (H como disparador), fase 6 original
(T > 120°), 6b, modelo lite y heavy, banda del disparador, 10 vías de reparar
el esqueleto; NoFall: descenso controlado, "erguido antes", recuperación rápida,
R, frenado (ver NOFALL.md).

## 2. Errores hoy (Mac; nube entre paréntesis)

- Caídas no detectadas 8 (3): A01-S4, A09-S2, A13-S4, A3-S2, A14-S1, A14-S2, A16-S2, A17-S4.
- Severidad errada 6 (7): A04-S1, A06-S3, A09-S1, A09-S3, A14-S4, A18-S3 (nube: + A06-S2, A09-S2).
- Falsos positivos 12 (12): sofá B05 ×3; inclinarse/agacharse B03-S3, B09-S4, B10-S1, B10-S3;
  levantarse B14-S2; acostarse B11-S3, B11-S4, B12-S1, B12-S4.

Diagnóstico de fondo: casi todo es la misma pregunta física —¿a qué altura
quedó el cuerpo respecto del piso?— respondida con un ángulo 2D. T no distingue
acostado-hacia-la-cámara de esqueleto invertido (T ≈ 175° en ambos), ni caer
alejándose de la cámara de estar de pie (A14-S4, T 28°), ni sentarse o agacharse
de caer.

## 3. Medida nueva: A — altura de la cabeza, 3D, en la dirección de la gravedad

A = (altura de la nariz sobre los tobillos a lo largo del "arriba") / (la misma
altura con el sujeto de pie). Coordenadas de mundo de MediaPipe (metros; el
§3.2 ya dice "infers 33 3D body landmarks"). El "arriba" y la altura de pie se
calibran con el propio sujeto parado (T < 20° en la imagen y rodillas ≥ 150° en
3D): no depende de la inclinación de la cámara. Física directa: una caída es
pérdida de altura contra la gravedad. Explicable: "la cabeza quedó al 12 % de su
altura de pie".

Distribución al final del clip (nube; Mac casi igual):

| postura final | A |
|---|---|
| de pie (Recovered) | 0.93 – 1.09 |
| sentado / arrodillado (Partially) | 0.28 – 0.70 |
| en el suelo (NotRecovered) | −0.21 – 0.25 (A17 contra la pared: 0.33 – 0.51) |

Usos:
- U1 severidad al final: bandas 0.26 / 0.82 (huecos de la partición de entrenamiento).
- U2 confirmación "¿llegó al suelo?": si A no bajó de 0.28 en [disparo − 1 s,
  disparo + 4 s] y la cadera terminó elevada (> 0.22), no es caída.
- U3 disparador complementario: A pasa de ≥ 0.7 a ≤ 0.3 en ≤ 1.5 s.

## 4. Efecto estimado U1 + U2 (fuera de línea, sin tocar el código)

| | nube | Mac |
|---|---|---|
| accuracy | 106 → 116/128 (90.6 %) | 102 → 112/128 (87.5 %) |
| caídas exactas | 62 → 66/72 | 58 → 61/72 |
| especificidad | 44 → 50/56 | 44 → 51/56 |
| sensibilidad | igual | igual |
| entrenamiento | 83.0 → 92.0 % | 81.8 → 90.9 % |
| prueba | 82.5 → 87.5 % | 75.0 → 80.0 % |

Única pérdida en ambas: A17-S1 (desplomado contra la pared, A = 0.33).
U3 en el Mac alcanzaría 7 de las 8 caídas no detectadas (todas menos A17-S4);
nuevos disparos en B07-S4, B09-S3, B10-S2, B11-S2. Hay que medirlo en el sistema real.

## 5. Lo que A no resuelve

- B11/B12 (acostarse a propósito): postura final idéntica a una caída. Límite.
- Clips sin el sujeto de pie (A02 sentado, A03 en el sofá, A09-S1/S3 entra tarde):
  sin calibración decide T como hoy. En despliegue la calibración es por cámara.
- El 3D de MediaPipe es estimado: los miembros ocultos se infieren.
- Se exploró mirando también clips de prueba; los umbrales coinciden con los huecos
  de entrenamiento, pero la cifra del paper sale de prueba con configuración congelada.
- Paper: A es una cantidad nueva (§3.4) y su uso (§3.5): handoff.

## 6. Plan

| fase | qué | acepta si |
|---|---|---|
| 8.1 | cantidad A + calibración, columna CSV, curva GUI; ninguna decisión cambia | cifras idénticas |
| 8.2 | U1 severidad | mejora en ambas plataformas sin perder caídas |
| 8.3 | U2 confirmación | ídem |
| 8.4 | U3 disparador | decisión del usuario sobre el costo en FP |
