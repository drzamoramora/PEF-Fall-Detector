# NOFALL.md — Fase N0: diagnóstico de los 12 falsos positivos (25/09)

Código: `ajustes2` con fase 6. Medido en las dos plataformas: nube x86
(`PEF-f6`) y Mac, corrida en terminal (`logs/cli-fase6`). Los mismos 12 FP en ambas:
especificidad 44/56 = 78.6 %.

## Familias (revisión visual del usuario + §3.3 del draft 4)

| familia | clips | actividad | partición |
|---|---|---|---|
| sofá | B05-S1, S2, S3 | "sitting down with rapid deceleration on a chair or sofa" | entrenamiento |
| inclinarse / cuclillas | B03-S3 (horno), B09-S4 (cuclillas, zapato), B10-S1, B10-S3 (zapato bajando escaleras) | "bending forward… to tie a shoe" / "crouching" | B03, B09 prueba |
| levantarse | B14-S2 (acostado en el suelo, se levanta y se va) | "rising from a lying position" | entrenamiento |
| acostarse | B11-S3, B11-S4, B12-S1, B12-S4 | "lying down intentionally" | B12 prueba |

## Medición en el disparo (nube / Mac casi idénticos)

| clip | V mínima (ventana del código) | tiempo bajo −1 tps | T mín. 2 s antes | veredicto |
|---|---|---|---|---|
| B05-S1 | −3.13 | 0.37 s | 2° | nullified/mild |
| B05-S2 | −2.81 | 0.30 s | 0° | confirmed/severe |
| B05-S3 | −2.5 | 0.40 s | 2° | confirmed/severe |
| B03-S3 | −5.8 / −6.2 | 0.20–0.30 s | 0° | nullified/mild |
| B09-S4 | −0.9 | 0.00 s | 1° | nullified/mild |
| B10-S1 | −1.16 | 0.17 s | 4° | confirmed/moderate |
| B10-S3 | −1.9 | 0.17–0.23 s | 1°–167° | confirmed/moderate |
| B14-S2 | −1.3 / −1.5 | 0.13–0.17 s | 0°–2° | severe (nube) / mild (Mac) |
| B11/B12 | −0.7 a −3.7 | 0.0–0.6 s | 0°–44° | confirmed/severe |

Caídas (69 eventos en Etapa 3): V mínima mediana −2.6 (nube) / −2.8 (Mac),
tiempo bajo −1 tps mediana 0.57 s.

## Las tres vías planeadas, medidas antes de programar — las tres fallan

1. **N1, descenso controlado (sofá).** B05 NO es un descenso suave: V mínima
   −2.5 a −3.1 y 0.30–0.43 s bajo −1 tps, igual que una caída típica. El §3.3
   lo diseñó así ("rapid deceleration"). Cualquier corte que rechace algo pierde
   caídas: con V > −2.0 y < 0.2 s, pierde 6 (Mac) a 8 (nube) caídas para quitar
   5 FP, y ninguno es de B05. Vía cerrada para B05.
2. **N2b, "estaba erguido antes del disparo" (levantarse).** B14-S2 NO disparó
   estando acostado: T mínima 0–2° en los 2 s previos (ya estaba de pie). Y
   caídas reales tienen T previa 127–180° por esqueleto invertido (A09-S1,
   A16-S3), que la regla perdería. Vía cerrada.
3. **N2, recuperación rápida después del disparo (inclinarse).** B03-S3 y
   B05-S1 están erguidos de inmediato (0.00 s, T nunca ≥ 60°), pero la caída
   Recovered A16-S2 también (0.00 s, T máx. 33°), y A16-S4 en 0.37 s. B09-S4
   tarda 2.67 s, dentro del rango de las caídas (0.8–5.4 s). No separa.

## Conclusión

Los 12 FP son justamente los ADL que el §3.3 incluyó como prueba de estrés
("closely mimics a genuine collapse"). En el disparo, en la Etapa 2 y en la
recuperación son cinemáticamente iguales a caídas reales, en las dos
plataformas. Con las cantidades del §3.4 la especificidad de 78.6 % es el
resultado honesto en este dataset.

## Lo que sí corresponde reportar (§4) — no es una mejora, es la lectura correcta

El §3.5 dice que ante una recuperación "the alarm is nullified", y
`alerts.dispatch_verdicts: [stage3_confirmed]` ya lo implementa: un evento
`stage3_nullified` no dispara alarma. Hay dos métricas distintas:

- **Etiqueta del clip (§3.3, 4 clases):** los 12 FP cuentan. Especificidad 78.6 %.
- **Alarma despachada (§3.5):** solo cuentan los `stage3_confirmed`. FP que
  alarman: 9 en la nube (47/56 = 83.9 %), 8 en el Mac (48/56 = 85.7 %).
  Pero la misma regla deja sin alarma a las caídas Recovered: eso también hay que
  reportarlo, porque es una decisión de diseño del §3.5, no un error.

Pendiente de decisión del usuario: si N1/N2 se cierran como límite documentado
(recomendado) y se pasa a consolidar y al Pi.
