# Fases 0 y 1 — medidas sobre los 128 (23/09, complexity 1, runner de etiquetado)

| | accuracy 4 clases | caídas exactas | sens | espec |
|---|---|---|---|---|
| Fase 0 (base) | 99/128 = 77.3 % | 55/72 = 76.4 % | 93.1 % | 78.6 % |
| Fase 1 | **101/128 = 78.9 %** | **57/72 = 79.2 %** | 93.1 % | 78.6 % |

Fase 0 reproduce la corrida de QuantityH del 11/09 clip a clip (los mismos 17
errores en caídas): el contenedor queda validado como control.

Fase 1 (T decide en `_is_lying` cuando está medida): gana A12-S1 y A12-S4, no
pierde nada. El costo temido en A14 (cámara de frente) no apareció. B05-S1..S3
pasan de NotRecovered a Recovered/PartiallyRecovered: siguen siendo falsos
positivos, porque el evento se dispara por puntaje, no por H.

## Fase 2 — el mecanismo real no es finalise()

`pipeline._abandon_pending_event()` borra el evento si el sujeto falta más de
`history_max_gap_s` (0.5 s). Existe por una razón real (multipersona: el cuerpo
que reaparece puede ser otro). Eventos que pasaron la Etapa 2 y fueron borrados:

| clip | verdad | última T vista antes de perderlo |
|---|---|---|
| A08-S3 | NotRecovered | 160° (en el suelo) |
| B07-S3 | NoFall | 2.2° |
| B08-S3 | NoFall | 3.0° |
| B10-S2 | NoFall | 3.7° |

Variante A (cerrar todo evento abandonado con la última postura vista):
+1 caída, −3 NoFall. Variante B (cerrar solo si se lo vio en el suelo):
+1 caída, 0 FP — pero diseñada después de ver estos 4 clips.

## Fase 3 — A17 por inmovilidad (§3.5)

% del tiempo post-disparo quieto (I ≥ 0.5 s): A17-S2 71 %, A17-S4 63 %;
21 PartiallyRecovered que terminan sentados 0–35 %; 22 Recovered 0–38 %.
A17 termina con T 2.7° y 25.3°: sentado contra la pared el tronco queda vertical.

---

# Fases 2 (variante B) y 3 — medidas sobre los 128 (23/09)

| | accuracy 4 clases | caídas exactas | sens | espec | FN | FP |
|---|---|---|---|---|---|---|
| Fase 0 (base) | 99/128 = 77.3 % | 55/72 = 76.4 % | 93.1 % | 78.6 % | 5 | 12 |
| Fase 1 | 101/128 = 78.9 % | 57/72 = 79.2 % | 93.1 % | 78.6 % | 5 | 12 |
| **Fases 2B + 3** | **104/128 = 81.2 %** | **60/72 = 83.3 %** | **94.4 %** | 78.6 % | **4** | 12 |

Ganancias, ninguna pérdida: A08-S3 (NoFall → NotRecovered, fase 2B),
A17-S2 y A17-S4 (PartiallyRecovered → NotRecovered, fase 3). B05-S2/S3 cambian
de severidad pero ya eran falsos positivos; ningún NoFall nuevo se volvió FP.

Implementado:
- `FallStateMachine.close_if_last_seen_down()` + `Stage3Evaluator.last_seen_down()`;
  `pipeline.analyze()` lo llama antes de abandonar por pérdida (no en saltos).
- `Stage3Evaluator._moderate_or_persistent()` en las dos salidas moderate de
  `finalise()`; `config.yaml` stage3: `persistent_still_s: 0.5`,
  `persistent_immobility_fraction: 0.5`.
- `tests/test_fases_2_3.py` (15 pruebas). 392 pruebas en total.
- Mutaciones: 9 probadas; la única que sobrevive (no reiniciar contadores en
  `reset()`) es equivalente — `start()` también los reinicia.

## Los 12 errores que quedan en caídas

| grupo | clips |
|---|---|
| caídas perdidas, causa sin mirar | A13-S4, A3-S2, A14-S2, A16-S1 |
| esqueleto infiel | A06-S14, A06-S2, A06-S3, A09-S1, A09-S2, A09-S3, A14-S4 |
| de pie sin tobillos visibles | A04-S1 |
