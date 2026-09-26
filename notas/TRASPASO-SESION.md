# TRASPASO-SESION.md — estado completo para seguir en un chat nuevo (25/09/2026; actualizado el 26/09 con la 8.4)

Documento de traspaso del chat largo de Cowork (fases 4 → 8.3). **Leer completo
antes de hacer nada.** Las notas del repositorio (`notas/`) tienen el detalle;
esto es el mapa.

---

## 1. Quién, qué y dónde

- Autor: Arthur (ULACIT). Paper: *"Beyond Black-Box AI: A Physics-Informed Edge Framework for Explainable Markerless Fall Detection Using MediaPipe Landmarks"*. El draft 4 manda.
- Repo: `PEF-Fall-Detector`.
  - Mac: `/Users/arthurjimenez/Paper Fall/PEF-Fall-Detector`.
  - Desde la VM de Cowork: `$HOME/mnt/Paper Fall/PEF-Fall-Detector`.
- Videos: `Paper Fall/recordings/` (128 clips: 72 caídas A01–A18 × S1–S4, clases Recovered / PartiallyRecovered / NotRecovered, y 56 NoFall B01–B14).
  - Errata de nombres: `A3-S2` es A13-S2; `A06-S14` y `A11-S14` son S4.
- Entorno del Mac: `venv/` con `requirements.txt` fijado (mediapipe 0.10.14, protobuf 4.25.9, opencv-contrib-python 4.13.0.92, numpy 2.4.4, PySide6 6.11.2, pyqtgraph 0.14.0, PyYAML 6.0.3). El `venv-viejo` ya se borró.
- Respaldo de la nube (fuera del repo): `Paper Fall/respaldo-nube/`.
  - `scripts/` con `verif2.py`, `diag81.py`, `sim81.py` y los demás; ver su `LEEME.md`.
  - `corridas/nube-8.1` y `corridas/nube-8.3`, que son registros x86 sin los CSV de 33 puntos.
  - `LEGACY.md` y este traspaso.

## 2. Reglas de trabajo del autor (obligatorias)

- **Nunca commit sin que el autor lo diga.** Mensaje siempre `update`; autor y committer solo él. Sin `Co-Authored-By` ni ninguna otra atribución.
  - Commit: `git -c user.name=Arthrrr -c user.email=102784141+Arthrrr@users.noreply.github.com commit -m "update"`.
- Usar `git --no-optional-locks`. Los push los hace él (la VM no tiene credenciales de GitHub).
- Desde la VM de Cowork, hacer commit con `-c core.createObject=rename -c maintenance.auto=false -c gc.auto=0`. Si no, quedan `.lock` y `tmp_obj` que después piden permiso de borrado.
- **Antes de crear o cambiar cualquier archivo: decir cuáles y por qué, y esperar aprobación.** Pide confirmación de cada cosa.
- No borrar nada que no pida. Regrabar video nunca es opción.
- **Prefiere que una caída se detecte, aunque la severidad salga mal, a que quede como NoFall.**
- El draft lo escribe él: los cambios al paper se entregan como sugerencias (`notas/POSIBLES-CAMBIOS-DEL-DRAFT.md`). Se permiten cambios al paper "siempre y cuando den sustento científico y vayan en la misma dirección".
- Idioma: habla español.
  - Notas en español.
  - Código, documentación del repo y GUI en inglés.
- Cuidar el uso de tokens. Pregunta seguido "¿cómo va?": dar estado corto.
- Antes de cada fase, dar por dónde empieza, qué hace, los diagnósticos y las predicciones prometedora, realista y pesimista.
- Objetivo: **90 % "cerrado"** (≥ 116/128), sin buscar el 100 %. Las cifras oficiales saldrán del Raspberry Pi 4B.

## 3. Estado del código

- Rama de trabajo `82-90`.
  - `8ccd27d`: fase 8.1 más el registro de 33 puntos.
  - `2cdd99b`: fases 8.2 + 8.3, la GUI rediseñada con textos en inglés, 562 pruebas.
  - `68d128e`: este traspaso en `notas/`.
  - Commit `update` del 26/09: fase 8.4 (A como disparador), la columna `trigger_source` y 594 pruebas.
- Rama `legacy-pre-cleanup`: la foto antes de la limpieza, con `LEGACY.md`.
  - Etiqueta `v0.1.0-legacy` (`7cc8ec5` = `2cdd99b` + `LEGACY.md`): el estado de la 8.3.
  - Etiqueta `v0.1.1-legacy` (26/09): el estado de la 8.4, con `LEGACY.md` actualizado.
- Push pendiente (lo hace el autor): `git push origin 82-90 legacy-pre-cleanup --tags`.
- Solo queda `.DS_Store` sin rastrear.

### Resultados de la 8.4 (PEF-FallDB, 128 clips, modo etiquetado)

| | nube x86 | Mac ARM |
|---|---|---|
| accuracy | 115/128 (89.8 %) | 110/128 (85.9 %) |
| entrenamiento / prueba (partición por actor) | 81/88 · 34/40 | 80/88 · 30/40 |
| caídas detectadas | 70/72 | 66/72 |
| especificidad etiqueta / alarma | 49/56 · 52/56 | 49/56 · 52/56 |

- Nube: corrida real con MediaPipe (`PEF-84/logs`, en la nube del chat), idéntica a la simulación.
- Mac: lote en terminal en `logs/cli-a84`. Nueve clips quedaron repetidos (de A01-S1 a A03-S1), idénticos byte a byte; los scripts toman el primero.
- La 8.3 del Mac está en los registros de la GUI (`logs/*-20260925-13*`, 109/128).
- Historia de la accuracy:

  | | fase 0 | 4 | 6 | 8.1 | 8.3 | 8.4 |
  |---|---|---|---|---|---|---|
  | nube | 99 | 105 | 106 | 106 | 114 | 115 |
  | Mac | — | 101 | 102 | 102 | 109 | 110 |

- La 8.1 solo registró A: el Mac quedó idéntico byte a byte a la fase 6.
- Nube y Mac difieren por la inferencia de MediaPipe (x86 vs ARM): las etiquetas coincidían en 120/128 con la fase 6. La GUI y la terminal dan lo mismo en la misma máquina.

### Errores que quedan

- **Nube:**
  - no detectadas: A13-S4 (cae sentada con el tronco vertical; su A oscila entre 0.29 y 0.34 y llega abajo a V −1.42) y A14-S2 (cae hacia la cámara y A no calibra);
  - severidad errada: A09-S1, A09-S3 (esqueleto infiel), A14-S4 y A17-S1;
  - falsos positivos: B09-S4, B10-S3, B11-S3, B11-S4, B12-S1, B12-S4 y B14-S2. Solo alarman los de B11 y B12: acostarse a propósito, un límite documentado.
- **Mac:**
  - no detectadas: 6, las 2 de la nube más A01-S4, A09-S2, A16-S2 y A17-S4;
  - severidad: A09-S1, A09-S3, A14-S1 (la detecta A, pero sale PartiallyRecovered), A17-S1 y A18-S3;
  - los mismos 7 falsos positivos.

## 4. Cantidad A (fase 8) — lo esencial

- **Definición:** A = altura de la nariz sobre los tobillos a lo largo del "arriba", dividida por la altura de pie. Usa las coordenadas de mundo de MediaPipe.
- **Calibración:** causal (`GravityCalibrator`), con el sujeto de pie (T < 20° en la imagen y rodillas ≥ 150° en 3D) durante 0.5 s o más.
  - Calibra 116/128 en la nube y 117/128 en el Mac.
  - Sin calibrar: A02-S1/S3/S4, A03 ×4, A09-S1/S3, A14-S2, B12-S4, y A14-S4 solo en la nube.
- **8.2, severidad desde A al final:**
  - ≤ 0.26: severe;
  - ≥ 0.82: mild;
  - en medio: moderate, o severe si la inmovilidad persiste;
  - la subida de cadera (6a) tiene prioridad;
  - costo conocido: A17-S1 (A = 0.28) pasa a moderate.
- **8.3, "¿llegó al suelo?":** si la cabeza nunca bajó de **0.31** entre 1 s antes y 4 s después del disparo, y la cadera terminó por encima de 0.22, el evento es `stage3_not_down` (NoFall, sin alarma).
  - Se abstiene si A no estaba calibrada al disparar. Sin esa regla, el Mac perdía A16-S3, que calibra 2.6 s tarde.
  - El corte es 0.31 y no 0.28 por dos márgenes: A18-S4 (caída real) está en 0.26 y B09-S4 en 0.28–0.29, y el ruido entre plataformas tiene p90 0.04.
  - La condición de la cadera protege a A17-S2 (cabeza 0.44, cadera 0.07).
- **El autor aceptó cambiar la frase del §3.2** (A usa las coordenadas de mundo solo como cociente autocalibrado). Con eso, C5 y C20 de `POSIBLES-CAMBIOS-DEL-DRAFT.md` pasan a ser obligatorios.
- **Las fases 3, 6a, 8.2 y 8.3 solo actúan en modo etiquetado** (en `Stage3Evaluator.finalise`, al final del clip). En vivo la Etapa 3 decide con la regla original (recuperación sostenida 1 s o quieto 5 s) y sin A. Pasarlas a vivo es la **fase L**: código nuevo más su medición, reproduciendo los clips en modo en vivo. Importa para el Pi y para el §3.6.

## 5. Fase 8.4: A como disparador (hecha y aceptada el 26/09)

A como disparador adicional para las caídas que no se ven.

Decisiones del autor (todas las recomendadas):
1. **Disparo:** la cabeza pasa de A ≥ 0.7 a A ≤ 0.3 en ≤ 1.5 s, solo con A calibrada. Es un disparo **además** del actual, no en su lugar.
2. **Criterio de aceptación:**
   - no perder ninguna caída;
   - subir la accuracy en la nube y en el Mac;
   - ningún NoFall con alarma nueva (un falso positivo sin alarma se tolera).
3. **H:** se queda como está; un cambio a la vez.
4. **Cuándo:** antes de la limpieza. Después se crea la etiqueta `v0.1.1-legacy` en `legacy-pre-cleanup`; `v0.1.0-legacy` no se mueve.

Primer paso acordado: **simular la 8.4 sobre las corridas existentes, sin tocar código**.
- En la nube: `respaldo-nube/corridas/nube-8.3`.
- En el Mac: los registros de la GUI del 25/09, que tienen `A_head_EXP`.
- Dar predicciones prometedora, realista y pesimista, y luego pedir aprobación para implementar.

Límites que ya se saben:
- A14-S2 no calibra en ninguna plataforma.
- En varios clips A calibra después de la caída, y ahí no puede disparar.
- Techo realista en la nube: +2 (A13-S4 y A3-S2 → 116/128 = 90.6 %) si no hay falsos positivos nuevos.
- El análisis viejo (fuera de línea) decía que en el Mac alcanzaba 7 de 8, con disparos nuevos en B07-S4, B09-S3, B10-S2 y B11-S2. Hay que rehacerlo con la calibración causal.

### Resultado de la simulación (26/09)

- **Herramienta: `rep84.py` ("replay").** Reconstruye cada cuadro desde el CSV de 33 puntos y lo pasa por `FramePipeline.analyze` del repo, sin MediaPipe.
  - Reproduce las corridas: nube 127/128 eventos idénticos y 128/128 etiquetas; Mac 128/128 etiquetas (GUI del 25/09 con los 33 puntos de `logs/cli-a81`, que son idénticos cuadro a cuadro).
  - Tarda 25 s en la nube y 8 s en la VM del Mac.
  - Hoy vive en la nube del chat y en la VM (`~/ana/`). Falta guardarlo en `respaldo-nube/scripts/` (pedir permiso).
- **La 8.4 como se acordó (A sola) no pasa el criterio.**
  - Nube 114 → 113; Mac 109 → 109.
  - B11-S2 y B12-S2 (acostarse a propósito) ganan alarma en las dos plataformas.
  - Ninguna ventana de tiempo los separa: bajan la cabeza en 0.57 y 0.77 s, más rápido que la caída A13-S4 (1.27 s).
- **Variante aprobada: A + V del cuadro.** Dispara si A ≤ 0.3, con A ≥ 0.7 en los 1.5 s previos, **y** V < −1.5 (el umbral del §3.4) en ese mismo cuadro, sostenido 0.1 s (`trigger_hold_s`). Es un disparo aparte, con su propio pestillo, como H.

  | | nube | Mac |
  |---|---|---|
  | accuracy | 115/128 (89.8 %) | 110/128 (85.9 %) |
  | entrenamiento / prueba | 81/88 · 34/40 | 80/88 · 30/40 |
  | caídas detectadas | 70/72 | 66/72 |
  | especificidad etiqueta / alarma | 49/56 · 52/56 | 49/56 · 52/56 |

  - Gana A3-S2 en las dos. En el Mac, A14-S1 pasa a detectada (PartiallyRecovered en vez de NotRecovered, con alarma).
  - Márgenes de V cuando la cabeza llega abajo:
    - acostarse: de −0.53 a +0.66 (a 0.97 o más del umbral);
    - A3-S2: −2.8 (1.3 de margen);
    - A14-S1 en el Mac: −1.74 y solo 0.13 s sostenido (frágil).
  - Sensibilidad (20 variaciones de los 5 parámetros por plataforma):
    - la nube da 115 en 20/20 y el Mac 110 en 20/20;
    - 0 alarmas nuevas y 0 caídas perdidas en las 40 corridas;
    - A14-S1 se detecta en 17/20.
  - **Declararlo en el paper:** con la V de la ventana de pico (la que usa H), B12-S2 vuelve a alarmar. La V del cuadro se eligió después de ver ese clip, y B12 es de la partición de prueba.
    - El argumento físico no depende de ese clip: una caída llega al suelo con velocidad, y un descenso controlado frena antes.
    - Sustento: Bourke et al. 2008 (*Med. Eng. Phys.* 30:937–946; umbral de velocidad vertical −1.3 m/s, 100 % de exactitud, unos 323 ms antes del impacto) y Wu 2000 (*J. Biomech.*).
  - **No llega al 90 % en la nube:** falta 1 clip para 116. A13-S4 queda fuera, porque su A oscila entre 0.29 y 0.34 y su V al llegar es −1.42.
  - **Riesgo para el Pi:** B07-S4 tiene un salto de A de 1 cuadro con V −3.1. Hoy lo filtra el sostenimiento de 0.1 s.
- **Implementación aprobada (26/09):**
  - `state_machine.py`, `pipeline.py` y `config.yaml`;
  - `audit_log.py`, `main.py` y `gui/lab_window.py`, para la columna `trigger_source` (score / H / A);
  - `tests/test_fase84.py`.

### Resultado real (26/09)

- **Nube 115/128 y Mac 110/128**, exacto lo simulado. Cumple los tres criterios en las dos plataformas y el autor la aceptó.
- 594 pruebas (32 nuevas en `tests/test_fase84.py`).
- Mutación: 42 mutantes, todos atrapados salvo 2 equivalentes (guardas de NaN defensivas).
- Qué regla levantó cada evento (columna `trigger_source`):

  | | puntaje | H | A |
  |---|---|---|---|
  | nube | 62 | 25 | 3 |
  | Mac | 65 | 21 | 4 |

  - El conteo viejo de H (13/89) contaba solo los eventos con puntaje < 2.6 en el cuadro del disparo.
  - H levanta 25 en la nube, pero en 12 de ellos el puntaje habría disparado poco después.
  - Hay que corregirlo en `POSIBLES-CAMBIOS-DEL-DRAFT.md` (C19) y en el bloque B5.
- Lo simulado con `rep84.py` y lo real coincidieron evento por evento. El replay sirve como verificación rápida para la limpieza: segundos en vez de 20 minutos por corrida.

## 6. Pendientes, en orden

1. Push (autor): `git push origin 82-90 legacy-pre-cleanup --tags`.
2. Actualizar `POSIBLES-CAMBIOS-DEL-DRAFT.md`: las cifras de la 8.2 a la 8.4, el disparo por A con su sustento (Bourke et al. 2008; Wu 2000) y el conteo corregido de H. El paper lo escribe el autor.
3. Guardar `rep84.py` (replay) y la corrida de la nube de la 8.4 en `respaldo-nube/` (pedir permiso).
4. Congelar la configuración antes de la evaluación final y reportar entrenamiento y prueba por separado (C31).
5. **Limpieza B0–B9** (plan abajo). Se hará después; posiblemente en una sesión de Claude Code en la nube, con los créditos.
6. **Fase L** (reglas de A y de etapa 3 en vivo), si el paper va a reportar el sistema en vivo.
7. Raspberry Pi 4B: dependencias sin PySide6, cifras oficiales, y rendimiento (FPS, CPU, memoria, latencia) en modo despliegue.

## 7. Plan de limpieza (aprobado como plan; no empezar sin confirmación)

- Una rama nueva por bloque, cada una desde la anterior.
- Cada bloque tiene que reproducir **exacto** (etiquetas, eventos y valores por cuadro) la referencia del código de ese momento.
- La referencia hoy es la 8.4: nube `PEF-84/logs` y Mac `logs/cli-a84`. El replay (`rep84.py`) la verifica en segundos sobre los 33 puntos, sin MediaPipe.

Decisiones de las 36 preguntas:

| Bloque | Qué |
|---|---|
| B0 | Scripts de análisis a `tools/` en inglés, un comparador de corridas de 128 clips, el commit de git y la versión en cada `meta.json`, CHANGELOG |
| B1 | Borrar la banda provisional (y `test_zona_gris`) y el rechazo por descenso (y `test_descent`); `sequential` y `simultaneous` a un módulo de ablaciones; `v_only` se queda |
| B2 | `config.yaml` en inglés, corregido y ordenado (pose → logging → stage1 → stage2 → stage3 → A → alerts → display); validación que falla ante claves faltantes o desconocidas; una sola fuente de valores; `config/partition.yaml`; modo investigación/despliegue con los 33 puntos apagados por defecto |
| B3 | `main.py --folder`, `--output-dir`, `--landmarks/--no-landmarks`, `--quiet`; un solo escritor de registros para la terminal y la GUI (la fila describe el evento que decidió la etiqueta; la GUI también guarda los eventos de Etapa 1 y los 33 puntos); columnas CSV en inglés sin compatibilidad con las viejas |
| B4 | Dividir `state_machine` (trigger, stage2, stage3, funnel, verdicts); `quantities/` con un archivo por cantidad y listas "paper"/"experimental" (promover A = cambiarla de lista); adelgazar `pipeline` (Timings, HUD, fábrica de la máquina, dataclass por cuadro hacia `FallStateMachine.update`) |
| B5 | Todo el código en inglés, incluidos identificadores; la historia de los docstrings a `notas/`; motivos de eventos en inglés; H documentada (sigue encendida y levanta 25/90 eventos en la nube y 21/90 en el Mac) |
| B6 | GUI dividida en paneles (widgets, curves, queue_panel); pruebas de la GUI más rápidas |
| B7 | `tests/` como paquete con `_helpers.py`; nombres por comportamiento en inglés; pruebas de punta a punta con el `config.yaml` real (hoy la config de prueba difiere: sequential 2.2, sin ventana de pico, sin H ni A); `simular_caida` convertido en prueba |
| B8 | ruff con línea de 100; deque; NaN con `math.isnan`; OSError; `with` y try/finally; logging. La mediana se deja exactamente igual (el cálculo actual toma el valor superior). |
| B9 | `pyproject.toml` (`pef-lab`, `pef-run`); dependencias núcleo/GUI/desarrollo + lock generado en el Mac; `.gitignore` completo; `notas/archivo/` + `notas/README.md`; archivar `replay_disparador` y `verificar_mediciones`; README reescrito en inglés |

Al terminar, actualizar `LEGACY.md` en `legacy-pre-cleanup` con lo realmente hecho.

Claude Code en la nube puede hacer el código y las pruebas unitarias, pero no tiene los videos. La verificación de los 128 clips la corre el autor en su Mac, contra una corrida de referencia hecha antes de empezar.

## 8. Cómo correr y verificar

- Pruebas: `python -m unittest discover -s tests`. Hoy son 594 y 1 omitida.
- GUI: `python main.py`.
- Lote en el Mac:
  `sed 's#output_dir: "logs"#output_dir: "logs/X"#' config.yaml > /tmp/c.yaml; for v in ../recordings/*.mp4; do python main.py --video "$v" --headless --config /tmp/c.yaml > /dev/null 2>&1; done`
- Métricas: `python3 respaldo-nube/scripts/verif2.py <repo> <logs> [<logs_b>]` (accuracy total y por partición, sensibilidad, falsos positivos, alarma, eventos disparados por H, concordancia).
- Diagnóstico de A: `diag81.py`. Simulación de reglas con A: `sim81.py`.
- Replay sin MediaPipe: `python3 rep84.py <repo> <logs>`.
  - Reconstruye cada cuadro desde el CSV de 33 puntos y pasa la corrida por el código del repo.
  - Con `--a ALTO BAJO VENTANA --vneed V` simula el disparo por A sobre código viejo.
  - Hoy vive en la nube del chat y en la VM (`~/ana/`).

## 9. Notas vigentes en `notas/`

- `ANALISIS-82-90.md`: el diagnóstico y la propuesta de A.
- `NOFALL.md`: por qué los falsos positivos son los ADL de estrés del §3.3.
- `POSIBLES-CAMBIOS-DEL-DRAFT.md`: 36 cambios al paper, con dos repasadas.
- `HALLAZGO-B05.md` y `CONVENCION-NOMBRES.md`.
- Las demás son históricas.
