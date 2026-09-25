# POSIBLES-CAMBIOS-DEL-DRAFT.md — qué se puede o se debería cambiar del draft 4

25/09/2026 · rama `82-90` (fases 0–6 + 8.1) · referencia: `4-paper-draft.docx` (draft 4).

Este documento **no escribe en el draft**. Lista cada cambio posible: dónde va,
por qué y con qué evidencia. El texto final lo escribe el autor. Las frases
propuestas van en inglés porque el draft está en inglés; son un punto de
partida, no un texto final.

## Cómo leer cada punto

- **Dice**: la cita literal del draft 4, con su línea en el texto plano del
  documento (`draft4.txt`).
- **Qué pasa**: lo que hace el código o lo que se midió.
- **Propuesta** y **por qué**.
- **Prioridad**:
  - **DEBE**: el draft afirma algo que los datos o el código contradicen.
  - **DEBERÍA**: el draft calla algo que hace falta para reproducir los resultados.
  - **PUEDE**: mejora la claridad o la fuerza del argumento.
  - **CONDICIONAL**: depende de una decisión pendiente (por ejemplo, adoptar A).
- **Verificación**, dos repasadas por punto:
  - **Repaso 1** (al redactar): la cita se buscó palabra por palabra en el
    draft y el dato se comparó con su fuente.
  - **Repaso 2** (relectura completa, 25/09):
    - cada cifra se **recalculó** desde los registros con un script
      (`verif2.py`, `diag81.py`), en las dos plataformas;
    - el código se releyó;
    - cada cita se releyó en su párrafo completo.

    Lo que este repaso cambió está marcado **Corregido** y resumido al final.
- Plataformas:
  - nube = x86 (`PEF-a`);
  - Mac = ARM (`logs/cli-a81`).

  Las dos son de la fase 8.1, que no cambia decisiones, así que las cifras son
  las de la fase 6.

---

## §1 Introducción y §3.1 Visión general

### C1 · "Cuatro cantidades", pero deciden más · **DEBE**
- **Dice** (l. 87): *"derives four closed-form physical quantities from the resulting 33-landmark skeleton"*; (l. 155) *"Each of the four named quantities…"*; (l. 169) *"a downstream reviewer can reconstruct the decision by reading the four quantities"*.
- **Qué pasa**: además de T, V, P e I, hoy deciden etiquetas:
  1. la extensión cadera-tobillo (leve vs moderada, `severity_uses_leg_extension: true`; ver C18);
  2. la cantidad H, que es experimental: dispara 13 de 89 eventos en la nube y 11 de 88 en el Mac (ver C19);
  3. la subida de la cadera al final (fase 6a, `getup_rise_torsos: 0.45`);
  4. y A, si se adopta (C20).
- **Propuesta**: nombrar en el §3.4 cada medida que decide, aunque sea auxiliar, con su papel. O apagar las que no se quieran defender (H) y volver a medir.
- **Por qué**: la afirmación central del paper es que cualquier decisión se reconstruye con cantidades nombradas. Hoy un revisor no podría reconstruir, por ejemplo, los disparos por H.
- **Verificación**:
  - Repaso 1: citas en l. 87, 155 y 169; `config.yaml`.
  - Repaso 2: disparos por H recalculados con el puntaje del cuadro de disparo (columna `trigger_score`, que es el puntaje sobre la ventana de pico). Con `trigger_score_provisional: 0.0`, solo el respaldo de H puede disparar con un puntaje menor a 2.6. **Corregido**: se agregó la cifra del Mac.

### C2 · "No exhibe sobreajuste" · **DEBE**
- **Dice** (l. 89): *"the system does not exhibit the overfitting seen in pipelines that report near-perfect metrics on proprietary corpora"*.
- **Qué pasa**:
  - todos los umbrales se ajustaron midiendo sobre los 128 clips, incluidos los actores de prueba (disparo 2.6, bandas 30/60, persistencia 50 %, subida de cadera 0.45, re-adquisición 0.5 s…);
  - con la partición provisional por actor:

    | | entrenamiento | prueba |
    |---|---|---|
    | Mac | 72/88 = 81.8 % | 30/40 = 75.0 % |
    | nube | 73/88 = 83.0 % | 33/40 = 82.5 % |

  - con solo 40 clips de prueba, el intervalo de Wilson al 95 % de 75.0 % va de 59.8 % a 85.8 %. El de 82.5 % va de 68.0 % a 91.3 %.
- **Propuesta**: bajar el tono, por ejemplo *"is designed to limit overfitting: thresholds are few, physically interpretable, and evaluated on held-out subjects (§3.7)"*. Y dejar que el §4 muestre entrenamiento y prueba por separado, con sus intervalos.
- **Por qué**: los datos no permiten afirmar ni negar el sobreajuste.
  - La diferencia entre entrenamiento y prueba es de 6.8 puntos en el Mac y de 0.5 en la nube, y las dos caben dentro de los intervalos.
  - Además, el propio §1 (l. 7) critica la evaluación *"on large in-house datasets curated by the same authors who defined the detection logic"*, y PEF-FallDB es exactamente eso (ver C31).
- **Verificación**:
  - Repaso 1: cita en l. 89; `notas/particion.yaml`.
  - Repaso 2: accuracies recalculadas por partición en las dos plataformas; intervalos calculados con `evaluation.wilson`; l. 7 releída. **Corregido**: la primera versión decía que las cifras mostraban sobreajuste. Lo correcto es que no permiten afirmar que no lo hay.

### C3 · Explicable ≠ correcto · **PUEDE**
- **Dice** (l. 89): *"a flagged event can be reviewed by reading off a trunk angle, a centroid velocity, a center-of-mass projection offset, and an immobility duration"*.
- **Qué pasa**:
  - cada evento se registra con su **motivo**, la regla que decidió con sus números (`EVENT_FIELDS` incluye `motivo`). La alerta lleva el mismo motivo (`alerts.py`, línea "why:");
  - pero cuando el esqueleto es infiel, la explicación puede ser coherente y a la vez falsa. En A14-S4, en las dos plataformas, el sistema escribe *"T final 28 deg < 30 con piernas extendidas: se levanto"* (Recovered), y la verdad es NotRecovered;
  - y MediaPipe no avisa:
    - en los 4 clips NotRecovered donde T terminó leyendo "de pie" con la persona en el suelo, la confianza del último segundo y medio fue de 0.998 a 1.000, igual que en los 19 bien leídos (mediana 0.998);
    - en el cuadro exacto en que el esqueleto de A09-S3 se invierte, la confianza sube de 0.957 a 0.961.
- **Propuesta**:
  - en el §3.5, decir que la alerta y el registro llevan la regla y los valores que la dispararon;
  - en la Discusión: *"explainability makes a decision auditable, not necessarily correct: an unfaithful skeleton yields internally consistent but false explanations."*
- **Verificación**:
  - Repaso 1: l. 89 y l. 169; `notas/ANALISIS-VIDEOS.md` (sección A14 y "Y MediaPipe no avisa").
  - Repaso 2: motivo de A14-S4 leído en el registro de eventos de las dos corridas actuales; `audit_log.py` (EVENT_FIELDS) y `alerts.py`. **Corregido**: el 27.8° que decía la primera versión venía de una corrida anterior; la actual dice 28°.

---

## §3.2 Front-end de pose

### C4 · Re-adquisición del rastreador (fase 4) · **DEBERÍA**
- **Dice**: nada; no hay mención de reinicio ni re-detección (búsqueda de "reacquir", "re-seed" y "tracker": sin resultados).
- **Qué pasa**:
  - si la visibilidad mínima de hombros y caderas queda bajo el umbral (0.5) durante 0.5 s mientras MediaPipe sigue reportando a la persona, se reinicia el grafo para que vuelva a detectar;
  - caso que lo motivó, A16-S1 antes de la fase 4: 5.3 s seguidos (de 2.3 a 7.6 s) con la persona "detectada" pero con visibilidad mediana 0.02, y el 82 % de los cuadros por debajo de 0.05. Mientras tanto, el sujeto caía y quedaba en el piso a la vista;
  - con la fase 4, ese tramo se corta a los 0.5 s.
- **Propuesta**: *"When the core landmarks remain below the visibility threshold for longer than a configurable time while the tracker keeps reporting a detection, the pose graph is reset so the detector searches for the subject again."*
- **Verificación**:
  - Repaso 1: búsqueda sin resultados en el draft; `config.yaml` (`pose.reacquire_after_s: 0.5`) y `pose_frontend.py` (`_watch_tracker`).
  - Repaso 2: el tramo se recalculó en los CSV de cuadros de A16-S1 antes (`PEF-f23`) y después (`PEF-a`) de la fase 4, y se revisaron los cuadros del video. **Corregido**: decía "5.4 s con visibilidad 0.00–0.05"; son 5.3 s, con mediana 0.02 y máximo 0.49.

### C5 · Exclusión de la profundidad monocular vs cantidad A · **CONDICIONAL (DEBE si se adopta A)**
- **Dice** (l. 95): *"the monocular z-coordinate is likewise excluded, because its estimated depth is not reliable as metric geometry"*. La l. 157 lo repite en la definición de T.
- **Qué pasa**:
  - A usa las coordenadas **de mundo** de MediaPipe (otra salida del mismo modelo, en metros), no la z de la imagen;
  - y solo como **cociente** contra la altura de pie del mismo sujeto, con el "arriba" calibrado por el propio sujeto;
  - fase 8.1: A calibra en 116/128 clips en la nube y 117/128 en el Mac;
  - en los 106 clips calibrados en ambas, el A final difiere entre plataformas en una mediana de 0.003 (p90 0.04);
  - en vivo contra el análisis fuera de línea, la diferencia mediana es de 0.001 o menos en las dos;
  - estimación de 8.2 + 8.3 con la calibración causal: unos +8 clips (nube ~114/128, Mac ~110–111/128), por confirmar.
- **Historia**: el 08/09, `DENTRO-FUERA-DRAFT4.md` §4.1 descartó usar el 3D para T porque compraba 4 clips a cambio de contradecir el §3.2. A es otra medida (no es T), con otro costo-beneficio, y el autor dio permiso explícito para proponer cambios.
- **Propuesta** si se adopta:
  - reescribir la frase para distinguir la z de la imagen, que sigue excluida, de las coordenadas de mundo, que se usan **solo** en forma de cociente autocalibrado;
  - citar la concordancia entre plataformas.
- **Verificación**:
  - Repaso 1: l. 95 y l. 157; `notas/ANALISIS-82-90.md`.
  - Repaso 2: `diag81.py` sobre `PEF-a` (nube) y `logs/cli-a81` (Mac), y comparación clip por clip entre ambas. **Corregido**: la estimación de "+10 en las dos plataformas" usaba la calibración fuera de línea. Con la causal de 8.1 baja a unos +8.

### C6 · La mitigación por "razones de desplazamiento" existe y conviene describirla · **DEBERÍA**
- **Dice** (l. 95): *"depth ambiguity from the monocular input is mitigated by tracking within-frame and across-frame landmark displacement ratios rather than relying on absolute 3D distances"*.
- **Qué pasa**: T pasa a NaN cuando el tronco proyectado cae por debajo del 15 % de su mediana en el último segundo (`min_trunk_ratio: 0.15`, `trunk_reference_window_s: 1.0`). Es la versión concreta de esa frase: antes de este mecanismo, el tronco de A14 pasó de 209 px a 2 px en 0.2 s.
- **Propuesta**: nombrar el mecanismo con su parámetro. También reemplaza la guarda de 0.001 px (C13).
- **Verificación**:
  - Repaso 1: l. 95; `pipeline.py` (`_trunk_is_plausible`) y `config.yaml`.
  - Repaso 2: `_trunk_is_plausible` releído (compara con la mediana reciente del mismo sujeto; si falla, `t_deg = NaN`); el dato de A14 está en `TRASPASO-DRAFT.md` l. 394 y en el docstring del código.

### C7 · El modelo "full" por defecto está respaldado por medición · **PUEDE**
- **Dice** (l. 97): *"…with the heavy variant reserved for low-light or multi-person scenarios where landmark drift is observed in practice"*.
- **Qué pasa**:
  - lite: 103/128, con 66/72 caídas detectadas (91.7 %), contra 106/128 y 69/72 (95.8 %) con full (nube);
  - heavy: 73.4 % contra 78.1 % con full, los dos con disparo 2.6 (medido el 11/09 con la configuración de entonces).
- **Propuesta**: una frase con esa evidencia, que refuerza la elección que el draft ya hace.
- **Verificación**:
  - Repaso 1: l. 97; `notas/ANALISIS-VIDEOS.md` ("El modelo pesado, en detalle").
  - Repaso 2: lite (`PEF-c0`) y full (`PEF-a`) recalculados desde los registros; tabla del pesado releída.

### C8 · Las cifras dependen de la plataforma · **DEBE**
- **Dice** (l. 173): *"The framework is deployed on a Raspberry Pi 4B (or an equivalent single-board computer)"*. Sobre la variabilidad entre equipos no dice nada.
- **Qué pasa**:
  - con código, configuración y versiones idénticas (mediapipe 0.10.14, OpenCV 4.13.0.92, numpy 2.4.4), x86 y ARM dan esqueletos distintos;
  - las etiquetas coinciden en 120/128 clips (117/128 antes de la fase 4);
  - de los 8 clips distintos, en 5 el Mac pierde una caída que la nube detecta: 64/72 caídas detectadas en el Mac contra 69/72 en la nube;
  - es una diferencia de inferencia, no de la interfaz. En el Mac, la GUI (antes de la fase 4) y la terminal (fase 4) dan cuadros idénticos en 109/128 clips, y los otros 19 divergen justo donde actúa la fase 4: 12 re-adquisiciones y 7 por quitar la espera tras un rechazo;
  - en la misma plataforma el resultado se repite: la 8.1 en el Mac reproduce la fase 6 con columnas comunes idénticas en 128/128 clips.
- **Propuesta**:
  - declarar en §4 la plataforma exacta de las cifras (el Pi 4B);
  - reportar la concordancia entre plataformas como resultado de reproducibilidad;
  - no afirmar mejoras menores que esa variación.
- **Verificación**:
  - Repaso 1: l. 173.
  - Repaso 2: etiquetas recalculadas (Mac `cli-a81` contra nube `PEF-a`: 120; GUI del Mac contra nube `PEF-f23`: 117). Cuadros de la GUI contra la terminal comparados columna por columna, con la causa de cada diferencia. **Corregido**: la primera versión decía "73/76 idénticos"; no se pudo reproducir y se reemplazó por la comparación completa de los 128.

---

## §3.3 Dataset y §3.6 Privacidad

### C9 · El detector escribe coordenadas y cantidades a disco · **DEBE (código)**
- **Dice** (l. 151): *"No landmark data is stored or distributed alongside the videos."*, y sigue: *"the deployed detector persists nothing"*; (l. 173) *"because no frames, landmark coordinates, or derived quantity values are written to persistent storage"*.
- **Qué pasa**:
  - el texto ya separa el dataset (se publican los videos) del detector desplegado (no guarda nada);
  - pero el software tiene un solo modo, y cada corrida escribe:
    - el CSV por cuadro, con coordenadas de cadera y hombros en píxeles y todas las cantidades derivadas;
    - el CSV de eventos;
    - con `save_landmarks: true`, además, los 33 puntos;
  - la sección `logging` de `config.yaml` no tiene forma de apagar el CSV por cuadro.
- **Propuesta**:
  - (a) código: un modo despliegue que no escriba nada a disco. La alerta ya lleva severidad, cantidades y motivo, así que la explicación sale con ella y no queda guardada en el dispositivo;
  - (b) texto: una frase que diga que las corridas del §4 usan el modo investigación, que registra por cuadro, y que esos registros no se publican con el dataset;
  - (c) las cifras de rendimiento del Pi (C30) deben medirse en modo despliegue, porque escribir un CSV por cuadro gasta E/S y CPU que el dispositivo desplegado no gastaría.
- **Por qué**: hoy el §3.6 afirma algo que el software, tal como corre, no cumple.
- **Verificación**:
  - Repaso 1: l. 151 y l. 173; `audit_log.py` (PHASE1_FIELDS con `mid_hip_x_px`…, PHASE2_FIELDS con T, V, P, I…).
  - Repaso 2: l. 151 releída entera; `main.py` (el CSV por cuadro se crea siempre, el de puntos solo con `save_landmarks`); `alerts.py` (la alerta lleva el motivo). **Corregido**: la primera versión proponía separar investigación de despliegue en el texto, pero el draft ya lo hace. Lo que falta es el código.

### C10 · Definición operativa de las tres severidades · **DEBERÍA**
- **Dice** (l. 145): *"…or remained on the ground (severe severity)"*; (l. 149) *"when the subject remains on the ground or fails to recover during the observation window"*. Pero el mismo §3.3 cuenta como recuperación (l. 111) *"recovering by sitting on the floor"* y (l. 117) *"recovering by kneeling"*.
- **Qué pasa**:
  - A17-S2 y A17-S4 quedan sentados en el piso, recostados contra la pared, y el dataset los etiqueta NotRecovered, confirmado por el autor;
  - "en el suelo" no los separa de "recuperarse sentándose en el piso";
  - lo que los separa es que no se mueven. T los lee erguidos (T final 3° y 25° en la nube: la espalda contra la pared mantiene el tronco vertical), y A los lee en la banda media (0.51 y 0.46);
  - el sistema llega a severe solo por la regla de persistencia (fase 3: 63 % y 58 % de la Etapa 3 quietos, nube). En el Mac, A17-S4 ni siquiera dispara.
- **Propuesta**: definir la severidad por la **recuperación de una postura sostenida por el propio sujeto**, no solo por "estar en el suelo":
  - severe: no recupera una postura activa (tumbado, o desplomado e inmóvil);
  - moderate: se incorpora a sentado o de rodillas;
  - mild: de pie sin ayuda.
- **Verificación**:
  - Repaso 1: l. 145 y l. 149; `config.yaml` (persistencia 0.5 / 50 %).
  - Repaso 2: l. 111 y l. 117 agregadas como el texto que choca; motivos, T y A finales de los cuatro A17 recalculados. A17-S1 y A17-S3 no son este caso: terminan con T de 61° y 73° y son severe por T. **Corregido**: decía que todo A17 se lee como postura media; son solo S2 y S4.

### C11 · Nombres y anotación del dataset · **ACCIÓN, no cambio de texto**
- Ya resuelto por plan (TRASPASO D9–D11): el dataset se renombra según `CONVENCION-NOMBRES.md` y el §3.3 no cambia. Se lista aquí para que no se reabra.
- **Verificación**:
  - Repaso 1: `TRASPASO-DRAFT.md` D9, D10 y D11, marcadas "RESUELTA POR PLAN".
  - Repaso 2: las rutas de origen en los `.meta.json` de la corrida actual siguen siendo `recordings/A14-S4-NotRecovered.mp4`, etc. El renombrado sigue pendiente.

---

## §3.4 Cantidades

### C12 · T: dos papeles, dos umbrales · **DEBERÍA**
- **Dice** (l. 157): *"a fall action produces a transition to T of 60° or greater within a short time window"*.
- **Qué pasa**: el 45° normaliza a T dentro del puntaje de disparo (T/45 + V/(−1.5) ≥ 2.6), y el 60° es la frontera de "tumbado" en la Etapa 3.
- **Propuesta**: decir explícitamente los dos papeles. Hoy el lector cree que el disparo exige T ≥ 60°.
- **Verificación**:
  - Repaso 1: l. 157; `config.yaml` (`threshold_T_deg: 45.0`, `state_display.lying_T_deg: 60.0`, `trigger_score: 2.6`).
  - Repaso 2: `pipeline.py` (`lying_t_deg=float(cfg.state_display.lying_T_deg)`, usado por la Etapa 3).

### C13 · Guarda de tronco degenerado: 0.001 px no protege nada · **DEBERÍA**
- **Dice** (l. 157): *"When |trunk| falls below 0.001 px…"*.
- **Qué pasa**: un tronco de 2 px pasa esa guarda. La que protege de verdad es la relativa de C6. El código tiene las dos.
- **Propuesta**: la frase de TRASPASO D7 (umbral relativo al torso reciente), unida a C6.
- **Verificación**:
  - Repaso 1: l. 157; `quantities.py` (`MIN_TRUNK_PX = 1e-3`).
  - Repaso 2: `pipeline.py` (`_trunk_is_plausible`), cuyo docstring registra el caso medido: tronco de 2 px con confianza 0.99.

### C14 · "Sentarse = V monótona y lenta": falso para el sentarse del propio dataset · **DEBE**
- **Dice** (l. 159): *"a sit-down event produces a monotonic negative V over a longer window, and a fall produces a large-magnitude negative V within a short window"*.
- **Qué pasa**:
  - el §3.3 incluye a propósito (l. 123) *"Sitting down with rapid deceleration on a chair or sofa"*;
  - B05 (sofá), con nube y Mac casi idénticos: V mínima de −2.5 a −3.1 torsos/s, y de 0.30 a 0.43 s por debajo de −1;
  - caídas (69 eventos en Etapa 3): V mínima mediana −2.6 en la nube y −2.8 en el Mac, y 0.57 s por debajo de −1 (mediana);
  - el sofá cae dentro del rango de las caídas.
- **Propuesta**: *"a slow sit-down produces a monotonic negative V over a longer window; a sit-down with rapid deceleration, included in PEF-FallDB as a stress test, produces a descent of comparable magnitude to a fall and is not separable by V alone."*
- **Verificación**:
  - Repaso 1: l. 159 y l. 123; tabla de `notas/NOFALL.md`.
  - Repaso 2: `NOFALL.md` releído contra la frase. **Corregido**: decía "son iguales en las dos plataformas" junto a dos medianas distintas. Lo casi idéntico entre plataformas es B05, no la mediana de las caídas.

### C15 · Premisa de la Etapa 2: al sentarse, el COM no queda dentro del apoyo · **DEBE**
- **Dice** (l. 167): *"…rather than a controlled sit-down in which COM stays within the support polygon"*.
- **Qué pasa**: al sentarse en el sofá, el peso pasa al mueble. En B05-S1, S2 y S3 el COM está fuera del polígono de los pies en el 100 % de las muestras de la Etapa 2 (11 de 11), en las dos plataformas. `TRASPASO-DRAFT.md` lo llama D16.
- **Propuesta**:
  - acotar lo que P puede separar: caídas contra acciones en las que el peso sigue sobre los pies;
  - declarar que sentarse sobre un mueble saca el COM del polígono;
  - y, si se adopta A, que la separación viene de "¿llegó al suelo?" (U2).
- **Verificación**:
  - Repaso 1: l. 167; `TRASPASO-DRAFT.md` ("Intento fallido — criterio de descenso controlado").
  - Repaso 2: `p_outside_fraction = 1.0` con `p_samples = 11` leído en los registros de eventos de B05-S1/S2/S3 en la nube y en el Mac. **Corregido**: la primera versión citaba una columna P de `NOFALL.md` que esa nota no tiene.

### C16 · I como umbral de confirmación: casi nunca se alcanza · **DEBE**
- **Dice** (l. 163): *"If I exceeds a confirmation threshold W, the event is classified as a confirmed fall and an alert with the appropriate severity tag is dispatched."*
- **Qué pasa**:
  - solo 2 de 89 eventos llegan a I ≥ W = 5 s en la nube (B05-S3 6.1 s y A17-S4 5.1 s), y 1 de 88 en el Mac (B05-S3 5.3 s);
  - la severidad sale de la postura al final;
  - I entra a la decisión por la regla de persistencia (fase 3): severe si al menos el 50 % de los cuadros de la Etapa 3 están quietos (I ≥ 0.5 s), cuando la postura final no es ni tumbada (T ≥ 60°) ni de pie con piernas extendidas. Es decir: sentado o de rodillas (30–60°), o tronco vertical sin piernas verificables, que es el caso de A17-S2 y A17-S4.
- **Propuesta**: la salida (a) de TRASPASO D6, I como evidencia de apoyo para la severidad severa y no como umbral por sí sola, más una frase sobre la persistencia.
- **Verificación**:
  - Repaso 1: l. 163; `config.yaml` (`threshold_W_seconds: 5.0`, `persistent_still_s: 0.5`, `persistent_immobility_fraction: 0.5`).
  - Repaso 2: `max_immobility_s` de todos los eventos recalculado en las dos plataformas; `Stage3Evaluator.finalise` y `_moderate_or_persistent` releídos. **Corregido**: decía que la persistencia solo actúa en la postura media, pero también actúa con T < 30° sin piernas verificables. Se agregó el Mac.

### C17 · Cómo se mide I en la práctica · **DEBERÍA**
- **Dice** (l. 163): *"…remain within a small radius ε around their average post-trigger position"*.
- **Qué pasa**: I corre en continuo, sin esperar al disparo, y ancla ε a la posición media del episodio de quietud actual, no al promedio desde el disparo. El promedio desde el disparo mezclaría la cola de la caída con la quietud.
- **Propuesta**: la frase que pide el docstring de `ImmobilityTimer`, anclando al episodio de quietud.
- **Verificación**:
  - Repaso 1: l. 163; `quantities.py` (docstring de `ImmobilityTimer`: *"The draft needs one sentence changed for the two to agree."*).
  - Repaso 2: docstring releído entero: corre en cada cuadro y ancla a la media del episodio.

### C18 · La extensión cadera-tobillo decide leve vs moderada · **DEBE (si sigue en uso)**
- **Dice**: nada; "extension" o "hip-to-ankle" no aparecen en el draft.
- **Qué pasa**: con `severity_uses_leg_extension: true`, la etiqueta leve exige piernas extendidas: en el último segundo, algún cuadro con la extensión entre 1.1 y 3.0 torsos. TRASPASO D8 lo midió con la configuración del commit `5b0fd62`: 74.2 % con la extensión contra 70.3 % usando P.
- **Propuesta**: la frase de D8 para el §3.4, o reemplazar la extensión por A si se adopta. A también distingue estar de pie de estar sentado o arrodillado.
- **Verificación**:
  - Repaso 1: búsqueda sin resultados en el draft; tabla de D8 en `TRASPASO-DRAFT.md`.
  - Repaso 2: `config.yaml` (`severity_uses_leg_extension: true`, `standing_extension: 1.1`, `max_extension_ratio: 3.0`) y `Stage3Evaluator.finalise`. **Corregido**: la fecha "03/09" no se pudo verificar y se quitó.

### C19 · La cantidad H (experimental) decide · **DEBE: documentarla o apagarla**
- **Dice**: nada; H no existe en el draft.
- **Qué pasa**: H participa en tres lugares.
  1. **Respaldo del disparo**: H ≤ 0.35 con V < −1.5, si H ≥ 0.85 en los 2 s previos.
     - Nube, 13 de 89 eventos:
       - 8 caídas bien etiquetadas: A02-S3, A02-S4, A07-S1, A08-S1, A12-S2, A12-S3, A12-S4 y A17-S2;
       - 1 caída detectada con severidad errada: A09-S2;
       - 3 falsos positivos: B05-S2, B05-S3 y B14-S2;
       - 1 evento que la Etapa 2 rechazó bien: B12-S2.
     - Mac, 11 de 88 eventos: 7 caídas bien etiquetadas, 2 falsos positivos (B05-S2 y B05-S3) y 2 rechazos de la Etapa 2 (B07-S2 y B12-S2).
  2. "Tumbado" cuando T no está medida.
  3. El chequeo de "erguido" en modo en vivo (H ≥ 0.75).

  En la evaluación (modo etiquetado) solo el 1 cambia cifras. El 2 y el 3 actúan en vivo.
- **Propuesta**:
  - (a) documentarla en el §3.4 y el §3.5, con sus límites; o
  - (b) apagarla (`trigger_H_ratio: 0`) y volver a medir. Algunas de esas caídas quizá disparen más tarde por el puntaje, así que la pérdida real no se conoce hasta medirla;
  - si se adopta A, A reemplazaría a H con una base más firme (H se envenena en B05; ver `HALLAZGO-B05.md`).
- **Verificación**:
  - Repaso 1: búsqueda de "height ratio" y "Quantity H" sin resultados; `config.yaml` (`experimental.trigger_H_ratio: 0.35`, `trigger_H_erect: 0.85`, `recovery_H_ratio: 0.75`).
  - Repaso 2: los eventos, recalculados con `trigger_score` en las dos plataformas y leídos uno por uno; `state_machine.py` (`_is_lying`: T decide cuando está medida; el chequeo de erguido con H solo pesa fuera del modo etiquetado). **Corregido**: la primera versión dejaba sin explicar 2 de los 13 eventos (A09-S2 y B12-S2) y no tenía el Mac.

### C20 · Nueva cantidad A · **CONDICIONAL**
- **Si 8.2/8.3 la confirman**:
  - (1) §3.4: definir A, su calibración y sus bandas.
    - A es la altura de la cabeza sobre los tobillos a lo largo del "arriba" calibrado con el sujeto de pie, dividida por esa misma altura de pie.
    - Las bandas son 0.26 / 0.82 / 0.28, tomadas de los huecos de la partición de entrenamiento.
  - (2) §3.5: su uso en la severidad y en "¿llegó al suelo?", con la regla de abstención: si A no estaba calibrada antes del disparo, "¿llegó al suelo?" no decide, igual que la Etapa 2 sin pies;
  - (3) Introducción, §3.1 y contribuciones: "four quantities" pasa a "five";
  - (4) §3.2: C5;
  - (5) §3.7: A en las ablaciones.
- **Por qué**: A mide directamente lo que define una caída, la pérdida de altura contra la gravedad. Se calcula en forma cerrada y se explica sola ("la cabeza quedó al 15 % de su altura de pie").
- **Verificación**:
  - Repaso 1: l. 87, 155 y 169 (conteos "four"); `ANALISIS-82-90.md`.
  - Repaso 2: corridas 8.1 en las dos plataformas. La verificación del Mac encontró que A16-S3 calibra 2.6 s más tarde que en la nube, ya con el sujeto levantado. Sin la regla de abstención, "¿llegó al suelo?" perdería esa caída en el Mac. Las cifras finales saldrán de 8.2/8.3; las de hoy son estimaciones. **Corregido**: se agregó la regla de abstención y se quitó "la cantidad más physics-informed posible", que no se puede demostrar.

---

## §3.5 Lógica de confirmación

### C21 · "Combinación instantánea" vs ventana de pico · **DEBE**
- **Dice** (l. 167): *"an event enters Stage 2 whenever an instantaneous combination of the two quantities exceeds a configurable trigger threshold while V remains negative (downward)"*.
- **Qué pasa**:
  - el puntaje usa el máximo de T y el mínimo de V dentro de los últimos 0.8 s (`trigger_peak_window_s: 0.8`);
  - y exige sostenerlo 0.1 s (`trigger_hold_s: 0.1`);
  - en una caída, T y V no alcanzan su pico en el mismo cuadro (`DENTRO-FUERA-DRAFT4.md` §2.1).
- **Propuesta**: *"…a combination of the two quantities, each taken at its extreme over a short trailing window, exceeds…"*
- **Verificación**:
  - Repaso 1: l. 167; `config.yaml` (0.8 y 0.1).
  - Repaso 2: `state_machine.py` (`Stage1Trigger._evaluate_formulation` pasa T y V por `PeakWindow` antes de combinarlos).

### C22 · La Etapa 2 se abstiene si no se ven los pies · **DEBERÍA**
- **Dice** (l. 167): la Etapa 2 *"evaluates P to ensure that the event is geometrically fall-like"*.
- **Qué pasa**:
  - sin muestras suficientes de P (pies no visibles), el veredicto es `stage2_inconclusive` y el evento **sigue** a la Etapa 3;
  - solo un rechazo explícito lo detiene.
- **Propuesta**: una frase que diga que la Etapa 2 veta cuando puede medir y se abstiene cuando no.
- **Verificación**:
  - Repaso 1: l. 167.
  - Repaso 2: `state_machine.py` (`if verdict == REJECTED or self.stage3 is None: resolve`; cualquier otro veredicto pasa a `stage3.start`).

### C23 · Cómo se asignan las tres severidades · **DEBE**
- **Dice** (l. 167): *"…as evidenced by torso and hip landmark recovery to upright, the alert's severity tag is downgraded or the alarm is nullified; if the immobility persists, the alert is dispatched with the appropriate severity tag"*.
- **Qué pasa**, en modo etiquetado:
  - se toma la mediana de T en el último segundo observado;
  - T ≥ 60° → severe, salvo que la cadera haya subido ≥ 0.45 torsos en los últimos 2 s: entonces moderate, "se está levantando" (fase 6a; es la "hip landmark recovery" del draft);
  - 30–60° → moderate, o severe si la inmovilidad persistió (≥ 50 % de la Etapa 3 quieta);
  - < 30° con piernas extendidas → mild (evento anulado, sin alarma);
  - < 30° sin piernas verificables → moderate, o severe si la inmovilidad persistió.
- **Propuesta**: declarar la regla como parámetros calibrados (TRASPASO D14), incluidas la persistencia y la subida de cadera. Si se adopta A, la postura sale de A.
- **Verificación**:
  - Repaso 1: l. 167; `config.yaml` (`upright_T_deg: 30`, `lying_T_deg: 60`, persistencia y `getup_*`).
  - Repaso 2: `Stage3Evaluator.finalise` releído rama por rama (`final_window_s = 1.0`). **Corregido**: faltaba la persistencia en la rama < 30° sin piernas verificables.

### C24 · Dos modos de resolución · **DEBE**
- **Dice**: el §3.3 (l. 145) define la severidad por cómo terminó el episodio, mientras que el §3.5 describe una resolución en línea.
- **Qué pasa**:
  - los clips grabados se etiquetan al final de la observación (`labelling=not source.is_live`);
  - la cámara en vivo resuelve en línea;
  - con resolución en línea, A13 se cerraba como severe 3 s antes de que el sujeto se levantara solo.
- **Propuesta**: la frase de TRASPASO D15. La evaluación del §3.7 usa el modo al cierre, y la latencia del §3.6 usa el modo en línea.
- **Verificación**:
  - Repaso 1: l. 145 y l. 167; `main.py` (l. 56: `FramePipeline(cfg, labelling=not source.is_live)`).
  - Repaso 2: D15 releída ("tres segundos antes", medido sobre A13).

### C25 · Pérdida de detección durante el juicio · **DEBERÍA**
- **Dice**: nada.
- **Qué pasa**:
  - un hueco de hasta 0.5 s no invalida el evento (TRASPASO D13: +11 puntos cuando se introdujo);
  - si el hueco es mayor y el sujeto se vio por última vez en el suelo, el evento se cierra con esa última postura (fase 2B); si no, se abandona.
- **Propuesta**: la frase de D13, más una sobre el cierre con la última postura vista.
- **Verificación**:
  - Repaso 1: búsqueda sin resultados; `config.yaml` (`history_max_gap_s: 0.5`).
  - Repaso 2: `pipeline.py` (tras un hueco mayor: `close_if_last_seen_down`, y si no, abandono) y `last_seen_down` (mediana de T ≥ 60° en lo último observado).

### C26 · Periodo refractario · **DEBERÍA**
- **Dice**: nada (búsqueda de "cooldown" y "refractory": sin resultados).
- **Qué pasa**:
  - después de un evento resuelto hay 3 s sin nuevos disparos;
  - después de un rechazo de la Etapa 2 no hay espera (fase 4), porque tapaba la caída real que venía detrás (A17-S3).
- **Propuesta**: una frase con ambos casos y su razón.
- **Verificación**:
  - Repaso 1: búsqueda sin resultados; `config.yaml` (`cooldown_seconds: 3.0`, `cooldown_after_rejection: false`).
  - Repaso 2: en las dos plataformas, A17-S3 tiene dos eventos: uno rechazado por la Etapa 2 a los 2.47 s y la caída confirmada a los 4.57 s.

### C27 · Alarma vs etiqueta: una caída "recuperada" no alarma · **DEBE**
- **Dice** (l. 167): *"…the alert's severity tag is downgraded or the alarm is nullified"*; (l. 145): mild = *"recovered on their own"*.
- **Qué pasa**: solo `stage3_confirmed` despacha alarma, así que un evento anulado (mild) no alarma.

  | | especificidad de etiqueta | especificidad de alarma | caídas que alarman | Recovered con alarma |
  |---|---|---|---|---|
  | nube | 44/56 (78.6 %) | 47/56 (83.9 %) | 45/72 | 1 de 24 |
  | Mac | 44/56 (78.6 %) | 48/56 (85.7 %) | 43/72 | 1 de 24 |
- **Propuesta**: el §3.7/§4 tiene que reportar **las dos** lecturas (etiqueta del §3.3 y alarma del §3.5) y declarar que "Recovered" no genera alarma por diseño.
- **Por qué**: sin esa distinción, la sensibilidad y la especificidad no significan lo mismo para un cuidador que para el dataset.
- **Verificación**:
  - Repaso 1: l. 145 y l. 167; `config.yaml` (`alerts.dispatch_verdicts: [stage3_confirmed]`).
  - Repaso 2: las cuatro columnas recalculadas en las dos plataformas.

### C28 · Despachar y degradar vs despachar una vez · **DEBERÍA**
- **Dice** (l. 167): la severidad *"is downgraded or the alarm is nullified"*, que sugiere despachar primero y corregir después.
- **Qué pasa**: el código despacha una sola vez, al resolver (TRASPASO D2).
- **Propuesta**: elegir y declararlo. Afecta la latencia del §3.6.
- **Verificación**:
  - Repaso 1: l. 167; TRASPASO D2.
  - Repaso 2: `pipeline.py`: `_dispatch` se llama solo sobre `just_resolved` y al cerrar tras un hueco (`close_if_last_seen_down`). Los dos casos son una resolución.

### C29 · Evento sin resolver = no caída · **DEBERÍA**
- **Dice** (l. 163): la confirmación depende de que I alcance W.
- **Qué pasa**: los eventos que no terminan el embudo se cuentan como NoFall, siguiendo esa frase (D12 cerrada de esa forma).
- **Propuesta**: una frase explícita.
- **Verificación**:
  - Repaso 1: l. 163.
  - Repaso 2: `classification.py` (`class_for_event(..., unresolved_as=NO_FALL)`; `classify_clip` usa ese valor por defecto).

---

## §3.6 Implementación en el borde

### C30 · Tasa de cuadros real del Pi · **DEBERÍA**
- **Dice** (l. 173): *"Sustained frame rate, CPU footprint, memory footprint, and end-to-end latency are reported for the target hardware in §4"*.
- **Qué pasa**: los umbrales se calibraron sobre clips a 30 fps. En vivo, el Pi procesará menos cuadros por segundo. Que todos los tiempos del código estén en segundos no garantiza la misma accuracy.
- **Propuesta**: agregar al §4 la accuracy a la tasa real del Pi (clips reproducidos a esa tasa), no solo sobre clips completos. Medir en modo despliegue (C9).
- **Verificación**:
  - Repaso 1: l. 173.
  - Repaso 2: `config.yaml` (todas las ventanas en segundos: `velocity_window_s`, `trigger_peak_window_s`, …); los `.meta.json` declaran 30 fps de origen.

---

## §3.7 Protocolo de evaluación

### C31 · La partición real y lo que se miró durante el desarrollo · **DEBE**
- **Dice** (l. 177): *"…by holding out settings, camera placements, and lighting conditions that were not used when the thresholds were calibrated"*.
- **Qué pasa**:
  - hoy la partición es por actor (cada tercer actor; prueba = A07, A16, A08, A17, A09, A18, B03, B06, B09, B12), hasta que exista la anotación de setting, cámara e iluminación;
  - todas las fases se midieron sobre los 128 clips;
  - además, durante el desarrollo se miraron clips de prueba: A16-S1 y A17-S3 en la fase 4, y los valores de A de A09 y A17 al explorar la cantidad A.
- **Propuesta**:
  - declarar la partición usada;
  - congelar la configuración antes de la evaluación final;
  - declarar honestamente qué se inspeccionó.
- **Por qué**: el §1 (l. 7) critica justamente la evaluación sobre datasets propios curados por los mismos autores que definieron la lógica. Declarar esto es lo que separa al paper de lo que critica.
- **Verificación**:
  - Repaso 1: l. 177; `notas/particion.yaml` (que dice que es provisional y que no debe reportarse como la partición del §3.7).
  - Repaso 2: l. 7 releída; `ANALISIS-82-90.md` §5 ("Se exploró mirando también clips de prueba").

### C32 · Métricas que faltan · **DEBERÍA**
- **Dice** (l. 181): accuracy, sensibilidad, especificidad, precisión, F1 y FP por hora.
- **Propuesta**: agregar:
  - intervalos de Wilson;
  - la plataforma;
  - las lecturas de etiqueta y de alarma (C27);
  - la concordancia entre plataformas (C8);
  - las cifras de entrenamiento y de prueba por separado.
- **Verificación**:
  - Repaso 1: l. 181.
  - Repaso 2: `evaluation.py` ya calcula Wilson (`wilson`) y FP por hora (`fp_per_hour`). **Corregido**: la primera versión citaba también `reporte.py`, que es un script de análisis de la nube y no está en el repo.

### C33 · Las líneas base "según se reportan" no son comparables · **DEBERÍA**
- **Dice** (l. 179): *"The proposed framework is compared against four representative baselines reported in the recent literature"*.
- **Qué pasa**:
  - esas cifras vienen de otros datasets (UR Fall y corpus propios);
  - el mismo draft (l. 7) advierte que el cambio de dataset degrada el desempeño;
  - ninguna línea base está implementada sobre PEF-FallDB.
- **Propuesta**: implementar al menos la línea base basada en reglas (Saraswat & Malathi, 2024) sobre PEF-FallDB, o declarar que la comparación con cifras publicadas es de contexto y no directa.
- **Verificación**:
  - Repaso 1: l. 179 y l. 7.
  - Repaso 2: l. 179 releída (las cuatro líneas base son "as reported"); el repo no tiene implementación de ninguna.

### C34 · Ablaciones acordes al sistema real · **PUEDE**
- **Dice** (l. 183): *"one quantity is removed at a time (T alone, V alone, P alone, I alone)"*.
- **Qué pasa**: T y V se combinan en un solo puntaje, e I entra como persistencia. Además deciden la extensión, H, la subida de cadera y quizá A.
- **Propuesta**: definir las ablaciones por mecanismo:
  - puntaje T+V;
  - P (Etapa 2);
  - persistencia de I;
  - cada medida auxiliar;
  - A.
- **Verificación**:
  - Repaso 1: l. 183.
  - Repaso 2: coherente con C1, C16, C18, C19 y C21 ya corregidos.

### C35 · Validación externa, solo de detección · **DEBERÍA**
- **Dice** (l. 177): *"PEF-FallDB is the only dataset on which the full framework can be assessed"*.
- **Propuesta**: agregar una evaluación **solo binaria** (caída sí/no, Etapas 1–2) sobre corpus públicos, corrida una sola vez y con la configuración congelada. Es coherente con la frase: la severidad sigue siendo exclusiva de PEF-FallDB.
- **Por qué**: es la única medida contra el ajuste al laboratorio (cámara, luz, casa, forma de actuar). Y es la respuesta directa a la crítica del propio §1 (l. 7).
- **Verificación**:
  - Repaso 1: l. 177.
  - Repaso 2: `TRASPASO-DRAFT.md` Parte 1b (hasta ~150 videos por dataset público, una sola corrida al final; la llama "la decisión metodológica más fuerte de todo el proyecto"). **Corregido**: se quitó una cifra de videos reunidos que no está en ninguna nota, y la prioridad sube de PUEDE a DEBERÍA por la l. 7.

---

## §5 Discusión (contenido sugerido)

### C36 · Limitaciones que conviene declarar · **DEBERÍA**
1. **Acostarse a propósito** (B11-S3, B11-S4, B12-S1 y B12-S4: 4 falsos positivos en las dos plataformas): la postura final es idéntica a una caída. Tres vías medidas y negativas (`HALLAZGO-B05.md`, `NOFALL.md`).
2. **Caídas hacia atrás hasta quedar sentado, con el tronco vertical** (A13-S4 y A3-S2, que es A13-S2): T máxima de 23° y 34°, puntaje máximo de 1.5 y 2.3; ninguno llega a 2.6. La cabeza sí baja al 17 % y al 9 % de su altura de pie (A).
3. **Caídas en el eje de la cámara** (A14): T se escorza. A no las rescata tal como está: con la calibración causal, A14-S2 no calibra en ninguna plataforma, y A14-S4 solo calibra en el Mac.
4. **Esqueletos infieles** (A09): el esqueleto se invierte (A09-S3: 163° en 33 ms, con la confianza subiendo) o pone de pie a alguien arrodillado (A09-S1/S2). Dan explicaciones coherentes y falsas (C3).
5. **Variabilidad entre plataformas** (C8).
6. **Sujetos que nunca aparecen de pie**, o que aparecen de pie menos de 0.5 s: no hay calibración de A.
   - En las dos plataformas: A02-S1/S3/S4, A03 ×4, A09-S1/S3, A14-S2 y B12-S4.
   - Solo en la nube: A14-S4.
- **Verificación**:
  - Repaso 1: cada punto con su clip y su nota.
  - Repaso 2: todas las cifras recalculadas:
    - falsos positivos y caídas perdidas en las dos plataformas;
    - T, puntaje y A extremos de A13-S4 y A3-S2 (y los cuadros de A13-S4);
    - clips sin calibrar con `diag81.py`: 12 en la nube y 11 en el Mac, donde A14-S4 sí calibra.

    **Corregido**:
    - se agregó A3-S2 al punto 2;
    - el punto 3 decía que A mitigaba A14;
    - el punto 4 nombraba A06, pero `ANALISIS-VIDEOS.md` atribuye A06 a la lógica (T no ve una recuperación que la cadera sí muestra, lo que resolvió la fase 6a), no al esqueleto.

---

## Resumen por prioridad

| prioridad | puntos |
|---|---|
| DEBE | C1, C2, C8, C9, C14, C15, C16, C18, C19, C21, C23, C24, C27, C31 |
| DEBERÍA | C4, C6, C10, C12, C13, C17, C22, C25, C26, C28, C29, C30, C32, C33, C35, C36 |
| PUEDE | C3, C7, C34 |
| CONDICIONAL (A) | C5, C20 |
| ACCIÓN, no texto | C11 |

## Lo que corrigió la 2.ª repasada

Dieciséis correcciones. Ninguna cambia la dirección de un punto; casi todas
precisan una cifra o su fuente.

1. **C2**: las cifras no "muestran" sobreajuste; no permiten afirmar que no lo hay (intervalos, 6.8 vs 0.5 puntos). Se agrega la crítica de la l. 7.
2. **C3**: A14-S4 dice 28° en la corrida actual (27.8° era de una corrida anterior). Se agregan la confianza de MediaPipe y el motivo en la alerta.
3. **C4**: 5.3 s con visibilidad mediana 0.02, no "5.4 s, 0.00–0.05".
4. **C5**: la estimación baja de +10 a unos +8 con la calibración causal. Se agregan la cobertura del Mac y la concordancia entre plataformas.
5. **C8**: el "73/76" no se pudo reproducir. Se reemplaza por la comparación completa: 109/128 idénticos, y las 19 diferencias explicadas por la fase 4.
6. **C9**: el draft ya separa dataset y detector (l. 151). La contradicción es del código.
7. **C10**: el caso son A17-S2 y A17-S4, no todo A17. Se agregan las l. 111 y 117 como el texto que choca.
8. **C14**: la frase "iguales en las dos plataformas" estaba junto a dos medianas distintas.
9. **C15**: la fuente citada (columna P de `NOFALL.md`) no existe. Se reemplaza por los registros de eventos.
10. **C16 y C23**: la persistencia también actúa con T < 30° sin piernas verificables. Se agrega el Mac (1 de 88 eventos llega a W).
11. **C18**: se quita una fecha que no se pudo verificar.
12. **C19**: los 13 eventos quedan todos explicados. Se agregan el Mac y que solo el disparo pesa en la evaluación.
13. **C20**: regla de abstención para "¿llegó al suelo?" (A16-S3 en el Mac). Se quita una afirmación que no se puede demostrar.
14. **C32**: `reporte.py` no está en el repo.
15. **C35**: se quita una cifra sin fuente. Sube a DEBERÍA.
16. **C36**: se agregan A3-S2 y los datos de A13-S4. A no rescata A14-S2 ni A14-S4. A06 no era un esqueleto infiel.
