# FASES.md — Plan de fases del PEF-Fall-Detector (guía de estudio)

Este documento acompaña el desarrollo del detector fase por fase. Está pensado
para leerse junto al código: cada fase dice **qué** se construye, **por qué**
(el concepto detrás), **qué archivos** toca, y **cómo se comprueba** que quedó
bien antes de avanzar. El idioma del código y sus docstrings es inglés (el
paper se publica en inglés); este documento es en español porque es la bitácora
de trabajo del proyecto.

**Regla de oro del proyecto:** no se avanza de fase sin cumplir el criterio de
salida de la anterior. Cada sub-parte se prueba sola antes de integrarse.

**Nota sobre esta carpeta (2026-08-15):** `notas/` estuvo en el `.gitignore` y
ahora está versionada. La razón del cambio: estos tres documentos son la
evidencia detrás de las afirmaciones de método del paper —qué se revisó, qué
se midió, y con qué script se reproduce cada cifra—. Fuera del repositorio esa
trazabilidad depende de que alguien reenvíe un archivo por correo; dentro, la
responde el repositorio.

---

## Mapa general

El sistema completo, cuando esté terminado, hace esto con cada frame de video:

```
Cámara o .mp4          (Fase 0/1: sources.py)
      │
      ▼
MediaPipe → 33 landmarks en píxeles + Paso 0
(punto medio cadera, hombro, longitud de torso)   (Fase 1: pose_frontend.py)
      │
      ▼
Cantidades físicas:  T (inclinación tronco)  V (velocidad centroide)   (Fase 2)
                     P (COM vs pies)                                    (Fase 4)
                     I (inmovilidad)                                    (Fase 5)
      │
      ▼
Máquina de estados de 3 etapas:                                  (Fases 3-5)
  Etapa 1: ¿T y V dispararon?  → Etapa 2: ¿P confirma geometría de caída?
  → Etapa 3: ¿sigue inmóvil? ¿intenta levantarse? → severidad (leve/moderada/severa)
      │
      ├── Pantalla: PEF-Lab dibuja esqueleto, valores y estado   (GUI, crece por fase)
      ├── CSV: una fila por frame con todos los valores          (audit_log.py)
      └── Alertas: buzzer / email / SMS                          (Fase 7)
```

Dos formas de correr lo mismo: `python main.py --camera 0` (laboratorio en
vivo, en el Mac) y modo headless por terminal (calibración batch y, al final,
la Raspberry Pi sin pantalla).

---

## Fase 0 — Fundaciones del repo ✅ COMPLETADA

**Objetivo:** que exista la estructura sobre la que todo lo demás se apoya,
sin una sola cantidad física todavía.

**Qué se construyó y por qué:**

- `main.py` — la puerta de entrada. Lee los argumentos de la terminal y decide
  GUI o headless. *Concepto: un solo punto de entrada, dos modos de ejecución.*
- `config.yaml` — TODOS los números calibrables del sistema en un archivo,
  comentados y agrupados por fase. *Concepto: la guía de implementación tiene
  una tabla de "parámetros a calibrar"; este archivo ES esa tabla. El paper
  (§3.5) exige reportar los umbrales usados — así quedan versionados.*
- `requirements.txt` — dependencias con versiones exactas. MediaPipe fijado en
  0.10.14 porque las versiones nuevas eliminaron la API `mp.solutions.pose`
  que asume la guía, y la 0.10.14 trae los modelos dentro del paquete
  (funciona sin internet — coherente con la promesa edge del §3.6).
- `pef_fall_detector/config.py` — lector del config.yaml para que cualquier
  módulo escriba `cfg.stage1.threshold_T_deg`.

**Criterio de salida:** `python main.py --video x.mp4` y `--camera 0` abren la
fuente y muestran frames. ✅

---

## Fase 1 — Front-end de pose (§3.2) ✅ COMPLETADA (falta su prueba en Mac)

**Objetivo:** ver el esqueleto de MediaPipe dibujado en vivo y tener el CSV
por frame funcionando. Es la materia prima de todo lo demás.

**Qué se construyó y por qué:**

- `sources.py` — `VideoFileSource` (lee un .mp4) y `CameraSource` (webcam).
  Ambas entregan lo mismo: (frame, timestamp). *Concepto clave de toda la
  arquitectura: el resto del programa no sabe de dónde vienen los frames, por
  eso "en vivo" y "offline" son el mismo código. También incluye la auditoría
  de FPS: si el .mp4 declara 30 fps pero se grabaron 20 reales, la velocidad V
  saldría inflada un 50% — hay que detectarlo antes de calibrar.*
- `pose_frontend.py` — MediaPipe adentro; salen los 33 landmarks convertidos a
  píxeles + el **Paso 0** de la guía: punto medio de cadera (landmarks 23,24),
  punto medio de hombro (11,12) y longitud de torso (distancia entre ambos).
  *Concepto: (a) los landmarks crudos de MediaPipe vienen normalizados por
  ancho y alto por separado — calcular ángulos ahí sale distorsionado, por eso
  se pasa a píxeles primero; (b) dividir toda distancia entre la longitud de
  torso hace las medidas adimensionales: no importa si la persona está cerca o
  lejos, es alta o baja. También calcula `core_visibility`: qué tan confiables
  son hombros y caderas en este frame — si están fuera de cuadro u ocluidos,
  la geometría de ese frame no es de fiar.*
- `audit_log.py` — el CSV, una fila por frame. *Concepto: la auditabilidad que
  promete el §3.5. De aquí salen además las gráficas de calibración y las
  tablas de la Sección 4.*
- `overlay.py` — dibuja esqueleto, línea del tronco y textos sobre la imagen.
  Separado de la GUI a propósito: el mismo dibujo sirve en pantalla y en PNGs
  exportados para figuras del paper.
- `gui/lab_window.py` — **PEF-Lab**, el laboratorio: abrir cámara o video,
  play/pausa, avanzar frame a frame, barra de tiempo. Solo muestra; no calcula
  nada. *Concepto: la GUI es un visor sobre el núcleo. La Raspberry Pi
  (screenless, por terminal) usa el mismo núcleo sin instalar Qt jamás.*
- `tests/test_pose_frontend.py` — 4 pruebas con esqueletos sintéticos armados
  a mano: torso de 200px debe medir 200px parado o acostado, y normalizar una
  distancia igual al torso debe dar exactamente 1.0. *Concepto: cada sub-parte
  se prueba con matemática pura antes de ver un solo video.*

**Verificado hasta ahora:** tests pasando; corrida headless completa sobre
`test-1.mp4` (248 frames @ 25 fps, ratio FPS correcto). Hallazgo: `test-1.mp4`
es un primer plano de la cara → hombros/caderas fuera de cuadro →
`core_visibility 0.00` marcado en rojo. El chequeo de confiabilidad funcionó a
la primera con un caso real.

**Criterio de salida (pendiente):** correr `python main.py --camera 0` en el
Mac y ver el esqueleto estable en vivo; grabar un clip de CUERPO COMPLETO y
comprobar que: (a) el torso en píxeles cambia *suavemente* al acercarse o
alejarse — crece al acercarse, y eso es correcto: el torso es la regla con la
que medimos, no lo medido (versión anterior de este criterio corregida: decía
"constante" y era un error); (b) el torso *métrico* (`world_torso_len_m`, en
metros, del segundo output de MediaPipe) sí se mantiene ~constante a cualquier
distancia; y (c) las cantidades normalizadas son estables.

---

## Fase 2 — Cantidades T y V + curvas de calibración (§3.4) — EN CURSO

**Objetivo:** las dos primeras cantidades físicas, VISIBLES antes de decidir
nada. Aquí nace la "teoría valiosa": ver las firmas cinemáticas en curvas.

**Avance registrado (2026-08-14):**

- ✅ **2.1 Cantidad T** (`quantities.py` + tests): ángulo del tronco contra la
  vertical, en píxeles 2D, con NaN para tronco degenerado. Incluye
  `trunk_band()` (etiqueta upright / moderate-lean / collapse-range del §3.4,
  SOLO display; el hueco 35–60° que el paper no define se absorbe en
  moderate-lean, documentado). El test `TestAspectRatioCorrection` deja
  clavada la decisión de píxeles: un tronco de 45° reales daría 29.35° en
  coordenadas normalizadas 16:9.
- ✅ **Refactor `FramePipeline`** (`pipeline.py`): TODA la orquestación por
  frame (pose → cantidades → HUD → CSV) vive en una sola clase del núcleo
  que GUI y CLI consumen. Razón: V arrastra estado (historial + EMA); si
  cada entrada orquestara por su cuenta, laboratorio y batch podrían dar
  números distintos para el mismo video, invalidando la calibración.
- ✅ **Política de visibilidad**: frame "confiable" = detección + visibilidad
  de hombros/caderas sobre el umbral. TODO se registra en el CSV (no se
  borran datos), pero las estadísticas del resumen usan solo frames
  confiables. Evidencia que motivó esto: en test-1.mp4 (primer plano de
  cara) MediaPipe extrapolaba caderas a 1958px en un video de 960px de alto.
- ✅ **`pose_world_landmarks` capturados desde el front-end**: el segundo
  output de la MISMA inferencia de MediaPipe — los 33 puntos en METROS,
  centrados en cadera. Diagnóstico, ninguna etapa lo consume. Columna
  `world_torso_len_m` en el CSV: el torso métrico debe ser ~constante a
  cualquier distancia (mejor chequeo de estabilidad que el torso en píxeles).
- ✅ **2.2 Centroide + EMA** (`quantities.py`: `centroid()`,
  `ExponentialMovingAverage`): el punto C del §3.4 suavizado ANTES de
  derivar — sin el filtro, el temblor de los landmarks produciría picos
  falsos de velocidad. Dibujado en el overlay como cruz magenta, con toggle.
- ✅ **2.3 Cantidad V** (`quantities.py`: `VelocityEstimator`): derivada
  discreta sobre ventana deslizante, en torsos/segundo (adimensional).
  Convención de signo del paper: NEGATIVO = hacia abajo (la y de imagen
  crece hacia abajo, así que se niega el dy/dt crudo). HALLAZGO al
  implementar: el pseudocódigo de la guía devuelve (y_actual−y_anterior)/dt
  SIN negar — positivo al caer — pero su umbral es negativo (V < −1.5):
  inconsistencia interna de la guía, resuelta a favor de la convención
  física. Incluye V horizontal (EXPERIMENTAL, fuera del paper) y guardia de
  gaps: un hueco de detección > 0.5 s limpia el historial para no fabricar
  un pico falso al reaparecer el sujeto.
- ✅ **2.4 Estados de la persona** (`person_state.py`, nuevo): las 8
  etiquetas acordadas (en código/CSV en inglés: STANDING, WALKING, LEANING,
  CROUCHING, SITTING, LYING, TRANSITION, NOT_DETECTED ↔ DE_PIE, CAMINANDO,
  INCLINADO, AGACHADO, SENTADO, ACOSTADO, EN_TRANSICION, NO_DETECTADO).
  Rasgo clave: extensión cadera–tobillo dividida entre torso — NO la altura
  del centroide en imagen. Tobillos ocluidos → extensión NaN → fallback
  grueso solo-tronco (oclusión = "no sé", nunca una postura inventada).
  Umbrales provisionales en config.yaml sección `state_display` (marcada
  display-only para no ensuciar la superficie de calibración del §3.5);
  se afinan en 2.6. Solo pantalla y CSV: ninguna etapa los consume.
- ✅ **2.5 Curvas en PEF-Lab** (pyqtgraph): T(t) y V(t) en vivo, con líneas
  punteadas en los umbrales vigentes del config (se VE dónde dispararía la
  Etapa 1), cursor de posición sincronizado con el video, huecos de
  detección mostrados como cortes en la curva (no puentes falsos), y
  toggles: Skeleton / Trunk vector / Centroid / Curves.
- ✅ **Convención `_EXP` en el registro (2026-08-15)**: las columnas que NO
  define el paper llevan sufijo `_EXP` en el CSV — `Vh_tps_EXP`,
  `world_torso_len_m_EXP`, `extension_ratio_EXP`, `state_EXP`. Motivo: el
  registro mezcla cantidades que el paper sostiene (§3.4) con señales que
  grabamos para averiguar si merecen entrar, y de los números solos no se
  distinguen; publicar una tabla donde ambas se ven igual invita a leer una
  columna exploratoria como resultado establecido. NO llevan marca el
  centroide (el §3.4 sí lo define — solo lo suavizamos, que es la divergencia
  P3, un arreglo de texto), ni `reliable`/`core_visibility` (higiene del
  registro, no cantidades bajo evaluación). Implementado como convención **del
  registro**, no del código: `csv_column()` en `audit_log.py` traduce al
  escribir la fila, así que el resto del sistema sigue diciendo `Vh_tps` y
  graduar un campo al paper es borrar una línea de un `frozenset`.
- ✅ **Columnas documentadas en el `README.md`** ("Outputs → Columns"): las 20
  columnas en tres tablas (Paso 0, cantidades del paper, experimentales), cada
  experimental con el porqué de su registro. Antes esa información solo existía
  en docstrings y en esta carpeta, que no viajaba con el repositorio: alguien
  que clonara el proyecto no tenía dónde leer qué significa `Vh_tps`.
- 📋 **Protocolo calibrar-vs-entrenar acordado** (pendiente de ejecutar en
  2.6 — su ✅ llega con el tag git `calib-v1`): NO se entrena nada (tesis del
  paper). Umbrales elegidos mirando curvas → commit de config.yaml + tag git
  (calib-v1) anotando qué clips los produjeron → el holdout se evalúa sin
  retocar → si hay que reajustar: calib-v2, y se reporta que hubo dos rondas.

**Qué se construirá:**

- `quantities.py` (nuevo) — funciones puras:
  - **T, ángulo del tronco:** ángulo entre el vector cadera→hombro y la
    vertical. De pie ≈ 0°; inclinado 15–35°; caída 60–90° en poco tiempo.
  - **V, velocidad vertical del centroide:** cuánto baja por segundo el punto
    cadera+hombro, en torsos/segundo (adimensional). Antes de derivar se
    suaviza el centroide (EMA) porque derivar amplifica el temblor natural de
    los landmarks — sin filtro habría picos falsos de velocidad.
  - **V horizontal (EXPERIMENTAL):** no está en el paper (§3.4 define V solo
    vertical); se registra como señal diagnóstica. Si en las pruebas separa
    casos que la vertical no separa, se propondrá agregarla al paper con esa
    evidencia.
- GUI: panel numérico con T y V en vivo + curvas T(t) y V(t) (pyqtgraph) con
  líneas horizontales en los umbrales del config. Toggle nuevo.
- CSV: columnas nuevas T, V, V_horizontal.
- **Estado de la persona (de pie / caminando / sentado / agachado / acostado):**
  etiqueta por reglas sobre T + altura del centroide + V. Solo visualización y
  columna en CSV — NO participa en la alarma (si algún día entra, se modifica
  el draft §3.4/3.5 primero, con evidencia).
- **Mini-suite de clips** (grabados por usted con PEF-Video-Tool): caminar,
  sentarse rápido, agacharse a recoger algo, acostarse, caída simulada
  adelante/atrás/lateral (en colchón). Son los datos de calibración.

**Criterio de salida:** las curvas de los clips muestran separación visible
entre caída y actividades normales en el plano (T, V); umbrales iniciales
elegidos MIRANDO las gráficas, no a ciegas; la etiqueta de estado coincide con
lo que la persona hace en el video.

**Pendientes concretos de 2.6, en orden:**

1. Procesar cada clip en headless y revisar el audit de FPS y la estabilidad
   del torso métrico ANTES de sacarle conclusiones a las curvas. Criterio de
   descarte a fijar mirando los datos; de referencia, `video__8` del dataset
   público tiene CV de torso métrico del 9.7 % contra 2.7 % del mejor clip.
2. Curvas T(t) y V(t) por clip, caídas superpuestas contra ADL, y elegir los
   umbrales mirando.
3. **Revisar `extension_ratio_EXP` negativo.** En dos clips de caída del
   dataset da −0.40 y −0.17. El cálculo es `(y_tobillo − y_cadera) / torso`,
   una diferencia CON SIGNO: negativo = tobillos por encima de las caderas,
   físicamente posible a media caída. El problema no es el dato sino su
   consumo: `classify_state` evalúa `extension_ratio < crouch_extension`, así
   que manda cualquier negativo a la categoría más agachada cuando la persona
   está más bien invertida. Hay que ver los frames antes de decidir si se
   corrige el clasificador, el nombre del campo, o ambos.
4. **Revisar el V de −12.76 torso/s de `video__8`**, muy fuera del rango −2.8
   a −5.2 de las caídas reales. Con el CV de torso del 9.7 % en el mismo clip,
   el sospechoso es el artefacto de salto de esqueleto ya documentado.
5. **Alinear el EMA con el draft.** El §3.4 de draft4 dice que se suaviza
   *"the vertical component of C"*; nuestro código suaviza el vector completo
   (x e y). Hay que alinear uno de los dos — el código es lo barato, y la
   componente x solo alimenta `Vh_tps_EXP`, que no está en el paper.
6. **Pasar a segundos los parámetros que quedaron en frames**:
   `stage1.min_consecutive_frames` y `stage2.com_eval_window_frames`. Es
   exactamente el hallazgo C5 sin aplicar, y estorba más si se arrastra a las
   Fases 3 y 4 en vez de arreglarse aquí.
7. **Reencuadrar el CSV** en el docstring de `audit_log.py` y en el README:
   draft4 funda la explicabilidad en leer las cantidades de los frames
   (§3.5), no en un registro persistido, y el §3.6 exige que el detector
   desplegado no escriba nada. El CSV es instrumento de investigación y
   calibración; el perfil de despliegue lo apaga. Son dos párrafos, no código.
8. Afinar los umbrales de `state_display` (en 2.4 quedaron provisionales) y
   comprobar clip por clip que la etiqueta coincide con lo que se ve.
9. Commit de `config.yaml` + tag `calib-v1` + bitácora de qué clips lo
   produjeron. **Ese tag es el ✅ de la sub-parte 2.6.**

---

## Sub-parte 2.7 — Especificación de la Etapa 1 y de la guarda de sujeto

**Depende de 2.6** (necesita las curvas y los umbrales calibrados).
**No escribe código del sistema.** Produce un documento de especificación y
los análisis que lo respaldan. Existe porque el §3.5 de draft4, tal como está
redactado, **no es implementable sin inventar el criterio** — y eso es
exactamente lo que un paper de método no debe permitir.

**El problema, textual.** §3.5 dice: *"an **instantaneous** combination of the
two quantities exceeds a configurable trigger threshold while V remains
negative (downward)"*. Dos huecos:

1. *"Instantaneous"* significa mismo frame. En las 6 caídas reales medidas, los
   picos de T y de V están desfasados entre 0.03 s y 5 s, y una regla de mismo
   frame perdía **3 de 6**. La redacción actual describe un sistema que
   nuestros propios datos muestran que falla.
2. *"A combination... exceeds a threshold"* no dice **cuál** combinación. Una
   conjunción, una suma, un score normalizado y una secuencia son cuatro
   sistemas distintos, todos compatibles con esa frase.

**Entregable A — definición del disparador.** Evaluar sobre los datos de 2.6
más las 6 caídas del dataset público, y elegir con evidencia entre:

- conjunción simultánea (la del draft y la guía) — línea base, ya sabemos que
  pierde 3 de 6, pero hay que reportar el número;
- disparo por V con confirmación de T en una ventana posterior (secuencial);
- score combinado normalizado de T y V que cruce un umbral único.

El criterio de elección se fija **antes** de mirar: sensibilidad sobre las
caídas con cero disparos en los clips de caminata, y a igualdad de eso, la
formulación más simple de explicar en el paper.

**Entregable B — guarda de plausibilidad de sujeto.** Es la respuesta a los
transeúntes del §3.3 sin agregar detección multi-persona, que contradiría el
compromiso con MediaPipe Pose del §3.2. Consiste en un umbral de
desplazamiento del esqueleto entre frames, en torsos/segundo, por encima del
cual se declara cambio de sujeto y se marca discontinuidad en vez de emitir V.

Se calibra con dos referencias que ya tenemos: el clip de dos personas, donde
el salto de esqueleto produjo **+26.6 torso/s** tras un micro-hueco de 0.02 s,
y las caídas reales, cuyo máximo legítimo observado es **−5.2 torso/s**. Entre
ambos hay un orden de magnitud, así que el umbral no es delicado. Señal
secundaria: la longitud del torso cambia suave con la distancia y salta cuando
el tracker cambia de cuerpo.

**Anclaje en el paper:** no hay que inventar nada. El §3.2 ya dice que la
ambigüedad de profundidad *"is mitigated by tracking within-frame and
**across-frame landmark displacement ratios**"*. Esta guarda es esa frase hecha
concreta.

**Entregable C — histéresis y enfriamiento en segundos**, no en frames
(hallazgo C5), con el valor justificado por la duración real de las caídas
medidas.

**Entregable D — redacción propuesta para el draft**: el párrafo de §3.5 con
el disparador definido, la guarda nombrada como parte de la Etapa 1, y la
concreción de la frase del §3.2. Con las cifras que respaldan cada número.
Cierra la divergencia P5.

**Criterio de salida:** una especificación que dos personas distintas
implementarían igual; cada umbral acompañado de la curva o la tabla que lo
justifica; y el texto listo para pegar en el draft. Solo entonces empieza la
Fase 3 — que pasa a ser transcripción de una decisión ya tomada, no diseño
sobre la marcha.

---

## Fase 3 — Etapa 1: disparador cinemático + máquina de estados (§3.5)

**Objetivo:** la primera decisión automática: "algo raro pasó".

**Qué se construirá:**

- `state_machine.py` (nuevo) — máquina de estados explícita:
  `monitoreando → evaluando_geometria → confirmando_inmovilidad`, con
  histéresis (el disparo debe sostenerse m frames seguidos, no 1) y cooldown
  (período refractario tras cada evento para no duplicar alarmas).
- Registro de eventos: frame exacto del disparo + valores de T y V que lo
  causaron (el inicio del log auditable de eventos).
- GUI: indicador de etapa actual y lista de eventos.

**Concepto:** la Etapa 1 es deliberadamente "paranoica": debe atrapar TODAS
las caídas aunque también dispare con sentarse rápido o agacharse. Filtrar
esos falsos positivos es trabajo de las Etapas 2 y 3 — cada etapa es un
embudo.

**Criterio de salida:** 0 disparos en clips de caminata; 100% de disparos en
clips de caída; documentado qué actividades "trampa" disparan (serán el caso
de prueba de las fases siguientes).

---

## Fase 4 — Cantidad P + Etapa 2: verificación geométrica (§3.4/3.5)

**Objetivo:** distinguir caída de movimiento controlado usando el equilibrio.

**Qué se construirá:**

- En `quantities.py`: **P** — proyección del centro de masa (COM, dominado por
  la cadera) contra el **polígono de soporte** (envolvente convexa de
  tobillos + pies). *Concepto físico: mientras el COM se proyecta dentro del
  área de los pies, hay equilibrio; cuando sale, geométricamente es caída.
  Al sentarse, el COM se mantiene sobre la base durante el descenso.*
- Correcciones sobre la guía (fundamentadas en el plan, §5.2 y 5.3):
  - P se evalúa sobre una **ventana de frames alrededor del disparo**
    (incluyendo frames de ANTES, guardados en un buffer circular), no en un
    solo frame después — la física del desequilibrio ocurre al inicio del
    evento, no tras el impacto. Se exige k-de-n frames, no 1.
  - **Pies ocluidos ≠ evidencia de caída.** Si la visibilidad de los pies es
    baja (camas y muebles los tapan — escenarios centrales del dataset), P
    devuelve "no concluyente" y el peso de la decisión pasa a la Etapa 3.
- GUI: toggle para dibujar el polígono de soporte y el punto COM.

**Criterio de salida:** los clips de sentarse rápido que disparaban la Etapa 1
quedan filtrados aquí; las caídas siguen pasando.

---

## Fase 5 — Cantidad I + Etapa 3: inmovilidad, intentos de levantarse y severidad

**Objetivo:** la confirmación final y la etiqueta de severidad.

**Qué se construirá:**

- En `quantities.py`: **I** — tiempo de inmovilidad post-disparo, redefinido
  como "tiempo desde el último movimiento" (comparar contra la posición de
  hace ~0.5 s, no contra el promedio global — corrección al pseudocódigo de la
  guía, que rompía el caso "forcejea y luego queda inmóvil").
- Sub-estados dentro de la ventana de observación (idea suya, 2026-08-14):
  - **inmóvil** — sin movimiento sobre ε
  - **forcejeando** — se mueve, pero el tronco no vuelve a vertical ni el
    centroide sube = **intento de levantarse fallido**
  - **recuperado** — postura vertical recuperada
- Severidad: recuperado → **leve**; intentos fallidos sin recuperación →
  **moderada**; inmovilidad sostenida ≥ W → **severa**. *Nota: el pseudocódigo
  de la guía solo devuelve leve/severa — omite la "moderada" que el paper SÍ
  define (§3.3, §3.7). Su observación sobre "intenta levantarse pero no puede"
  cierra exactamente ese hueco.*
- GUI: contador de inmovilidad, contador de intentos, severidad final.

**Criterio de salida:** pipeline completo correcto sobre toda la mini-suite,
incluido el colapso tipo síncope (quieto 4–8 s); cada evento con su registro
auditable (T, V, P, I + frames).

---

## Fase 6 — Harness de evaluación para PEF-FallDB (§3.7)

**Objetivo:** la máquina de generar la Sección 4 del paper.

**Qué se construirá:**

- `evaluation.py` (nuevo) — modo batch: procesa carpetas enteras de videos,
  lee la etiqueta verdadera desde la ruta (protocolo de anotación §3.3:
  `[setting]/[event]-[recovery]`, p. ej. `hallway/fall-recovered-001.mp4`), y
  produce matriz de confusión, accuracy, sensibilidad, especificidad,
  precisión, F1 y falsos positivos por hora.
- Ablaciones del §3.7: quitar T/V/P/I de a una y medir la caída de desempeño;
  quitar la Etapa 3; colapsar todo a umbral único.
- A partir de aquí usted graba PEF-FallDB con el video tool (usando la
  convención de nombres desde el primer video) y cada tanda se evalúa con un
  comando.

**Criterio de salida:** un comando procesa una carpeta del dataset y emite el
reporte de métricas + ablaciones listo para la Sección 4.

---

## Fase 7 — Tiempo real pulido, alertas y Raspberry Pi (§3.6)

**Objetivo:** el producto final del paper corriendo en el edge.

**Qué se construirá:**

- Modo servicio headless continuo (cámara + detección + alertas, por
  terminal/CLI — la Pi es screenless; jamás instala Qt).
- Despachador de alertas: buzzer local primero; email/SMS después (§3.6).
- Port a Raspberry Pi + mediciones que exige el §3.7: FPS sostenido, latencia
  por frame, CPU, memoria — datos que van directo a la Sección 4.

**Criterio de salida:** el sistema corre en la Pi con métricas medidas y
registradas.

---

## Estado actual y pendientes inmediatos

| # | Fase | Estado |
|---|------|--------|
| 0 | Fundaciones | ✅ Completada |
| 1 | Front-end de pose + PEF-Lab | ✅ Código completo — **nunca se ha corrido en el Mac**: la fase está marcada completa por código, no por observación |
| 2 | T y V + calibración | 🔶 Código completo (2.1–2.5 ✅), revisión cerrada (lotes A–D ✅), convención `_EXP` ✅, columnas documentadas ✅, **73 tests** — faltan **2.6** (calibración) y **2.7** (especificación de la Etapa 1) |
| 3–7 | — | No iniciadas |

**Adherencia a draft4 (revisado el 15 de agosto de 2026).** Draft4 cerró cuatro
de las cinco divergencias que este proyecto había catalogado: el centroide ya
está definido como punto medio 0.5/0.5 (P1), el conteo de landmarks es correcto
(P2), el EMA quedó **dentro de la definición de la cantidad V** (P3), y la
geometría en píxeles 2D con vertical (0,−1) es ahora la del paper (P4). Además
cerró el hueco de 35–60° en las bandas de T, que ahora son 15 / 15–60 / ≥60 —
justo lo que `trunk_band()` ya implementaba. Tres argumentos de este proyecto
llegaron al texto: el del aspecto (45° reales leídos como 29° en 16:9), el del
recorte del coseno, y el umbral de tronco degenerado de 0.001 px.

**Queda abierta P5**, la formulación de la Etapa 1, y es el objeto de 2.7.

**Consecuencia sobre el EMA:** ya no se puede quitar por configuración sin
contradecir el §3.4, que lo declara parte de la definición de V. `tau = 0`
sigue siendo necesario, pero como **ablación** del §3.7, no como opción de
despliegue.

*Actualizado 2026-08-15. La suite pasó de 67 a 73 tests con las pruebas de la
convención `_EXP`, que además cerraron un hueco anterior: nada cruzaba las
claves que emite el pipeline contra el esquema del CSV, así que una clave
desalineada solo reventaba al correr un video real, nunca en la suite.*

## Lo que falta antes de la Fase 3

**Suyo, y bloquea el resto:**

1. **Grabar la mini-suite de cuerpo completo**, con tobillos visibles en
   cuadro: caminar de frente, caminar alejándose, sentarse rápido, agacharse a
   recoger algo, acostarse despacio, levantarse del suelo, y 3 caídas sobre
   colchón (adelante, atrás, lateral). Unos 10–12 clips de ~20 s. Los tobillos
   importan porque `extension_ratio_EXP` se calcula cadera-a-tobillo; sin ellos
   el clasificador cae al fallback grueso solo-tronco.
2. **Correr `python main.py --camera 0` una vez** y reportar qué se ve. Es el
   criterio de salida de la Fase 1, que nadie ha verificado.
3. **Limpiar `logs/`**: 31 CSV, de los cuales 3 vacíos y ~20 anteriores al
   Lote C (sin `.meta.json`, así que no se sabe con qué configuración
   salieron). Los del dataset se reprocesan en minutos.

**Mío, una vez existan los clips:** los 9 puntos de 2.6 listados arriba, y
después **2.7** — la especificación de la Etapa 1 y de la guarda de sujeto.
La Fase 3 no empieza hasta que 2.7 esté cerrada, porque el §3.5 de draft4 no
se puede implementar sin decidir antes qué significa su "combination".

**Mío, sin esperar clips:**

- **Escribir la especificación de la Etapa 1 secuencial.** Las 6 caídas reales
  mostraron que la conjunción de la guía (T>45 **y** V<−1.5 simultáneos) falla
  en 3 de 6 porque los picos están desfasados entre 0.03 y 5 s. La Fase 3
  codifica `state_machine.py` a partir de esa especificación, así que tiene que
  estar escrita y discutida antes, no durante. Es también el P5 del draft.
- Re-medir sobre metraje real las dos cifras que hoy solo existen en versión
  sintética (ver `REVISION-CODIGO.md`, sección de re-verificación) — esto sí
  depende de los clips.
