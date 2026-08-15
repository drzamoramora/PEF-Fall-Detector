# Revisión de código — consolidado de 5 pasadas

**Proyecto:** PEF-Fall-Detector
**Fecha de la revisión:** 14 de agosto de 2026
**Alcance revisado:** todo el código de las Fases 0 a 2 (12 módulos, ~1 500
líneas, 39 tests) antes de comenzar la Fase 3.

## ✅ ESTADO: REVISIÓN CERRADA

**19 de los 21 hallazgos están corregidos y verificados.** Los 2 restantes
(F1 y F2) se difirieron a la Fase 7 de forma deliberada, porque solo tienen
sentido cuando exista la Raspberry Pi.

La suite pasó de **39 a 73 tests**. Cada corrección se escribió con su prueba
—en los lotes B y C, la prueba **antes** que el arreglo, de modo que primero
fallara y luego pasara— y cada lote se sometió a prueba de mutación.

> **Léase primero la sección "Re-verificación (2026-08-15)" al final.** Al
> volver a ejecutar todo, cinco cifras de este documento NO se reprodujeron en
> su decimal exacto, porque los scripts que las produjeron no se guardaron.
> Ninguna conclusión cambia de dirección, pero las cifras afectadas están
> marcadas en el texto con ⚠, y ahora existe
> [`verificar_mediciones.py`](verificar_mediciones.py) para que cada número se
> regenere con un comando.

| Lote | Contenido | Estado |
|---|---|---|
| A — Tiempo y suavizado | C5, M7a, M3 | ✅ |
| B — Discontinuidad | C1, C2 | ✅ |
| C — Integridad del registro | M1, M4, M8, fps, C4 | ✅ |
| D — Mecánica de acceso | M2, C3, M9, M7b | ✅ |
| Sueltos | M5, docstrings | ✅ |
| Fase 7 | F1, F2 | ⏸ diferidos |
| Draft | P1–P5 | 📋 esperan la Fase 3 |

Aparte quedan las **cinco divergencias con el paper** (P1–P5), que no son de
software: se resuelven editando el draft, y P5 en particular necesita la
Fase 3 implementada para respaldarse con números propios.

---

Cada pasada usó una técnica distinta a propósito, para que no se solaparan.
Ninguna pasada contradijo a las anteriores.

El documento tiene **dos vistas del mismo material**: el catálogo de hallazgos
clasificado por severidad (para juzgar importancia) y, al final, la
agrupación en **lotes de trabajo** con sus dependencias (para ejecutar). Los
hallazgos individuales indican a qué lote pertenecen cuando no es evidente.

| Pasada | Técnica | Hallazgos nuevos |
|---|---|---|
| 1ª | Razonamiento sobre el diseño | 7 |
| 2ª | Lectura fresca + verificación contra datos reales | 3 |
| 3ª | Verificación automática, rendimiento y mutación | 4 |
| 4ª | Ejercicio de la interfaz + auditoría contra el paper | 4 |
| 5ª | Determinismo, integridad y robustez | 3 |

---

## CRÍTICO — corrompe datos de calibración

### C1. La guardia de huecos no protege contra micro-huecos
*(1ª pasada, confirmado empíricamente)*

El reinicio del estado de movimiento se dispara por tiempo transcurrido
(0.5 s), no por pérdida de detección. Los dos picos imposibles observados en
datos reales (+17.3 y +26.6 torsos/s en el clip con dos personas) ocurrieron
tras huecos de **0.02 s** — un solo frame perdido. Reproducido de forma
sintética: un frame perdido más un salto de esqueleto produce V = +10.0.

**Corrección:** reiniciar el estado de movimiento ante **cualquier** pérdida
de detección. Si se perdió a la persona, aunque sea un frame, el sistema no
puede saber si recuperó a la misma.


> **✅ RESUELTO (Lote B).** Ahora *cualquier* pérdida de detección marca
> discontinuidad, sin importar su duración. El test que reproducía el fallo
> daba +10.0 torso/s antes de la corrección y NaN después. Verificado también
> en la aplicación real.

### C2. La interfaz no puede reiniciar el pipeline al saltar
*(1ª pasada)*

`FramePipeline` solo expone `_reset_motion_state` en privado. Al saltar hacia
adelante menos de 0.5 s, la ventana de velocidad mezcla frames de dos
posiciones distintas del video y produce velocidades inventadas — justo
durante la calibración, que es cuando más se usa el scrub.

**Corrección:** exponer `reset()` público y llamarlo explícitamente en cada
salto.


> **✅ RESUELTO (Lote B).** `FramePipeline.reset()` es público y la GUI lo
> llama desde un `_seek_to()` común que sirve a la barra de scrub y al paso
> frame a frame. Verificado en la GUI real: tras saltar al frame 150, la V
> del siguiente frame sale NaN.

### C3. El paso frame a frame está desfasado en ambas direcciones
*(2ª pasada)*

La aritmética `target = frame_index - 1 + delta` trata `frame_index` como "el
siguiente a leer" cuando en realidad es "el último mostrado". Trazado:

- Desde el frame 50, **"Frame ▶" vuelve a mostrar el 50** (no avanza).
- **"◀ Frame" salta al 48** (retrocede dos).

**Corrección:** `target = frame_index + delta`. Verificado.


> **✅ RESUELTO (Lote D).** Corregido a `frame_index + delta`. Verificado en
> la aplicación real: 49 → 50 → 49 → 48, y sin reventar en el frame 0. Se
> corrigió **después** de M2, según el vínculo causal.

### C4. El scrub duplica filas en el registro CSV
*(2ª pasada, reproducido en la 4ª)*

Cada salto manual vuelve a escribir al CSV. Reproducido: una sesión de
ejercicio dejó **30 filas con 5 `frame_index` duplicados**. Cualquier análisis
posterior contaría frames dos veces.

**Corrección:** no registrar frames que provienen de un salto manual; el CSV
queda reservado para reproducción continua.

> **Pertenece al Lote C** (integridad del registro), no al de discontinuidad.
> El invariante que rompe es *"el CSV es un registro fiel"* — el mismo de M1,
> M4 y M8 — aunque comparta disparador con C1 y C2. Depende del mecanismo del
> Lote B para saber cuándo un frame viene de un salto, así que va después de
> ambos.


> **✅ RESUELTO (Lote C).** Los frames alcanzados por un salto manual ya no
> se registran: el CSV es la crónica de una observación continua, no de lo
> que uno miró. Verificado con scrub agresivo sobre frames ya vistos —
> 40 filas, 0 duplicadas, monótonas.

### C5. La ventana de velocidad está en frames, no en tiempo
*(2ª pasada)*

La misma caída física medida a distintos frame rates:

| fps | V pico | % del valor a 60 fps |
|---|---|---|
| 60 | −5.03 | 100 % |
| 30 | −4.10 | 81 % |
| 25 | −3.81 | 76 % |
| 15 | −2.83 | 56 % |
| 10 | −2.11 | 42 % |

Un umbral calibrado a 30 fps **perdería caídas en la Raspberry Pi**. El mismo
problema afecta al filtro EMA (su constante de tiempo depende del frame rate).
Expresar la ventana en segundos reduce la dispersión al 59 % en el peor caso,
aunque no la elimina.

**Corrección:** expresar ventana y suavizado en unidades de tiempo, y
convertirlos a frames usando el fps de la fuente. Registrar además el fps en
el CSV (hoy no se guarda).

> **✅ RESUELTO (Lote A).** Ventana y suavizado pasaron a expresarse en
> segundos, con el coeficiente del filtro recalculado en cada frame a partir
> del `dt` real. **La dispersión entre 60 y 10 fps cayó de 138 % a 5.7 %** ⚠.
> Fijado por el test P-01, que además exige que todos los frame rates crucen
> un umbral común. El fps quedó registrado en el `.meta.json` de cada corrida.

---

## MEDIO — fragilidad y deuda técnica

### M1. `extrasaction="ignore"` silencia errores de tipeo *(1ª)*
Una clave mal escrita desaparece del CSV sin aviso. Verificado
automáticamente: **hoy no hay ningún desajuste**, pero el riesgo crece con
cada fase. Validar las claves contra el esquema, al menos en los tests.


> **✅ RESUELTO (Lote C).** Ahora `extrasaction="raise"`: una clave mal
> escrita revienta con mensaje claro en vez de desaparecer.

### M2. La interfaz lee un atributo privado de la fuente *(1ª)*
`getattr(source, "_next_index")`. Debería ser una propiedad pública.

> **No es cosmético: es la causa de C3.** El desfase del paso frame a frame
> nació precisamente de que la interfaz *reconstruye* el índice actual con
> `_next_index - 1` sobre un atributo privado. Corregir C3 sin corregir M2
> deja intacta la condición que lo produjo, y el mismo error puede repetirse
> la próxima vez que alguien necesite saber en qué frame está.
> **Orden obligado: M2 primero, C3 después.**


> **✅ RESUELTO (Lote D).** `VideoFileSource.current_index` es una propiedad
> pública, y su docstring cita el bug que su ausencia produjo. Ya no queda
> ningún acceso privado desde la GUI.

### M3. Mezcla de suavizado en `extension_ratio` *(1ª)*
Usa torso filtrado con posiciones de cadera y tobillo crudas. Definir una
política y documentarla.

> **Pertenece al Lote A** (tiempo y suavizado), no al de discontinuidad pese a
> vivir en el mismo archivo: es una decisión sobre *qué se filtra*, que es el
> subsistema que C5 va a tocar.


> **✅ RESUELTO (Lote A), con la justificación corregida.** La razón se
> calcula entera con datos crudos y se filtra el resultado. Medido: suavizar
> solo el denominador aportaba **16 %**; filtrar la salida aporta **80 %** ⚠.
> La mezcla no empeoraba las cosas — casi no servía. El filtro quedó
> **encendido** por defecto: sin él, un sujeto quieto en el límite de postura
> cambia de etiqueta 7.1 veces más.

### M4. La ruta de logs depende del directorio de invocación *(3ª)*
`config.yaml` se resuelve de forma absoluta, pero `logging.output_dir` es
relativo: ejecutando desde `/tmp`, los CSV van a `/tmp/logs`. Anclarlo al
repositorio o documentar el comportamiento.


> **✅ RESUELTO (Lote C).** Una ruta relativa se ancla a la raíz del
> repositorio, igual que ya hacía `config.yaml`.

### M5. Falta test del recorte del coseno *(3ª, prueba de mutación)*
De cinco regresiones introducidas a propósito, la suite detectó cuatro. La no
detectada: eliminar `np.clip(cos_theta, -1, 1)`, la protección contra que la
deriva de punto flotante haga que `arccos` devuelva NaN.


> **✅ RESUELTO, y el hallazgo resultó ser otro.** El recorte **viene de la
> guía** (§3.4, "evitar errores de redondeo"), no del código nuestro. Y no es
> alcanzable con esta formulación: al ser unitario el eje vertical dividimos
> entre una sola norma, y `sqrt(vx²+vy²)` nunca queda bajo `|vy|`. Medido
> sobre 400 000 troncos casi verticales en 12 órdenes de magnitud: **cero
> desbordamientos**. Por eso ninguna mutación podía cazarlo — no era un hueco
> de cobertura. Se conserva porque la fórmula **general** sí desborda (13 % ⚠ de
> los pares paralelos), y se añadieron dos pruebas de propiedad sobre 40 000
> casos que fijan la garantía para el día que el cálculo pase a 3D.

### M6. Cobertura de pruebas concentrada en la matemática pura *(1ª)*
39 tests cubren cantidades, estados y Paso 0. **No hay ningún test de
`pipeline.py`, `audit_log.py`, `sources.py` ni `config.py`** — precisamente
los módulos con estado, donde viven C1, C2 y C4.


> **✅ RESUELTO.** Existen `tests/test_pipeline.py` y `tests/test_audit_log.py`.
> Fue posible gracias a separar `FramePipeline.analyze()` de `process()`: el
> front-end de pose se crea de forma perezosa, así que toda la lógica con
> estado se prueba con esqueletos sintéticos, sin MediaPipe y en milisegundos.

### M7. Temporizador de cámara fijo y precedencia silenciosa *(3ª)*
El modo cámara usa 33 ms sin importar el frame rate real. Si se pasan
`--video` y `--camera` a la vez, el video gana sin avisar.


> **✅ RESUELTO, y M7a cambió de enfoque al medir.** La propuesta original era
> derivar el intervalo del fps de la cámara; los datos mostraron que **el
> cuello de botella es el procesamiento** (80 ms logrados contra un
> temporizador de 33 ms en el Mac del autor), así que eso habría empeorado el
> encolado. Se implementó **auto-programación**: al terminar un frame se
> agenda el siguiente. M7b se resolvió con el grupo de exclusión mutua de
> `argparse` — dos líneas, error automático y documentado en el `--help`.

### M8. Se crean CSV vacíos *(4ª)*
El registro se abre al abrir la fuente, antes de leer un frame: abrir y cerrar
sin reproducir deja archivos de cero filas.


> **✅ RESUELTO (Lote C).** El archivo se crea con la primera fila, no al abrir
> la fuente.

### M9. Import sin usar *(1ª)*
`math` en `lab_window.py`.

---


> **✅ RESUELTO (Lote D).**

## PARA LA FASE 7 — anotado, no se corrige ahora

### F1. No existe modo headless en vivo *(1ª)*
`--headless` exige `--video`. La Raspberry Pi necesita exactamente lo
contrario: cámara y detección sin pantalla.

> **⏸ DIFERIDO a la Fase 7, junto con F2.** Son un solo trabajo de diseño: un
> bucle de procesamiento independiente de Qt sirve a los dos.

### F2. El procesamiento corre en el hilo de la interfaz *(3ª)*
Medido: mediana **24.7 ms/frame** y p95 **36.4 ms** a 1280×960 (unos 40 FPS
teóricos en servidor). El temporizador dispara cada 33 ms, así que el
presupuesto ya se desborda en el p95: la ventana pierde frames y queda
momentáneamente sorda. Para la Pi, **bajar la resolución de entrada es la
palanca principal** (640×480 cuesta aproximadamente una cuarta parte).

> **⏸ DIFERIDO a la Fase 7, con evidencia nueva.** Al medir los registros
> reales se confirmó en la máquina del autor: las sesiones en vivo corrieron a
> **12.5 fps**, no a 30 — el temporizador pedía cada 33 ms y se lograban 80.
> Mitigado parcialmente en M7a con la auto-programación, que impide que se
> acumule cola, pero **no aumenta el rendimiento**: eso exige sacar el
> procesamiento del hilo de la interfaz. Pendiente de medir cuánto de esa
> carga son las gráficas y cuánto la detección (P-62).

---

## DIVERGENCIAS CON EL PAPER — deuda editorial, no de software
*(4ª pasada)*

### P1. El centroide no está definido de forma verificable
El §3.4 dice *"the position-by-distance-weighted mean of the hip and shoulder
clusters"*. Esa frase no especifica ponderada por qué distancia ni respecto a
qué. El código implementa una media ponderada simple de los dos puntos medios
(peso 0.5). **Un revisor no puede reproducir el método a partir del texto.**
Reescribir el párrafo para que describa exactamente lo implementado.

### P2. El conteo de landmarks del §3.2 es incorrecto
El texto dice "11 upper-body landmarks (face, shoulders, elbows, wrists, hips)
y 22 lower-body (knees, ankles, heels, foot indices)". Suman 33 en total, pero
la repartición es imposible: MediaPipe tiene 11 puntos solo de cara, y
rodillas, tobillos, talones y pies son 8, no 22.

### P3. El filtro EMA no aparece en el paper
El §3.4 describe V como derivada directa. El suavizado existe, es
indispensable (la prueba de mutación confirma que quitarlo rompe tests),
**modifica el valor de V, introduce latencia de detección (~0.1 s) y tiene un
parámetro calibrable**. Debe documentarse y reportarse en la Sección 4.

### P4. La normalización usa distancia 2D, la guía pide 3D
Divergencia deliberada y ya documentada en el código (la z monocular no es
fiable). Debe quedar explícita en el paper.

### P5. La Etapa 1 del §3.5, tal como está formulada, pierde caídas
Detalle completo en `HALLAZGOS-Y-PROPUESTAS.md`: la conjunción instantánea de
T y V no se cumple porque los picos están desfasados entre 0.03 s y 5 s. En
6 clips reales, 3 no se detectarían.

---

## VERIFICADO CORRECTO — no requiere acción

- **Determinismo**: procesar el mismo video dos veces produce CSV **byte a
  byte idénticos** (248 filas, 0 diferencias). La calibración es reproducible,
  requisito para la Sección 4.
- **Coherencia de configuración**: las 11 claves que el código lee existen
  todas en `config.yaml`; ninguna clave escrita al CSV queda fuera del
  esquema.
- **Robustez de la interfaz**: 18 maniobras hostiles (reabrir sin cerrar,
  cerrar dos veces, scrub fuera de rango, paso sin fuente, archivo
  inexistente) sin un solo fallo.
- **Robustez ante entradas**: procesa correctamente video vertical de celular,
  resolución de 64×48 y videos de un solo frame; rechaza con error claro los
  archivos inexistentes y los que no son video.
- **Separación núcleo/interfaz**: ningún módulo del núcleo importa Qt; el port
  a la Raspberry Pi sigue limpio.
- **Compatibilidad**: numpy 2.4.4 convive con mediapipe 0.10.14 (ejercitado
  decenas de veces). Única restricción declarada: protobuf < 5.
- **Política de NaN**: consistente en todo el sistema; nunca se escribe un
  cero fingido.
- **Documentación**: **89 % de las funciones públicas con docstring (47/53)**,
  los 12 módulos con docstring de módulo. Las 6 restantes son implementaciones
  concretas que heredan su documentación de la clase abstracta `FrameSource`;
  duplicarla crearía dos versiones que pueden divergir. Lo que falta es trivial (`close`,
  `closeEvent`, `main`) o hereda sentido de la clase abstracta. Las decisiones
  difíciles están explicadas con su porqué y citando la sección del paper.
- **Prueba de mutación**: acumuladas a lo largo de la revisión y de la
  ejecución, **20 de 21 mutaciones válidas fueron detectadas**. La única que
  no —volver a mezclar etapas de suavizado— corresponde a un cambio cuyo
  efecto medido es del 3.2 %, por debajo del umbral que alteraría cualquier
  decisión.

---

## LIMITACIONES CONOCIDAS — documentar, no corregir

- **Mono-persona por diseño.** El backend sigue a un solo individuo; con dos
  personas el esqueleto puede saltar de una a otra. Coherente con el §3.2 y
  con multi-persona declarado como trabajo futuro (Sección 6).
- **`extension_ratio` se acerca a 0 al estar acostado**, así que alguien
  incorporado sobre los codos (con T bajo 60°) se etiquetaría `CROUCHING`.
  Solo afecta a la visualización.
- **El modo en vivo no guarda video**, solo el CSV. Deliberado.
- **Licencia de PySide6: LGPL/GPL.** El repositorio es MIT y las demás
  dependencias son permisivas (Apache, BSD, MIT), pero PySide6 es
  LGPL-3.0/GPL. No afecta al uso académico ni a la distribución del código
  fuente, pero conviene saberlo si alguna vez se distribuye un binario
  empaquetado. La interfaz es opcional: el núcleo y el modo headless no
  dependen de ella.

---

---

## Lo que la ejecución enseñó

Las correcciones no salieron como se planearon. Vale la pena registrar en qué
se desviaron, porque el patrón se repitió y volverá a repetirse.

### Cuatro hallazgos cambiaron al medirlos

**M7a cambió de solución.** La propuesta era derivar el intervalo del
temporizador del fps de la cámara. Al medir los registros reales apareció que
el cuello de botella no era el temporizador sino el procesamiento — 80 ms
logrados contra 33 ms pedidos — así que aquella propuesta habría **empeorado**
el encolado. Se implementó auto-programación en su lugar.

**M3 se implementó con la justificación invertida.** Se había argumentado que
mezclar etapas de suavizado "hereda el ruido de la entrada más ruidosa". La
simulación mostró algo más preciso: suavizar solo el denominador aporta un
16 %, filtrar la salida un 80 %. La mezcla no empeoraba nada — casi no servía.
Y el filtro, que se pensaba dejar apagado por defecto, quedó **encendido**: sin
él la etiqueta de estado cambia 7.1 veces más con un sujeto quieto.

**M5 resultó no ser un hueco de cobertura.** El recorte del coseno viene de la
guía de implementación, y con la formulación 2D el desbordamiento que previene
es inalcanzable — medido sobre 400 000 casos. Ninguna mutación podía cazarlo
porque no hay comportamiento que cambiar.

**El EMA estuvo a punto de eliminarse por una medición mal hecha.** Una primera
medición concluyó que no reducía el ruido; usaba como métrica la dispersión
total de V, que incluye el movimiento real del sujeto. Con la métrica correcta
—la componente de alta frecuencia— resultó que reduce el temblor un 63 % ⚠.

### El patrón: sospechar del instrumento antes que del sistema

Durante la ejecución, **cinco mutaciones salieron "no detectadas"**. Al
investigarlas una por una:

| Caso | Diagnóstico real |
|---|---|
| M3 — mezclar etapas | mutación inválida: pasaba `dt=0`, que es pasa-todo por diseño |
| M3 — segundo intento | efecto real de solo 3.2 %, por debajo del umbral de etiqueta |
| Lote B — reset de suavizadores | código redundante: el `dt=0` ya resiembra el filtro |
| M1 — `extrasaction` | mutación inválida: reemplazó el texto **de un comentario** |
| Metadatos — versión del código | **la única debilidad real**: `assertIn` solo comprobaba la clave, no su valor |

Cuatro de cinco eran defectos del instrumento de medición, no de los tests.
**Regla que queda:** cuando una mutación no se detecta, primero verificar que
de verdad cambió el comportamiento; solo después dudar de la cobertura.

### Dos refactorizaciones que no estaban en el plan

Separar `FramePipeline.analyze()` de `process()` —con creación perezosa del
front-end de pose— fue necesario para poder simular M3 sin MediaPipe. De paso
**resolvió M6**, la brecha estructural de cobertura, y habilitó todos los tests
de los lotes B y C.

Exponer `current_index` en la fuente (M2) no era cosmético: era la condición
que produjo el desfase del paso frame a frame (C3). Corregir el síntoma sin la
causa habría dejado la trampa puesta.

---

## Lotes de trabajo y orden de ejecución

Los 21 hallazgos se reagruparon por **invariante violado** en lugar de por
severidad. La agrupación se derivó dos veces de forma independiente —una por
archivo tocado, otra por invariante— y los lotes donde ambos criterios
coinciden son los más sólidos. La reagrupación reveló que varios hallazgos
listados por separado son en realidad **una misma decisión de diseño**, y que
dos de ellos están causalmente encadenados.

### Lote A — Tiempo y suavizado

**Contiene:** C5, M7a (temporizador fijo de cámara), M3
**Invariante:** el sistema respeta el ritmo real de la fuente en lugar de
imponer el suyo.

**Va primero, y es obligatorio:** cambia la firma del estimador de velocidad.
Hacerlo después obligaría a reescribir los tests de todos los demás lotes.

### Lote B — Discontinuidad

**Contiene:** C1, C2
**Invariante:** la velocidad refleja el movimiento de un cuerpo continuo.

No son dos defectos sino **un mecanismo con dos disparadores** (pérdida de
detección y salto manual). Corregirlos por separado replicaría la misma lógica
en dos sitios distintos, que es exactamente como nació el problema. Los tests
se escriben antes de la corrección, para que quede demostrada y no asumida.

### Lote C — Integridad del registro

**Contiene:** M1, M4, M8, la columna de fps de C5, y C4
**Invariante:** el CSV es un registro fiel e interpretable sin contexto
externo.

**Depende de A** (para conocer el fps) **y de B** (para distinguir los frames
que vienen de un salto).

Beneficio adicional: escribir al abrir cada registro una cabecera con la
instantánea de la configuración, el fps de la fuente y la versión del código
resuelve los cinco puntos **y** entrega la trazabilidad que exige el protocolo
de calibración (`calib-v1`). El método `Config.as_dict()` ya existe para esto.

### Lote D — Mecánica de acceso

**Contiene:** M2 → C3 (en ese orden, por el vínculo causal), M9, M7b
(precedencia silenciosa del CLI, en `main.py`)
**Invariante:** cada módulo expone su propio estado; nadie lo reconstruye
desde fuera.

Independiente de los demás; puede ejecutarse en cualquier momento.

### Sueltos

Test del recorte del coseno (M5) y relleno de los docstrings faltantes.

### Dependencias

```
A ──> B ──> C            D  (independiente)
```

El orden A → B → C es obligatorio por dependencias reales de código, no por
preferencia.

### Fuera de este ciclo

- **F1 + F2 son un solo trabajo de Fase 7**, no dos: el modo headless en vivo
  y sacar el procesamiento del hilo de la interfaz se resuelven con el mismo
  diseño — un bucle de procesamiento independiente de Qt.
- **P1 + P3 + P4 son una sola sesión de edición del draft** (§3.2–3.4:
  centroide indefinido, EMA ausente, normalización 2D frente a 3D).
  **P2** es un error factual aparte, se corrige en un párrafo.
  **P5** no puede tocarse todavía: reformular la Etapa 1 requiere la Fase 3
  implementada para respaldarse con números propios.

---

*Cinco pasadas, cinco técnicas distintas. Los rendimientos son decrecientes en
el software —la quinta no encontró defectos nuevos de código, solo confirmó
propiedades— pero abrió el frente editorial: las secciones 3.5 a 3.7 del draft
aún no han sido auditadas contra ninguna implementación.*

*La agrupación en lotes se derivó dos veces de forma independiente. La segunda
derivación corrigió tres clasificaciones (C4 y M3 cambiaron de lote, M7b de
archivo) y descubrió el vínculo causal entre M2 y C3, que las dos revisiones
anteriores habían tratado como hallazgos sin relación.*

*Ejecución cerrada el 14 de agosto de 2026: 19 de 21 hallazgos corregidos, la
suite de 39 a 73 tests, y cuatro conclusiones revisadas al medirlas. Lo que
queda son F1 y F2 —un solo trabajo de la Fase 7— y las cinco divergencias con
el draft, que se resuelven escribiendo, no programando.*

---

# Re-verificación (2026-08-15)

La pregunta que la motivó: *"¿realizó todas las pruebas que hizo para escribir
esta revisión?"* Se volvió a ejecutar todo desde cero. La respuesta corta es
sí, pero con una distinción que importa más que el sí.

## Reproduce exacto

| Afirmación | Al re-ejecutar |
|---|---|
| Suite completa | 73 tests OK (67 antes de la convención `_EXP`) |
| Determinismo del registro | dos corridas del mismo clip: 248 filas, mismo md5 |
| Docstrings 47/53 = 89 % | idéntico, y las 6 faltantes son exactamente las implementaciones triviales de `sources.py` (`fps`/`read`/`release` ×2) |
| Recorte del coseno inalcanzable | 0 desbordamientos en 400 000 troncos casi verticales |
| Mutaciones | 5 de 5 válidas detectadas |

## NO reproduce el decimal exacto ⚠

| | En este documento | Al re-medir |
|---|---|---|
| Fórmula general desborda | 13 % | 22 % |
| EMA quita temblor | 63 % | 77.5 % |
| EMA atenúa el pico | 4 % | 21.7 % |
| M3: denominador / salida | 16 % / 80 % | 23 % / 59 % |
| C5: dispersión frames → segundos | 138 % → 5.7 % | 152.6 % → 18.7 % |

**Causa.** Los scripts ad-hoc que produjeron esas cifras no se guardaron. Cada
una depende del ruido sintético y de la forma de caída que se simuló, y esos
parámetros no quedaron escritos en ningún lado. Además, dos de ellas —el 63 %
del EMA y parte de la dispersión de C5— se midieron sobre **metraje real**, y
un script sintético no puede reproducirlas por construcción: el ruido de
MediaPipe no es gaussiano y una caída real no es medio coseno.

**Qué se sostiene y qué no.** Ninguna conclusión cambia de dirección: el EMA sí
quita la mayor parte del temblor, filtrar la salida sí es muy superior a
filtrar solo el denominador, la ventana en segundos sí colapsa la dispersión, y
la fórmula general sí desborda mientras la nuestra no. Lo que no se sostiene es
la **precisión** con que este documento las citaba. Un número que nadie puede
volver a producir no es evidencia, es una afirmación — y en un paper eso es
justo lo que un revisor pide primero.

**Corrección aplicada.** Existe [`verificar_mediciones.py`](verificar_mediciones.py):
`python notas/verificar_mediciones.py` regenera cada cifra e imprime al lado el
valor documentado, para que la comparación sea explícita. Su encabezado advierte
que las mediciones son sintéticas y que las de metraje real deben re-medirse
sobre la mini-suite —corriendo el pipeline con `ema_time_constant_s: 0.0` y con
`0.1`— cuando exista. Hasta entonces, las dos cifras de metraje real quedan
marcadas como pendientes, no como establecidas.

## La trampa de las mutaciones, comprobada por segunda vez

Al repetir las mutaciones, la de `extrasaction="raise"` salió **"no
detectada"**. Aplicando la regla que este mismo documento dejó escrita —*primero
verificar que la mutación de verdad cambió el comportamiento*— apareció el
motivo: el reemplazo había tocado la **primera** aparición del texto, que está
en un comentario, no en el código. Aplicada sobre el código real, la suite la
cazó de inmediato.

Es exactamente el error catalogado en "Lo que la ejecución enseñó", cometido de
nuevo sin querer y por quien lo había escrito. Vale como confirmación de que la
regla no es teórica: **si una mutación no se detecta, el sospechoso número uno
es el instrumento, no la cobertura.**

## Hallazgo nuevo, abierto

`extension_ratio_EXP` sale **negativo** en dos clips de caída del dataset
público (−0.40 y −0.17). El cálculo es `(y_tobillo − y_cadera) / torso`, una
diferencia **con signo**: negativo significa tobillos por encima de las
caderas, físicamente posible a media caída, así que el dato probablemente es
correcto. Lo que no está claro es su consumo: `classify_state` evalúa
`extension_ratio < crouch_extension`, de modo que manda cualquier negativo a la
categoría más agachada cuando la persona está más bien invertida. **No se
clasifica como defecto todavía porque no se han visto los frames**; queda como
punto 3 de la sub-parte 2.6.
