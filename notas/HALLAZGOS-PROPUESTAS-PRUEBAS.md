# Hallazgos, propuestas y catálogo de pruebas

**Proyecto:** PEF — *Beyond Black-Box AI: A Physics-Informed Edge Framework for
Explainable Markerless Fall Detection Using MediaPipe Landmarks* (ULACIT)
**Fecha:** 14 de agosto de 2026 · *actualizado 15 de agosto de 2026*
**Estado del software:** Fase 2 con código completo (cantidades T y V, estados,
curvas), revisión de código cerrada, 73 tests; Fase 3 (lógica de decisión) aún
no implementada. Falta la sub-parte 2.6 (calibración), bloqueada por la
mini-suite de clips.

**Naturaleza de este documento.** Tiene tres partes:

1. **Hallazgos** (§1–§7): lo que los datos reales ya dijeron, con sus cifras.
2. **Propuestas** (§8): qué hacer al respecto, en orden de prioridad.
3. **Catálogo de pruebas** (§9): el menú de verificaciones que dan evidencia
   de calidad, con su estado y el momento en que cada una puede ejecutarse.

**Nada de lo propuesto está implementado todavía**; el documento existe para
discutirlo en equipo antes de decidir. El catálogo de pruebas se mantiene como
lista viva: una prueba anotada ahí no se pierde en el camino, y su estado dice
si ya dio evidencia o si sigue pendiente.

> **Actualización del 15 de agosto de 2026 — dos cambios que afectan la lectura
> de este documento.**
>
> **1. Las columnas experimentales ahora se llaman así en el CSV.** Los campos
> que este documento discute como candidatos —velocidad horizontal, torso
> métrico, razón de extensión, etiqueta de estado— llevan sufijo `_EXP` en el
> registro: `Vh_tps_EXP`, `world_torso_len_m_EXP`, `extension_ratio_EXP`,
> `state_EXP`. El motivo es exactamente el asunto de este documento: un CSV que
> mezcla cantidades del paper con señales bajo evaluación, sin distinguirlas,
> invita a leer una columna exploratoria como resultado establecido. Cuando una
> de ellas se gradúe al paper —con la evidencia que se discute aquí— pierde el
> sufijo. Las columnas están documentadas en el `README.md`.
>
> **2. Varias cifras de `REVISION-CODIGO.md` resultaron no reproducibles en su
> decimal exacto** al re-ejecutarlas, porque los scripts que las produjeron no
> se guardaron. Las conclusiones se sostienen; la precisión citada no. Afecta a
> este documento en un punto concreto: **el "63 %" del EMA** que aparece en §5 y
> §10 se midió sobre metraje real y queda **pendiente de re-medición** con la
> mini-suite. Ver la sección "Re-verificación" de `REVISION-CODIGO.md`.

---

## 0. De dónde salen estos datos

Se procesaron **8 registros CSV** generados por el detector, uno por corrida:

| Tipo | Fuente | Cantidad | Duración |
|---|---|---|---|
| Caídas | Clips de un dataset público descargado | 6 clips | 1.8 s – 11 s |
| Actividad normal (ADL) | Cámara en vivo, sujeto propio | 2 sesiones | 41 s y 24 s |

Cada CSV contiene una fila por frame con las cantidades físicas calculadas
(§3.4), la visibilidad de los landmarks y una bandera de confiabilidad. Todas
las estadísticas de este documento usan **solo frames confiables** (persona
detectada y landmarks de hombros/caderas por encima del umbral de
visibilidad).

Advertencia metodológica: la muestra es pequeña y las caídas provienen de
clips actuados de un dataset público, no de PEF-FallDB. Las conclusiones son
**indicios sólidos que justifican decisiones de diseño**, no resultados
publicables. La validación formal corresponde a la Fase 6 con el dataset
propio.

---

## 1. Lo que los datos confirman del paper

### 1.1 La cantidad V separa caídas de actividad normal

| Población | V mínima observada (torsos/segundo) |
|---|---|
| 6 clips de caída | de **−2.82** a **−5.19** |
| 66 s de actividad normal en vivo | nunca por debajo de **−1.28** |

No hay solapamiento entre ambas poblaciones. La afirmación del §3.4 —que la
velocidad vertical del centroide captura la firma cinemática de una caída— se
sostiene con datos reales. El umbral de referencia de la guía (−1.5 torsos/s)
cae dentro del margen limpio que separa ambos grupos.

Detalle de cada clip de caída (frames que superan cada umbral candidato):

| Clip | V mínima | frames < −1.5 | < −2.0 | < −2.5 |
|---|---|---|---|---|
| 20240912_101331 | −4.22 | 13 | 12 | 9 |
| 20240914130701 | −5.19 | 17 | 16 | 14 |
| 20240915183518 | −4.64 | 12 | 10 | 9 |
| S_N_568 | −3.19 | 39 | 29 | 16 |
| video__1_ (a) | −2.82 | 11 | 7 | 3 |
| video__1_ (b) | −3.24 | 21 | 16 | 4 |
| **ADL en vivo (2 sesiones)** | **−1.28** | **0** | **0** | **0** |

### 1.2 La firma de estados coincide con lo descrito en el §3.3

Las seis caídas muestran la misma secuencia alrededor del pico de velocidad:
`STANDING` o `WALKING` → `TRANSITION` sostenido → `LYING` (o `CROUCHING` en el
clip donde el sujeto queda recogido). Es la trayectoria postural que el paper
describe, observable directamente en el registro.

### 1.3 La auditabilidad prometida en el §3.5 es real

El evento se puede reconstruir leyendo el CSV, sin depender de ninguna caja
negra. Extracto textual de una caída (`20240915183518`):

| frame | T (grados) | V (torsos/s) | Vh (torsos/s) | estado |
|---|---|---|---|---|
| 24 | 28.7 | −0.69 | −0.48 | LEANING |
| 28 | 32.1 | −0.84 | −1.04 | TRANSITION |
| 31 | 38.4 | −1.25 | −1.57 | TRANSITION |
| 34 | 44.3 | −2.20 | −2.14 | TRANSITION |

El tronco se abre de 28° a 44° mientras la velocidad de descenso se triplica.
Cualquier revisor clínico puede seguir el razonamiento.

---

## 2. Hallazgo crítico: la Etapa 1 de la guía pierde caídas reales

### 2.1 El problema

La guía de implementación define el disparador de la Etapa 1 como una
**conjunción instantánea**:

```
disparar  si  T > 45°  Y  V < −1.5   (en el mismo frame)
```

Evaluada contra los seis clips de caída:

| Clip | frames con T>45 | frames con V<−1.5 | **ambos a la vez** | ¿dispararía? |
|---|---|---|---|---|
| 20240912_101331 | 12 | 13 | 12 | sí |
| 20240914130701 | 0 | 17 | **0** | **NO** |
| 20240915183518 | 21 | 12 | 10 | sí |
| S_N_568 | 43 | 39 | 21 | sí |
| video__1_ (a) | 108 | 11 | **2** | **no** (1) |
| video__1_ (b) | 90 | 21 | **2** | **no** (1) |

(1) La configuración exige que el disparo se sostenga 3 frames consecutivos
(histéresis, `stage1.min_consecutive_frames`) para evitar alarmas por ruido.
Con solo 2 frames coincidentes, estos clips no alcanzan el disparo.

**Resultado: 3 de 6 caídas reales no se detectarían.**

### 2.2 La causa: T y V no ocurren al mismo tiempo

Desfase medido entre el instante de V mínima y el de T máxima:

| Clip | pico de V | pico de T | desfase |
|---|---|---|---|
| 20240912_101331 | 1.73 s | 1.77 s | 0.03 s |
| 20240914130701 | 1.63 s | 1.83 s | 0.20 s |
| 20240915183518 | 1.33 s | 1.63 s | 0.30 s |
| video__1_ (a) | 6.37 s | 6.92 s | 0.54 s |
| S_N_568 | 1.70 s | 2.40 s | 0.70 s |
| video__1_ (b) | 7.40 s | 2.32 s | 5.08 s |

El desfase es **sistemático y físicamente esperable**: primero el cuerpo cae
(pico de velocidad), y *después* termina de rotar contra el suelo (pico de
inclinación). Exigir que ambos máximos coincidan en un mismo frame equivale a
exigir que dos eventos secuenciales sean simultáneos.

Caso especial, el clip `20240914130701`: su tronco **nunca superó los 39°**
pese a tener la V más negativa de toda la muestra (−5.19). Corresponde a un
desplome con el torso relativamente erguido —el arquetipo *síncope* que el
propio §3.3 exige incluir en PEF-FallDB—, o bien a una caída en dirección de
la cámara (ver §4.2). En cualquiera de los dos casos, la regla actual es
ciega a él.

### 2.3 El contraste

Si el disparador fuera **únicamente** `V < −1.5`:

- Detecta **6 de 6** caídas.
- Produce **0 falsos positivos** en 66 segundos de actividad normal
  (caminar, inclinarse, sentarse, acostarse intencionalmente).
- Con umbral −2.0 mantiene 6 de 6 con mayor margen de seguridad.

### 2.4 Propuesta

Reformular la Etapa 1 como una **secuencia**, no como una conjunción:

1. **Disparo por V**: la velocidad vertical cruza el umbral (evento rápido).
2. **Confirmación por T dentro de una ventana posterior**: se exige que la
   inclinación del tronco alcance el rango de colapso en los siguientes
   *N* frames tras el disparo.

Esto conserva las cuatro cantidades del §3.4, mantiene intacta la
explicabilidad (la alerta sigue justificándose con valores físicos con nombre)
y elimina la exigencia de simultaneidad que la física no cumple.

**Implicación para el paper:** el §3.5 debe actualizarse. La reformulación no
debilita el aporte, lo refuerza: aporta un resultado propio con evidencia —
*la conjunción instantánea usada en la literatura previa pierde las caídas
verticales; la formulación secuencial las recupera sin costo en falsos
positivos*. Es exactamente el tipo de hallazgo que justifica un marco
físicamente fundamentado frente al ajuste empírico de umbrales que el §2.5
critica.

---

## 3. Segundo hallazgo: velocidades físicamente imposibles por salto de esqueleto

En el clip `S_N_568` —el único con **dos personas en escena**— se registró una
velocidad de **+26.6 torsos/segundo**. Es un valor imposible (un cuerpo en
caída libre no supera el orden de 10 torsos/s).

**Causa:** el backend de pose empleado sigue a una sola persona. Cuando el
sujeto cae, el rastreador se re-engancha a la persona que sigue de pie; el
centroide "se teletransporta" de un cuerpo a otro y la derivada temporal
fabrica una velocidad enorme. La protección existente contra huecos no lo
detecta, porque **la detección nunca se pierde**: el esqueleto salta sin
interrupción.

**Propuesta:** añadir un **límite de plausibilidad física**. Toda velocidad
cuya magnitud supere un máximo biomecánico (orden de 8–10 torsos/s) debe
marcarse como no confiable en lugar de propagarse. Sin esta guarda, la Etapa 1
disparará por artefactos de rastreo y no por caídas — un falso positivo
particularmente difícil de explicar ante un auditor clínico.

**Nota sobre multi-persona:** resolver el problema de raíz exigiría migrar a
una API multi-pose, construir asociación de identidad entre frames y duplicar
el estado del pipeline por persona. Se recomienda **no hacerlo ahora**: el
paper declara explícitamente el escenario mono-persona (§3.2) y sitúa
multi-persona como trabajo futuro (Sección 6). Los *bystanders* previstos en
el §3.3 sirven precisamente para **medir** esta limitación, que es material
legítimo para la Sección 5. Si en el futuro se implementa, el registro debe
adoptar formato largo (una fila por persona por frame, con columna
`person_id`), nunca columnas ensanchadas por persona.

---

## 4. La componente de profundidad (z): situación real y oportunidad

### 4.1 Qué se está registrando hoy

**No hay ninguna columna de z en el registro.** La estimación de profundidad
se calcula durante el procesamiento y se descarta al escribir la fila. Lo
único derivado de 3D que sobrevive es `world_torso_len_m`, que es una
*longitud*, no una posición.

Restricción importante que conviene conocer antes de decidir: las coordenadas
3D métricas de MediaPipe están **centradas en la cadera** — el punto medio de
caderas es el origen en cada frame. Por lo tanto **no permiten saber si la
persona se acercó o se alejó de la cámara**: describen la postura del cuerpo
en 3D, no su posición en la habitación. El único indicio de traslación en
profundidad disponible es indirecto (el torso en píxeles crece al acercarse).

Lo que la profundidad **sí** puede aportar es la componente **sagital** de la
postura: si el tronco se inclina hacia la cámara o alejándose de ella.

### 4.2 Hipótesis a verificar

El clip `20240914130701` es el que falló la Etapa 1 con un tronco que nunca
superó los 39°, pese a la velocidad más negativa de la muestra. Una
explicación plausible es que el sujeto **cae en la dirección del eje de la
cámara**. En esa dirección el ángulo T calculado en 2D es **estructuralmente
ciego**: el tronco no rota dentro del plano de la imagen, se escorza.

Un ángulo de tronco calculado en 3D capturaría esa rotación. Si la hipótesis
se confirma, se obtienen dos resultados: una explicación física de por qué la
conjunción de la guía falla, y evidencia para incorporar la profundidad al
paper de forma justificada.

**Para verificarla se requiere:** registrar dos columnas nuevas — ángulo de
tronco en 3D y ángulo sagital (hacia/desde la cámara) — y volver a procesar
los seis clips de caída.

### 4.3 Efecto de escorzo detectado

Durante una caída medida, el torso en píxeles se acortó un **8 %** (247 → 228
px) mientras el torso métrico varió solo un **4 %** (0.468 → 0.447 m). Como la
velocidad V se normaliza dividiendo por el torso en píxeles, una unidad que se
encoge **infla ligeramente las velocidades justo durante las caídas**.

No es un error: es geometría de la proyección. Pero conviene tenerlo presente
al fijar umbrales, y abre una línea de mejora: normalizar contra el torso
métrico podría dar una V menos sesgada. Sería, además, un uso justificado de
la información 3D dentro del marco monocular del paper.

---

## 5. El filtro de suavizado (EMA): lo medido y lo que falta por medir

La velocidad V se calcula sobre un centroide previamente suavizado con un
filtro exponencial. Como el paper no lo menciona (ver §4 de este documento y
P3 en `REVISION-CODIGO.md`), conviene tener medido qué aporta y qué cuesta.

### 5.1 Lo que aporta, medido

Aislando la componente de alta frecuencia de la señal —la desviación de cada
posición respecto a la recta que trazan sus vecinas inmediatas, que vale ~0
para movimiento suave por rápido que sea— el filtro reduce el temblor de los
landmarks de forma consistente:

| Clip | Temblor sin filtro | Con filtro | Reducción |
|---|---|---|---|
| 20240912_101331 | 0.38 px | 0.25 px | 34 % |
| 20240914130701 | 0.28 px | 0.09 px | 69 % |
| 20240915183518 | 0.78 px | 0.37 px | 53 % |
| S_N_568 | 0.25 px | 0.11 px | 58 % |
| ADL en vivo (a) | 0.25 px | 0.07 px | 72 % |
| ADL en vivo (b) | 1.06 px | 0.39 px | 63 % |
| video__1_ (a) | 0.08 px | 0.02 px | 75 % |
| video__1_ (b) | 0.35 px | 0.11 px | 69 % |

**Reducción mediana del 63 %.** El costo: el pico de velocidad en las caídas
se atenúa un **4 %** y se introducen unos **0.1 s de latencia** de detección.

### 5.2 Por qué en material de buena calidad parece prescindible

En todos los clips disponibles el temblor absoluto es de 0.25 a 1 píxel, es
decir entre **0.001 y 0.003 longitudes de torso**. Frente a un pico de caída
del orden de 4 torsos/segundo, eso es despreciable: el filtro está limpiando
un ruido que no llegaba a molestar. En condiciones buenas, el sistema
funcionaría igual sin él.

### 5.3 Lo que no se puede decidir con los datos actuales

La hipótesis relevante es que el temblor crece cuando la calidad de imagen se
degrada (poca luz, desenfoque de movimiento, oclusión parcial), y que ahí el
filtro sí sería determinante. **Con el material disponible esa hipótesis no es
verificable**: la visibilidad de los landmarks en los ocho registros va de
0.980 a 0.999 — todos son condiciones favorables, no hay variación de calidad
que analizar. Cualquier correlación que se calcule sobre ese rango es ruido de
medición.

### 5.4 Experimento propuesto con PEF-FallDB

El §3.3 exige grabar en condiciones diurnas, con luz artificial y en
nocturno/baja luz, además de sujetos parcialmente visibles. Ese diseño
proporciona exactamente el contraste que falta. Con el dataset disponible:

1. **Medir el temblor por condición de iluminación y por grado de oclusión**,
   con la métrica de alta frecuencia descrita en 5.1. Se espera que crezca al
   degradarse la imagen; si no crece, el filtro deja de justificarse.
2. **Calibrar la constante de tiempo del filtro con los clips difíciles, no
   con los fáciles.** Calibrarla sobre material de buena calidad daría un
   valor engañosamente bajo, porque ahí el filtro parece sobrar.
3. **Cuantificar el compromiso** entre reducción de temblor y latencia de
   detección en cada condición, para reportarlo en la Sección 4.
4. **Evaluar suavizado adaptativo** (opcional): el sistema ya calcula la
   visibilidad de los landmarks en cada frame, así que el filtro podría
   endurecerse cuando la detección es dudosa y aligerarse cuando es nítida.
   Daría latencia mínima en condiciones buenas y robustez en las malas, que
   es precisamente el doble objetivo del paper. No se implementa hasta tener
   los clips difíciles con los que calibrarlo.

### 5.5 Advertencia metodológica

Una primera medición de este mismo asunto arrojó la conclusión opuesta —que
el filtro no reducía el ruido— por usar como métrica la dispersión de V
durante actividad normal. **Esa métrica es incorrecta**: cuando el sujeto
camina o se inclina, V varía legítimamente, de modo que estaba midiendo
señal real y no temblor. Queda registrado para que la medición no se repita
con el mismo error: el temblor debe aislarse por su componente de alta
frecuencia, nunca por la dispersión total de la señal.

---

## 6. Elementos posiblemente redundantes

El balance es favorable: sobra poco.

- **La `z` de imagen** de los landmarks se calcula y se guarda en memoria pero
  no la consume ningún cálculo, y ya se sabe que la estimación métrica es
  superior. Es peso muerto.
- **Columnas del punto medio de hombros y del centroide** son parcialmente
  redundantes entre sí (el centroide se deriva de los puntos medios). Se
  recomienda **conservarlas**: cuestan pocos bytes y permiten auditar el
  filtrado de suavizado a posteriori.
- **`extension_ratio`** (rasgo de postura basado en la extensión
  cadera–tobillo) aparece **vacío en todos los clips del dataset** porque los
  tobillos no alcanzan el umbral de visibilidad. No sobra, pero implica que
  las etiquetas de estado operan en modo degradado —solo tronco— hasta que se
  grabe con cuerpo completo. Es una razón adicional para que PEF-FallDB
  encuadre el cuerpo entero.

---

## 7. Observación sobre el dataset público empleado

Los clips descargados duran entre 1.8 y 11 segundos y **terminan en el momento
del impacto o poco después**. Con ellos es imposible medir la cantidad I
(duración de inmovilidad post-caída) ni evaluar la etiqueta de severidad.

Esto constituye **respaldo empírico directo a la crítica del §3.3**: los
corpus públicos no preservan la continuidad post-evento, y por eso PEF-FallDB
—con ventanas de 8–12 s y coreografías de recuperación— es necesario y no un
capricho metodológico. Es un argumento verificable que puede citarse en el
paper con datos propios.

---

## 8. Trabajo propuesto, en orden de prioridad

### Análisis (no requieren escribir código nuevo del sistema)

1. **Barrido de umbrales** sobre los 8 registros: recorrer V de −1.0 a −3.0 y
   medir detección frente a falsos positivos en cada valor, para elegir el
   umbral con evidencia en lugar de por inspección visual.
2. **Cuantificación del desfase T–V** en las seis caídas, para proponer con un
   número la ventana de confirmación de la Etapa 1 reformulada.
3. **Curvas superpuestas** de T(t), V(t) y Vh(t): las seis caídas contra las
   dos sesiones de actividad normal. Es material directo para la Sección 4.
4. **Aporte de la velocidad horizontal**: verificar si separa caídas laterales
   que la componente vertical no separa. En la caída examinada, la componente
   horizontal alcanzó una magnitud comparable a la vertical (−2.14 frente a
   −2.20), indicio de que aporta información.

### Análisis que requieren PEF-FallDB (no pueden hacerse antes)

5. **Utilidad del filtro de suavizado en condiciones degradadas** (§5.4):
   medir el temblor por condición de iluminación y oclusión, calibrar la
   constante de tiempo con los clips difíciles, cuantificar el compromiso
   entre reducción de ruido y latencia, y evaluar suavizado adaptativo según
   la visibilidad. Es el experimento que decide si el filtro se mantiene, se
   endurece o se elimina.

### Implementación

5. **Registrar la profundidad**: ángulo de tronco 3D y ángulo sagital, para
   poner a prueba la hipótesis del §4.2. Requiere además volver a procesar los
   clips (los videos deben estar accesibles al entorno de trabajo).
6. **Guarda de plausibilidad física** en la velocidad (§3).
7. **Reformulación secuencial de la Etapa 1** (§2.4), con la ventana de
   confirmación derivada del análisis 2.
8. **Modo de comparación por lotes** para procesar carpetas de clips y emitir
   métricas automáticamente (anticipo de la Fase 6).

---

## 9. Catálogo de pruebas

Lista viva de las verificaciones que dan evidencia de calidad del software.
Existe para que ninguna se pierda en el camino: cada entrada dice qué
verifica, cómo se ejecuta hoy y cómo se ejecutaría con hardware real, y en
qué estado está.

### 9.1 El principio: sintética hoy, real después

Casi toda prueba tiene dos versiones. La **sintética** se construye con datos
fabricados y puede correr hoy, en segundos, sin equipo ni sujetos: aísla la
lógica y prueba que la matemática es correcta. La **real** usa grabaciones o
hardware, tarda más, exige coordinación, y es la única que demuestra que el
sistema funciona en el mundo.

Las dos son necesarias y ninguna sustituye a la otra: la sintética atrapa
errores de implementación con precisión quirúrgica, la real atrapa supuestos
falsos sobre la realidad. El caso que originó este catálogo lo ilustra bien:

> **Invariancia al frame rate (P-01).** *Versión sintética:* simular la misma
> caída fabricada a 60, 30 y 15 fps y exigir que V dé aproximadamente lo
> mismo. Corre en milisegundos y hoy fallaría con más del doble de
> diferencia. *Versión real:* grabar una misma caída actuada con cámaras
> configuradas a distintos frame rates, o remuestrear un mismo clip, y
> comprobar la invariancia sobre imágenes verdaderas — con el ruido de
> landmarks, el desenfoque de movimiento y el escorzo que ninguna simulación
> reproduce.

**Convención de estado:** ✅ ejecutada y con resultado · 🔶 ejecutable ya, no
ejecutada · ⏳ bloqueada esperando datos o hardware.

### 9.2 Pruebas unitarias — sintéticas, ejecutables hoy

Corren en segundos, sin cámara ni video. Son la primera línea de defensa.

| ID | Qué verifica | Estado |
|---|---|---|
| P-01 | **Invariancia al frame rate**: misma caída a 60/30/15 fps → misma V | ✅ dispersión 138 % → **5.7 %** |
| P-02 | **Micro-hueco**: pérdida de un solo frame + salto de esqueleto no debe producir V imposible | ✅ el test reproducía +10.0; tras C1, la ventana se reinicia |
| P-03 | **Reinicio en salto**: saltar en el video reinicia la ventana de velocidad | ✅ `reset()` público, llamado desde la GUI |
| P-04 | **Paso frame a frame**: avanzar uno avanza exactamente uno | ✅ verificado en la GUI real: 49→50→49→48 |
| P-05 | **Sin duplicados en el registro**: hacer scrub no repite filas | ✅ tras C4: scrub agresivo → 40 filas, 0 duplicadas, monótonas |
| P-06 | **Invariantes del ángulo T** (prueba de propiedad): nunca NaN, siempre en [0,180], a cualquier escala | ✅ 40 000 casos generados. El recorte no es alcanzable con esta fórmula (medido: 0 desbordamientos); protege un cambio futuro a 3D o a la fórmula general, donde el 13 % de los pares paralelos sí desborda |
| P-07 | **Esquema del registro**: una clave mal escrita debe fallar ruidosamente | ✅ `extrasaction="raise"` + test; mutación detectada |
| P-08 | **Plausibilidad física**: \|V\| por encima del límite biomecánico se marca no confiable | ⏳ propuesta, sin implementar |
| P-09 | **Cantidades sobre esqueletos sintéticos**: T a 0/45/90/180°, simetría, casos degenerados | ✅ 16 tests pasando |
| P-10 | **Paso 0**: longitud de torso invariante a rotación; normalización adimensional | ✅ 4 tests pasando |
| P-11 | **Velocidad y estados**: caída sintética, guardia de huecos, EMA, 8 etiquetas | ✅ 19 tests pasando |
| P-12 | **Suavizado de la razón de extensión (M3)**: filtrar la salida en vez de mezclar etapas; oclusión no contamina el filtro; parpadeo de etiqueta; invariancia a la distancia | ✅ 8 tests; parpadeo −7.1×; sesgo por mezcla de etapas 3.2 % detectado |
| P-14 | **Registro auto-descriptivo**: cada CSV lleva su `.meta.json` con fuente, fps, versión del código y la configuración completa | ✅ 4 tests; da la trazabilidad de `calib-v1` |
| P-15 | **Creación perezosa del registro**: abrir y cerrar sin grabar no deja archivos vacíos | ✅ 2 tests |
| P-16 | **Ruta de salida anclada**: una ruta relativa no depende del directorio de invocación | ✅ 2 tests |
| P-13 | **Histéresis en la etiqueta de estado**: un valor exactamente sobre el umbral no debe alternar; el suavizado reduce el parpadeo pero no lo elimina | ⏳ propuesta nueva, sin implementar |

### 9.3 Pruebas sobre datos grabados

Requieren clips. Las que ya se hicieron usaron material del dataset público y
sesiones propias; las pendientes esperan la mini-suite o PEF-FallDB.

| ID | Qué verifica | Estado |
|---|---|---|
| P-20 | **Determinismo**: procesar el mismo video dos veces da CSV idénticos | ✅ 248 filas, 0 diferencias |
| P-21 | **Separabilidad**: caídas y actividad normal se separan en el plano (T, V) | ✅ sin solapamiento; margen 12 % |
| P-22 | **Barrido de umbrales**: detección frente a falsos positivos para V de −1.0 a −3.0 | 🔶 datos disponibles |
| P-23 | **Desfase T–V**: cuantificar la separación entre picos para dimensionar la ventana de confirmación | ✅ 0.03 s a 5 s; motiva reformular la Etapa 1 |
| P-24 | **Aporte de la velocidad horizontal**: ¿separa caídas laterales que la vertical no separa? | 🔶 indicio fuerte (−2.14 vs −2.20) |
| P-25 | **Hipótesis de caída axial**: ¿el ángulo de tronco 3D recupera la caída que el 2D no ve? | ⏳ requiere registrar la z y reprocesar |
| P-26 | **Escorzo**: torso en píxeles frente a torso métrico durante la caída | ✅ 8 % contra 4 % |
| P-27 | **Estabilidad del Paso 0 con cuerpo completo**: torso métrico ~constante a cualquier distancia | ⏳ requiere clip de cuerpo completo |
| P-28 | **Etiquetas de estado con tobillos visibles**: ¿la etiqueta coincide con lo que hace la persona? | ⏳ requiere clip de cuerpo completo |
| P-29 | **Filtro de suavizado en condiciones degradadas**: temblor por iluminación y oclusión (§5.4) | ⏳ requiere PEF-FallDB |
| P-30 | **Auditoría de FPS**: fps declarado frente al efectivo en las grabaciones del video tool | ✅ ratio 0.99, sin desfase |

### 9.4 Pruebas de sistema

Verifican el conjunto, no las piezas.

| ID | Qué verifica | Estado |
|---|---|---|
| P-40 | **Ciclo de vida de la interfaz**: 18 maniobras hostiles sin fallo ni fuga | ✅ 18/18 |
| P-41 | **Robustez de entrada**: video vertical, 64×48, un solo frame, archivo corrupto | ✅ procesa o rechaza con error claro |
| P-42 | **Mutación**: la suite detecta regresiones introducidas a propósito | ✅ 9 de 10 acumuladas (el hueco es P-06); las 6 del Lote A, detectadas |
| P-43 | **Coherencia configuración ↔ código**: toda clave leída existe; toda clave escrita está en el esquema | ✅ sin desajustes |
| P-44 | **Cobertura de documentación**: funciones públicas con docstring | ✅ 78 % (39/50) |
| P-45 | **Equivalencia vivo ↔ grabado**: procesar en vivo y procesar la grabación del mismo evento deben coincidir | ⏳ requiere grabar y procesar en paralelo |
| P-46 | **Estabilidad en ejecución prolongada**: horas de operación sin fugas de memoria ni deriva | ⏳ para la beta |

### 9.5 Pruebas de hardware — la evidencia concreta

Son las que el §3.7 del paper exige reportar, y las que ninguna simulación
puede sustituir.

| ID | Qué verifica | Estado |
|---|---|---|
| P-60 | **Rendimiento por frame**: mediana, p95 y máximo | ✅ 24.7 ms / 36.4 ms en servidor a 1280×960 |
| P-61 | **Rendimiento real en la máquina del usuario**: cuadros por segundo efectivamente logrados | ✅ 12.5 fps con curvas activas — el procesamiento es el cuello de botella |
| P-62 | **Costo del dibujo**: cuánto de esa carga es detección y cuánto son las gráficas | 🔶 medible apagando las curvas |
| P-63 | **Efecto de la resolución**: detección y latencia a 1280×960 frente a 640×480 | ⏳ pendiente |
| P-64 | **Raspberry Pi**: fps sostenido, latencia extremo a extremo, CPU, memoria, consumo | ⏳ Fase 7; requerido por el §3.7 |
| P-65 | **Cámaras distintas**: ¿los umbrales calibrados transfieren entre modelos de cámara? | ⏳ para la beta |
| P-66 | **Iluminación real**: diurna, artificial y nocturna sobre el mismo escenario | ⏳ requiere PEF-FallDB |
| P-67 | **Despacho de alertas**: fiabilidad del zumbador, correo y SMS | ⏳ Fase 7 |

### 9.6 Pruebas de evaluación del paper (§3.7)

Producen las tablas de la Sección 4. Todas requieren PEF-FallDB.

| ID | Qué verifica | Estado |
|---|---|---|
| P-80 | **Métricas estándar**: matriz de confusión, exactitud, sensibilidad, especificidad, precisión, F1 | ⏳ |
| P-81 | **Falsos positivos por hora** de material sin caídas | ⏳ |
| P-82 | **Ablaciones**: quitar T, V, P o I de a una y medir la caída de desempeño | ⏳ |
| P-83 | **Ablación de la Etapa 3**: aumento de falsos positivos al quitar la confirmación por inmovilidad | ⏳ |
| P-84 | **Umbral único frente a tres etapas**: pérdida de precisión al colapsar la lógica | ⏳ |
| P-85 | **Exactitud de la etiqueta de severidad**: leve, moderada y severa frente a lo anotado | ⏳ |
| P-86 | **Generalización por holdout**: escenarios, cámaras e iluminaciones no usados al calibrar | ⏳ |
| P-87 | **Comparación con líneas base** de la literatura sobre el mismo material | ⏳ |

### 9.7 Estado del catálogo

De **49 pruebas** catalogadas: **27 ejecutadas** con resultado, **0
pendientes de ejecutar**, y **22 bloqueadas** esperando PEF-FallDB o la
Raspberry Pi. **La revisión de código está cerrada**: los 21 hallazgos
quedaron resueltos o diferidos explícitamente a la Fase 7 (F1 y F2).

**Pruebas agregadas el 15 de agosto (suite de 67 a 73):** seis que fijan la
convención `_EXP` en las dos direcciones —que ninguna columna experimental
quede sin marcar y que ninguna cantidad del paper quede marcada de más— más el
cruce entre las claves que emite el pipeline y el esquema del CSV. Esta última
tapó un hueco real: `extrasaction="raise"` ya rechazaba una clave desconocida,
pero solo al escribir una fila, es decir corriendo un video, nunca en la suite.
Una clave mal escrita se descubría en mitad de una sesión de calibración.
Las tres mutaciones probadas contra estas pruebas —quitar el sufijo, marcar
`V_tps` por error, y dejar el nombre viejo en el pipeline— fueron detectadas.

El desbloqueo mayor es **grabar la mini-suite de clips con cuerpo completo**:
libera de golpe las pruebas de Paso 0, etiquetas de estado, barrido de
umbrales y buena parte de la calibración. El segundo es **PEF-FallDB**, que
libera todo el bloque de evaluación del paper.

---

## 10. Resumen para discusión en equipo

| Tema | Estado | Acción propuesta |
|---|---|---|
| Cantidad V como señal de caída | **Validada** con datos reales | Mantener; fijar umbral con barrido |
| Etapa 1 tal como la define la guía | **Falla: pierde 3 de 6 caídas** | Reformular como secuencia V → T; actualizar §3.5 del paper |
| Velocidades imposibles por multi-persona | **Detectado** | Añadir guarda de plausibilidad física |
| Multi-persona | Limitación conocida y declarada | No abordar ahora; medirla y reportarla (Sección 5) |
| Profundidad (z) | **Ya se registra** como `world_torso_len_m_EXP` (torso métrico 3D) | Probar la hipótesis de caída axial cuando existan clips propios |
| Escorzo del torso | **Cuantificado** (8 % vs 4 %) | Considerar normalización métrica |
| Filtro de suavizado (EMA) | **Reduce el temblor ~63 %** ⚠ (cifra pendiente de re-medición); cuesta ~4 % de señal y 0.1 s de latencia. Su utilidad en malas condiciones **no es verificable con los datos actuales** | Mantenerlo; re-medir sobre la mini-suite y calibrarlo con los clips difíciles de PEF-FallDB (§5.4) |
| Dataset público sin post-evento | **Confirmado** | Citar como respaldo a la necesidad de PEF-FallDB |
| Etiquetas de estado | Funcionan; degradadas sin tobillos visibles | Grabar con cuerpo completo |
| `extension_ratio_EXP` negativo | **Abierto.** Da −0.40 y −0.17 en dos clips de caída. Es una diferencia con signo (tobillos por encima de las caderas), así que el dato es plausible; lo dudoso es que `classify_state` mande todo negativo a la categoría más agachada | Ver los frames en 2.6 y decidir si se corrige el clasificador, el nombre, o ambos |
| V de −12.76 torso/s en `video__8` | **Abierto.** Muy fuera del rango −2.8 a −5.2 de las caídas reales; el mismo clip tiene CV de torso métrico del 9.7 % contra 2.7 % del mejor | Sospechoso: artefacto de salto de esqueleto (§3). Revisar en 2.6 |
| Marca `_EXP` en el registro | **Implementada** (15 ago) | Quitar el sufijo a cada campo que se gradúe al paper |

---

*Documento de trabajo. Las cifras provienen de 8 registros reales procesados
el 14 de agosto de 2026 y son reproducibles a partir de los CSV en `logs/` —
con la salvedad de que esos CSV usan los nombres de columna anteriores al
sufijo `_EXP`; reprocesar los clips los regenera con el esquema actual.
Salvo la marca `_EXP`, ninguna de las propuestas ha sido implementada.*

*El catálogo de pruebas (§9) es una lista viva: se actualiza cada vez que una
prueba se ejecuta o se descubre una nueva verificación que valga la pena.
Toda prueba nueva que aparezca durante el desarrollo debe anotarse aquí
aunque no pueda ejecutarse todavía — ese es su propósito.*
