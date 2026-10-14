# TRASPASO-DRAFT.md — Todo lo que falta, y todo lo que no se apega al draft 4

**Fecha:** 3 de septiembre de 2026
**Estado del código:** commit `5b0fd62`, accuracy 74.2 % (95/128) sobre PEF-FallDB.
**Fuente:** `4-paper-draft.docx`, §3.1–3.7, leído párrafo por párrafo contra el
código en `5b0fd62`. No es una lista de memoria: cada cita de abajo está
verificada contra el archivo.

**Nota sobre la numeración.** Una lista anterior de "trece divergencias" existió
sólo en conversación y se perdió. Esta auditoría se renumera desde cero
(D1, D2, …) y reemplaza a aquella. La numeración vieja no debe citarse.

---

# PARTE 1 — El camino al 90 %, paso por paso

**Punto de partida:** 95/128 = 74.2 % (commit `5b0fd62`, verificado dos veces).
**Meta:** 116/128 = 90 %. **Presupuesto: 12 errores.** Hay que eliminar 21 de
los 33 actuales.

Los 33, agrupados por causa, medidos sobre la corrida del 27/08:

| grupo | n | lo ataca |
|---|---|---|
| falsos positivos sobre NoFall | 12 | Paso 3 |
| caídas que nunca dispararon | 14 | Pasos 2 y 4 |
| confusiones de severidad en la frontera | 7 | Paso 5 |

Y no están repartidos al azar: **nueve actores concentran 22 de los 33 fallos**
(A06, A09, A14, B10 con 3 de 4 cada uno; A12, A16, A17, B11, B12 con 2). Son
unos ocho **comportamientos** sistemáticos, no 33 casos sueltos. Por eso el 90 %
es alcanzable: cada arreglo mueve dos o tres clips juntos.

---

## Paso 1 — Partición y métricas *(Fase 9a + 9b)*

**Qué:** separar calibración de prueba, y construir `metrics.py` con lo que el
§3.7 pide: sensibilidad sobre la clase caída, especificidad, precisión, F1,
falsos positivos por hora, y la fila no-determinada visible. Con intervalos de
confianza.

**Por qué primero:** todo lo que viene después ajusta algo. Sin partición, esos
ajustes se miden sobre los mismos clips que los produjeron y el número no se
puede publicar.

**Partición provisional** (por actor, regla fija para que no la elija el
desempeño — cada tercero dentro de cada clase):
prueba = `A07 A16 A08 A17 A09 A18 B03 B06 B09 B12` → 40 clips.
Se reemplaza por la partición real del §3.7 cuando el dataset esté renombrado
(ver `CONVENCION-NOMBRES.md`).

**No toca el paper. Ganancia esperada: 0 puntos** — esto no mejora nada, hace
publicable lo que sigue.

**Criterio de salida:** un comando emite la tabla del §3.7 sobre entrenamiento y
prueba por separado.

---

## Paso 2 — Barrer las formulaciones del disparador *(Fase 11a)*

**Qué:** medir las cuatro lecturas de la frase del §3.5 (`score`, `sequential`,
`simultaneous`, `v_only`) **ya con la Fase 8 puesta**, sobre la partición de
calibración.

**Por qué:** varias de las 14 caídas mudas tienen T **y** V cruzando sus
umbrales, pero en momentos distintos — A09-S2 llega a T=173° y V=−2.48 y no
dispara, porque `score` exige la combinación en un mismo cuadro. `sequential`
existe justamente para el desfase.

Lo medido antes de la Fase 8 (contaminado por el defecto 8a, hay que rehacerlo):

| formulación | accuracy |
|---|---|
| score (2.2) | 68.2 % |
| score (2.6) | 70.5 % |
| sequential | 68.2 % |
| v_only | 62.1 % |
| simultaneous | 55.3 % |

**No toca el paper** — comparar lecturas de la misma frase es lo que el §3.7
pide como ablación. **Ganancia esperada: 3–6 clips.** Barato: es un bucle.

---

## Paso 3 — Guarda de plausibilidad *(Fase 10, divergencia D3)*

**Qué:** implementar lo que el §3.2 ya promete — *"tracking within-frame and
across-frame landmark displacement ratios"*— como defensa contra geometría
imposible. Un tronco de 2 px con confianza 0.99 no es una observación.

Dos mitades, y sólo la segunda ajusta:

1. **La guarda geométrica** — tope inferior al torso proyectado relativo al alto
   del cuadro. Estructural, no ajusta nada, y ya está autorizada por el §3.2.
2. **Discriminación de ADL** — caracterizar los 12 falsos positivos con sus
   trazas. Sentarse o acostarse da una V negativa sostenida y moderada; una
   caída da un pico corto y grande. La firma temporal difiere aunque el máximo
   se parezca. **Esta mitad sí ajusta**, y por eso necesita el Paso 1 hecho.

**No toca el paper** — al contrario, salda una promesa incumplida.
**Ganancia esperada: 7–9 clips.** Es el grupo más grande.

**Criterio de salida:** NoFall ≥ 52/56 en el conjunto de prueba, sin perder
recall de caídas.

---

## Intento fallido — criterio de descenso controlado *(03/09)*

**Se probó y se revirtió.** Queda escrito para que no se repita igual.

**La idea.** El §3.5 encarga a la Etapa 2 separar una caída de *"a controlled
sit-down in which COM stays within the support polygon"*, y le da P como única
herramienta. Medido, esa premisa es falsa: tres falsos positivos de calibración
tenían `p_samples=11` con el COM **fuera** del apoyo en el 100 % de los cuadros.
Sentarse en un sofá saca el centro de masa de sobre los pies porque el peso pasa
al mueble — la geometría dice "caída" con razón. Ver **D16**.

Lo que sí separaba, medido sobre los 40 clips de calibración que dispararon:

```
caídas            V_min mediana -3.00   tiempo bajo -1 tps  0.77 s
falsos positivos  V_min -1.16 y -1.39   tiempo             0.17 y 0.20 s
```

Se implementó rechazar cuando el descenso fuera **poco profundo Y breve** (las
dos condiciones: un síncope es corto pero violento, y deslizarse por una pared
es suave pero largo).

**Lo que pasó.** Calibrado predijo "pierde 1 caída de 40". Medido de punta a
punta sobre los 128: **perdió 4 caídas y ganó 3 ADL.**

```
                    guarda sola    + descenso
accuracy            97/128 75.8%   96/128 75.0%
sensibilidad (prueba)  70.8%          66.7%
especificidad (prueba) 75.0%          93.8%
```

**Por qué falló, y es un error de método, no de idea.** El barrido midió el
descenso sobre el **clip entero** — el mínimo de V y la racha más larga de todo
el video. El código lo evalúa sobre una **ventana móvil de 2 s que cierra 0.33 s
después del disparo**. No son la misma medición: en varias caídas el descenso
fuerte ocurre después de ese cierre, así que el código ve un descenso más débil
que el que la calibración midió.

**Por qué se revierte aunque el accuracy sólo baje 0.8 puntos** (que es ruido
con 128 clips): el intercambio va en la dirección equivocada. Cambió
sensibilidad por especificidad, y para un detector de caídas una alarma falsa
molesta mientras que una caída perdida es alguien en el suelo sin auxilio. El
§3.7 pide sensibilidad sobre la clase caída precisamente por eso.

**Qué queda.** `DescentTracker` y sus 16 pruebas se conservan; el instrumento es
correcto y la separación medida es real. El rechazo está desactivado con
`descent_min_depth_tps: 0.0`. **Para reactivarlo hay que re-calibrarlo con la
misma ventana que usa el código**, no con el clip entero.

---

## Paso 4 — Caídas axiales *(Fase 11b, divergencia D4)*

**Qué:** **primero contar** cuántos de los 128 son caídas en el eje de la cámara
(el tronco se proyecta casi a un punto y T nunca sube: A17-S2 llega a 9.1°,
A13-S4 a 23.1°). Recién con ese número se decide.

- **Si son pocos** → se documenta como limitación del método 2D. El §3.2 ya
  reconoce la ambigüedad de profundidad; basta una frase en §3.4.
- **Si son muchos** → hay que calcular T desde los landmarks de mundo cuando el
  torso proyectado sea implausible. Medido: el torso 3D cae 0.458 → 0.209 m
  (−54 %) donde el de píxeles cae −99 %. La información existe y ya la
  registramos. **Pero contradice el §3.2**, que excluye la z explícitamente, y
  requiere reescribir ese párrafo.

**Puede tocar el paper. Ganancia esperada: 4–8 clips**, o cero si se opta por
documentar. **Este paso decide si el 90 % es alcanzable sin cambiar el §3.2.**

---

## Paso 5 — Calibración *(Fase 12)*

**Qué:** sobre la partición de calibración únicamente:
`upright_T_deg`, `lying_T_deg`, `threshold_T_deg` (ver **D1**: el paper dice
60°, el config usa 45°), `trigger_score`, `trigger_hold_s`, `confirm_window_s`,
`com_outside_fraction`, `com_eval_window_s`, `contact_band_torso`, `epsilon`,
`threshold_W_seconds`, `com_hip_weight`.

Cada valor queda en `config.yaml` con su número, su fecha y la partición sobre
la que se calibró — eso es lo que hace reproducible la Sección 4.

**Barrido exploratorio ya hecho (03/09), sobre los 128 — insumo, NO decisión.**
Se guarda aquí para no repetirlo; aplicarlo requiere la partición del Paso 1.

```
 score   hold     acc    NoFall   caidas mudas
   2.2  0.100   74.2%    45/56       16      <- hoy
   2.2  0.033   75.8%    43/56       10
   2.0  0.100   75.0%    41/56       11
   2.0  0.050   75.8%    40/56        9
   1.8  0.100   69.5%    31/56        7
```

Dos lecturas que ya se pueden dar por firmes porque no dependen del ajuste
fino: **bajar `trigger_score` es un mal negocio** (a 1.8 rescata 9 caídas y
pierde 14 NoFall — el umbral no separa, las caídas débiles caen en el mismo
rango que los ADL), y **bajar `trigger_hold_s` es buen negocio pero chico**
(+6 caídas, −2 NoFall, neto +2). Ojo: `hold = 0.033 s` a 30 fps es un solo
cuadro, o sea eliminar el mecanismo, no calibrarlo.

**Ganancia esperada: 3–5 clips.** **Riesgo:** son ~12 parámetros sobre 88 clips
de calibración. Es fácil llegar al 95 % en calibración y perder en prueba. La
regla: si prueba no sube junto con calibración, el parámetro se revierte.

---

## Paso 6 — Ablaciones *(Fase 9c)*

**Qué:** lo que el §3.7 exige — quitar T, V, P e I de a una; quitar la Etapa 3 y
reportar el aumento de falsos positivos; colapsar a umbral único y reportar la
caída de precisión. Incluye la ablación de `extension_ratio_EXP` que ya está
medida y que cierra **D8**.

**Ganancia: 0 puntos.** Es contenido de la Sección 4, no mejora del sistema.

---

## Paso 7 — Validación externa *(ver Parte 1b)*

Una sola corrida, al final, sobre el conjunto de datasets públicos. No se toca
antes.

---

## Suma

| paso | ganancia esperada | acumulado |
|---|---|---|
| — | — | 95/128 (74.2 %) |
| 2 · formulaciones | +3 a +6 | 98–101 |
| 3 · guarda + ADL | +7 a +9 | 105–110 |
| 4 · axiales | 0 a +8 | 105–118 |
| 5 · calibración | +3 a +5 | 108–123 |

**116/128 cae dentro del rango, pero no está garantizado.** El paso que decide
es el 4: si las axiales son muchas y no se autoriza tocar el §3.2, el techo
queda alrededor de 110/128 ≈ 86 %.

**Sólo el punto de partida está medido.** Todas las ganancias de la tabla son
estimaciones basadas en el tamaño de cada grupo de error, no mediciones.

---

## Lo que el paper necesita y no mueve el accuracy

Estas no suman puntos pero el draft las afirma en presente, así que sin ellas el
paper describe un sistema que no existe.

- **Fase 13 — Baselines (§3.7).** Cuatro comparaciones prometidas: MediaPipe +
  LSTM, MediaPipe + Transformer, el pipeline de reglas de Saraswat & Malathi, y
  una CNN liviana de borde. Hay que decidir si se implementan o se citan cifras
  publicadas — pero el §3.7 dice *"compared against"*, y eso compromete a una
  comparación medida.
- **Fase 14 — Despliegue en el borde (§3.6).** Modo servicio headless sin Qt;
  despachador real de SMS, email y buzzer (hoy existe la costura, ningún
  transporte); **modo desplegado que no escriba nada a disco**, que el §3.6
  afirma y hoy no existe; port a Raspberry Pi 4B con sus mediciones de latencia,
  CPU, memoria y energía.
- **Fase 15 — Rediseño del GUI.** Navegación y filtros para revisar 128 clips,
  comparación traza-contra-video, y el diagnóstico pendiente de los **5 fps en
  cámara viva** (la instrumentación está puesta hace semanas y nunca se leyó;
  `velocity_window_s: 0.167` es más corto que un intervalo de cuadro a 5 fps,
  con lo cual V degenera a una diferencia de un solo cuadro).
- **Deudas menores:** README desactualizado; transeúntes / multi-persona nunca
  discutido; `git gc --prune=now`.

---

# PARTE 1b — El conjunto de validación externa

Reunir datasets públicos de caídas, hasta ~150 videos de cada uno, y correrlos
**una sola vez**, cuando el plan de arriba se dé por terminado.

**Es la decisión metodológica más fuerte de todo el proyecto.** Una partición
interna controla contra ajustar a un sujeto; un dataset externo controla contra
ajustar al *laboratorio* — a la cámara, a la luz, a la casa, a la forma en que
estos actores actúan una caída. Ninguna partición de PEF-FallDB puede hacer eso.

## Qué puede y qué no puede medir

**No puede medir las cuatro clases.** El propio §3.7 lo dice:

> *"Because no existing public corpus preserves the post-event continuity that
> the severity logic depends on, PEF-FallDB is the only dataset on which the
> full framework can be assessed."*

Los corpus públicos cortan el clip en el impacto. Sin secuencia post-evento no
hay Recovered / PartiallyRecovered / NotRecovered que anotar, y forzar esas
etiquetas sobre ellos produciría números sin significado.

**Sí puede medir el binario caída / no-caída** — Etapa 1 más Etapa 2. Que es
exactamente el eje donde el §3.7 pide sensibilidad, especificidad, precisión y
F1, y donde están las cuatro comparaciones con baselines. Es la mitad del
sistema, pero es la mitad que se compara con la literatura.

**Y mide algo que ningún otro experimento mide:** que las cantidades sean
adimensionales de verdad. Los corpus públicos vienen en otras resoluciones y
otras tasas de cuadro. Si la normalización por torso y las ventanas en segundos
funcionan, los umbrales calibrados aquí deberían transferir sin tocarlos. Si no
transfieren, eso es un resultado — y respalda o refuta la afirmación de
robustez a *dataset shift* del §3.1.

## Las reglas, y son estrictas

1. **No se mira antes.** Ni una corrida exploratoria, ni "sólo para ver".
2. **No se calibra nada contra él. Nunca.** En el momento en que un umbral se
   mueve por un resultado de ahí, deja de ser validación externa y pasa a ser
   un segundo conjunto de calibración.
3. **Se corre una sola vez**, con la configuración congelada, y se reporta lo
   que salga — incluso si sale mal. Un resultado externo peor que el interno es
   un hallazgo publicable; correrlo cinco veces ajustando en medio, no.
4. **Se congela el commit** con el que se corre, y se cita en la Sección 4.

## Cómo elegir los corpus

- **Que tengan video RGB monocular.** Los basados en acelerómetro no aplican.
- **Que tengan ADL además de caídas**, o el binario mide sólo la mitad.
- **Que la licencia permita uso en investigación** y que la cita esté clara. El
  §3.3 de este paper es cuidadoso con consentimiento y privacidad; usar material
  ajeno sin respetar su licencia contradiría esa sección.
- **Cuantos más orígenes, mejor que más videos de uno solo.** Tres corpus de 100
  dicen más sobre generalización que uno de 300.
- **Anotar con la misma convención** (`CONVENCION-NOMBRES.md`), con `recovery`
  en `nofall` o en un valor `unknown` para las caídas sin continuidad
  post-evento, de modo que el runner los procese sin código especial.

## Cuándo

Después del Paso 6. Si se corre antes, se pierde: es irrepetible por diseño.

# PARTE 2 — Lo que no se apega al draft 4

Cuatro grupos, por naturaleza del desacuerdo y no por gravedad.

---

## GRUPO A — El código no hace lo que el paper dice. Arreglable en código.

### D1 — El umbral de T no es el que el paper nombra

> §3.4: *"A normal upright subject has T below 15°, a moderate lean (reaching
> for an object) has T in the range 15°–60°, and **a fall action produces a
> transition to T of 60° or greater** within a short time window."*

`config.yaml` usa `threshold_T_deg: 45.0`.

**Por qué está así:** 45° se eligió antes de existir el dataset, para que el
disparador se activara temprano en el descenso. El paper describe 60° como la
firma de la caída consumada.

**¿Se puede apegar?** Sí, es un número en el config. **Pero hay que medirlo
antes de cambiarlo**: subir a 60° hace el disparador más estricto y las 14
caídas que hoy no disparan podrían volverse 20. Es tarea de la Fase 12, y el
resultado puede ser evidencia para cambiar el paper en vez del código.

### D2 — El paper describe despachar y luego degradar; el código espera

> §3.5: *"si se detecta una secuencia de levantada exitosa dentro de la ventana
> de inmovilidad […] **la etiqueta de severidad se degrada o la alarma se
> anula**; si la inmovilidad persiste, la alerta se despacha con la etiqueta
> apropiada."*

El código nunca despacha dos veces: espera a resolver y despacha una sola vez,
con la etiqueta final.

**Por qué está así:** despachar y luego degradar significa que el cuidador
recibe una alarma que después se retracta. Se implementó la lectura
conservadora. Pero el paper describe la otra.

**¿Se puede apegar?** Sí, y probablemente **debe**, porque la lectura del paper
es mejor para el §3.6: una alarma que llega tarde no sirve. La costura de
alertas ya soporta múltiples envíos. Va con la Fase 14.

### D3 — La mitigación de ambigüedad de profundidad que el §3.2 promete no existe

> §3.2: *"depth ambiguity from the monocular input is mitigated by **tracking
> within-frame and across-frame landmark displacement ratios** rather than
> relying on absolute 3D distances."*

El código no implementa nada que corresponda a esa frase. No hay seguimiento de
razones de desplazamiento entre cuadros como defensa contra la ambigüedad de
profundidad.

**Consecuencia medida:** en A14 el torso pasó de 209 px a **2 px** en 0.2 s
mientras `core_visibility` reportaba 0.99, y T produjo 176.9°, 65.7°, 131.1°,
26.0°, 144.0° en cuadros consecutivos. Justamente lo que esa frase promete
evitar.

**¿Se puede apegar?** Sí. Es la guarda de plausibilidad de la Fase 10, y la
frase del §3.2 ya la autoriza — no hace falta texto nuevo. **Es la divergencia
más importante del Grupo A** porque es una promesa incumplida, no una elección.

### D4 — Caídas axiales: el paper no las excluye y el código no las ve

Ni el §3.2 ni el §3.4 reconocen que T es indefinida cuando el tronco apunta
hacia la cámara. El §3.4 afirma que el sistema *"preserva la firma cinemática
de una caída"*, sin excepción.

**Medido:** al menos 3 de los 4 clips de A14 y varios de A17 y A13 son caídas en
el eje de la cámara, donde T nunca supera 25°.

**¿Se puede apegar?** **Dos caminos, y la decisión no es de código.**

- **(a) Usar los landmarks de mundo (3D) para T cuando el torso proyectado sea
  implausible.** Medido: el torso 3D cae de 0.458 a 0.209 m (−54 %) donde el de
  píxeles cae −99 %. La información existe y ya la registramos
  (`world_torso_len_m_EXP`). **Pero contradice el §3.2**, que excluye la
  coordenada z explícitamente y por una razón declarada. Requiere reescribir ese
  párrafo.
- **(b) Documentarlo como limitación.** El §3.2 ya reconoce la ambigüedad de
  profundidad; bastaría una frase en §3.4 diciendo que T es indefinida bajo
  escorzo axial y que esos casos caen en la clase no determinada.

**Antes de decidir hay que contar cuántas son.** Si son 4 de 128, (b). Si son
20, (a) deja de ser opcional.

### D5 — El §3.6 y el §3.7 están sin implementar

El paper afirma en presente cosas que no existen: el despachador de SMS/email/
buzzer, el despliegue en la Pi, las cuatro comparaciones con baselines, y todas
las métricas del §3.7 salvo accuracy.

**¿Se puede apegar?** Sí — son las Fases 9, 13 y 14. No hay conflicto
conceptual, sólo trabajo. Se listan aquí porque **hoy el paper describe un
sistema más completo que el que existe**, y eso es una divergencia aunque sea
temporal.

---

## GRUPO B — El paper dice algo que no se puede cumplir tal cual. Necesita texto.

### D6 — La Cantidad I no decide nada, y no es calibración

> §3.4: *"**Si I excede un umbral de confirmación W**, el evento se clasifica
> como caída confirmada y se despacha una alerta con la etiqueta de severidad
> apropiada."*

**Medido sobre los 15 clips NotRecovered que confirmaron:** I máximo promedio
**1.12 s**, máximo absoluto **3.57 s**, y **ninguno** alcanzó W = 5.0 s. Todos
los veredictos `severe` salieron de la regla de postura al final del clip, no de
la inmovilidad. La Cantidad I, hoy, no decide nada.

**Por qué no es sólo recalibrar.** El desplazamiento por cuadro de un sujeto
inmóvil tiene mediana 0.038–0.044 torsos contra un ε = 0.05: el umbral está en
el piso de ruido. Pero al subir ε aparece un conflicto peor: con ε = 0.15, el
clip A13 —donde el señor se queda quieto seis segundos y **luego se levanta
solo**— pasa de `moderate` a `severe`, porque la rama de inmovilidad alcanza los
5 s antes de que se levante. **Las dos etiquetas más distantes del conjunto.**
Las dos ramas de la Etapa 3 compiten, y la que dispare primero gana.

**¿Se puede apegar?** No con un número. Dos salidas, ambas de texto:

- **(a)** Que el §3.4 diga que **I es evidencia de apoyo para la severidad
  severa, no un umbral de confirmación por sí solo**, y que el desenlace lo
  determina la postura al cierre de la ventana de observación. Es lo que el
  código hace hoy de facto, y es más fiel al §3.3, que define la severidad por
  **recuperación** — un enunciado sobre el final del episodio, no sobre un
  instante intermedio.
- **(b)** Mantener el umbral y aceptar que compite con la rama de recuperación,
  declarando explícitamente cuál tiene precedencia.

Lo que no se puede es dejar el §3.4 definiendo un umbral que en la práctica no
se alcanza nunca.

### D7 — La guarda de tronco degenerado del §3.4 es correcta en el código y demasiado permisiva en el paper

> §3.4: *"Cuando |trunk| cae por debajo de **0.001 px**, indicando un esqueleto
> degenerado en el que los grupos de hombro y cadera coinciden, T se retorna
> como indefinida (NaN)."*

El código la implementa **exactamente**: `MIN_TRUNK_PX = 1e-3`.

**El problema es el número del paper.** Un tronco de 2 px es 2000 veces mayor que
el umbral, así que pasa la guarda — y produjo los 176.9° y 26.0° consecutivos de
A14. Un umbral absoluto en píxeles no puede funcionar: 0.001 px no es alcanzable
por ninguna pose real, ni siquiera una totalmente colapsada.

**¿Se puede apegar?** El código **ya está apegado**. Apegarse más sería empeorar.
Lo que hay que cambiar es el paper: el umbral debería ser **relativo** — una
fracción del alto del cuadro, o del torso suavizado reciente — para que
signifique algo. Frase propuesta:

> *Cuando |trunk| cae por debajo de una fracción configurable del torso
> observado recientemente —indicando escorzo axial o un esqueleto degenerado en
> el que los grupos de hombro y cadera coinciden en proyección— T se retorna
> como indefinida (NaN) en lugar de asignarle un ángulo, de modo que las etapas
> posteriores descarten el cuadro en vez de actuar sobre un valor espurio.*

### D8 — Un `_EXP` decide una etiqueta de severidad del §3.3

`severity_uses_leg_extension: true` mete `extension_ratio_EXP` —una columna
marcada como experimental— en la decisión leve/moderada. La regla de este
proyecto es que ninguna señal `_EXP` entra a una decisión antes de que el draft
la autorice.

**Medido:**

| criterio | accuracy | Recovered | Partially |
|---|---|---|---|
| `extension_ratio_EXP` (hoy) | **74.2 %** | 19/24 | 14/24 |
| P: COM de vuelta dentro del apoyo (§3.4) | 70.3 % | 16/24 | 12/24 |
| P o extensión | 72.7 % | 19/24 | 12/24 |
| apagado, duda → Recovered | 68.8 % | 20/24 | 6/24 |
| apagado, duda → Partially | 60.2 % | 0/24 | 15/24 |

Se probó la salida elegante —usar **P**, que sí es del paper— y cuesta 4 puntos.
No es equivalente.

**¿Se puede apegar?** El §3.5 dice *"torso and hip landmark recovery to
upright"*, que es lo más cerca que el paper llega, pero no nombra la extensión
cadera-tobillo. **Sugerencia: quedarse con la extensión y convertir la
infracción en un resultado reportado.** El §3.3 ya se comprometió con tres
etiquetas; un paper que nombra tres desenlaces le debe al lector la señal que
separa los dos primeros, porque con T sola no se separan — alguien sentado
erguido en el piso y alguien de pie dan el mismo ángulo de tronco. La tabla de
arriba **es** la ablación que el §3.7 exige. Frase propuesta para el §3.4:

> *La razón de extensión cadera-tobillo, normalizada por la longitud del torso,
> complementa a T en la Etapa 3: distingue una recuperación completa (el sujeto
> se puso de pie) de una parcial (se sentó o quedó de rodillas), que T por sí
> sola no puede separar porque ambas posturas presentan el tronco erguido.*

Con esa frase la divergencia se cierra. Sin ella, el número honesto a reportar
es **70.3 %**, no 74.2 %.

### D9 — La partición que el §3.7 especifica no se puede hacer con el material entregado

> §3.7: *"La generalización a través de condiciones de grabación se prueba
> dentro de PEF-FallDB, **reteniendo settings, ubicaciones de cámara y
> condiciones de iluminación** que no se usaron al calibrar los umbrales."*

Los 128 clips entregados se llaman `A01-S1-Recovered.mp4`. **No codifican
setting, ni ubicación de cámara, ni iluminación.** La partición que el paper
especifica no es ejecutable sobre este material.

**RESUELTA POR PLAN (03/09).** El material existe con esas dimensiones; sólo no
está anotado en el nombre. El dataset se va a normalizar contra el §3.3, así que
**el paper no cambia**: la partición del §3.7 se vuelve ejecutable en cuanto el
nombre lleve setting, cámara e iluminación. Ver `notas/CONVENCION-NOMBRES.md`.

Mientras tanto, para no bloquear las Fases 10–12, la partición provisional es
**por actor**, por regla fija para que no la elija el desempeño (cada tercero
dentro de cada clase): prueba = **A07, A16, A08, A17, A09, A18, B03, B06, B09,
B12** → 40 clips. Es más débil que lo que el paper promete —controla identidad
del sujeto, no condición de grabación— y se reemplaza por la partición real
cuando la anotación esté lista. **No debe reportarse como si fuera la del §3.7.**

### D10 — El dataset descrito en §3.3 y el dataset entregado no son el mismo

El §3.3 describe PEF-FallDB con: **cinco settings** (dormitorio, baño, sala,
cocina, pasillo) con al menos cuatro escenarios cada uno; **tres categorías de
sujeto** por setting (activo, transeúntes caminando a distintas velocidades,
sujetos parcialmente visibles); **seis ADL nombrados** para falsos positivos;
**cuatro arquetipos de caída** (frontal, hacia atrás, lateral, síncope);
**múltiples alturas y ubicaciones laterales de cámara**; y **tres condiciones de
iluminación** (día, luz artificial, poca luz / noche).

De los 128 clips entregados no se puede verificar ninguna de esas dimensiones:
sólo actor, número de escena y clase de recuperación.

**Lo que sí coincide:** la ventana de 8–12 s (los clips duran 10.4 s) y la
continuidad post-evento, que es la característica definitoria y sí está
presente.

**RESUELTA POR PLAN (03/09).** El material existe sin anotar y queda material por
grabar; el conjunto se normalizará contra el §3.3. **El §3.3 no se toca.**

Consecuencia operativa mientras tanto: no se puede decir cuál de los 56 clips
NoFall es cada ADL, así que la Fase 10 no puede reportar en qué actividad falla
el sistema — sólo cuántas veces. Los cuatro falsos positivos `NotRecovered/
severe` con 99–100 % de confiabilidad son, por su forma, compatibles con
`adl.liedown`, pero eso es hipótesis y no medición hasta que exista la
anotación.

### D11 — El protocolo de anotación del §3.3 no es el que usa el dataset

> §3.3: *"La etiqueta sigue el formato **[setting]-[event]-[recovery status]**.
> Por ejemplo, videos grabados en un pasillo pueden etiquetarse
> `hallway-fall-recovered`…"*

El dataset usa `[actor]-[escena]-[recovery]`: `A01-S1-Recovered.mp4`.

**RESUELTA POR PLAN (03/09).** Se renombra el dataset. La convención propuesta
mantiene los tres campos del §3.3 como los tres primeros y añade sujeto, cámara
e iluminación detrás: ver `notas/CONVENCION-NOMBRES.md`. **El §3.3 no se toca.**

*(Nota aparte, no es divergencia: cuatro nombres tienen erratas —`A06-S14`,
`A11-S14`, `A3-S2`, y `Recovery` ×4—. Sólo se normaliza `Recovery`→`Recovered`,
y esa normalización está justificada por conteo: hay exactamente seis actores
por clase de caída, y sólo bajo esa lectura las cuentas dan 24/24/24.)*

---

## GRUPO C — El paper calla y el código tuvo que decidir. Necesita texto.

Estas no son incumplimientos: son huecos. El código hace algo razonable donde el
paper no dice nada, y el paper tiene que decirlo para que el resultado sea
reproducible.

### D12 — La clase "no determinada" no existe en el paper

El §3.5 describe tres etapas y el §3.3 tres severidades. Ninguno contempla
*"ninguna etapa pudo juzgar"*. El código lo produce: el disparador se activa,
los pies nunca son visibles, la Etapa 2 no puede votar y el evento queda sin
resolver.

**Por qué el código lo hace:** colapsarlo en "no hubo caída" contaría un fallo
como acierto en los clips NoFall y escondería la diferencia entre "nunca disparó"
y "disparó pero no pudo juzgar" — dos defectos con causas y arreglos distintos.

**Frase propuesta para el §3.5:**

> *Un evento cuya evidencia geométrica o de inmovilidad resulte insuficiente
> para que alguna etapa emita veredicto se registra como no determinado, en vez
> de asimilarse a una no-caída. Esta categoría se reporta por separado en §4,
> porque un sistema que no puede juzgar y uno que juzga que no hubo caída
> constituyen modos de falla distintos.*

### D13 — Qué pasa cuando se pierde el seguimiento durante la confirmación

El paper no dice nada sobre oclusión o pérdida de detección mientras un evento
está siendo juzgado. El código, hasta la Fase 8, abandonaba el evento ante
cualquier cuadro perdido: **46 eventos abortados en 37 de 128 clips**, y ocurría
justo en el impacto, que es cuando la pose es menos confiable. Tolerar lagunas
de hasta 0.5 s valió **+11 puntos de accuracy**.

**Es un resultado citable**, no sólo un detalle. Frase propuesta para el §3.5:

> *Una pérdida breve de detección no invalida un evento en curso: las Etapas 2 y
> 3 tratan la entrada faltante como desconocida y sus ventanas continúan. Sólo
> una laguna mayor que el umbral de continuidad abandona el evento, porque más
> allá de ese tiempo no hay garantía de que el cuerpo que reaparece sea el
> mismo. Sobre PEF-FallDB, las lagunas de detección tienen mediana 0.10 s y el
> 82 % dura menos de 0.5 s.*

### D14 — Cómo se separan las tres severidades por postura

El §3.3 define tres desenlaces y el §3.5 dice que la Etapa 3 los distingue, pero
ninguno dice **cómo**. El código usa tres bandas del ángulo de tronco al cierre:
`< 30°` con piernas extendidas → leve; `30–60°` → moderada; `≥ 60°` → severa.

**Medido:** con un corte binario en 30°, 9 de los 24 clips PartiallyRecovered
salían NotRecovered. Sus ángulos finales se agrupan entre 35° y 55° — alguien
que se sentó en el piso o quedó de rodillas. Con tres bandas, PartiallyRecovered
pasa de 8/24 a 14/24.

**Frase propuesta para el §3.4 o §3.5:** declarar las dos fronteras angulares
como parámetros calibrados, igual que los demás umbrales.

### D15 — Dos modos de resolución: en línea y al cierre

El §3.3 define la severidad por **recuperación** —un enunciado sobre cómo
terminó el episodio— mientras que el §3.5 describe una resolución **en línea**,
mientras el evento ocurre. Para un despliegue en vivo la segunda es obligatoria;
para etiquetar un clip grabado la primera es la correcta.

El código implementa ambas y elige según la fuente: cámara viva → resolución
inmediata; video grabado → decisión al cierre.

**Por qué importa:** medido sobre A13, un resolvedor glotón cierra el evento
como `severe` **tres segundos antes** de que el sujeto se levante solo. Truth:
Recovered. Es la diferencia entre las dos etiquetas más distantes del conjunto.

**Frase propuesta:** que el §3.5 declare explícitamente los dos modos y que la
evaluación del §3.7 use el de cierre, mientras que las métricas de latencia del
§3.6 usan el de línea.

---

## GRUPO D — Divergencias ya cerradas

Se listan para que no se vuelvan a abrir, y porque el código las documenta.

| # | qué era | cómo se cerró |
|---|---|---|
| — | La lectura de *"combinación instantánea"* del §3.5 | Formulación `score`: ambas cantidades sobre sus umbrales, en un mismo cuadro, con V negativa |
| — | El radio ε de la Cantidad I anclado al primer cuadro | Anclado a la **posición promedio** del episodio, como dice el §3.4 |
| — | El polígono de apoyo usaba todos los pies visibles | Filtro de **contacto bípedo**, como dice el §3.4. Medido: anchos absurdos de 21 clips → 0, cobertura de P 54 % → 70 % |
| — | El EMA sobre el centroide no estaba en el paper | El draft 4 ya lo incluye en la definición de V |

---

# PARTE 3 — Resumen para quien escribe

**Cambios de texto que el código necesita, en orden de importancia:**

1. **D6** — la Cantidad I: umbral de confirmación → evidencia de apoyo.
2. **D8** — autorizar la extensión cadera-tobillo (frase lista arriba).
3. **D7** — el umbral de tronco degenerado: absoluto → relativo (frase lista).
4. **D12, D13, D14, D15** — cuatro huecos del §3.5 (frases listas arriba).
5. ~~**D9, D10, D11**~~ — **resueltas 03/09 sin tocar el paper.** El material
   existe sin anotar y falta grabar más; el dataset se normaliza contra el §3.3
   y se renombra según `notas/CONVENCION-NOMBRES.md`.
6. **D4** — caídas axiales: limitación documentada, o reescribir el §3.2 para
   admitir los landmarks de mundo. **Contar primero.**

**Lo que es sólo trabajo de código, sin tocar el paper:** D1, D2, D3, D5.
