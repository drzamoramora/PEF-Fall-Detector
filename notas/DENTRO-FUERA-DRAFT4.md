# DENTRO / FUERA DEL DRAFT 4 — qué se puede tocar sin tocar el paper

Rama `camino-90`. Escrito el 08/09/2026, después de releer el §3.1–§3.7 completo
del `4-paper-draft.docx` contra el código en su estado actual (97/128 = 75.8 %;
calibración 78.4 %, prueba 70.0 %; 267 pruebas).

Este documento responde una sola pregunta, la que usted hizo: **para llegar al
90 %, ¿qué se hace por dentro del paper y qué obliga a cambiarlo?** El catálogo
completo de divergencias vive en `TRASPASO-DRAFT.md` (D1–D16); esto es el
recorte operativo: qué se puede implementar mañana sin escribirle una línea al
profesor, y qué no.

Regla de lectura: **el draft 4 es la autoridad.** Cuando el código y el paper no
coinciden, la pregunta no es "¿cuál gana?" sino "¿cuál de los dos está
equivocado sobre la física?", y esa pregunta se contesta con la medición, no con
la antigüedad del texto.

---

## NIVEL 1 — Dentro del draft 4. Cero cambios de texto.

Lo que sigue ya está escrito en el paper. Implementarlo no es divergir: es
**terminar de implementar lo que el paper dice**.

### 1.1 Contracción del polígono de apoyo *(el siguiente paso)*

El §3.4, Cantidad P, define la firma geométrica de una caída con **dos**
señales, unidas por un "or":

> *"A fall is geometrically characterized by projection of COM outside the
> support polygon, **or by a sudden contraction of the support polygon from a
> full two-foot footprint to a heel-only contact pair**."*

El código implementa la primera y sólo la primera. La segunda existe como
medición —`support_width()` en `quantities.py`, registrada en cada cuadro como
`P_support_width`— pero **ninguna etapa la consulta**. Está deliberadamente
desconectada desde la Fase 4.

Medido sobre la partición de calibración: la anchura del apoyo cae a **0.32**
longitudes de torso en las caídas contra **1.04** en los ADL. La separación es
real. El problema es la cobertura: sólo 20 de 88 clips tienen suficientes
cuadros con los pies medibles para evaluarla.

Por qué es el siguiente paso de todos modos: es la única mejora disponible que
está **completamente dentro del texto existente**, y su costo es acotado
—no puede empeorar los clips donde P no es medible, porque ahí ya no decide
nada—. Si sube el accuracy, sube gratis. Si no, cierra la mitad del §3.4 que hoy
está sin evidencia y el §3.7 gana una ablación que reportar.

### 1.2 Grabar más material

El §3.3 ya describe el dataset que hace falta: cinco settings, seis ADL
nombrados uno por uno, cuatro arquetipos de caída, múltiples alturas de cámara,
tres condiciones de iluminación, transeúntes y sujetos parcialmente visibles.
Nada de eso necesita escribirse: **necesita grabarse**.

Tres prioridades, en orden de cuánto desbloquean:

1. **Horas de metraje ADL largo.** Hoy hay 9.7 minutos. Con eso, un solo falso
   positivo se extrapola a 6.2 alarmas por hora, y el §3.7 pide reportar
   "false-positive rate per hour of non-fall footage". Esa cifra **hoy no es
   reportable**, y no por un defecto del detector.
2. **Los seis ADL del §3.3, anotados como tales.** El paper los nombra; el
   dataset entregado no distingue cuál es cuál. Sin eso no hay forma de decir
   "el agacharse a recoger es el que falla", que es exactamente la afirmación
   que el §3.3 anticipa al llamarlo *"a stringent stress test"*.
3. **Caídas axiales deliberadas.** De 72 caídas, sólo **4** son axiales
   (T_max < 40°: el cuerpo cae hacia o desde la cámara, y en 2D el tronco casi
   no rota). Son el 5.6 % del material, y son el techo estructural: la
   sensibilidad 2D no puede pasar de ~94.4 % mientras existan. Con 4 casos no
   se puede ni medir el problema ni justificar una solución.

### 1.3 Documentar las caídas axiales como límite, no como falla

El §3.2 ya reconoce la ambigüedad de profundidad del monocular y dice cómo se
mitiga: *"by tracking within-frame and across-frame landmark displacement ratios
rather than relying on absolute 3D distances"*. Reportar en la Sección 4 que 4
de 72 caídas son axiales y que ahí el detector es ciego **no contradice nada**:
es la consecuencia esperada del párrafo que el paper ya escribió.

---

## NIVEL 2 — Dentro del espíritu, pero pide UNA oración.

Cambios que el paper no prohíbe, y que en un caso el propio paper ya contradice
en otra sección. No son reescrituras: son una frase.

### 2.1 Disparador por ventana de pico *(la mejora grande, y la que hay que decidir)*

**El paper se contradice consigo mismo aquí.** El §3.4 dice que una caída
produce *"a transition to T of 60° or greater **within a short time window**"* y
que V produce *"a large-magnitude negative V **within a short window**"*. El
§3.5 dice que el evento pasa a la Etapa 2 cuando *"an **instantaneous**
combination of the two quantities exceeds a configurable trigger threshold"*.

No es una sutileza de redacción. **T es una posición y V es una velocidad, y no
llegan a su máximo en el mismo cuadro**: el cuerpo alcanza su mayor velocidad de
descenso cuando todavía está relativamente vertical, y alcanza su mayor
inclinación cuando ya casi se detuvo. Exigir que ambas sean extremas en el mismo
instante es exigir algo que la física de una caída no produce.

Medido, sin implementar todavía (barrido sobre la partición de calibración,
evaluado después sobre la de prueba):

| Configuración | Calibración | Prueba | Sensibilidad |
|---|---|---|---|
| Actual (instantánea) | 80.7 % | 72.5 % | — |
| Ventana 0.8 s / umbral 3.0 | 81.8 % | **87.5 %** | — |
| Ventana 0.8 s / umbral 2.2 | 47/48 | 23/24 | **96 %** |

Es la única palanca medida que llega sola a la meta. **Lo que cuesta:** una
oración en el §3.5 que cambie "instantaneous combination" por la ventana corta
que el §3.4 ya nombra dos veces. No se está inventando un mecanismo nuevo; se
está alineando el §3.5 con el §3.4.

**Decisión pendiente suya:** adoptarlo o no. Y con él, cuál de las dos filas es
el titular: 87.5 % de accuracy, o 96 % de sensibilidad. No son la misma
configuración y no se pueden reportar las dos como si fueran una.

### 2.2 Profundidad y duración del descenso en la Etapa 2

El §3.5 le encarga a la Etapa 2 separar una caída de *"a controlled sit-down in
which COM stays within the support polygon"*, y le da **P como única
herramienta**.

Esa premisa es empíricamente falsa, y está medida: los ADL que disparan ponen el
COM **fuera** del polígono de apoyo en el **100 %** de los cuadros medibles
(p_samples = 11). La razón es mecánica y obvia una vez vista: sentarse traslada
el peso al mueble, así que el centro de masa deja de estar sobre los pies. La
geometría dice "caída" con toda la razón. Esto es **D16** en el traspaso.

Lo que sí separa es la **magnitud del descenso**, y el §3.4 ya lo describe al
definir V: *"a sit-down event produces a monotonic negative V over a longer
window, and a fall produces a large-magnitude negative V within a short
window"*. El material está escrito; está en la sección equivocada.

Estado del código: `DescentTracker` existe, tiene 15 pruebas, y está
**deliberadamente desactivado** (`descent_min_depth_tps: 0.0`). El intento de
03/09 costó −4 clips netos, y la causa raíz fue mía y está documentada: calibré
sobre el clip entero mientras el código evalúa una ventana rodante de 2 s que
cierra 0.33 s después del disparo. El instrumento no está mal; la calibración
lo estuvo. **Cuesta:** una oración en el §3.5 que le dé a la Etapa 2 la segunda
herramienta que el §3.4 ya definió.

---

## NIVEL 3 — Ya está fuera del draft 4, hoy, corriendo.

### 3.1 `extension_ratio_EXP` decide una etiqueta de severidad *(D8)*

El §3.3 define las tres severidades por **recuperación**, no por postura:
recuperado solo = leve, parcialmente = moderada, quedó en el suelo = severa. El
código, para resolver un clip al cierre, usa la extensión cadera–tobillo —una
cantidad **experimental, con sufijo `_EXP`, que el paper no nombra en ninguna
parte**— como criterio de desempate entre moderada y nula.

Vale **5.4 puntos** de accuracy. Está corriendo ahora mismo. Es la divergencia
más grande que el sistema tiene en producción y la que menos se nota, porque no
rompe nada: produce una etiqueta plausible por un camino que el paper no
autoriza.

Dos salidas, y hay que elegir una antes de la Sección 4:

- **Legitimarla:** el §3.4 gana una quinta cantidad derivada, o el §3.3 gana una
  oración que diga que la severidad se resuelve por postura cuando la
  recuperación no es observable dentro de la ventana. Es honesto y es barato.
- **Quitarla:** −5.4 puntos, y el camino al 90 % se vuelve considerablemente más
  estrecho.

Lo que **no** es una salida es dejarla como está y reportar el 90 % sin
mencionarla.

---

## NIVEL 4 — Contradice el draft 4. No se hace.

### 4.1 Landmarks 3D del mundo para T en caídas axiales

Resolvería las 4 caídas axiales. Y el §3.2 lo prohíbe explícitamente, dos veces:

> *"the monocular z-coordinate is likewise excluded, because its estimated depth
> is not reliable as metric geometry"*

y el §3.4 lo repite al definir T. Además, la razón numérica que el §3.4 da para
dividir por el escalar `|trunk|` —que acota el argumento del arccoseno a [−1, 1]
por construcción, *"which a component-wise 3D normalization does not
guarantee"*— dejaría de aplicar.

**Compra 4 clips de 128 (3.1 puntos) a cambio de contradecir el argumento
central del §3.2.** No se hace. Se documenta como límite (ver 1.3).

---

## Lo que la relectura del 08/09 encontró de nuevo

Releí el §3.1–§3.7 completo contra el código buscando divergencias no
catalogadas. **El catálogo D1–D16 está completo**: no apareció ninguna
divergencia estructural nueva. Se verificaron uno por uno, y coinciden:

- COM con la cadera dominante (`com_hip_weight: 0.65`) contra *"a weighted 2D
  ground-plane projection in which the hip cluster carries the dominant
  contribution"* — coincide. El centroide C de la Cantidad V sí es 0.5/0.5,
  como el §3.4 pide por separado. **Son dos puntos distintos y el código los
  distingue.**
- Polígono de apoyo sobre los seis landmarks 27–32 (tobillos, talones, índices
  del pie) contra *"the ankle landmark positions plus their subsequent foot
  landmarks in bipedal contact"* — coincide, incluida la cláusula "in bipedal
  contact", que `feet_in_contact()` implementa como filtro y no como adorno.
- EMA sobre el componente vertical de C antes de derivar, con constante de
  tiempo calibrable (`ema_time_constant_s: 0.1`) — coincide, y el §3.4 tiene
  razón en declararla parte de la definición y no un detalle: cambia la
  magnitud de V.
- V como derivada sobre una ventana deslizante **en segundos**, no en cuadros —
  coincide con *"computed over a sliding window of frames"* y mejora sobre ella
  por la lección C5.
- Guarda de tronco degenerado a 0.001 px — coincide literalmente con el §3.4.
  (Que sea *demasiado permisiva* es D7, y sigue siendo cierto.)
- Etapa 1 exige V negativa, y el constructor **rechaza** un umbral no negativo
  con `ValueError` — coincide con *"while V remains negative (downward)"*.
- Cantidad I sobre nariz + hombros + caderas — coincide con *"the head and
  torso landmark positions"*.
- Persistencia: el §3.6 y el §3.3 prometen que el detector desplegado no
  escribe nada a disco, y `main.py` escribe siempre el CSV de auditoría. Ya
  estaba anotado en el traspaso como deuda del §3.6; **no es nuevo, pero sigue
  abierto y es una promesa de privacidad, no una preferencia.**

**Un hallazgo menor, y es del código, no del paper.** El docstring de
`trunk_band()` en `quantities.py` dice que *"§3.4 define 15–35° como inclinación
moderada y 60–90° como firma de caída, dejando 35–60° indefinido"*, y explica
que la banda `moderate-lean` absorbe ese hueco. **Eso era el draft 3.** El draft
4 dice 15°–60° y cierra el hueco. Los números del código (15 / 60) ya coinciden
con el draft 4; lo que quedó desactualizado es la justificación escrita al lado.
Cambio de comentario, sin efecto sobre ninguna decisión — pero conviene
arreglarlo antes de que alguien lo lea como si describiera el paper vigente.

---

## Orden propuesto

1. **Contracción del polígono de apoyo** (Nivel 1.1). Cero texto. Es lo que
   sigue.
2. **Decisión suya:** ¿accuracy o sensibilidad como titular? De eso depende si
   el disparador por ventana de pico se calibra a 3.0 o a 2.2.
3. **Decisión suya:** ¿se adopta el disparador por ventana de pico? Es la única
   palanca medida que llega sola al 90 %, y cuesta una oración en el §3.5.
4. **Decisión suya:** ¿se legitima o se quita `extension_ratio_EXP`? Hay que
   resolverlo antes de escribir la Sección 4, no después.
5. Grabar (Nivel 1.2). Es lo único que mueve el techo estructural, y es lo único
   que no depende de mí.
