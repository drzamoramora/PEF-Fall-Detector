# PLAN-90.md — De 59 % a 90 % de accuracy

**Fecha:** 27 de agosto de 2026
**Base de medición:** PEF-FallDB, 128 clips, 1280×720 @ 30 fps, ~10.4 s cada uno.
Verdad en el nombre del archivo. Reparto: NoFall 56, Recovered 24,
PartiallyRecovered 24, NotRecovered 24.

Este documento continúa `FASES.md` (fases 0–7) con las fases **8–14**. Todo lo
que aquí se propone es coherente con el draft 4; donde una fase toca texto del
paper se dice explícitamente y se separa del código.

---

## 1. La aritmética del 90 %

128 clips. **90 % = 116/128**, es decir un presupuesto de **12 errores**.

| momento | aciertos | accuracy | estado |
|---|---|---|---|
| corrida del 26/08 | 76/128 | 59.4 % | medido en vivo |
| + Fase 8 | 95/128 | **74.2 %** | ✅ **VERIFICADO 27/08** — corrida completa de 128 clips, dos veces en máquinas distintas (contenedor y Mac del autor), mismo resultado |
| + Fase 10 | ~103/128 | ~80 % | objetivo |
| + Fase 11 | ~113/128 | ~88 % | objetivo |
| + Fase 12 | ~117/128 | ~91 % | objetivo |

Los 33 errores que quedan después de la Fase 8, **contados sobre la corrida
real del 27/08**, y qué fase ataca cada grupo:

| grupo | n | fase |
|---|---|---|
| falsos positivos sobre NoFall | 12 | Fase 10 |
| caídas sin ningún evento (Etapa 1 nunca disparó) | 14 | Fase 11 |
| confusiones de severidad en la frontera | 7 | Fase 12 |

**Los errores se agrupan por actor, no son aleatorios.** Nueve actores
concentran 22 de los 33 fallos:

```
A06: 3 de 4    A09: 3 de 4    A14: 3 de 4    B10: 3 de 4
A12: 2    A16: 2    A17: 2    B11: 2    B12: 2
```

Eso cambia la naturaleza del trabajo restante: no son 33 casos sueltos sino
unos ocho **comportamientos** sistemáticos. Arreglar uno arregla dos o tres
clips de golpe, y es la razón por la que el 90 % sigue siendo alcanzable.

Matriz de la corrida verificada del 27/08:

```
verdad \ detectada        NoFall  Recovered  Partially  NotRecov  Undeterm
NoFall                        44          3          2         4         3
Recovered                      4         19          1         0         0
PartiallyRecovered             5          1         14         4         0
NotRecovered                   5          0          0        18         1
```

**Advertencia que gobierna todo el plan:** calibrar sobre los mismos 128 clips
con los que se mide el accuracy es sobreajuste, y el número resultante no se
puede publicar. Por eso la **Fase 9 va antes que cualquier calibración**.

---

## Fase 8 — Recuperar lo ya medido (59 % → 74 %) ✅ COMPLETADA 27/08/2026

**Nada de esto toca un umbral ni el paper.** Son tres defectos de
implementación, los tres medidos sobre el dataset.

### 8a — Tolerar lagunas de detección cortas *(+11 puntos)*

**El defecto:** el pipeline abandona el evento que está siendo juzgado ante
*cualquier* cuadro sin detección. Medido: **46 eventos abortados a media
evaluación, en 37 de los 128 clips**.

**Por qué es un defecto y no una decisión:** el razonamiento original es
correcto para el **estado de movimiento** — derivar la velocidad a través de
una laguna produce picos falsos, y eso ya se midió (+26 torso/s en una laguna
de un solo cuadro). Pero ahí se mezclaron dos cosas: reiniciar los filtros es
una cosa, y **botar una evaluación en curso es otra**. Las Etapas 2 y 3 ya
saben manejar entradas NaN — ese fue exactamente el criterio de "pies ocluidos
no significa 'no hubo caída'".

Y ocurre en el peor momento posible: la laguna aparece en el impacto, que es
cuando la pose es menos confiable y cuando el embudo más necesita mirar.

**La medición:** lagunas de detección, n=309 — mediana **0.10 s**, p90 1.30 s,
**82 % por debajo de 0.5 s**.

| tolerancia | accuracy |
|---|---|
| hoy (aborta con cualquier cuadro perdido) | 58–64 % |
| 0.20 s | 65.9 % |
| **0.35 s** | **69.0 %** |
| 0.50 s | 69.0 % |
| nunca aborta | 69.8 % |

**Qué se hace:** separar las dos responsabilidades en `pipeline.py`. El estado
de movimiento sigue reiniciándose siempre; el evento pendiente sólo se abandona
si la laguna supera `history_max_gap_s`, **parámetro que ya existe en
`config.yaml` con valor 0.5**.

**Criterio de salida:** accuracy ≥ 69 % en replay, y una prueba que falle si se
vuelve a abortar un evento por una laguna de un cuadro.

### 8b — Tres bandas de postura final *(+5 puntos)*

**El defecto:** `upright_T_deg = 30°` parte el mundo en dos — erguido o tirado
— y manda "se recuperó a medias" al cajón de "quedó tirado". De los 24 clips
PartiallyRecovered, **9 salen NotRecovered**.

Ángulo mediano de tronco al final del clip:

```
aciertos:  0.8   3.9   8.4  11.0  13.9  14.8  17.1  22.9 grados
fallos:   34.9  39.5  40.7  45.7  47.0  54.9  72.4  126.2  176.0
```

Alguien que se sentó en el piso, quedó de rodillas o se apoyó en un mueble
termina entre **35° y 55°**. Esa postura intermedia es literalmente lo que el
§3.3 llama *partially recovered*; un corte binario en 30° no puede
representarla.

**Qué se hace:** tres bandas al cerrar el clip, usando el `lying_T_deg = 60`
que ya está en `config.yaml`:

```
T_final <  30°  y piernas extendidas -> Recovered
T_final 30-60°                       -> PartiallyRecovered
T_final >= 60°                       -> NotRecovered
```

PartiallyRecovered pasa de **8/24 a 14/24**. Probado también a 55° (igual) y a
70° (peor: le cuesta a NotRecovered).

**Coherencia con el paper:** el §3.3 ya define tres desenlaces. Esto hace que
el código tenga tres salidas en vez de dos. Va **hacia** el paper.

### 8c — Tope de plausibilidad en `extension_ratio_EXP` *(0 puntos)*

**20 de 128 clips** registran valores físicamente imposibles, hasta **183.5
longitudes de torso**. Ocurre cuando el torso está escorzado y se divide por
pocos píxeles.

**No cambia ni una sola etiqueta** — lo verifiqué clip por clip. No es un
arreglo de exactitud, es de **integridad del registro**: esos CSV son el
instrumento de investigación que respalda la afirmación de auditabilidad del
§3.5, y un revisor que abra uno y encuentre una pierna de 183 cuerpos tiene
razón para desconfiar de las demás columnas.

Va de paquete porque son dos líneas.

---

## Fase 9 — Protocolo de evaluación honesto (§3.7)

**Va antes que cualquier calibración. Sin esto, los números de las Fases
10–12 no son publicables.**

### 9a — Partición por actor, no por clip

Los 128 clips vienen de **33 actores con 4 clips cada uno**, y cada actor tiene
una sola clase. Partir por clip metería clips del mismo actor a ambos lados y
el modelo aprendería a esa persona, no al fenómeno. La partición tiene que ser
**por actor**.

Propuesta: 22 actores para calibrar, 11 para medir (≈ 2/3 – 1/3), estratificada
por clase, con la lista de actores **escrita en un archivo versionado** para
que la partición sea reproducible y citable en la Sección 4.

### 9b — Métricas que el §3.7 pide, no sólo accuracy

Accuracy sobre 4 clases esconde lo que importa clínicamente. Hay que reportar
además:

- **Binario caída / no-caída**: sensibilidad y especificidad. Un sistema que
  clasifica mal la severidad pero detecta todas las caídas sirve; uno que
  acierta severidades y pierde caídas, no.
- **Recall por clase**, que es donde está el problema real (PartiallyRecovered).
- **Falsos positivos por hora**, la métrica que el §3.6 necesita para el borde.
- La fila `Undetermined` **completa y visible**, nunca colapsada en NoFall.

### 9c — Arnés de ablaciones (era la Fase 6)

El §3.7 exige quitar T, V, P e I de a una. Con el runner de lote ya construido,
esto es un bucle sobre configuraciones. Incluye la ablación que ya tenemos
medida y que resuelve la divergencia #12 (ver Fase 13).

**Criterio de salida:** un comando produce la tabla de la Sección 4 completa,
sobre la partición de prueba, con ablaciones.

---

## Fase 10 — Falsos positivos: la guarda de plausibilidad (divergencia #4)

**El problema:** 11 clips NoFall reciben etiqueta de caída. Cuatro salen
`NotRecovered/severe` con 99.7–100 % de cuadros confiables (B11-S3, B11-S4,
B12-S1, B12-S4). Disparan entre t=3 y t=6.6 s. **No es ruido: son ADL que
producen una firma T+V real.** Ninguno de los arreglos de la Fase 8 los mueve.

**Coherencia con el paper:** esto **no es una función nueva**. El draft 4
promete una guarda de plausibilidad que el código nunca implementó — es la
divergencia #4, y la #9 depende de ella. Aquí el código se pone al día con el
paper, no al revés.

**Qué se construye:**

1. **Guarda de plausibilidad sobre la geometría**: un torso de 2 px con
   confianza 0.99 no es una observación, es el modelo extrapolando. Un tope
   inferior en `torso_length_px` relativo al alto del cuadro, y un tope
   superior a la razón de extensión (ya en 8c). Esto también evita que la
   Etapa 1 dispare sobre ángulos calculados de un vector de 2 píxeles.
2. **Discriminación de ADL**: caracterizar los 11 con las trazas por cuadro —
   sentarse tiene V negativa sostenida y moderada; una caída tiene un pico
   corto y grande. La firma temporal difiere aunque el máximo se parezca. Sale
   de los datos, no de la intuición.
3. **Calibrar sobre los 56 NoFall** — que es lo que estaba esperando desde que
   dijimos "eso lo vemos con el dataset".

**Criterio de salida:** NoFall ≥ 52/56 en la partición de prueba, sin perder
recall de caídas.

---

## Fase 11 — Las 16 caídas que nunca dispararon

**Son dos problemas distintos y hay que medirlos por separado antes de tocar
nada.**

### 11a — Desfase entre T y V

Varias de las 16 tienen T **y** V cruzando sus umbrales, pero en momentos
distintos. A09-S2 llega a T=173° y V=−2.48 y aun así no dispara. La
formulación `score` exige la combinación **en un mismo cuadro**, y en una caída
real el pico de V (el descenso) y el pico de T (después del impacto) están
separados en el tiempo.

`sequential` existe precisamente para eso, con `confirm_window_s`. **Hay que
medir cuántas de las 16 recupera** — es una tarde de trabajo y ya tenemos el
barrido armado.

Lo medido hasta ahora sobre los 128 clips, sin los arreglos de la Fase 8:

| formulación | accuracy |
|---|---|
| score (2.2) | 68.2 % |
| score (2.6) | 70.5 % |
| sequential | 68.2 % |
| v_only | 62.1 % |
| simultaneous | 55.3 % |

Hay que repetirlo **después** de la Fase 8 y **sobre la partición de
calibración**, porque estos números están contaminados por el defecto 8a.

### 11b — Caídas axiales

En otras, T **nunca sube**: A17-S2 llega a 9.1°, A13-S4 a 23.1°, A16-S2 a
32.2°. El sujeto cae hacia o desde la cámara, el tronco se proyecta casi a un
punto y la inclinación 2D deja de existir. En A14 el torso pasó de 209 px a
**2 px** con el modelo reportando 0.99 de confianza.

**El dato que abre la puerta:** el torso en coordenadas 3D de MediaPipe
(`world_torso_len_m_EXP`, que ya registramos) aguanta mucho mejor — cae de
0.458 a 0.209 m (−54 %) donde el de píxeles cae −99 %. La información está; el
sistema no la usa.

**Dos caminos, y la decisión es de quien escribe el draft:**

- **(A)** Calcular T desde los landmarks de mundo cuando el torso proyectado
  sea implausible. Toca una cantidad definida por el paper (§3.2 fija espacio
  de píxeles) → **necesita texto en el draft**.
- **(B)** Documentarlo como limitación del método 2D. El §3.2 ya reconoce la
  ambigüedad de profundidad.

Antes de decidir hay que **contar cuántos de los 128 son axiales**. Si son 4,
se documenta; si son 15, (A) deja de ser opcional. Ese conteo es la primera
tarea de la fase.

**Criterio de salida:** las 16 clasificadas en 11a / 11b con números, y el
recall de caídas subiendo al menos 10 clips.

---

## Fase 12 — Calibración sobre la partición de entrenamiento

Sólo después de las Fases 9–11, y **sólo sobre los 22 actores de calibración**.

- `upright_T_deg` y `lying_T_deg` — las fronteras de las tres bandas.
- `trigger_score`, `trigger_hold_s`, `confirm_window_s`.
- `com_outside_fraction`, `com_eval_window_s`, `contact_band_torso`.
- `epsilon` y `threshold_W_seconds` — ver Fase 13.
- `com_hip_weight`: hoy 0.65 "porque la anatomía"; ahora se puede medir.

Cada valor queda en `config.yaml` con el número, la fecha y la partición sobre
la que se calibró. Es lo que hace reproducible la Sección 4.

**Criterio de salida:** accuracy ≥ 90 % en la partición de **prueba**, no en la
de calibración.

---

## Fase 13 — Las dos deudas con el draft

Ninguna se resuelve con código. Ambas son de quien escribe.

### 13a — La Cantidad I hoy no decide nada

En los 15 clips NotRecovered que confirmaron: I_max promedio **1.12 s**, máximo
**3.57 s**, y **ninguno** alcanzó el umbral W = 5.0 s. Todos los veredictos
`severe` salieron de la regla de postura final, no de la inmovilidad. El
desplazamiento por cuadro tiene mediana 0.038–0.044 torsos contra un ε = 0.05:
**el umbral está por debajo del piso de ruido**.

Peor: subir ε no arregla nada, lo voltea. Con ε=0.15, el clip A13 —donde el
señor se levanta solo— pasa de `moderate` a `severe`, porque la rama de
inmovilidad alcanza los 5 s *antes* de que se levante. Las dos etiquetas más
distantes del conjunto.

**Dos salidas:**
- Recalibrar ε y W hasta que la rama vuelva a existir, sabiendo que compite con
  la rama de recuperación.
- O que el draft diga que **I es evidencia de apoyo y no un umbral de
  confirmación**, que es lo que el código hace hoy de facto.

Lo que no se puede es dejar el §3.4 definiendo un umbral que en la práctica no
decide nada.

### 13b — La divergencia #12: un `_EXP` decide una etiqueta del paper

`severity_uses_leg_extension: true` mete `extension_ratio_EXP` en la decisión
leve/moderada. Medido:

| criterio | accuracy | Recovered | Partially |
|---|---|---|---|
| `extension_ratio_EXP` (hoy) | **74.2 %** | 19/24 | 14/24 |
| P: COM de vuelta dentro del apoyo (§3.4) | 70.3 % | 16/24 | 12/24 |
| P o extensión | 72.7 % | 19/24 | 12/24 |
| apagado, duda → Recovered | 68.8 % | 20/24 | 6/24 |
| apagado, duda → Partially | 60.2 % | 0/24 | 15/24 |

Probé la salida elegante — usar **P**, que sí es del paper — y **cuesta 4
puntos**. No la vendo como equivalente.

**Sugerencia:** quedarse con la extensión y convertir la infracción en un
resultado reportado. El §3.3 ya se comprometió con tres etiquetas; un paper que
nombra tres desenlaces le debe al lector la señal que separa los dos primeros,
porque con T sola no se separan. La tabla de arriba **es** la ablación que el
§3.7 exige. Frase propuesta para el §3.4:

> *La razón de extensión cadera-tobillo, normalizada por la longitud del torso,
> complementa a T en la Etapa 3: distingue una recuperación completa (el sujeto
> se puso de pie) de una parcial (se sentó o quedó de rodillas), que T por sí
> sola no puede separar porque ambas posturas presentan el tronco erguido.*

Con esa frase, `extension_ratio_EXP` deja de ser `_EXP` y la #12 se cierra. Sin
ella, el número honesto a reportar es 70.3 %, no 74.2 %.

---

## Fase 14 — Rediseño del GUI *(pendiente desde hace días)*

Pedido explícito y nunca empezado. Ahora tiene requisitos que antes no existían:

- El panel de cola creció encima de una ventana pensada para un solo clip.
- Revisar 128 clips necesita navegación, filtros (mostrar sólo los fallos) y
  comparación lado a lado de la traza contra el video.
- Los tiempos por sección ya se instrumentaron pero nunca se analizaron.
- **El problema de 5 fps en cámara viva sigue sin diagnosticar.** La
  instrumentación está puesta; falta correr una sesión y leerla.
  `velocity_window_s: 0.167` es **más corto que un intervalo de cuadro a 5 fps**,
  con lo cual V degenera a una diferencia de un solo cuadro. No afecta la
  evaluación en video, pero sí al §3.6.

---

## 2. Deudas abiertas que no son fases

Cosas dichas en el camino y nunca cerradas. Van aquí para que no se pierdan.

| # | deuda | estado |
|---|---|---|
| 1 | **Trabajo sin commit** — el último commit es `143c697`. Todo lo demás (Fases 3, 4, 5, alertas, clasificación, cola, README) está sin versionar | **Requiere su autorización** |
| 2 | Entregable D de la sub-parte 2.7 — el paquete de traspaso al draft, que sólo existe en el chat | Pendiente |
| 3 | Divergencias del paper: cerradas #1, #2, #3, #11. Abiertas: **#4** (Fase 10), #5, #6, #7, #8, #9 (depende de #4), **#12** (Fase 13b), #13 | Parcial |
| 4 | Transeúntes / multi-persona — usted lo planteó, nunca se discutió | Sin abrir |
| 5 | §3.8 ética y privacidad | Diferido |
| 6 | README desactualizado otra vez (cola, clasificación, cuatro clases) | Pendiente |
| 7 | Cuatro nombres anómalos en el dataset: `A06-S14`, `A11-S14`, `A3-S2`, y `Recovery` ×4. Sólo se normaliza `Recovery`→`Recovered` | Documentado, no tocado |
| 8 | Fase 1 "completa por código, nunca observada en el Mac" | Ya superada de hecho |

---

## 3. Orden recomendado

1. **Fase 8** — dos días. Recupera 15 puntos que ya están medidos. Sin riesgo,
   sin tocar el paper.
2. **Fase 9** — antes de calibrar nada. Es lo que hace publicable todo lo demás.
3. **Fase 11a** — barata: barrer formulaciones ya con la Fase 8 puesta.
4. **Fase 10** — la más grande. Es la divergencia #4, o sea deuda con el paper.
5. **Fase 11b** — necesita el conteo de caídas axiales antes de decidir.
6. **Fase 12** — calibración final.
7. **Fase 13** — al traspaso, en paralelo con todo lo demás.
8. **Fase 14** — GUI, cuando el motor esté quieto.

## 4. Riesgos

- **El 90 % puede no alcanzarse sin la opción (A) de la Fase 11b.** Si muchas
  de las 128 son axiales, la limitación es del método 2D y ningún umbral la
  arregla. Por eso el conteo va primero.
- **Sobreajuste.** Con 128 clips y ~20 parámetros es fácil llegar al 95 % sobre
  el conjunto entero y no poder publicarlo. La Fase 9 es la defensa.
- **Los objetivos de las Fases 10–12 son objetivos, no mediciones.** Lo único
  medido en este documento es la Fase 8 (74.2 %) y la tabla de la 13b.
