# B05 — por qué tres de cuatro sujetos salen `NotRecovered`

*Medido 23/09 sobre los cuatro clips de B05, con `model_complexity: 2` y el
runner de etiquetado (`labelling=True` + `finalise()`), que es el único modo
que reproduce las etiquetas del §3.3.*

## El resultado

| clip | veredicto | T final (mediana) | H final (mediana) | hombro mínimo |
|---|---|---|---|---|
| B05-S1 | `stage3_confirmed` **severe** ✗ | 30.0° | 0.227 | **1.7 px** |
| B05-S2 | `stage3_confirmed` **severe** ✗ | 47.8° | 0.294 | **7.9 px** |
| B05-S3 | `stage3_confirmed` **severe** ✗ | 40.5° | 0.118 | **0.3 px** |
| B05-S4 | sin evento — `NoFall` ✓ | 37.5° | 0.408 | 16.3 px |

**En ningún clip de B05 el tronco pasa de 49°.** T máxima: 30.4 / 49.0 / 43.9 /
37.9. Cero cuadros en estado `LYING`; el estado dominante es `LEANING` (188–211
cuadros de 313). Detección 100 % en los cuatro. El sujeto se inclina y se vuelve
a enderezar: nunca está en el suelo.

Y aun así tres salen **severe**, que es «permaneció en el suelo» — la etiqueta
más grave del §3.3.

## La causa: el ancho de hombros se proyecta a casi nada

`H_raw = extensión_vertical / ancho_de_hombros`, y la única guarda es
`shoulder_width_px <= 0.0` (`quantities.py:422`).

En B05 el sujeto queda de frente a la cámara. Los dos hombros se proyectan
casi uno encima del otro, el divisor colapsa, y `H_raw` explota:

```
B05-S1, t=2.63:  torso 171 px,  hombro implícito 1.7 px  ->  H_raw = 98.5
B05-S3, t=2.80:  torso 179 px,  hombro implícito 0.3 px
```

Un hombro humano mide del orden de 0.5–1.0 largos de torso. 1.7 px con 171 px
de torso no es un cuerpo: es el divisor deshaciéndose.

## Y la línea base se traga la basura

`HeightBaseline` es una EMA con τ = 2.0 s que solo se mueve en cuadros que el
llamador juzgó «de pie con confianza» — **por postura, no por plausibilidad**.
En B05-S1 durante ese tramo T vale 3–25°, así que califica como de pie y la
línea base ingiere los valores disparatados:

```
  t      T    H_raw   H_base  H_ratio
1.87    5.3   77.98    24.03   3.245
2.63    7.4   98.53    35.32   ...      <- el pico
2.93   25.1   15.06    35.32   0.426    <- H_raw vuelve a su escala real
3.07   17.7    6.89    35.32   0.195
3.80   14.2    6.33    27.49   0.230
```

A partir de t = 2.97 `H_ratio` queda bajo 0.35 **para siempre** — 7 de los 10 s
del clip — porque el numerador volvió a ~6 y el denominador quedó anclado en 35.

## El disparo final

`Stage3Evaluator.finalise()` pregunta `_is_lying(final_t, final_h)`:

```python
def _is_lying(self, t_deg, h_ratio):
    if not math.isnan(t_deg) and t_deg >= self.lying_t_deg:   # 60°, NO se cumple
        return True
    return (self.threshold_h_ratio_lying > 0.0 and not math.isnan(h_ratio)
            and h_ratio <= self.threshold_h_ratio_lying)      # 0.35, SÍ se cumple
```

T final 30–48° nunca llega a 60. Pero H final 0.12–0.29 sí cae bajo 0.35, así
que `_is_lying` devuelve True y el veredicto es `(CONFIRMED_FALL, SEVERE)`.

Sin H, la regla de tres bandas habría leído T final entre 30° y 60° y dicho
`moderate` («se sentó, se arrodilló»), o incluso `mild`. Cualquiera de las dos
está mucho menos equivocada que `severe`.

**B05-S4 es el control perfecto**: mismo escenario, mismo sujeto-situación, pero
su hombro nunca baja de 16.3 px. La línea base no se contamina, H final da 0.408
(por encima de 0.35), y el clip sale correcto sin siquiera disparar.

## Tres arreglos, en orden de importancia

**1 · Guarda de plausibilidad sobre `H_raw`.** Exactamente el tratamiento que
`extension_ratio` ya recibe con `max_extension_ratio` (y por la misma razón: ahí
se registraron 183 largos de torso). Si la extensión vertical supera unos 4
anchos de hombro, el divisor está escorzado y la lectura debe ser **NaN
(desconocido)**, nunca un número que parece medido.

Mejor aún, la guarda natural es sobre el divisor y relativa al torso: si
`ancho_hombros < ~0.15 × torso_length_px`, no es un hombro medible. Los cuatro
clips de B05 separan limpio con ese corte — los tres que fallan bajan de 8 px
con torsos de 115–180 px; el que funciona se queda en 16.3.

`MIN_SHOULDER_PX = 1e-3` no sirve para esto: solo atrapa el cuadro degenerado.

**2 · Que la línea base no ingiera cuadros no plausibles.** Es la que carga el
peso real. Aunque (1) deje el *ratio* en NaN, si la base ya se comió el pico el
daño persiste los siguientes segundos. El filtro de postura no basta: hay que
filtrar también por plausibilidad antes de alimentar la EMA.

**3 · Separar `threshold_h_ratio_lying` de `trigger_H_ratio`.** Hoy
`pipeline.py:295` reusa el mismo 0.35 para las dos. Son dos preguntas distintas
—«¿se está desplomando?» contra «¿está en el suelo?»— y atarlas a un número
significa que calibrar una rompe la otra. Además impide apagar una sin apagar la
otra, que fue justo lo que me estorbó al intentar aislar las mitades de H.

## Alcance

De los 12 falsos positivos de la corrida de QuantityH, **ocho salen
`NotRecovered`**: B05-S1, B05-S2, B05-S3, B11-S3, B11-S4, B12-S1, B12-S4,
B14-S2. B05 está medido y explicado. Los otros cinco tienen la misma firma
—NoFall que termina en la etiqueta más grave— y valen la misma revisión antes
de tocar ningún umbral.

Si el arreglo recupera los ocho, la especificidad sube de 78.6 % a ~92 % sin
tocar la sensibilidad de 93.1 %.

---

# CORRECCIÓN — revisados los otros nueve falsos positivos (23/09)

Me equivoqué al estimar el alcance. **El arreglo de plausibilidad recupera tres
clips, no ocho.** Medido, no supuesto.

*(Estas corridas son con `model_complexity: 2`, que es lo que hay hoy en
`config.yaml`. La corrida de los 128 fue con complexity 1, así que las
clasificaciones no son directamente comparables clip a clip.)*

| clip | Tmax | **T final** | H final | hombro mín | LYING | veredicto |
|---|---|---|---|---|---|---|
| B05-S1 | 30.4 | **30.0** | 0.227 | 1.7 px | 0 | severe |
| B05-S2 | 49.0 | **47.8** | 0.294 | 7.9 px | 0 | severe |
| B05-S3 | 43.9 | **40.5** | 0.118 | 0.3 px | 0 | severe |
| B05-S4 | 37.9 | 37.5 | 0.408 | 16.3 px | 0 | ✓ sin evento |
| B11-S3 | 97.3 | **84.9** | 1.649 | 7.8 px | 204 | severe |
| B11-S4 | 98.3 | **90.4** | 0.005 | 7.8 px | 271 | severe |
| B12-S1 | 107.1 | **88.1** | 0.076 | 0.6 px | 163 | severe |
| B12-S4 | 106.6 | **89.0** | 0.039 | 2.2 px | 202 | severe |
| B09-S4 | 103.5 | 11.1 | 1.304 | 4.0 px | 70 | nullified/mild |
| B10-S1 | 87.9 | 11.0 | 0.820 | 4.0 px | 51 | confirmed/moderate |
| B10-S3 | 176.3 | 6.4 | 0.603 | 1.4 px | 85 | confirmed/moderate |
| B14-S2 | 113.9 | 6.8 | 0.742 | 3.5 px | 115 | nullified/mild |
| B03-S3 | 74.0 | 7.2 | 1.446 | 7.8 px | 4 | ✓ sin evento |

## Son tres familias distintas, no una

**Familia 1 — B05 (×3): H envenenada.** T final 30–48°, **cero cuadros LYING**.
`_is_lying` sale True *solo* por H. Es el caso descrito arriba y la guarda de
plausibilidad los arregla.

**Familia 2 — B11-S3, B11-S4, B12-S1, B12-S4: el sujeto SÍ termina horizontal.**
T final **84.9–90.4°**, y entre 163 y 271 de ~313 cuadros en LYING. `_is_lying`
devuelve True **por T ≥ 60°, sin que H intervenga**. Estos salen `severe` con o
sin Quantity H — y probablemente ya eran falsos positivos antes de H.

No es un defecto de medición: el sistema mide bien que la persona está tumbada.
Es que **acostarse a propósito es indistinguible de una caída por pura postura**,
que es literalmente lo que dice el docstring de Quantity R: *"T, V, P, I and H
all read POSTURE... None reads INTENT"*. Estos cuatro son el caso que R fue
inventada para atacar, y R hoy solo se registra.

**Acción pendiente: ver B11 y B12 en video.** Hace falta saber qué hace el
sujeto — acostarse en una cama, tumbarse a hacer ejercicio, agacharse a recoger
algo del suelo. Sin eso no se puede decidir si hay señal que los separe.

**Familia 3 — B09-S4, B14-S2, B10-S1, B10-S3: el veredicto no depende de
`_is_lying`.** T final 6–11°, el sujeto termina de pie. B09-S4 y B14-S2 salen
`nullified/mild`; B10-S1 y B10-S3 caen en la rama de «no puedo verificar las
piernas» → `moderate`. Son fallos del **disparador**, no del veredicto final.

## Alcance corregido

| | especificidad | sensibilidad |
|---|---|---|
| QuantityH hoy | 78.6 % (44/56) | 93.1 % |
| con la guarda de plausibilidad | **~83.9 % (47/56)** | 93.1 % (sin tocar) |

Iguala la especificidad de la referencia anterior **conservando los 93.1 % de
sensibilidad** — que es el intercambio que llevábamos meses sin conseguir. Pero
no llega a 92 %: eso lo dije de más.

Los otros nueve necesitan trabajo distinto, y dos vías separadas:
Quantity R / intención para la familia 2, y el disparador para la familia 3.

---

# B11 y B12 — límite del método, documentado (23/09)

Revisión visual del usuario:

- **B11**: cámara un poco inclinada; la persona va hacia el sofá y se acuesta a dormir.
- **B12**: cámara no inclinada; la persona se acuesta a dormir en el suelo.

Por el §3.3, `NotRecovered` es *"a fall occurred and the subject remained on the
floor"*. Alguien que se acuesta a dormir en el suelo cumple eso punto por punto
para T, V, P, I y H: las cinco leen postura, y la postura final es la misma.

## Tres vías medidas, las tres negativas

Contra 8 caídas `NotRecovered` reales (A02 ×4, A08 ×4):

**Velocidad de descenso** (V media mínima sostenida 0.2 s) — se solapa.
ADL −3.19 / −1.22 / −1.04 / −1.48; caídas −1.44 / −3.64 / −1.55 / −1.52 / −6.11 /
−18.24 / −5.78 / −6.00. Un corte en −2.0 rechaza 3 ADL pero pierde 3 caídas
(A02-S1, S3, S4). Mismo empate que el piso de V medido en agosto.

**Descenso por etapas** — no existe. El tiempo de T de 30° a 60° da 0.00 s en 3
de los 4 ADL: el esqueleto salta igual que en una caída.

**Aproximación horizontal** (desplazamiento 2 s antes) — no separa. B12-S1 da
0.22 torsos, igual que A08-S1 (0.30) y A08-S4 (0.14).

## Lo que separa y NO se usa

Altura final del centroide en la imagen: ADL 0.667–0.694, caídas 0.691–0.877.
Separa casi limpio en esta muestra, pero depende de la posición e inclinación de
la cámara —y B11 tiene la cámara inclinada—. `person_state.py` ya lo descarta
explícitamente como rasgo de postura. Se ajustaría a estos clips y se rompería en
cualquier otra grabación. La versión física válida (altura sobre la superficie de
apoyo) exige estimar el plano del piso: fuera de un montaje monocular sin calibrar.

## Decisión

**Se documentan como limitación del método en el Discussion, no se persiguen.**
Un sistema que mide postura no puede distinguir intención. Valen 4 clips (3.1
puntos) de los 128.

---

# Errores en caídas — dos mecanismos más, medidos (23/09, complexity 2)

**A12-S1, A12-S4 (PartiallyRecovered → NotRecovered): H manda sobre una T válida.**
T final 35.0° y 42.5° → la regla de tres bandas diría `moderate` (correcto).
Pero H final 0.344 y 0.213 ≤ 0.35 → `_is_lying` True → `severe`. En A12-S1 no hay
pico de H_raw (hombro mín 13.9 px): H está baja *legítimamente* porque el sujeto
está sentado en el suelo. La guarda de plausibilidad NO arregla A12-S1; lo que lo
arregla es que H solo decida «tumbado» cuando T no esté disponible (NaN), que es
su propósito declarado de respaldo.

**A06-S14, A06-S2, A06-S3, A09-S3 (PartiallyRecovered → NotRecovered): esqueleto.**
T final 148 / 133 / 93 / 175°. Probé «T > 120° al final = esqueleto invertido →
moderate»: arregla A06-S14, A06-S2, A09-S3 pero ROMPE A08-S3 y A08-S4, caídas
NotRecovered reales que también terminan en 164° y 175°. Neto +1. Descartado.

**A08-S1, A08-S2 (NotRecovered → NoFall con complexity 2): se pierde el esqueleto
estando en el suelo.** Último esqueleto válido con T ≈ 176° (tumbado), y luego
MediaPipe no detecta a nadie durante 5.5 s y 3.7 s hasta el final del clip.
`finalise()` mira solo el último segundo de reloj → sin ángulos → UNRESOLVED →
NoFall. Una caída real, con la persona en el suelo, etiquetada «no pasó nada».
Con complexity 1 estos dos se detectaban y fallaba A08-S3: **el modelo cambia qué
caídas se pierden.**
