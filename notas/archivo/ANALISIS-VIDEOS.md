# ANALISIS-VIDEOS.md — qué se aprendió mirando los clips que fallan

Rama `camino-90`. Abierto el 10/09/2026, durante la revisión clip por clip de
los errores de la corrida con ventana de pico y umbral 2.6.

**Para qué existe:** la revisión visual produjo diagnósticos que ninguna
medición mía había alcanzado, y varios de ellos contradicen conclusiones que
yo había dado por buenas. Este documento guarda lo aprendido y el estado de
cada arreglo candidato, para implementarlos cuando la revisión termine.

**Estado de la revisión:** 11 de 28 clips revisados (A09 ×3, A14 ×3, A17 ×3,
A06 ×2).

---

## Lo primero, porque cambia cómo leer todo lo demás

**Ninguna de las tres sospechas de "clip mal anotado" resistió la revisión.
Cero de tres.**

Yo había marcado A09-S3, A06-S14 y A06-S3 como probables errores de
anotación, apoyado en que los tres terminan con el tronco a 72°–176° y por lo
tanto "el sujeto sigue en el suelo". Al verlos:

| clip | lo que yo medí | lo que pasa de verdad |
|---|---|---|
| A09-S3 | T final 176° | el esqueleto se invirtió; el sujeto está de rodillas |
| A06-S14 | T final 130° | sí se recupera, en el segundo 9 |
| A06-S3 | T final 72° | sí se recupera, en el segundo 7–8 |

La causa del error es la misma en los tres: **medí la postura final con T, y T
se calcula desde un esqueleto que en esos clips no describe el cuerpo.** Una
cantidad derivada no puede validar su propia entrada. Cualquier diagnóstico
que se apoye solo en las cuatro cantidades hereda ese punto ciego.

De ahí la regla de trabajo para lo que queda: **la revisión visual decide qué
pasó; la medición decide cuánto y dónde.** Ninguna de las dos sola alcanza.

---

## Hallazgos por situación

### A09 — el esqueleto se invierte o se estira

Los tres clips fallidos terminan con el sujeto **de rodillas, apoyando una
mano en el suelo**. MediaPipe no representa esa postura.

En A09-S3 la falla es medible y brutal:

```
t = 4.03 s   T =   2.5°    confianza = 0.957
t = 4.07 s   T = 175.0°    confianza = 0.961
```

**163 grados en 33 milisegundos, y la confianza SUBE.** El esqueleto queda
invertido los seis segundos restantes y nunca vuelve.

En A09-S1 y A09-S2 no hay inversión sostenida: MediaPipe le ajusta un
esqueleto de persona de pie a alguien arrodillado, con extensión de piernas
entre 1.57 y 1.91 (o sea "piernas extendidas"). El sistema lee esa postura
inventada y concluye `Recovered`.

**Causa raíz: esqueleto.** No hay nada que arreglar en el §3.4 ni en el §3.5.

### A14 — el cuerpo cae hacia la cámara y el tronco se escorza

Observación: *"cae de frente sin rotar"*, *"los pies no son visibles"*.

```
                torso inicio -> final   ratio    T final
A14-S4              119 px -> 68 px      0.57     27.8°
A14-S1              113 px -> 96 px      0.85     98.5°
A14-S3  (acierta)   111 px -> 109 px     0.98     86.7°
A14-S2              107 px -> 132 px     1.23     10.5°
```

En A14-S4 el tronco pierde el 43 % de su largo proyectado mientras T cae a
27.8°: el cuerpo se acuesta **alejándose de la cámara**, así que en 2D el
tronco se acorta en vez de rotar. Un tronco corto y vertical es indistinguible
de una persona de pie.

**Causa raíz: límite 2D** en S4 (el escorzo domina; aunque los pies se vieran,
seguiría leyéndose como alguien de pie), **encuadre** en S1.

### A17 — el sujeto queda recostado en una pared

Observación: *"cae hacia una pared, queda recostado en ella, pero eso no
significa que se recupera; el tronco casi siempre queda vertical"*.

**Este caso no es un fallo del esqueleto.** Una persona derrumbada contra una
pared tiene el tronco tan vertical como una de pie. El esqueleto es correcto y
T también: lo que falla es que **T no puede distinguir "de pie" de "en el
suelo apoyado"**, y ninguna de las cuatro cantidades del §3.4 puede.

Dato adicional: A17 tiene el torso más grande del dataset (159–214 px contra
~100 en el resto), o sea la cámara mucho más cerca. A17-S3 y A17-S4 **empiezan
el clip con T ≈ 174°**, con el sujeto todavía de pie — sin verificar cuadro a
cuadro, pero es sospechoso.

### A06 — la recuperación está grabada, pero no en T

A06-S14, últimos dos segundos:

```
   t      T      cadera_y
 8.10   148.8      515
 8.70   121.0      480
 9.30   123.6      454
 9.90   134.8      418
10.20   149.7      392
```

**T se queda entre 121° y 150° —"tumbado"— mientras la cadera sube 123 píxeles
de forma monótona.** Esa subida es la recuperación del segundo 9. Está en el
registro, es limpia, y ninguna etapa la consulta.

A06-S3 es distinto: el sujeto llega a una postura estable (T entre 72° y 84°,
cadera quieta) que es de rodillas o apoyado. El corte de 60° lo manda a
"tumbado".

**En los dos, `extension_ratio_EXP` está en NaN de principio a fin** — los pies
nunca son medibles, así que el único desempate entre "apoyado" y "tumbado" no
existe y T decide solo.

**Causa raíz: lógica** en ambos. La información estaba disponible.

---

## El hallazgo estructural

De las **24 caídas `NotRecovered`** —donde el sujeto está en el suelo al
terminar el clip, por definición de la etiqueta:

```
esqueleto fiel al final (T>=60)    19 clips    aciertan 19   (100 %)
esqueleto infiel        (T<60)      4 clips    aciertan  0   (0 %)
```

Sin excepciones en ninguno de los dos lados.

Hay circularidad parcial y conviene decirlo antes de que la pregunten: la
regla de severidad *es* "T final ≥ 60° → severa". Lo que **no** es circular es
que en **4 de 24 clips el esqueleto diga "de pie" mientras la persona está
demostrablemente en el suelo** — 17 % de fidelidad perdida, medida contra
verdad verificada por un humano, sin que la lógica intervenga.

La conclusión que sí se sostiene: **la lógica del §3.4 y §3.5 nunca falló
sobre un esqueleto fiel, y nunca salvó uno infiel.**

### Y MediaPipe no avisa

```
confianza reportada en el último segundo y medio

  esqueleto fiel    (19 clips)   mediana  0.998
  esqueleto infiel   (4 clips)   0.998, 1.000, 1.000, 1.000
```

Los esqueletos equivocados reportan **más** confianza que los correctos, y en
el cuadro exacto de la inversión de A09-S3 la confianza sube de 0.957 a 0.961.

**Esto toca la afirmación central del §3.1.** Esa sección sostiene que el marco
es explicable porque *"a flagged event can be reviewed by reading off a trunk
angle, a centroid velocity, a center-of-mass projection offset, and an
immobility duration"*. En A14-S4 ese ejercicio se puede hacer perfectamente:
ángulo 27.8°, velocidad razonable, inmovilidad medida. **La explicación es
completamente coherente y completamente falsa**, y nada en la cadena lo
delata.

O sea: la explicabilidad garantiza que la decisión sea **auditable**, no que
sea **correcta**. Es demostrable con estos cuatro clips y ningún benchmark que
corte en el impacto lo puede mostrar, porque el fallo solo aparece en la
ventana post-evento que PEF-FallDB agrega. Material para la Discusión.

---

## Vías cerradas — diez, todas con medición

Ninguna se descartó por intuición.

| vía | por qué no |
|---|---|
| Contracción del polígono de apoyo | no separa: caídas 0.41, ADL que alarman 0.39, ADL que no 0.36 |
| Guarda por rotación imposible de T | 4 958 °/s aparecen en clips **correctos**; con umbral 800 °/s se marcarían 48 clips, 41 de ellos caídas |
| Guarda por inversión sostenida | 25 clips tienen un salto >120° que no se deshace, y **19 clasifican bien** |
| Escorzo del tronco como señal | 11 de 16 esqueletos fieles caen dentro del rango de los infieles |
| Confianza de MediaPipe | no hay señal; sube en el cuadro del fallo |
| Descenso de cadera como regla única | 78.9 % contra 87.7 % de lo que ya hace |
| **Banda del disparador (zona gris 2.2–2.6)** | neutra-negativa: 91/120 = 75.8 % contra 100/128 = 78.1 % sin banda |
| **`model_complexity: 2` (pesado)** | 94/128 = 73.4 % contra 78.1 % del liviano; y `full` a 2.2 da mejor sensibilidad Y mejor especificidad |
| Bajar el corte de 60° | las bandas 30/60 ya eran óptimas entre 20 combinaciones — **pero esa medición asumía que T es fiel, así que conviene rehacerla** |

**La razón de fondo por la que ninguna de las primeras cinco podía funcionar:**
las cuatro cantidades se calculan *a partir del* esqueleto. Un esqueleto
equivocado produce cantidades internamente consistentes y perfectamente
plausibles. No hay anomalía que detectar río abajo.

---

### La banda del disparador, en detalle *(11/09)*

La idea era del usuario y era buena: si el puntaje se comporta como una
confianza —y lo hace, la proporción de caídas sube monótonamente de 5.3 % a
90.9 % en cinco tramos— entonces tratarlo por tramos en vez de por un corte
es legítimo. Un evento débil entra al embudo pero debe ganarse la alarma.

```
banda con el defecto de la marca    83/128 = 64.8 %   sens 59.7 %
banda con la promoción arreglada    91/120 = 75.8 %   sens 80.6 %
sin banda                          100/128 = 78.1 %   sens 81.9 %
```

**Por qué no funciona, y es la lección que queda:** la banda le exige a los
eventos débiles que la inmovilidad los confirme (`I >= W`, que es lo que el
§3.4 condiciona). Sobre los 128 clips rechazó 17 eventos — **7 eran ADL
(correcto) y 10 eran caídas (incorrecto)**.

La exigencia de inmovilidad **no separa un ADL de una caída. Separa a quien se
queda quieto de quien no.** Y eso corta por el lugar equivocado: alguien que
se acuesta en la cama se queda quieto, y alguien que se cae y trata de
levantarse se mueve. Es el mismo hallazgo que D16 por otra vía — la Etapa 2 y
la Etapa 3 no tienen con qué discriminar estos casos.

El mecanismo quedó implementado, probado (18 pruebas, 6 mutaciones
detectadas) y **apagado con `trigger_score_provisional: 0.0`**. Vuelve a tener
sentido el día que la Etapa 2 consiga un discriminador real.

Dos detalles de implementación que costaron encontrar y conviene no repetir:

* **El enganche corre sobre el borde inferior**, así que el evento se levanta
  mientras el puntaje todavía sube. Sin una promoción explícita, un evento con
  pico 3.32 quedaba marcado "débil" porque el hold terminó 0.13 s antes de que
  el puntaje cruzara el umbral pleno. La marca la decidía una carrera de
  milésimas, no la evidencia. Con ese defecto la sensibilidad cayó a 59.7 %.
* **La promoción termina con la Etapa 2.** En OBSERVING el sujeto ya está en
  el suelo y T ronda 180°, que da 4.0 de puntaje por sí solo: promover ahí
  ascendería cualquier evento provisional por el mero hecho de estar acostado.

### El modelo pesado, en detalle *(11/09)*

El §3.2 lo autoriza literalmente para el drift que medimos, así que era la
única vía pendiente que atacaba la causa en vez de compensarla río abajo.

```
              accuracy    sens     espec    FN   FP
full  2.6      78.1 %    81.9 %   83.9 %   13    9
heavy 2.6      73.4 %    87.5 %   73.2 %    9   15
heavy 2.8      74.2 %    81.9 %   80.4 %   13   11
```

Barrí el umbral sobre los datos del pesado por si el 2.6 heredado lo estaba
perjudicando: **el óptimo en calibración es 2.6 o 2.8, o sea que el umbral
prestado ya era el correcto.** No había nada que recuperar.

**El pesado es más sensible —pierde 9 caídas en vez de 13— pero paga 6 falsos
positivos más.** Igualado en sensibilidad (2.8 contra 2.6) sigue teniendo peor
especificidad. Y `full` con umbral 2.2 da 93.1 % de sensibilidad con 71.4 % de
especificidad, mejor en ambas que el pesado a 2.6 — y corre tres veces más
rápido.

Sí arregla casos individuales: **A09-S2 pasó de `NoFall` a correcto**, que es
justo el clip donde el usuario vio el esqueleto de pie sobre alguien
arrodillado. Y A14-S2 y A14-S4 pasaron de invisibles a detectadas. Pero rompe
otros en la misma proporción — el mismo barajado que se midió entre las dos
máquinas. **No es una mejora, es un reacomodo con más costo de cómputo.**

Nota operativa: `model_complexity: 2` no se puede correr en el contenedor de
desarrollo. MediaPipe no trae el modelo pesado; lo descarga de
`storage.googleapis.com`, que el proxy de ese entorno bloquea con 403.

---

## La meseta

Doce configuraciones medidas, el mismo rango:

```
ventana de pico sí / no
umbral 2.2 / 2.4 / 2.5 / 2.6 / 2.8 / 3.0 / 3.2 / 3.5
compuerta mínima de T 0 / 45 / 50 / 60
banda encendida / apagada / rota
model_complexity 1 / 2

accuracy siempre entre 64.8 % y 78.1 %
```

**El disparador solo puede canjear caídas perdidas por falsos positivos.** No
puede subir el accuracy, porque las severidades equivocadas no las toca y cada
caída que gana la paga en especificidad.

La mejor configuración medida sigue siendo la del 10/09:

```
full + ventana de pico 0.8 + umbral 2.6 + Undetermined -> NoFall
100/128 = 78.1 %   sens 81.9 %   espec 83.9 %
errores: 14 caídas perdidas · 8 severidades · 7 falsos positivos
```

Para 90 % hacen falta 115 de 128. No hay configuración de umbral que los
tenga. Lo que queda está en lo que la revisión visual está desenterrando.

---

## Vías abiertas — para implementar cuando termine la revisión

### 1 · Cadera que SUBE al final del clip *(sin medir)*

Lo que A06-S14 muestra: si la cadera asciende de forma sostenida en los
últimos segundos, el sujeto se está levantando, diga lo que diga T.

Distinto de lo que ya medí, que fue el **descenso neto** desde el inicio. Esto
es la **tendencia al final**.

Dentro del §3.2, que ya promete *"tracking within-frame and across-frame
landmark displacement ratios"*. Cierra la divergencia **D3**. Sin medir: no
prometer nada hasta hacerlo.

### 2 · Guarda de cadera baja con tronco erguido *(medido: +1 clip)*

Si T lee erguido (<60°) pero la cadera quedó más de 1.5–2.0 torsos por debajo
de donde empezó, la postura no es de pie → severidad `NotRecovered`.

```
sin guarda    35/40 calibración   50/57 total
X = 1.5–2.0   36/40 calibración   51/57 total
```

Pequeño en este dataset. Su valor real es otro: arregla los dos casos que el
usuario describió —recostado en pared, cuerpo escorzado— que en una casa real
son el caso común y no la excepción.

**Prioridad baja bajo el criterio del usuario**, que prefiere detectar una
caída con severidad equivocada antes que perderla.

### 3 · `model_complexity: 2` *(sin probar)*

El §3.2 lo autoriza literalmente: *"with the heavy variant reserved for
low-light or multi-person scenarios where **landmark drift is observed in
practice**"*. Se observó drift, está medido y confirmado visualmente.

Una línea de `config.yaml`, cero texto, y **es lo único pendiente que ataca la
causa en vez de compensarla río abajo**. Cuesta velocidad de inferencia, que
toca el §3.6.

### 4 · Rehacer la calibración de las bandas 30/60

La medición que las declaró óptimas se hizo sobre los 128 clips **incluyendo
aquellos cuyo esqueleto es infiel**. Repetirla sobre el subconjunto con
esqueleto fiel diría si 60° es el corte correcto para el problema real o solo
el que mejor compensa los errores de MediaPipe.

---

## Lo que NO es una opción

**Regrabar.** El usuario lo descartó explícitamente: *"nunca es una opción, lo
ideal es que se pueda detectar sin las condiciones óptimas"*. Es un requisito
de diseño, no una preferencia — y coincide con el argumento de robustez del
§3.1. Cualquier arreglo que dependa de mejor encuadre está fuera.

---

## Criterio de prioridad del usuario

> *"prefiero que se detecte una caída, ya sea que la califique mal, a que
> salga como NoFall"*

Bajo ese criterio el orden de importancia de los errores es:

1. caída que sale `NoFall` — grave
2. falso positivo — molesto
3. severidad equivocada — menor

Y eso implica que **el accuracy de cuatro clases es la métrica equivocada para
optimizar**, porque castiga igual (1) y (3). El §3.7 pide accuracy,
sensibilidad, especificidad, precisión y F1: elegir la sensibilidad como la
cifra que se optimiza es una decisión de diseño declarada, no un ajuste
oculto.

Queda sin resolver, y hay que resolverlo antes de la Sección 4: el umbral 2.6
se eligió rompiendo un empate **por especificidad**, que apunta al lado
contrario de este criterio. La curva medida:

```
umbral   caídas perdidas   falsos positivos   sens     espec
  2.2           5                 16          93.1 %   71.4 %
  2.5          12                 11          83.3 %   76.8 %
  2.6          13                  9          81.9 %   83.9 %
  2.8          17                  6          76.4 %   91.1 %
```

*(2.2 medido en el contenedor; 2.5 y 2.6 en la máquina del usuario)*

De 2.6 a 2.2 se recuperan 8 caídas a cambio de 7 alarmas falsas.
