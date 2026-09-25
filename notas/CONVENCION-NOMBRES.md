# CONVENCIÓN DE NOMBRES — PEF-FallDB

**Propósito:** que el nombre del archivo cargue todo lo que el paper necesita,
de modo que el §3.3 quede literalmente cierto y el §3.7 sea ejecutable sin
tocar el texto.

Fijar esto **antes de grabar el material restante**: cada video grabado con
otro formato es uno más que habrá que anotar a mano después.

---

## Formato

```
[setting]-[event]-[recovery]-[sujeto]-[camara]-[luz].mp4
```

Los **tres primeros campos son exactamente el protocolo del §3.3**
(*"The label follows the format [setting]-[event]-[recovery status]"*). Los
tres últimos son las dimensiones que el §3.7 pide retener para probar
generalización (*"holding out settings, camera placements, and lighting
conditions"*) y que hoy no están codificadas en ninguna parte.

Sin guiones dentro de un campo. Los subtipos usan punto (`fall.forward`), para
que partir por `-` siempre dé exactamente seis campos.

### Ejemplos

```
kitchen-fall.forward-recovered-A01-cam2-day.mp4
hallway-fall.syncope-notrecovered-A14-cam1-night.mp4
bedroom-fall.lateral-partially-A09-cam3-artificial.mp4
livingroom-adl.crouch-nofall-B03-cam1-day.mp4
bathroom-adl.liedown-nofall-B07-cam2-night.mp4
```

---

## Vocabularios

### setting — los cinco del §3.3

`bedroom` · `bathroom` · `livingroom` · `kitchen` · `hallway`

### event

**Caídas — los cuatro arquetipos del §3.3:**

| token | §3.3 |
|---|---|
| `fall.forward` | impacto, decúbito prono, intento de empuje, rodada opcional, levantada opcional |
| `fall.backward` | impacto, decúbito supino, intento de sentarse, levantada opcional |
| `fall.lateral` | impacto, decúbito lateral, intento de apoyarse en mueble, levantada opcional |
| `fall.syncope` | caída vertical silenciosa, quietud prolongada 4–8 s, recuperación parcial o total |

**ADL — los seis nombrados en el §3.3:**

| token | §3.3 |
|---|---|
| `adl.walk` | caminar a paso normal |
| `adl.walkobject` | caminar cargando un objeto |
| `adl.sit` | sentarse con desaceleración rápida en silla o sofá |
| `adl.crouch` | agacharse a recoger un objeto del suelo |
| `adl.tieshoe` | inclinarse con rodillas extendidas para amarrarse un zapato |
| `adl.liedown` | acostarse intencionalmente en cama o sofá |
| `adl.rise` | levantarse desde estar acostado |

> El §3.3 señala `adl.crouch` como la prueba de estrés más exigente: *"una caída
> vertical rápida del grupo cadera-hombro seguida de una postura asimétrica de
> tren inferior imita de cerca un colapso genuino"*. Conviene grabarlo en los
> cinco settings.

### recovery — las cuatro clases que el sistema emite

`recovered` · `partially` · `notrecovered` · `nofall`

`nofall` va en todos los ADL. El §3.3 etiqueta los ADL sólo por setting y
actividad; poner `nofall` explícito mantiene los seis campos siempre presentes
y hace el parser trivial.

### sujeto

`A01`…`Ann` para sujetos con caída, `B01`…`Bnn` para ADL. Dos dígitos siempre
(`A03`, no `A3`).

Si el §3.3 mantiene las tres categorías de sujeto (activo, transeúnte,
parcialmente visible), conviene un prefijo: `A01` activo, `T01` transeúnte,
`P01` parcialmente visible.

### camara

`cam1`, `cam2`, `cam3`… — **un token por combinación distinta de altura y
ubicación lateral**, no por altura sola. Codificar dos dimensiones en un campo
las vuelve ambiguas; una tabla en el README del dataset dice qué es cada `camN`
(altura en cm, posición, ángulo).

### luz

`day` (natural diurna) · `artificial` · `night` (poca luz / nocturna)

---

## Impacto en el código

`dataset.py::truth_from_name()` lee hoy **el último token** del nombre. Con este
formato el último token pasa a ser la luz, así que hay que extenderlo:

- Si el nombre parte en **seis campos** por `-`, leer por posición.
- Si no, caer al comportamiento actual (último token), para que los 128 clips ya
  grabados sigan funcionando mientras se renombran.

Con eso el runner de lote puede además reportar por setting, por cámara y por
condición de luz, que es lo que el §3.7 pide, y la partición de entrenamiento y
prueba puede hacerse por la dimensión que el paper especifica en vez de por
actor.

Media hora de trabajo. Conviene hacerlo el mismo día que se fije la convención,
para que el material nuevo entre ya medido.

---

## Estado de los 128 nombres actuales

Esta sección es un registro, no una tarea. **Nada de lo pendiente afecta el
accuracy**: el código ya lee correctamente los tres casos, y así se ha medido
en todas las corridas. Se anota para que no se pierda el día que se congele el
dataset para el paper.

### Ya normalizado

**Sufijo `Recovery` → `Recovered`** *(10/09/2026)*. Eran cuatro clips, todos del
actor A13: `A13-S1`, `A13-S3`, `A13-S4` y `A3-S2`. Renombrados en `recordings/`
y en la carpeta `videos mal etiquetados`. El dataset entero usa ahora las cuatro
clases canónicas.

Con eso, la entrada `"recovery"` de la tabla de alias en
`dataset.py::_ALIASES` quedó sin uso sobre este material. Conviene **dejarla**
de todos modos: es la clase de error de tipeo que vuelve a aparecer en material
nuevo, y borrarla convierte un typo futuro en una clase vacía silenciosa en vez
de en una lectura correcta.

### Pendiente, sin prioridad

**`A3-S2-Recovered` → `A13-S2-Recovered`.** El actor lleva un dígito en vez de
dos. Hoy funciona por el alias `A3 -> A13` de `evaluation.py::_ACTOR_ALIASES`,
que existe para que ese clip caiga del lado correcto de la partición en vez de
aparecer como un actor propio de un solo clip. Hay una prueba que fija ese
comportamiento (`test_the_A3_typo_resolves_to_its_real_actor`). Si se renombra,
el alias y su prueba se pueden borrar; mientras no se renombre, **no se toquen**.

**`A06-S14` y `A11-S14` → `S4`.** Todos los demás actores tienen escenas `S1` a
`S4`; sólo estos dos tienen `S14`, lo que casi con seguridad es un dedazo. No
hay alias ni caso especial para esto porque el número de escena no entra en
ninguna decisión: sólo se lee el actor y la clase. Es cosmético hasta el día en
que se quiera reportar por escena.

### Por qué esto importa aunque no mueva el número

Los dos alias que quedan son **parches para typos del dataset viviendo dentro
del código de inferencia**. Mientras existan, el parser carga una tabla de
excepciones cuyo único motivo es que unos archivos están mal escritos, y el
§3.3 no describe ninguna excepción: promete un formato. Normalizar los nombres
es lo que permite que el código haga literalmente lo que el paper dice, sin
notas al pie.
