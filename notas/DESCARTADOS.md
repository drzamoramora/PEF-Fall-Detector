# DESCARTADOS.md — mecanismos medidos y retirados del código (B1, 26/09)

La limpieza B1 borró del código dos mecanismos que llevaban semanas apagados en
`config.yaml`. No cambiaron ninguna decisión (se verificó con el replay de las
dos plataformas: idéntico). Esta nota guarda **por qué** se apagaron, con sus
cifras, porque el paper puede necesitarlas (§3.5, §4, ablaciones) y el código
ya no las lleva en comentarios. El código original está en la etiqueta
`v0.1.1-legacy`.

---

## 1. Banda provisional del disparador (`stage1.trigger_score_provisional`)

Apagada (0.0) desde las mediciones de agosto/septiembre. Lo que decía
`config.yaml`:

```
Banda del disparador: el borde inferior, por debajo del umbral pleno.
Medido sobre PEF-FallDB, la proporcion de caidas crece de forma monotona
con el puntaje, sin inversiones:
    < 1.8      5.3 % caidas      2.6 - 3.5   57.7 %
  1.8 - 2.2    7.1 %             > 3.5       90.9 %
  2.2 - 2.6   35.7 %
O sea que el puntaje se comporta como una confianza y no como un
interruptor, y eso es lo que hace legitimo tratarlo por tramos.
Un puntaje en la banda levanta un evento PROVISIONAL: entra al embudo
igual que cualquier otro, pero solo puede terminar en alarma si la
inmovilidad lo confirma (stage3_confirmed). Si resuelve por postura, o no
resuelve, queda en 'provisional_unconfirmed' y se lee como NoFall.
Esto no inventa mecanismo: el §3.5 ya describe un embudo donde la
sospecha se confirma o se descarta, y el §3.4 ya condiciona la
confirmacion a que I supere W. La banda solo le exige a la evidencia mas
debil la prueba mas fuerte que el paper ya define.
0.0 desactiva la banda -> corte unico, que es la ablacion del §3.7.
MEDIDO Y APAGADO. Con la banda en 2.2, sobre los 128 clips:
  banda con el defecto de la marca   83/128 = 64.8 %  (sens 59.7 %)
  banda con la promocion arreglada   91/120 = 75.8 %  (sens 80.6 %)
  sin banda                         100/128 = 78.1 %  (sens 81.9 %)
La banda arreglada quedo neutra-negativa: rechazo 17 eventos, de los
cuales 7 eran ADL (bien) y 10 eran caidas (mal). La exigencia de
inmovilidad no separa ADL de caida, separa a quien se queda quieto de
quien no — y varias caidas reales se mueven despues.
El mecanismo queda implementado y probado por si Stage 2 consigue algun
dia un discriminador de verdad (ver el criterio de descenso). Hasta
entonces, 0.0.
    trigger_score_provisional: 0.0
```

Detalle de la "promoción" (método `FallStateMachine._promote_if_strong`):

```
Un evento que nacio debil y despues se fortalecio deja de ser debil.

POR QUE HACE FALTA. El enganche corre sobre el borde INFERIOR de la
banda, asi que el evento se levanta cuando el puntaje lleva
``hold_s`` por encima de 2.2 — que en una caida real ocurre mientras
el puntaje todavia esta subiendo. Medido en A04-S1: cruza 2.2 en
t=2.60, dispara en t=2.70, cruza 2.6 en t=2.73 y llega a 3.32. Sin
esta promocion ese evento queda marcado provisional para siempre por
trece centesimas, y la marca termina dependiendo de una carrera entre
el reloj del enganche y la pendiente de la señal en vez de la fuerza
de la evidencia.

SOLO EN CONFIRMING. En OBSERVING el sujeto ya esta en el suelo y T
ronda 180, que da 4.0 por si solo: promover ahi ascenderia cualquier
evento provisional por el mero hecho de estar acostado. La ventana de
promocion es aquella en que la evidencia del DISPARO todavia se
forma.

El puntaje se lee del disparador, que se sigue alimentando en cada
cuadro mientras el evento se juzga — no se recalcula aqui, para que
haya un solo sitio donde la formulacion viva.
```

## 2. Rechazo por descenso controlado en la Etapa 2 (`stage2.descent_*`)

Apagado (`descent_min_depth_tps: 0.0`) desde el 03/09. Lo que decía
`config.yaml`:

```
Descenso controlado vs caida (§3.4). El §3.5 le da a esta etapa el trabajo
de rechazar "a controlled sit-down in which COM stays within the support
polygon", pero medido sobre PEF-FallDB esa premisa es falsa: los ADL que
disparan ponen el COM FUERA del apoyo en el 100 % de los cuadros medibles,
porque sentarse traslada el peso al mueble. Lo que si separa es la
magnitud del descenso, que es como el §3.4 describe V: "a sit-down event
produces a monotonic negative V over a longer window, and a fall produces
a large-magnitude negative V within a short window".
Se rechaza solo si el descenso fue POCO PROFUNDO **y** BREVE: un descenso
corto puede ser violento (sincope) y uno lento puede ser una caida; solo
fallar en ambas a la vez indica control.
DESACTIVADO 03/09 tras medirlo de punta a punta. Ver el intento fallido en
notas/TRASPASO-DRAFT.md: calibrado dio "pierde 1 caida de 40", medido dio
"pierde 4", porque la calibracion midio el descenso sobre el CLIP ENTERO y
el codigo lo evalua sobre una ventana movil que cierra 0.33 s despues del
disparo. En varias caidas el descenso fuerte ocurre despues de ese cierre.
Sobre los 128: gano 3 NoFall y perdio 4 caidas. Especificidad 75 -> 94 %,
sensibilidad 71 -> 67 %. Para un detector de caidas ese intercambio va en
la direccion equivocada: una alarma falsa molesta, una caida perdida es
alguien en el suelo sin auxilio.
La separacion medida (0.82 s en caidas contra 0.32 s en ADL) es real; lo
que esta mal es donde se evalua. Reactivar solo tras re-calibrar con la
MISMA ventana que usa el codigo.
    descent_min_depth_tps: 0.0 # 0 desactiva el rechazo por descenso
    descent_floor_tps: -1.0
    descent_window_s: 2.0
    descent_min_sustained_s: 0.40
```

Lo que decía la clase `DescentTracker`:

```
Qué tan profundo y qué tan sostenido fue el descenso (§3.4).

POR QUÉ EXISTE. El §3.5 le da a la Etapa 2 el trabajo de separar una caída
de *"a controlled sit-down in which COM stays within the support polygon"*,
y le da P como única herramienta. Medido sobre PEF-FallDB, esa premisa es
falsa: en los ADL que producen falso positivo, la Etapa 2 midió P sobre 11
cuadros y encontró el centro de masa **fuera** del polígono de apoyo en el
100 % de ellos. No es un fallo de medición — es que sentarse en un sofá
saca el COM de sobre los pies, porque el peso pasa al mueble. La geometría
dice "caída" con razón, y P sola no puede resolverlo.

QUÉ SÍ SEPARA, y el §3.4 ya lo dice: *"a sit-down event produces a
monotonic negative V over a longer window, and a fall produces a
large-magnitude negative V within a short window."* La distinción es de
MAGNITUD. Medido sobre los 40 clips de calibración que dispararon:

    caídas          V_min mediana -3.00, tiempo bajo -1 tps  0.77 s
    falsos positivos V_min  -1.16 y -1.39, tiempo           0.17 y 0.20 s

Se exigen las DOS condiciones para rechazar — poco profundo Y breve — no
una. Un descenso puede ser corto y aun así violento (una caída de síncope),
y puede ser lento y aun así ser una caída. Sólo cuando falla en las dos a la
vez se trata de un descenso controlado.

LO QUE NO RESUELVE. Un clip de calibración (B05-S1) llega a V_min = -3.13
con el COM fuera del apoyo: por profundidad, duración y geometría es
indistinguible de una caída. Probablemente alguien dejándose caer en un
sofá. Ninguna de las cuatro cantidades del §3.4 lo separa, y este criterio
lo deja pasar a propósito en lugar de apretarse hasta perder caídas reales.
```

Relación con el paper: C14 y C15 de `POSIBLES-CAMBIOS-DEL-DRAFT.md` (el sofá
B05 baja con V de caída y saca el COM del apoyo) y `NOFALL.md` (vía N1,
cerrada).

## 3. Lo que NO se borró

`sequential` y `simultaneous` siguen disponibles para las ablaciones del §3.7:
se movieron a `pef_fall_detector/ablations.py` y se eligen igual que antes con
`stage1.trigger_formulation`. `v_only` sigue junto a `score`.
