# Historia en los comentarios del código

Desde B5 (versión 0.2.0.dev5) el código está en inglés. Los comentarios en español que contaban la historia de una fase (qué se midió, en qué clip, con qué números) salieron del código y quedan aquí. En el código quedó una versión corta en inglés que dice qué hace la regla y apunta a esta nota.

**Cómo está hecha esta nota:**

- Cada bloque es el comentario original **copiado tal cual** de la versión 0.2.0.dev4 (B4, rama `cleanup-b4`), con la línea donde empezaba. No se corrigió nada dentro de los bloques: ni acentos, ni erratas.
- Solo se trasladaron los comentarios de historia de fases. Los demás comentarios y docstrings en español (la ventana de PEF-Lab, `evaluation.py`, `review_cache.py`, `theme.py`, `requirements.txt`) se tradujeron directamente: explicaban qué hace el código, no cómo se llegó a él.
- Las historias de `config.yaml` están en `CONFIG-HISTORIA.md`.

**Qué cambió en los resultados:** nada. Los motivos de los eventos y de las etiquetas pasaron al inglés (`tools/maps/b5.json` traduce los viejos para comparar corridas de antes de B5).

## Fase 4 — re-adquisición y enfriamiento

### `pef_fall_detector/state_machine/machine.py`, línea 42

```text
#: Fase 4. El enfriamiento existe para que UNA caida no llegue como
#: rafaga de eventos. Un candidato que la Etapa 2 rechazo no es una
#: caida: el §3.5 lo descarta ahi. Dejar al sistema ciego 3 s despues
#: de descartarlo no protege de ninguna rafaga (el pestillo de
#: Stage1Trigger ya impide re-disparar mientras la condicion siga
#: cierta) y si puede tapar la caida real que viene detras (A17-S3:
#: rechazo a los 2.5 s, caida real a los 4.0-5.0 s, enfriamiento
#: hasta 5.5 s). False: tras un rechazo se vuelve a MONITORING.
```

### `pef_fall_detector/pipeline.py`, línea 230

```text
# Fase 4: el rastreador se re-sembro (pose_frontend). Los landmarks de
# este cuadro vienen de una deteccion nueva, no de seguir al anterior:
# misma ruptura de continuidad que una perdida breve.
```

### `pef_fall_detector/pipeline.py`, línea 235

```text
# Y la ventana de pico del disparador: lo anterior al reinicio era
# un esqueleto pegado, no el mismo cuerpo. (Una perdida comun NO
# la limpia: medido, eso perdio 4 caidas reales cuya deteccion
# parpadea durante la caida.)
```

### `pef_fall_detector/pose_frontend.py`, línea 179

```text
# Re-adquisición (fase 4). MediaPipe Pose es detector + rastreador: el
# detector corre solo cuando el rastreo se pierde, y el rastreo se da
# por bueno mientras la *presencia* supere min_tracking_confidence.
# La presencia y la visibilidad de los landmarks son salidas
# distintas: en A16-S1 el rastreador quedó pegado 5.4 s a una región
# vieja (la del sujeto pasando pegado a la cámara) con presencia alta
# y visibilidad del núcleo en 0.00-0.05, mientras el cuerpo entero
# caía y yacía a la vista. Ninguna etapa del §3.5 puede ver una caída
# que el front-end no entrega. Si el núcleo sigue por debajo de
# visibility_threshold durante reacquire_after_s, se reinicia el
# grafo para que el siguiente cuadro pase otra vez por el detector.
# No cambia ningún landmark de un cuadro bueno; 0 lo apaga.
```

## Fase 6a — levantarse en curso

### `pef_fall_detector/state_machine/stage3.py`, línea 150

```text
# Fase 6a: levantarse en curso al terminar la observacion. El §3.5
# baja la severidad ante "torso and hip landmark recovery to
# upright"; hasta aqui solo se leia el torso (T). Si al final T dice
# "en el suelo" pero la CADERA viene subiendo de forma sostenida, el
# sujeto se esta levantando: moderate, no severe. Medido en A06-S14
# (sube 0.70 torsos en 2 s en la nube, 0.81 en el Mac) contra un
# maximo de 0.21 / 0.13 en las caidas NotRecovered con T >= 60.
# 0 lo apaga.
```

### `pef_fall_detector/state_machine/stage3.py`, línea 160

```text
# Fase 8 (cantidad A, EXPERIMENTAL): altura 3D de la cabeza contra la
# gravedad, dividida por la altura de pie del mismo sujeto. Solo en
# modo etiquetado (finalise), igual que las fases 3 y 6a.
#   8.2 (a_band_floor > 0): la postura final sale de A cuando esta
#       calibrada; <= a_band_floor en el suelo, >= a_band_standing de
#       pie, en medio sentado o arrodillado.
#   8.3 (a_reach_floor > 0): si la cabeza nunca bajo de a_reach_floor
#       entre a_reach_before_s antes y a_reach_after_s despues del
#       disparo, y la cadera termino sobre a_hip_raised, no hubo caida.
#       Se abstiene si A no estaba calibrada al disparar.
# 0 en a_band_floor / a_reach_floor apaga cada regla.
```

### `pef_fall_detector/state_machine/stage3.py`, línea 485

```text
# Fase 8.2: con A calibrada, la postura final sale de A, no del angulo
# 2D. T no distingue caer alejandose de la camara de estar de pie
# (A14-S4: T 28 deg, A 0.14), ni sentado contra la pared de tumbado.
# La subida de cadera (6a, arriba) y la persistencia (fase 3) siguen.
```

### `pef_fall_detector/pipeline.py`, línea 386

```text
# Fase 6a: la altura de la cadera, para leer un levantarse en
# curso al final (Stage3Evaluator.getup_rise).
```

## Fase 8 — cantidad A

### `pef_fall_detector/state_machine/stage3.py`, línea 447

```text
# Fase 8.3: "¿llego al suelo?". Una caida es perdida de altura contra
# la gravedad; si la cabeza nunca bajo de a_reach_floor de su altura de
# pie alrededor del disparo y la cadera termino elevada, el cuerpo no
# llego al piso: se sento en un mueble o se agacho. La condicion de la
# cadera es la que protege al desplomado contra la pared (A17-S2:
# cabeza 0.44, cadera 0.07). Sin A calibrada al disparar, se abstiene.
```

### `pef_fall_detector/state_machine/build.py`, línea 22

```text
# Fases 8.2 y 8.3: la Etapa 3 lee A (ver Stage3Evaluator.finalise).
# Cada regla tiene su interruptor; apagadas, A solo se registra (8.1).
```

### `pef_fall_detector/state_machine/events.py`, línea 46

```text
#: Fase 8.3 (cantidad A): la cabeza nunca bajo cerca del suelo y la cadera
#: termino elevada, asi que no hubo caida. Cuenta como NoFall y no despacha
#: alarma. Ver Stage3Evaluator.finalise.
```

### `pef_fall_detector/classification.py`, línea 132

```text
# Fase 8.3: la Etapa 3 midio que el cuerpo no llego al suelo (cantidad A).
# Es un juicio activo, como el rechazo de la Etapa 2, no una falta de
# evidencia: NoFall siempre, sin importar ``unresolved_as``.
```

### `pef_fall_detector/pipeline.py`, línea 392

```text
# Fases 8.2/8.3: la cantidad A. Se anota si estaba calibrada en el
# cuadro del disparo; al empezar la Etapa 3 se le entrega el
# segundo previo, y despues cada cuadro observado.
```
