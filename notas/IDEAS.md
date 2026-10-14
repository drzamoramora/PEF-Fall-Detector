# Ideas

Ideas que salieron trabajando y que todavía **no están aprobadas ni en el plan**.
Cada una lleva su análisis (a favor, en contra y qué falta para decidir), para que
no se pierdan entre sesiones. Cuando una se aprueba, pasa al plan de
`TRASPASO-SESION.md` y aquí queda marcada con la fecha. Las descartadas se
quedan abajo, con el motivo.

---

## 1. Glosario de métricas en el README

*Propuesta: Arthur, 14/10/2026. Estado: idea.*

Una sección del `README.md` que explique cada métrica de `metrics.csv` / `metrics.txt`
en lenguaje simple, para que alguien externo (revisor, coautor) no se confunda.
Por ejemplo:

- **TP** (verdadero positivo): era caída y se detectó caída.
- **FP** (falso positivo): no era caída y se detectó caída.
- **TN** (verdadero negativo): no era caída y no se detectó.
- **FN** (falso negativo): era caída y no se detectó; es el error más grave.
- Sensibilidad = TP / (TP + FN); especificidad = TN / (TN + FP); y así con el resto.

- **A favor:** las métricas se leen sin contexto (en una hoja de cálculo, en el
  paper). Hay dos lecturas, de etiqueta y de alarma, que confunden si no se
  explican.
- **En contra:** hay que mantenerlo al día cuando se agregue una métrica.
- **Falta para decidir:** si va en el `README.md` o en un `docs/METRICS.md` aparte.

## 2. Decisión diferida en vivo ("10 s de retraso")

*Propuesta: Arthur, 14/10/2026. Estado: idea; candidata para resolver la fase L.*

En el dataset, la etapa 3 decide con el clip completo: reglas como "A final ≥ 0.82"
miran el final del clip. En vivo no hay final. La idea original era recibir el
video en vivo con unos 10 s de retraso, para analizar cada caída como un clip
de 10 s.

**Ajuste propuesto:** no retrasar el video, solo la decisión. MediaPipe y las
cantidades corren en vivo, cuadro por cuadro. Cuando la etapa 1 dispara en *t*,
el sistema espera a *t + X* y evalúa el evento como si fuera un clip. Solo se
guardan landmarks y cantidades en memoria, unos pocos KB.

- **A favor:**
  - Las reglas validadas con los 128 clips se aplican igual en vivo, y los
    números del dataset dicen algo del sistema real (argumento para el paper).
  - Es coherente con la §3.5: la confirmación ya depende de ver si la
    inmovilidad persiste.
  - No guarda imágenes (§3.6).
  - Esperar antes de alertar es práctica común: se espera a ver si la persona
    se levanta.
- **En contra:**
  - La alarma llega unos X s tarde; hay que reportarlo como latencia de alerta.
  - Hay que elegir X con datos. En los clips, el disparo cae entre 0.3 y 5.8 s,
    así que lo observado después del evento va de unos 4.6 a 10 s.
  - Casos que el dataset no tiene: eventos solapados, la persona que sale de
    cuadro, horas sin caídas.
  - Retrasar el video completo costaría unos 830 MB de RAM en el Pi (300
    cuadros a 720p) y no aporta nada sobre retrasar solo la decisión.
- **Cómo validarlo:** pasar los 128 clips por el modo "en vivo con decisión
  diferida" y verificar que dan las mismas etiquetas que el modo etiquetado.

## 3. Comparar dos corridas en la GUI, con datos para machine learning

*Propuesta: Claude; ampliada por Arthur, 14/10/2026. Estado: aprobada en principio,
se diseña después del diagnóstico x86/ARM.*

Abrir dos corridas ya hechas (por ejemplo Mac y nube) del mismo clip. El video
se reproduce con los dos esqueletos a la vez y con las curvas superpuestas, y
queda marcado el primer cuadro donde divergen. Además genera un conjunto de
datos para analizar la diferencia con machine learning.

- **Falta para decidir:** qué datos exporta para ML y con qué objetivo
  (predecir la divergencia, agrupar clips por sensibilidad a la plataforma…).
  Se diseña con los datos del formato nuevo en mano.

## 4. Veredicto provisional en la GUI

*Propuesta: Claude, 14/10/2026. Estado: aprobada, sin orden todavía.*

Mientras la etapa 3 observa, mostrar qué veredicto saldría si el clip terminara
en ese cuadro, en vez de "— analyzing…". Sirve para ver cómo cambia el resultado
con el tiempo y para elegir la X de la idea 2.

## 5. La misma imagen en todas las plataformas (opción A del diagnóstico x86/ARM)

*Propuesta: Claude, 14/10/2026. Estado: idea; falta la prueba barata.*

La Mac y la nube x86 decodifican el mismo video con FFmpeg distintos (los trae la rueda de OpenCV), y casi toda la diferencia de resultados (110 contra 115) sale de ahí (ver `DIAGNOSTICO-X86-ARM.md`). Idea: decodificar a YUV (exacto por norma en H.264) y hacer la conversión de color en el código, con aritmética entera.

- **A favor:** la Mac, la nube y el Pi le pasarían a MediaPipe la misma imagen; queda solo la diferencia de CPU (~0.03 px de mediana). El número del paper sería uno solo y reproducible.
- **En contra:** una dependencia nueva para leer archivos (probablemente PyAV, `av`); las cifras actuales cambian (hay que volver a correr y aceptar la referencia nueva); solo afecta al dataset, no al vivo.
- **Falta para decidir:** decodificar a YUV en la Mac y en la nube y comparar hashes. Si no coinciden, la idea no sirve tal cual.

---

## Descartadas

- **Ramas por arquitectura (ARM / x86).** La diferencia entre plataformas viene
  de los landmarks de MediaPipe, no del código: con los mismos landmarks, el
  código decide igual en las dos. Ramas separadas duplicarían el mantenimiento
  sin arreglar la causa.
- **Retrasar el video completo en vivo.** Reemplazada por la decisión diferida
  de la idea 2: mismo efecto, sin guardar imágenes y con KB en vez de cientos
  de MB.
