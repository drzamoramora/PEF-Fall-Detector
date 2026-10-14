# Diagnóstico x86 / ARM (14/10/2026)

Por qué la misma versión del detector da 110/128 en la Mac y 115/128 en la nube x86.
Medido con el formato de registros nuevo (0.3.0.dev0, rama `registros`).

## Corridas comparadas

| | Mac | Nube x86 |
|---|---|---|
| Corrida | `logs/2026-10-14_0411-macos-arm64-recordings` | `logs/2026-10-14_0413-linux-x86_64-recordings` |
| CPU | Apple M4 | Intel Xeon @ 2.10 GHz (2 núcleos) |
| Resultado | 110/128 (85.9 %) | 115/128 (89.8 %) |
| mediapipe · opencv-contrib-python · numpy · protobuf | 0.10.14 · 4.13.0.92 · 2.4.4 · 4.25.9 | iguales |
| Decodificador (backend de OpenCV) | FFMPEG | FFMPEG |
| FFmpeg dentro de la rueda de OpenCV | avcodec 61.19.101 · swscale 8.3.100 | avcodec 62.11.100 · swscale 9.1.100 |
| Huella del dataset (`run.json`) | `8535297257b5…` | `8535297257b5…` (los mismos bytes) |

Verificación de cada corrida contra su referencia anterior:

- **Mac:** idéntica a `logs-historicos/cli-a84` (fase 8.4, Mac), 128/128 clips.
- **Nube x86:** idéntica a `respaldo-nube/corridas/nube-8.4` (fase 8.4, x86), 128/128 clips, con `--landmarks`.

## 1. Las corridas completas

- **Imagen decodificada:** 0 de 40 080 cuadros tienen el mismo `frame_sha1`. Desde el cuadro 0, MediaPipe recibe imágenes distintas en cada plataforma.
- **Landmarks:** ningún cuadro es igual bit a bit. Distancia del punto más alejado en x, y:
  - cuadro 0: mediana 9.9 px, p90 35 px;
  - todos los cuadros: mediana 9 px, p90 54 px.
- **Detección:** la persona se detecta en una plataforma y no en la otra en 1036 cuadros de 71 clips.
- **Etiquetas:** 7 clips cambian. La x86 acierta en 6 (A01-S4, A09-S2, A14-S1, A16-S2, A17-S4, A18-S3) y la Mac en 1 (A14-S4).

## 2. Las imágenes decodificadas

Se exportó en PNG (sin pérdida) el cuadro 0 de cada clip tal como lo decodificó la Mac; los 128 hashes coinciden con su corrida. Comparado con la decodificación x86:

- el 96 % de los píxeles difiere (mediana por clip);
- la diferencia es de 1 a 5 niveles de 255, con un promedio de 0.66: invisible.

Lo más probable es que venga de la conversión de color YUV→BGR (`swscale` distinto) y no de la decodificación H.264, que es exacta por norma. Falta confirmarlo (ver "Próximo paso").

## 3. Separar decodificador y CPU

MediaPipe x86 sobre la imagen que decodificó la Mac, comparado con los landmarks de la Mac. Son los 68 clips con persona detectada en el cuadro 0, en las dos corridas y en la prueba.

| Cuadro 0 | Mediana | p90 | Máximo |
|---|---|---|---|
| Corrida Mac vs corrida x86 (decodificador + CPU), x, y | 9.87 px | 35.3 px | 169 px |
| Misma imagen, MediaPipe Mac vs x86 (solo CPU), x, y | 0.033 px | 0.27 px | 5.95 px |
| Mundo: corrida vs corrida | 40 mm | 139 mm | 397 mm |
| Mundo: misma imagen | 0.18 mm | 0.73 mm | 2.5 mm |

- **Cada plataforma es determinista:** MediaPipe x86 sobre la imagen x86 reproduce bit a bit la corrida x86 (70/70 clips).
- **Con la misma imagen, MediaPipe no es bit a bit igual entre CPUs** (XNNPACK con instrucciones distintas), pero la diferencia es ~300 veces menor.
- **Conclusión:** el decodificador explica ~99.6 % de la diferencia del cuadro 0 (mediana de la razón por clip). El seguimiento cuadro a cuadro de MediaPipe la arrastra por el clip, y alcanza para cambiar 7 de 128 etiquetas.

## Lectura para el paper (sugerencia; el texto lo decide el autor)

Cambios de 1–5 niveles de gris, imperceptibles, mueven los landmarks unos 10 px en el primer cuadro y cambian 7 de 128 etiquetas. Es una medida de la sensibilidad del sistema a la entrada, y conviene reportarla junto con las dos cifras (110 y 115). En vivo no aplica igual: la cámara entrega la imagen sin decodificar video.

## Próximo paso propuesto (opción A, ver `IDEAS.md`)

Que todas las plataformas le pasen a MediaPipe la misma imagen: decodificar a YUV y hacer la conversión de color en el propio código. Antes de cambiar nada, una prueba barata: decodificar a YUV en la Mac y en la nube y comparar los hashes. Si coinciden, la conversión propia deja la entrada idéntica y solo queda la diferencia de CPU.

## Cómo se midió

- Herramientas: `run.json` (versiones, decodificador), `frame_sha1` de `frames.csv`, `landmarks/npz` exactos.
- Los scripts de la comparación se corrieron fuera del repo (la VM y el entorno en la nube de Claude); los PNG temporales se borraron.
