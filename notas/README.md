# notas/ — el registro de desarrollo

Las notas están en español, el idioma de trabajo del proyecto; el código y su
documentación están en inglés, el idioma del paper. Son la evidencia detrás de
las afirmaciones de método: qué se midió, con qué código y qué se descartó.

## Vigentes

| nota | para qué |
|---|---|
| `TRASPASO-SESION.md` | **El mapa.** Estado, reglas de trabajo, resultados y pendientes. Leer primero. |
| `POSIBLES-CAMBIOS-DEL-DRAFT.md` | Los cambios propuestos al draft 4, con evidencia y prioridad. |
| `ANALISIS-82-90.md` | El diagnóstico de los errores y la propuesta de la cantidad A (fase 8). |
| `NOFALL.md` | Por qué los falsos positivos son los ADL de estrés del §3.3. |
| `HALLAZGO-B05.md` | Por qué H falla en B05 (el sofá). |
| `CONVENCION-NOMBRES.md` | La convención de nombres de PEF-FallDB. |
| `DESCARTADOS.md` | Mecanismos medidos y retirados del código (banda provisional, rechazo por descenso). |
| `CONFIG-HISTORIA.md` | Por qué cada valor de `config.yaml` es el que es. |
| `CODIGO-HISTORIA.md` | Las historias que contaban los comentarios del código antes de pasarlo a inglés. |

## archivo/

Notas de fases ya cerradas. Se conservan porque las vigentes y el paper las
citan como fuente de cifras, pero describen el código y los números de su
momento, no los de hoy: el plan de fases original (`FASES.md`, `PLAN-90.md`),
la revisión de código (`REVISION-CODIGO.md`), el catálogo de hallazgos y
pruebas (`HALLAZGOS-PROPUESTAS-PRUEBAS.md`), el traspaso del draft
(`TRASPASO-DRAFT.md`), el análisis de los clips que fallaban
(`ANALISIS-VIDEOS.md`), qué se podía tocar sin tocar el paper
(`DENTRO-FUERA-DRAFT4.md`) y las medidas de las fases 0–3
(`NOTA-FASES-0-1.md`).

Los tres scripts (`replay_disparador.py`, `simular_caida.py`,
`verificar_mediciones.py`) regeneran las cifras que esas notas citan. Usan los
módulos de antes de la limpieza, así que corren sobre el código de la etiqueta
`v0.1.1-legacy`, no sobre el actual. El plan de B7 pedía convertir
`simular_caida.py` (caídas sintéticas por el embudo real) en prueba; no se
convirtió tal cual, pero su propósito lo cubren `tests/test_main_headless.py`
(caídas sintéticas por `main.py` y el pipeline real) y `tests/test_end_to_end.py`
(doce clips reales con la configuración real).

Cuando una nota vigente cita `notas/X.md` y X está aquí, es la misma nota.
