"""Análisis congelado de clips ya corridos en PEF-Lab. Sin Qt, para probarse solo.

EL PROBLEMA QUE RESUELVE
------------------------
Antes, al terminar un clip la ventana no guardaba nada de lo que había
calculado. Moverse cuadro a cuadro llamaba ``pipeline.reset()`` y volvía a
procesar el cuadro desde cero: V volvía a «filling window», I reiniciaba, la
etapa cambiaba, las curvas se sobrescribían y hasta podían dispararse eventos
que en la pasada real no existieron. Lo que se veía al revisar no era lo que el
sistema vio al analizar. Y navegar a otro clip de la cola lo re-analizaba
entero y escribía un CSV nuevo cada vez.

LA REGLA
--------
* Un clip analizado de principio a fin, sin saltos, queda CONGELADO en memoria
  con el resultado de cada cuadro tal como salió de la pasada.
* Volver a ese clip, con la MISMA configuración y sin cerrar PEF-Lab, lo
  muestra congelado: nada se recalcula y no se escribe ningún registro.
* Si la configuración cambió, lo guardado corresponde a otra configuración y
  mostrarlo sería engañoso: el clip se re-analiza.
* «Re-analizar desde 0» e «Iniciar cola» son decisiones explícitas de correr
  de nuevo: descartan lo guardado y producen un registro nuevo.
* Nada de esto sobrevive a cerrar PEF-Lab. Es a propósito: el código puede
  haber cambiado entre sesiones, y re-analizar produce datos del código actual.

Una pasada con saltos (el usuario movió la barra a mitad del análisis) NO se
congela: sus cuadros posteriores al salto salieron de un pipeline reiniciado,
no de una observación continua, y congelarlos presentaría como definitivo algo
que no lo es. El clip se re-analiza la próxima vez que se abra.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def config_key(config: dict[str, Any]) -> str:
    """Huella estable de una configuración: mismo contenido, misma clave."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _clip_id(path: str | Path) -> str:
    return str(Path(path).resolve())


@dataclass
class ClipAnalysis:
    """Todo lo que una pasada completa calculó sobre un clip."""

    path: str
    config_key: str
    fps: float
    #: índice de cuadro -> FrameResult, tal como salió de la pasada.
    frames: dict[int, Any]
    #: los eventos de la máquina al terminar (con veredicto y motivo).
    events: list = field(default_factory=list)
    #: la fila de etiqueta de la cola, si el clip se corrió en una cola.
    label: dict | None = None
    #: el registro de cuadros que esta pasada escribió.
    record_path: str = ""

    @property
    def frame_count(self) -> int:
        return len(self.frames)


class PassRecorder:
    """Junta una pasada continua; cualquier salto la invalida.

    Los cuadros deben llegar en orden, empezando en 0 y sin huecos. Eso es lo
    que distingue una observación continua de una que se reinició a medias.
    """

    def __init__(self, path: str | Path, cfg_key: str, fps: float) -> None:
        self.path = _clip_id(path)
        self.cfg_key = cfg_key
        self.fps = float(fps)
        self._frames: dict[int, Any] = {}
        self._next = 0
        self.invalid_reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None

    def record(self, index: int, result: Any) -> None:
        if not self.valid:
            return
        if index != self._next:
            self.invalidate(f"cuadro {index} llego cuando se esperaba el {self._next}")
            return
        self._frames[index] = result
        self._next += 1

    def invalidate(self, reason: str) -> None:
        if self.invalid_reason is None:
            self.invalid_reason = reason

    def finish(self, events: list, record_path: str = "",
               label: dict | None = None) -> ClipAnalysis | None:
        """Cierra la pasada. None si no se puede congelar (saltos o vacía)."""
        if not self.valid or not self._frames:
            return None
        return ClipAnalysis(path=self.path, config_key=self.cfg_key, fps=self.fps,
                            frames=self._frames, events=list(events),
                            label=label, record_path=record_path)


class ReviewCache:
    """Los clips congelados de esta sesión de PEF-Lab, por (video, configuración)."""

    def __init__(self) -> None:
        self._clips: dict[tuple[str, str], ClipAnalysis] = {}

    def get(self, path: str | Path, cfg_key: str) -> ClipAnalysis | None:
        return self._clips.get((_clip_id(path), cfg_key))

    def store(self, analysis: ClipAnalysis) -> None:
        self._clips[(analysis.path, analysis.config_key)] = analysis

    def drop(self, path: str | Path, cfg_key: str) -> None:
        self._clips.pop((_clip_id(path), cfg_key), None)

    def __len__(self) -> int:
        return len(self._clips)
