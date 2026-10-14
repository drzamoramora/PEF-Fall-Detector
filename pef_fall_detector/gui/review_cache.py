"""Frozen analysis of clips already run in PEF-Lab. Qt-free, so it can be tested alone.

THE PROBLEM IT SOLVES
---------------------
Before, when a clip ended the window kept nothing of what it had computed.
Stepping frame by frame called ``pipeline.reset()`` and processed the frame
again from scratch: V went back to "filling window", I restarted, the stage
changed, the curves were overwritten, and events that never existed in the
real pass could even fire. What was seen while reviewing was not what the
system saw while analysing. And moving to another clip of the queue
re-analysed it whole and wrote a new CSV every time.

THE RULE
--------
* A clip analysed from start to end, with no jumps, is FROZEN in memory with
  the result of every frame exactly as the pass produced it.
* Going back to that clip, with the SAME configuration and without closing
  PEF-Lab, shows it frozen: nothing is recomputed and no record is written.
* If the configuration changed, what was stored belongs to another
  configuration and showing it would mislead: the clip is re-analysed.
* "Re-analyze" and the queue's "Start" are explicit decisions to run again:
  they drop what was stored and produce a new record.
* None of this survives closing PEF-Lab. On purpose: the code may have
  changed between sessions, and re-analysing produces data from the current
  code.

A pass with jumps (the user moved the slider mid-analysis) is NOT frozen: its
frames after the jump came from a reset pipeline, not from a continuous
observation, and freezing them would present as final something that is not.
The clip is re-analysed the next time it is opened.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def config_key(config: dict[str, Any]) -> str:
    """Stable fingerprint of a configuration: same content, same key."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _clip_id(path: str | Path) -> str:
    return str(Path(path).resolve())


@dataclass
class ClipAnalysis:
    """Everything a full pass computed on one clip."""

    path: str
    config_key: str
    fps: float
    #: frame index -> FrameResult, exactly as the pass produced it.
    frames: dict[int, Any]
    #: the machine's events at the end (with verdict and reason).
    events: list = field(default_factory=list)
    #: the queue's label row, if the clip ran in a queue.
    label: dict | None = None
    #: the per-frame record this pass wrote.
    record_path: str = ""

    @property
    def frame_count(self) -> int:
        return len(self.frames)


class PassRecorder:
    """Collects a continuous pass; any jump invalidates it.

    Frames must arrive in order, starting at 0 and with no gaps. That is what
    tells a continuous observation from one that restarted halfway.
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
            self.invalidate(f"frame {index} arrived when {self._next} was expected")
            return
        self._frames[index] = result
        self._next += 1

    def invalidate(self, reason: str) -> None:
        if self.invalid_reason is None:
            self.invalid_reason = reason

    def finish(self, events: list, record_path: str = "",
               label: dict | None = None) -> ClipAnalysis | None:
        """Close the pass. None if it cannot be frozen (jumps or empty)."""
        if not self.valid or not self._frames:
            return None
        return ClipAnalysis(path=self.path, config_key=self.cfg_key, fps=self.fps,
                            frames=self._frames, events=list(events),
                            label=label, record_path=record_path)


class ReviewCache:
    """The frozen clips of this PEF-Lab session, by (video, configuration)."""

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
