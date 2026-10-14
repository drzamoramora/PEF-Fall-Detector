"""The 33 landmarks of a run as exact binary arrays (``landmarks/npz/<clip>.npz``).

Schema 1 kept the landmarks only as text rounded to 0.01 px, 0.001 of
visibility and 0.1 mm. That was enough to look at them and not enough to
replay a run exactly: the phase-8.4 replay changed one verdict and had to
"nudge" visibilities near the threshold. Nor is it enough to find where two
platforms start to disagree, which happens below 0.01 px.

MediaPipe returns float32 numbers. Kept as float32 here, they are exactly
what the model produced: nothing rounded, nothing converted. The image
coordinates are kept NORMALISED, as MediaPipe gives them; the pixels the
detector works in are ``x * width, y * height, z * width`` (pose_frontend),
and :func:`pixel_landmarks` recomputes them with the very same operation, so
a replay gets bit-identical pixels.

One file per clip, written when the clip's records close; ``numpy.load``
reads it back in one line, with the shape a skeleton model expects
(frames x 33 x coordinates). The same values, exact, are also written as
text (``landmarks/csv``) for anyone who wants to open them in a spreadsheet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: What each array holds; written into ``run.json`` as the npz schema.
NPZ_LAYOUT = {
    "format": "pef-landmarks/1",
    "frame_index": "int32 (N,)",
    "timestamp_s": "float64 (N,)",
    "detected": "bool (N,): MediaPipe found a person",
    "frame_size": "int32 (N, 2): width, height in pixels",
    "image": "float32 (N, 33, 3): MediaPipe's normalised x, y, z, exactly as returned; "
             "pixels = x*width, y*height, z*width. NaN when not detected",
    "visibility": "float32 (N, 33)",
    "world": "float32 (N, 33, 3): metric 3D landmarks, metres, hip-centred. NaN when absent",
}
FORMAT = NPZ_LAYOUT["format"]

_NAN3 = np.full((33, 3), np.nan, dtype=np.float32)
_NAN1 = np.full(33, np.nan, dtype=np.float32)


class LandmarkArrays:
    """Collects one clip's landmarks frame by frame; :meth:`save` writes the npz."""

    def __init__(self) -> None:
        self.frame_index: list[int] = []
        self.timestamp: list[float] = []
        self.detected: list[bool] = []
        self.frame_size: list[tuple[int, int]] = []
        self.image: list[np.ndarray] = []
        self.visibility: list[np.ndarray] = []
        self.world: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.frame_index)

    def add(self, pf) -> None:
        """One PoseFrame."""
        self.frame_index.append(int(pf.frame_index))
        self.timestamp.append(float(pf.timestamp))
        self.frame_size.append((int(pf.frame_size[0]), int(pf.frame_size[1])))
        detected = bool(pf.detected) and len(pf.landmarks) == 33
        self.detected.append(detected)
        if not detected:
            self.image.append(_NAN3)
            self.visibility.append(_NAN1)
            self.world.append(_NAN3)
            return
        self.image.append(normalised_landmarks(pf))
        self.visibility.append(np.asarray(pf.visibility, dtype=np.float32))
        world = pf.world_landmarks
        self.world.append(np.asarray(world, dtype=np.float32) if len(world) == 33 else _NAN3)

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self)
        return {
            "format": np.array(FORMAT),
            "frame_index": np.asarray(self.frame_index, dtype=np.int32),
            "timestamp_s": np.asarray(self.timestamp, dtype=np.float64),
            "detected": np.asarray(self.detected, dtype=bool),
            "frame_size": np.asarray(self.frame_size, dtype=np.int32).reshape(n, 2),
            "image": np.stack(self.image) if n else np.empty((0, 33, 3), np.float32),
            "visibility": np.stack(self.visibility) if n else np.empty((0, 33), np.float32),
            "world": np.stack(self.world) if n else np.empty((0, 33, 3), np.float32),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"landmark record already exists: {path}")
        with open(path, "xb") as fh:
            np.savez_compressed(fh, **self.arrays())
        return path


def normalised_landmarks(pf) -> np.ndarray:
    """The frame's landmarks as MediaPipe returned them (float32, normalised).

    A PoseFrame from the front end carries them (``landmarks_norm``). One
    built another way (a test, a schema-1 replay) only has pixels; those are
    divided back, which is exact only to float32.
    """
    norm = getattr(pf, "landmarks_norm", None)
    if norm is not None and len(norm) == 33:
        return np.asarray(norm, dtype=np.float32)
    w, h = pf.frame_size
    scale = np.array([w, h, w], dtype=np.float64)
    return (np.asarray(pf.landmarks, dtype=np.float64) / scale).astype(np.float32)


def pixel_landmarks(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """``(33, 3)`` pixels from normalised float32, the way pose_frontend computes them.

    ``lm.x * width`` there multiplies a Python float holding the float32 value
    by an int; this does the same in float64, so the result is bit-identical.
    """
    return np.asarray(image, dtype=np.float32).astype(np.float64) * np.array(
        [width, height, width], dtype=np.float64)


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Every array of a landmark file."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    fmt = str(out.get("format", ""))
    if fmt != FORMAT:
        raise ValueError(f"{path}: unknown landmark format {fmt!r}")
    return out
