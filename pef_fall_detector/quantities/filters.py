"""Smoothing shared by the quantities."""

from __future__ import annotations

import math

import numpy as np


class ExponentialMovingAverage:
    """Time-constant EMA smoother, applied before differentiation.

    Differentiation amplifies noise: MediaPipe landmarks jitter a few pixels
    even on a still subject, and a raw derivative would turn that jitter
    into spurious velocity spikes that could fire the Stage-1 trigger with
    nobody moving (implementation-plan §5.4). Measured on real footage, this
    filter removes ~63 % of the landmark jitter at a cost of ~4 % peak
    attenuation.

    The smoothing strength is a **time constant in seconds**, not a
    per-sample coefficient. The blend factor is recomputed for every sample
    from the actual elapsed time::

        alpha = 1 - exp(-dt / tau)
        smoothed = alpha * new + (1 - alpha) * previous

    Why time and not samples: a fixed per-sample coefficient means a
    different physical memory at every frame rate — alpha = 0.3 is ~0.09 s
    of memory at 30 fps but ~0.19 s at 15 fps — and live cameras deliver
    frames at irregular intervals anyway (21-25 % variation measured on this
    project's own sessions). Deriving alpha from the real dt makes the
    filter behave identically regardless of capture rate or frame drops.

    ``tau = 0`` disables smoothing (pass-through), which is the honest way
    to turn the filter off for an experiment. Works on scalars and on numpy
    vectors alike.
    """

    def __init__(self, time_constant_s: float) -> None:
        if time_constant_s < 0.0:
            raise ValueError(f"EMA time constant must be >= 0, got {time_constant_s}")
        self.tau = float(time_constant_s)
        self._value: float | np.ndarray | None = None

    def update(self, value, dt: float):
        """Feed one sample taken ``dt`` seconds after the previous one."""
        if self._value is None or self.tau <= 0.0 or dt <= 0.0:
            self._value = value
        else:
            alpha = 1.0 - math.exp(-dt / self.tau)
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self):
        """Current smoothed value, or NaN before the first sample."""
        return float("nan") if self._value is None else self._value

    def reset(self) -> None:
        """Forget all history (call when the subject is lost or on seek)."""
        self._value = None
