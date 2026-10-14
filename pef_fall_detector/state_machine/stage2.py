"""Stage 2 of §3.5: the geometric check on Quantity P."""

from __future__ import annotations

import math

from .events import CONFIRMED, INCONCLUSIVE, REJECTED


class Stage2Evaluator:
    """The geometric check of §3.5, on Quantity P.

    §3.5: *"Stage 2 evaluates P to ensure that the event is geometrically
    fall-like rather than a controlled sit-down in which COM stays within the
    support polygon."* That sentence names one test — where the COM sits
    relative to the support — so that is the only test implemented here.

    §3.4 also characterises a fall by *"a sudden contraction of the support
    polygon from a full two-foot footprint to a heel-only contact pair"*, and
    the width is recorded every frame in ``P_support_width``. It is
    deliberately **not** consumed: §3.4 offers it as a description of falls,
    §3.5 does not put it in Stage 2, and wiring it in would be this code
    deciding something the paper did not. The data is there for the day that
    changes, with evidence.

    **Why a vote and not a single frame.** At the instant Stage 1 fires the
    body is still moving, and one frame of landmark noise can put the COM on
    either side of a foot. The verdict is taken over a window: the fraction
    of *measurable* frames in which the COM projects outside must reach
    ``outside_fraction``.

    **Why the window is in seconds.** Same reason as everywhere else in this
    project: a frame count is a different physical duration on a laptop and
    on the §3.6 Raspberry Pi. A vote count would be worse still — "5 of 10
    frames" is unreachable at 10 fps over a third of a second, so the
    threshold is a fraction of whatever was actually observed.

    **Occluded feet do not silently kill the event.** When too few frames of
    the window carried a measurable P, the verdict is ``INCONCLUSIVE`` rather
    than ``REJECTED``. Feet leave the frame constantly — furniture, cropping,
    the subject's own body — and treating "I could not look" as "no fall"
    would turn the most common failure of the sensor into silence. An
    inconclusive event passes to Stage 3, which can still confirm on
    immobility, and the verdict is recorded so §3.7 can report how often it
    happens instead of it hiding inside the aggregate.
    """

    def __init__(self, window_s: float = 0.33, outside_fraction: float = 0.5,
                 min_samples: int = 3) -> None:
        if window_s <= 0.0:
            raise ValueError("window_s must be > 0")
        if not 0.0 < outside_fraction <= 1.0:
            raise ValueError("outside_fraction must be in (0, 1]")
        self.window_s = float(window_s)
        self.outside_fraction = float(outside_fraction)
        self.min_samples = int(min_samples)
        self._started_at: float | None = None
        self._outside = 0
        self._measured = 0

    def start(self, timestamp: float) -> None:
        """Open the evaluation window at the moment Stage 1 fired."""
        self._started_at = float(timestamp)
        self._outside = 0
        self._measured = 0

    def observe(self, p_offset: float) -> None:
        """Record one frame's P. NaN frames count as unmeasured, not as inside."""
        if math.isnan(p_offset):
            return
        self._measured += 1
        if p_offset > 0.0:
            self._outside += 1

    def ready(self, timestamp: float) -> bool:
        """Whether the window has elapsed and a verdict can be taken."""
        if self._started_at is None:
            return False
        return timestamp - self._started_at >= self.window_s

    @property
    def outside_fraction_seen(self) -> float:
        """Share of measurable frames with the COM outside; NaN if none."""
        if self._measured == 0:
            return float("nan")
        return self._outside / self._measured

    @property
    def samples(self) -> int:
        return self._measured

    def verdict(self) -> str:
        if self._measured < self.min_samples:
            return INCONCLUSIVE
        return (CONFIRMED if self.outside_fraction_seen >= self.outside_fraction
                else REJECTED)

    def reset(self) -> None:
        self._started_at = None
        self._outside = 0
        self._measured = 0
