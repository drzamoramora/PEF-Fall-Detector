"""Trigger formulations kept only for the ablations of §3.7.

The detector runs ``score`` (``config.yaml``, ``stage1.trigger_formulation``).
Two other readings of §3.5's "instantaneous combination of the two
quantities" live here, so that the ablation study is a configuration change
and not a code change:

* ``simultaneous``: T and V past their own thresholds on the same frame, the
  conjunction this project assumed first (8 of 13 real falls when measured);
* ``sequential``: V sustained arms the event, and T must confirm it within
  ``confirm_window_s`` (12 of 13).

``v_only`` (V alone) is not here: it is the single-threshold pipeline §3.7
names explicitly, and it stays beside ``score`` in :mod:`.state_machine`.

Both rules use :class:`.state_machine.Stage1Trigger`'s own hold and latch,
passed in as ``held``: a formulation changes what is combined, never how long
the condition must hold or how often it may fire.
"""

from __future__ import annotations

from collections.abc import Callable

SIMULTANEOUS = "simultaneous"   # T and V above threshold in the same frame
SEQUENTIAL = "sequential"       # V fires, T confirms within a later window
ABLATION_FORMULATIONS = (SIMULTANEOUS, SEQUENTIAL)

#: ``Stage1Trigger._held``: True on the frame a condition completes its hold.
Held = Callable[[float, bool], bool]


def simultaneous(held: Held, timestamp: float, t_hit: bool, v_hit: bool) -> bool:
    """Both quantities past their thresholds on the same frame, held."""
    return held(timestamp, t_hit and v_hit)


class Sequential:
    """V sustained arms the event; T has ``confirm_window_s`` to confirm it.

    Measured on real falls, the peaks of T and V are offset by 0.03 s to
    1.8 s, which is why requiring them on the same frame loses events that
    both quantities clearly witnessed.
    """

    def __init__(self, confirm_window_s: float = 1.0) -> None:
        self.confirm_window_s = float(confirm_window_s)
        #: When V armed the event, or None while nothing is armed.
        self.armed_at: float | None = None

    def step(self, held: Held, timestamp: float, t_hit: bool, v_hit: bool) -> bool:
        """Feed one frame; True on the frame T confirms an armed V event."""
        if held(timestamp, v_hit):
            self.armed_at = timestamp
        if self.armed_at is None:
            return False
        if timestamp - self.armed_at > self.confirm_window_s:
            self.armed_at = None
            return False
        if t_hit:
            self.armed_at = None
            return True
        return False

    def reset(self) -> None:
        """Disarm. An armed V event must not survive a break in observation
        and be confirmed by a T measured on what may be a different body."""
        self.armed_at = None
