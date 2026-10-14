"""Stage 1 and the confirmation state machine (§3.5).

This is where the system stops measuring and starts **deciding**. Everything
before it produces numbers; this module turns numbers into events.

The three stages of §3.5 are a funnel. Stage 1 is deliberately paranoid: it
must catch every fall even at the cost of also firing on a fast sit-down or
a crouch. Filtering those is the job of Stage 2 (geometry, via P) and Stage 3
(immobility, via I). A Stage 1 with some false positives is therefore working
as designed — a Stage 1 that misses a fall is not, because nothing
downstream can recover an event that was never raised.

**Why the trigger formulation is a configurable choice.** §3.5 says an event
enters Stage 2 "whenever an instantaneous combination of the two quantities
exceeds a configurable trigger threshold while V remains negative". That
sentence does not determine an implementation: a conjunction, a weighted sum
and a sequence are all "combinations", and they behave very differently.
Measured on 13 real falls with the trigger required to hold for 0.1 s, the
same underlying thresholds gave:

    score  13/13 falls     simultaneous  8/13     sequential  12/13     V alone  13/13

``score`` is the default because it is the reading that follows the sentence
most closely — a *combination* of the two quantities, evaluated
*instantaneously*, against a *configurable threshold*, with V required to be
negative on that same frame — and because it happens to lose no falls. The
conjunction is the reading this project assumed first, and it is the worst
of the four.

Two cautions belong with those numbers. The false-positive column (0 for
every formulation except ``v_only``, which had 1) comes from four non-fall
sessions: far too small to say anything about specificity, and no curated
ADL footage exists yet. And ``score`` buys its sensitivity by letting one
quantity carry the sum — which is also what a tracking artefact does. A
spurious −13 torso/s spike scores 8.7 on the velocity term alone and fires
regardless of the trunk. That is an argument for the subject-plausibility
guard, not for capping the terms: capping would be a new departure from the
paper, and the point of this formulation is to remove one.

All four are selected from ``config.yaml``; ``simultaneous`` and
``sequential``, kept only for the ablations, live in :mod:`.ablations`. The
configuration also supplies the ablation §3.7 requires — "the three-stage logic is replaced
with a single-threshold pipeline and the precision drop is reported" —
instead of it needing separate throwaway scripts. The table came from
``notas/archivo/replay_disparador.py``, which runs on the code it was written
for (tag ``v0.1.1-legacy``).

**Time, never frames.** The hold and cooldown durations are in seconds.
Expressed in frames they would mean different physical durations on the
author's machine and on the §3.6 Raspberry Pi, and the measured sensitivity
of the trigger to that parameter (8/13 against 12/13 above) is precisely
what a frame count would make device-dependent.

**Layout (0.2.0.dev4).** One module per piece of the funnel: :mod:`.events`
(formulations, stages, verdicts and the event record), :mod:`.trigger`
(Stage 1), :mod:`.stage2`, :mod:`.stage3` and :mod:`.machine` (the funnel
that joins them), plus :mod:`.build`, which assembles it from the
configuration. Every name is re-exported here, so
``from pef_fall_detector.state_machine import X`` reads as it always did.
"""

from __future__ import annotations

from ..ablations import SEQUENTIAL, SIMULTANEOUS, Sequential, simultaneous
from .events import (
    A_FALLBACK,
    CONFIRMED,
    CONFIRMED_FALL,
    FORMULATIONS,
    H_FALLBACK,
    INCONCLUSIVE,
    MILD,
    MODERATE,
    NOT_DOWN,
    NULLIFIED,
    REJECTED,
    SCORE,
    SEVERE,
    UNRESOLVED,
    V_ONLY,
    Stage,
    TriggerEvent,
)
from .machine import FallStateMachine
from .stage2 import Stage2Evaluator
from .stage3 import Stage3Evaluator
from .trigger import PeakWindow, Stage1Trigger

__all__ = [
    "A_FALLBACK", "CONFIRMED", "CONFIRMED_FALL", "FORMULATIONS", "H_FALLBACK",
    "INCONCLUSIVE", "MILD", "MODERATE", "NOT_DOWN", "NULLIFIED", "REJECTED", "SCORE",
    "SEQUENTIAL", "SEVERE", "SIMULTANEOUS", "UNRESOLVED", "V_ONLY",
    "FallStateMachine", "PeakWindow", "Sequential", "Stage", "Stage1Trigger",
    "Stage2Evaluator", "Stage3Evaluator", "TriggerEvent", "simultaneous",
]
