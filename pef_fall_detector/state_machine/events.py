"""The vocabulary of the funnel: formulations, stages, verdicts and the event.

Every other module of the package speaks in these names; keeping them here
lets each stage be read on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..ablations import SEQUENTIAL, SIMULTANEOUS

#: Trigger formulations. See the module docstring for why there are four.
SCORE = "score"                 # T and V combined into one instantaneous score
V_ONLY = "v_only"               # V alone; the single-threshold ablation
# SIMULTANEOUS and SEQUENTIAL (ablations only) are imported from .ablations.
FORMULATIONS = (SCORE, SIMULTANEOUS, SEQUENTIAL, V_ONLY)

#: Stage1Trigger.last_source / TriggerEvent.trigger_source when one of the two
#: EXPERIMENTAL fallbacks fired instead of the formulation (whose own name is
#: used otherwise): H + V (Quantity H) and the A drop (phase 8.4).
H_FALLBACK = "H"
A_FALLBACK = "A"


class Stage(Enum):
    """Where an observation sits in the §3.5 funnel."""

    MONITORING = "MONITORING"       # watching T and V, nothing pending
    CONFIRMING = "CONFIRMING"       # Stage 1 fired; Stage 2 judges the geometry
    OBSERVING = "OBSERVING"         # Stage 3 watches for immobility or recovery
    COOLDOWN = "COOLDOWN"           # refractory period after an event


#: Stage-2 outcomes.
CONFIRMED = "stage2_confirmed"
REJECTED = "stage2_rejected"
INCONCLUSIVE = "stage2_inconclusive"


#: Stage-3 outcomes and the severity vocabulary of §3.3.
CONFIRMED_FALL = "stage3_confirmed"
NULLIFIED = "stage3_nullified"
UNRESOLVED = "stage3_unresolved"
#: Phase 8.3 (Quantity A): the head never came near the floor and the hip
#: ended raised, so there was no fall. Counts as NoFall and raises no alarm.
#: See Stage3Evaluator.finalise.
NOT_DOWN = "stage3_not_down"
MILD, MODERATE, SEVERE = "mild", "moderate", "severe"


@dataclass
class TriggerEvent:
    """One Stage-1 firing, with the evidence that caused it.

    This is the auditable record §3.5 promises when it claims "a downstream
    reviewer can reconstruct the decision by reading the four quantities from
    the corresponding frames". Storing the values that fired the trigger —
    not just the fact that it fired — is what makes that reconstruction
    possible without re-running the video.
    """

    frame_index: int
    timestamp: float
    t_deg: float
    v_tps: float
    formulation: str
    #: Verdict after the funnel has run as far as it currently goes. Set to
    #: ``stage1_only`` while Stage 2 is still collecting, then to one of
    #: ``stage2_confirmed`` / ``stage2_rejected`` / ``stage2_inconclusive``.
    #: It never says "confirmed fall": Stage 3 does not exist yet, so no
    #: record can be mistaken for one that passed the whole funnel.
    verdict: str = "stage1_only"
    #: Stage-2 evidence, kept alongside the verdict for the same reason the
    #: trigger keeps T and V: a verdict without its evidence cannot be
    #: audited, only believed.
    p_outside_fraction: float = float("nan")
    p_samples: int = 0
    #: Stage-3 outcome. ``severity`` follows §3.3 and is defined by RECOVERY,
    #: not by impact: recovered alone (mild), partially recovered (moderate),
    #: remained down (severe). Empty until Stage 3 has run.
    severity: str = ""
    max_immobility_s: float = float("nan")
    #: Quantity H (EXPERIMENTAL, not in §3.4) at the firing frame, kept for
    #: the same auditability reason as t_deg/v_tps — including when it was
    #: NOT what fired the event, so a reviewer can see whether the H+V
    #: fallback (Stage1Trigger.threshold_h_ratio) or T/V's own formulation
    #: was responsible.
    h_ratio: float = float("nan")
    #: Which Stage-1 condition raised the event: the formulation's name
    #: (``score``...), :data:`H_FALLBACK` or :data:`A_FALLBACK`. The quantity
    #: values above say what the frame looked like; this says which rule
    #: read them as a fall, so the record needs no re-derivation.
    trigger_source: str = ""
    #: Why the verdict is what it is: the rule that decided, with the numbers
    #: it read. §3.5 promises that "a downstream reviewer can reconstruct the
    #: decision"; without this the reviewer has to re-derive which branch of
    #: the funnel closed the event. Empty until something resolves it.
    reason: str = ""
