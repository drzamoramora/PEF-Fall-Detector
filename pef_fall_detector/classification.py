"""Clip-level classification: the four labels of the PEF-FallDB protocol.

WHY THIS IS A SEPARATE LAYER, AND NOT A RENAME
----------------------------------------------
The paper's §3.3 vocabulary — mild / moderate / severe — describes ONE EVENT.
The four labels here describe ONE CLIP. They are not the same statement, and
collapsing them would cost something real:

* ``NoFall`` has no severity at all. A clip with no event is not a "mild"
  anything; it is the absence of the thing severity describes.
* A clip may contain more than one event. Severity cannot answer "what is
  this clip", only "what was that event".
* §3.3 is what a reader of the paper can check the code against. Renaming
  ``severe`` to ``NotRecovered`` inside the state machine would leave the
  events record unreadable next to the section that defines it.

So the paper's names stay where the paper can see them — in the per-event
record — and these four are the clip label, derived from them by the single
mapping in :func:`class_for_event`. One place to change, and the §3.3
correspondence is written down rather than remembered.

THE END-OF-CLIP RULE, AND WHY IT DIFFERS FROM THE LIVE ONE
----------------------------------------------------------
§3.3 defines severity by RECOVERY: recovered alone, partially recovered,
remained down. Those are statements about how the episode ENDED.

The live state machine resolves greedily — the first branch to fire wins —
because on the §3.6 device an alert that arrives late is worthless. That is
correct there and wrong here. Measured on clip A13, where the subject lies
still for about six seconds and then stands up unaided: with the immobility
radius calibrated to the real noise floor, the immobility branch reaches its
threshold BEFORE the subject rises, the event closes as ``severe``, and the
recovery three seconds later is never seen. Truth: Recovered. Verdict with a
greedy resolver: NotRecovered. The two most distant labels in the set.

For labelling, therefore, the clip is watched to the end and the label comes
from how the subject ENDED — which is what §3.3 asked for in the first place.
Immobility becomes evidence supporting ``NotRecovered``, not the trigger that
closes the case.
"""

from __future__ import annotations

from .state_machine import (
    CONFIRMED_FALL,
    MILD,
    MODERATE,
    NOT_DOWN,
    NULLIFIED,
    SEVERE,
)

#: No fall occurred in this clip.
NO_FALL = "NoFall"
#: A fall occurred and the subject got up unaided (§3.3 "mild").
RECOVERED = "Recovered"
#: A fall occurred; the subject is not incapacitated but did not fully get
#: up — sat up, knelt, propped themselves against furniture (§3.3 "moderate").
PARTIALLY_RECOVERED = "PartiallyRecovered"
#: A fall occurred and the subject remained on the floor (§3.3 "severe").
NOT_RECOVERED = "NotRecovered"
#: Something happened that no stage could judge — the trigger fired but the
#: feet were never visible, or the clip ended mid-verdict.
#:
#: This label is NOT one of the four, on purpose. Folding it into ``NoFall``
#: would count a failure as a correct negative on the NoFall clips and as a
#: plain miss everywhere else, and the confusion matrix would then hide the
#: difference between "never fired" and "fired but could not judge". Those
#: are two defects with different causes and different fixes. §3.7 asks for
#: an evaluation that can tell them apart.
UNDETERMINED = "Undetermined"

CLASSES = (NO_FALL, RECOVERED, PARTIALLY_RECOVERED, NOT_RECOVERED, UNDETERMINED)

#: Severity ranking used when a clip contains several events: the worst one
#: names the clip. A clip where someone recovered from one fall and stayed
#: down after a second is a NotRecovered clip.
_RANK = {
    NO_FALL: 0,
    RECOVERED: 1,
    PARTIALLY_RECOVERED: 2,
    NOT_RECOVERED: 3,
    # Ranked above NoFall so it can never be silently outvoted by "nothing
    # happened", and below the real outcomes so a decided event always wins
    # over an undecided one.
    UNDETERMINED: 1,
}

#: The §3.3 correspondence, written down once.
_FROM_SEVERITY = {
    (NULLIFIED, MILD): RECOVERED,
    (NULLIFIED, MODERATE): RECOVERED,
    (CONFIRMED_FALL, MODERATE): PARTIALLY_RECOVERED,
    (CONFIRMED_FALL, SEVERE): NOT_RECOVERED,
    (CONFIRMED_FALL, MILD): RECOVERED,
}


def class_for_event(verdict: str, severity: str,
                    unresolved_as: str = NO_FALL) -> str:
    """Map one resolved event to one of the labels.

    ``stage2_rejected`` is NoFall: a stage actively judged the geometry and
    found no topple.

    Everything that never reached the immobility confirmation —
    ``stage1_only``, ``stage2_confirmed``, ``stage2_inconclusive``,
    ``stage3_unresolved``, ``provisional_unconfirmed`` — is decided by
    ``unresolved_as``, and the default is **NoFall because that is what §3.4
    says**:

        *"If I exceeds a confirmation threshold W, the event is classified
        as a confirmed fall and an alert with the appropriate severity tag
        is dispatched."*

    Confirmation is conditional on I reaching W. When the subject stops
    being observable before that — which is what every one of these verdicts
    means — the condition is not met, so the event is not a confirmed fall
    and no alert goes out. Reporting it as a fifth class instead was this
    project's invention (divergence **D12**), and it made the record
    disagree with the deployed system: ``dispatch_verdicts`` only despatches
    ``stage3_confirmed``, so an unresolved event already produces no alarm
    and nobody attends. The label now says what the system does.

    ``unresolved_as=UNDETERMINED`` restores the previous behaviour. Keep it
    for diagnosis — "the funnel never finished" and "the funnel decided no"
    are different failures, and a run that cannot tell them apart hides
    clips where the subject simply left the frame.
    """
    if verdict == "stage2_rejected":
        return NO_FALL
    # Fase 8.3: la Etapa 3 midio que el cuerpo no llego al suelo (cantidad A).
    # Es un juicio activo, como el rechazo de la Etapa 2, no una falta de
    # evidencia: NoFall siempre, sin importar ``unresolved_as``.
    if verdict == NOT_DOWN:
        return NO_FALL
    return _FROM_SEVERITY.get((verdict, severity), unresolved_as)


def classify_clip(events, unresolved_as: str = NO_FALL) -> tuple[str, str]:
    """Label a whole clip from its resolved events.

    Returns ``(label, reason)`` — the reason is the short evidence string that
    goes into the record beside the label, so a disagreement between the
    machine and a human annotator can be examined without re-running the clip.
    """
    if not events:
        return NO_FALL, "sin eventos"

    labelled = [(class_for_event(e.verdict, e.severity, unresolved_as), e)
                for e in events]
    label, event = max(labelled, key=lambda pair: _RANK[pair[0]])
    reason = (f"evento t={event.timestamp:.2f}s "
              f"{event.verdict}/{event.severity or '-'}")
    if len(events) > 1:
        reason += f" (el peor de {len(events)})"
    return label, reason
