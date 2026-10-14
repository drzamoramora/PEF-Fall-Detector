"""The §3.5 funnel: the three stages joined, one event at a time."""

from __future__ import annotations

from .events import CONFIRMED, INCONCLUSIVE, REJECTED, Stage, TriggerEvent
from .stage2 import Stage2Evaluator
from .stage3 import Stage3Evaluator
from .trigger import Stage1Trigger


class FallStateMachine:
    """The §3.5 funnel: Stages 1 and 2 today, Stage 3 plugged in later.

    Stage 1 raises an event; the machine then sits in ``CONFIRMING`` while
    Stage 2 collects Quantity P over its window, and resolves with a
    geometric verdict. Stage 3 (immobility, recovery, severity) is Phase 5b
    and attaches at the same seam.

    **The verdict is asynchronous, and that is not an implementation
    convenience.** At the instant Stage 1 fires, the geometry has not
    happened yet — the body is still on its way down. Deciding "is the COM
    outside the feet" on the trigger frame would test the posture the subject
    was leaving, not the one they are arriving at. So the event is raised
    immediately (that is when it happened) and its verdict is filled in when
    the window closes. Consumers get the event object on the firing frame and
    can read ``verdict`` later; the record is written once everything has
    settled.

    Cooldown is the refractory period after an event resolves, so that one
    fall produces one alarm rather than a burst of them.
    """

    def __init__(self, trigger: Stage1Trigger,
                 stage2: "Stage2Evaluator | None" = None,
                 stage3: "Stage3Evaluator | None" = None,
                 cooldown_s: float = 3.0,
                 cooldown_after_rejection: bool = True) -> None:
        self.trigger = trigger
        self.stage2 = stage2
        self.stage3 = stage3
        self.cooldown_s = float(cooldown_s)
        #: Phase 4. The cooldown keeps ONE fall from arriving as a burst of
        #: events. A candidate Stage 2 rejected is not a fall, and the
        #: trigger's latch already prevents re-firing, so cooling down after
        #: it only risks hiding the real fall behind it. False: after a
        #: rejection, straight back to MONITORING. Measurements:
        #: notas/CODIGO-HISTORIA.md.
        self.cooldown_after_rejection = bool(cooldown_after_rejection)
        self.stage = Stage.MONITORING
        self.events: list[TriggerEvent] = []
        self._resolved_at: float | None = None
        self._pending: TriggerEvent | None = None
        #: The event whose verdict landed on THIS frame, or None. Separate
        #: from update()'s return value, which reports the frame an event was
        #: RAISED on: those are different moments, and the alert belongs to
        #: the second one. A consumer that dispatched on the raising frame
        #: would alert before any stage had judged the event.
        self.just_resolved: TriggerEvent | None = None

    def update(self, frame_index: int, timestamp: float,
               t_deg: float, v_tps: float,
               p_offset: float = float("nan"),
               i_still_s: float = float("nan"),
               extension_ratio: float = float("nan"),
               h_ratio: float = float("nan"),
               a_head: float = float("nan")) -> TriggerEvent | None:
        """Advance the machine one frame; returns a NEW event, or None.

        Only the frame on which Stage 1 fires returns an event. Later frames
        return None even while that event's verdict is still being decided —
        the object already handed out is updated in place.

        ``h_ratio``: Quantity H (EXPERIMENTAL, not in §3.4/§3.5). Threaded
        through to :class:`Stage1Trigger`'s OR'd fallback and to
        :class:`Stage3Evaluator`'s recovery check; NaN (the default) reduces
        both to their pre-H behaviour exactly. See the ``quantities`` package
        docstring for what it is proposed to cover.

        ``a_head``: Quantity A (EXPERIMENTAL, phase 8.4), threaded through
        to :class:`Stage1Trigger`'s A-drop condition only. NaN (the default)
        can never satisfy it, so omitting it reproduces 8.3 exactly.
        """
        self.just_resolved = None
        if self.stage is Stage.OBSERVING:
            self._advance_observing(timestamp, t_deg, i_still_s, extension_ratio, h_ratio)
            self.trigger.update(timestamp, t_deg, v_tps, h_ratio, a_head)
            return None

        if self.stage is Stage.CONFIRMING:
            self._advance_confirming(timestamp, p_offset, t_deg,
                                     i_still_s, extension_ratio, h_ratio)
            # The trigger keeps seeing frames so its own hold state stays
            # coherent, but a second firing cannot start while one event is
            # still being judged.
            self.trigger.update(timestamp, t_deg, v_tps, h_ratio, a_head)
            return None

        if self.stage is Stage.COOLDOWN:
            if self._resolved_at is not None and (
                timestamp - self._resolved_at >= self.cooldown_s
            ):
                self.stage = Stage.MONITORING
                self._resolved_at = None
            else:
                self.trigger.update(timestamp, t_deg, v_tps, h_ratio, a_head)
                return None

        if self.stage is not Stage.MONITORING:
            return None

        if not self.trigger.update(timestamp, t_deg, v_tps, h_ratio, a_head):
            return None

        event = TriggerEvent(
            frame_index=frame_index,
            timestamp=timestamp,
            t_deg=t_deg,
            v_tps=v_tps,
            formulation=self.trigger.formulation,
            h_ratio=h_ratio,
            trigger_source=self.trigger.last_source,
        )
        self.events.append(event)

        if self.stage2 is None:
            # No geometric check configured: the event stands on Stage 1
            # alone and says so. This is the §3.7 ablation "remove Stage 2".
            event.reason = "Stage 1 only (no Stage 2 configured)"
            self._resolve(timestamp, event)
            return event

        self._pending = event
        self.stage2.start(timestamp)
        self.stage2.observe(p_offset)
        self.stage = Stage.CONFIRMING
        return event

    def _advance_confirming(self, timestamp: float, p_offset: float,
                            t_deg: float, i_still_s: float,
                            extension_ratio: float,
                            h_ratio: float = float("nan")) -> None:
        assert self.stage2 is not None
        self.stage2.observe(p_offset)
        if not self.stage2.ready(timestamp):
            return
        verdict = self.stage2.verdict()
        if self._pending is not None:
            self._pending.reason = self._stage2_text(verdict)
            self._pending.verdict = verdict
            self._pending.p_outside_fraction = self.stage2.outside_fraction_seen
            self._pending.p_samples = self.stage2.samples
        self.stage2.reset()

        # A geometric rejection ends the funnel: §3.5 puts Stage 2 there
        # precisely to stop a controlled sit-down before it can be confirmed
        # by lying still afterwards — which a sit-down does.
        if verdict == REJECTED or self.stage3 is None:
            self._resolve(timestamp)
            return

        self.stage3.start(timestamp)
        self.stage3.observe(timestamp, t_deg, i_still_s, extension_ratio, h_ratio)
        self.stage = Stage.OBSERVING

    def _stage2_text(self, verdict: str) -> str:
        """Stage 2's outcome in words. Stage 3 overwrites it if it runs."""
        n = self.stage2.samples
        if verdict == INCONCLUSIVE:
            return (f"Stage 2: feet not measurable ({n} samples of P "
                    f"< {self.stage2.min_samples})")
        frac = 100.0 * self.stage2.outside_fraction_seen
        cut = 100.0 * self.stage2.outside_fraction
        rel = ">=" if verdict == CONFIRMED else "<"
        return f"Stage 2: COM outside the support {frac:.0f}% {rel} {cut:.0f}% ({n} samples)"

    def _advance_observing(self, timestamp: float, t_deg: float,
                           i_still_s: float, extension_ratio: float,
                           h_ratio: float = float("nan")) -> None:
        assert self.stage3 is not None
        self.stage3.observe(timestamp, t_deg, i_still_s, extension_ratio, h_ratio)
        if not self.stage3.ready(timestamp):
            return
        verdict, severity = self.stage3.verdict()
        if self._pending is not None:
            self._pending.verdict = verdict
            self._pending.severity = severity
            self._pending.max_immobility_s = self.stage3.max_immobility_s
            self._pending.reason = self.stage3.reason
        self.stage3.reset()
        self._resolve(timestamp)

    def close_if_last_seen_down(self, timestamp: float) -> TriggerEvent | None:
        """Detection lost during Stage 3: close the event if it was seen down.

        Called by the pipeline when the subject has been missing longer than
        the gap threshold, instead of abandoning the event outright. Returns
        the closed event, or None when the event must still be abandoned.

        Why not simply abandon, as before 23/09: a subject lying on the floor
        is exactly when MediaPipe loses them. Measured on A08-S3 (truth
        NotRecovered): trigger at 4.43 s, seen on the floor at T = 160 deg,
        then 4.7 s undetected to the end of the clip. Abandoning erased a
        fall that had already passed Stages 1 and 2, and the clip read
        "nothing happened". §3.5 only downgrades or nullifies an alert on
        evidence of a get-up — *"as evidenced by torso and hip landmark
        recovery to upright"* — and §3.3 labels not-recovered as *"remains
        on the ground or fails to recover during the observation window"*.

        Why only when last seen DOWN: the abandon rule exists because the
        body that reappears may be another person, so nothing seen after
        the loss is used — the event is closed on what was observed of the
        original body. When that last observation was upright, recovery was
        already visible and there is no alarm to keep, so abandoning (as
        before) costs nothing. Measured on the three NoFall clips whose
        events were abandoned (B07-S3, B08-S3, B10-S2): last seen at
        2-4 deg; closing them too would have turned all three into false
        positives. This asymmetry was chosen after seeing those four clips.

        Only from OBSERVING: an event still in Stage 2 never had its
        geometry judged, and a truncated window is not a verdict.
        """
        if (self._pending is None or self.stage is not Stage.OBSERVING
                or self.stage3 is None or not self.stage3.last_seen_down()):
            return None
        closed = self.finalise(timestamp)
        if closed is not None:
            closed.reason = f"detection lost on the floor; {closed.reason}"
        return closed

    def finalise(self, timestamp: float) -> TriggerEvent | None:
        """Close an event still open when the source runs out.

        Only meaningful for a recorded clip, which HAS an end; a live camera
        does not, and calling this there would resolve an episode that is
        still happening. Returns the event that was closed, or None.

        An event still in CONFIRMING — Stage 2 never collected its window —
        is closed too: the clip ended mid-verdict, and a labelling pass has
        to say something about it. It says ``stage2_inconclusive``, which the
        clip label reads as Undetermined, not as NoFall.
        """
        if self._pending is None:
            return None
        if self.stage is Stage.OBSERVING and self.stage3 is not None:
            self.stage3.finalise()
            verdict, severity = self.stage3.verdict()
            self._pending.verdict = verdict
            self._pending.severity = severity
            self._pending.max_immobility_s = self.stage3.max_immobility_s
            self._pending.reason = self.stage3.reason
            self.stage3.reset()
        elif self.stage is Stage.CONFIRMING and self.stage2 is not None:
            self._pending.reason = "the clip ended during Stage 2"
            self._pending.verdict = INCONCLUSIVE
            self._pending.p_outside_fraction = self.stage2.outside_fraction_seen
            self._pending.p_samples = self.stage2.samples
            self.stage2.reset()
        closed = self._pending
        self._resolve(timestamp)
        return closed

    def _resolve(self, timestamp: float, event: TriggerEvent | None = None) -> None:
        settled = event if event is not None else self._pending
        self.just_resolved = settled
        self._pending = None
        if (not self.cooldown_after_rejection and settled is not None
                and settled.verdict == REJECTED):
            self.stage = Stage.MONITORING
            self._resolved_at = None
            return
        self.stage = Stage.COOLDOWN
        self._resolved_at = timestamp

    def reset(self) -> None:
        """Discontinuity: forget pending state, keep the event history.

        An event still in ``CONFIRMING`` is abandoned rather than judged on a
        truncated window. Its verdict stays ``stage1_only``, which is the
        truth: Stage 2 never finished looking at it.

        Note on ``self._pending = None`` below: it is **defence in depth, not
        the mechanism**. What actually abandons the event is returning to
        ``MONITORING``, since ``_advance_confirming`` only runs from
        ``CONFIRMING`` and the next firing overwrites the field anyway — a
        mutation removing this line passes every test, verified. It stays
        because a future stage added between these two states would make the
        stale reference reachable, and the failure would be an event judged
        with another event's evidence.
        """
        if self._pending is not None:
            self._pending.reason = (
                "abandoned at a discontinuity (detection lost or jump); "
                + (self._pending.reason or "no stage verdict"))
        self.trigger.reset()
        if self.stage2 is not None:
            self.stage2.reset()
        if self.stage3 is not None:
            self.stage3.reset()
        self._pending = None
        self.stage = Stage.MONITORING
        self._resolved_at = None
