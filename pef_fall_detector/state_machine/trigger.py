"""Stage 1 of §3.5: the kinematic trigger and its peak window."""

from __future__ import annotations

import math
from collections import deque

from ..ablations import SIMULTANEOUS, Sequential, simultaneous
from .events import A_FALLBACK, FORMULATIONS, H_FALLBACK, SCORE, V_ONLY


class PeakWindow:
    """The most extreme T and V seen inside a short trailing window.

    WHY THIS EXISTS. §3.5 asks Stage 1 to fire when *"an instantaneous
    combination of the two quantities exceeds a configurable trigger
    threshold"*. Requiring the same frame is an extra condition, and §3.4
    does not imply it: §3.4 describes each quantity's own temporal
    signature — T reaching the fall range *"within a short time window"*,
    V going strongly negative *"within a short window"* — and says nothing
    about the two coinciding.

    They do not coincide, and the reason is mechanical rather than
    incidental. T is a position and V is a velocity of the same body: the
    centroid moves fastest while the trunk is still near vertical, and the
    trunk reaches its greatest inclination once the motion has almost
    stopped. Measured over PEF-FallDB, the peak of T and the peak of V are
    separated by 0.6 s to 3.1 s on every fall the trigger missed — never
    less than half a second.

    This class widens "the same frame" to "the same short window" without
    touching what is combined or how. §3.5 already grants each stage a
    tunable window length (*"Each stage's input variables, threshold
    values, and window lengths are tunable at deployment time"*), so the
    window is a parameter the paper already declared, not new machinery.

    Time, never frames (the C5 lesson): the same "24 frames" spans 0.8 s at
    30 fps and 1.6 s at 15 fps, and this project has clips at both.

    NaN is skipped, not read as zero. A frame whose quantity could not be
    computed is an absence of evidence; folding it in as 0.0 would let an
    occlusion lower a maximum or raise a minimum, which is the same class of
    error the immobility timer avoids by tracking episodes.
    """

    def __init__(self, window_s: float) -> None:
        if window_s <= 0.0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self.window_s = float(window_s)
        self._samples: deque[tuple[float, float, float]] = deque()

    def update(self, timestamp: float, t_deg: float, v_tps: float) -> tuple[float, float]:
        """Feed one frame; return ``(max T, min V)`` over the window.

        Either component is NaN when no frame in the window carried a usable
        value for it, which downstream reads as "condition not met" exactly
        as a NaN from the quantity itself would.
        """
        self._samples.append((timestamp, t_deg, v_tps))
        cutoff = timestamp - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        ts = [t for _, t, _ in self._samples if not math.isnan(t)]
        vs = [v for _, _, v in self._samples if not math.isnan(v)]
        return (max(ts) if ts else float("nan"),
                min(vs) if vs else float("nan"))

    def reset(self) -> None:
        """Forget the window (subject lost, seek, discontinuity).

        Load-bearing: without it the peak of T from before an occlusion could
        combine with a V measured after it, on what may be a different body.
        """
        self._samples.clear()


class Stage1Trigger:
    """The kinematic trigger of §3.5, in the selected formulation.

    Holds the state the formulations need: how long the condition has been
    continuously true, and (for the ``sequential`` ablation, in
    :mod:`.ablations`) whether a V event is armed and waiting for T.

    Args:
        formulation: one of :data:`FORMULATIONS`.
        threshold_t_deg: trunk angle above which T counts as triggered.
        threshold_v_tps: centroid velocity below which V counts as triggered
            (negative — §3.5 requires "V remains negative (downward)").
        hold_s: how long the condition must persist before firing. A single
            frame is noise; this is the hysteresis.
        confirm_window_s: for ``sequential`` only, how long after the V event
            T may still arrive and confirm it.
        threshold_score: for ``score`` only, the value the combined score
            must reach. 2.0 means "both quantities exactly at their own
            threshold"; above that, the two together must be more extreme.
        peak_window_s: how far back T and V may each be taken from. **0.0
            is the literal reading of §3.5** — both quantities from the same
            frame — and reproduces this class's behaviour before the
            parameter existed, which is what makes the §3.7 ablation a
            configuration change rather than a code change. Above 0.0 the
            formulation is fed the window's maximum T and minimum V instead
            of the current frame's, per :class:`PeakWindow`.
        threshold_h_ratio: EXPERIMENTAL, not in §3.4/§3.5. Quantity H
            (``quantities.body_height_ratio``) below this value, combined
            with V still falling, fires a SEPARATE edge-triggered condition,
            OR'd with whichever formulation above decided — never mixed into
            the T/V arithmetic those already have calibrated. It exists for
            one measured gap: a fall whose rotation axis points at the
            camera projects the trunk vector to almost nothing, so T reads a
            falsely low angle and every formulation above can miss the fall
            outright. ``<= 0.0`` disables it — H is always positive when
            defined, so no real reading can satisfy the condition, the same
            "off by an impossible threshold" convention used across
            ``config.yaml``.
        threshold_h_erect / h_sequence_window_s: EXPERIMENTAL. When
            ``threshold_h_erect > 0``, the H+V fallback above additionally
            requires H to have read at/above ``threshold_h_erect`` (subject
            confirmed standing) at some point in the last
            ``h_sequence_window_s`` seconds — encoding the STORY §3.4 already
            tells for T and V separately (upright, then a short-window
            collapse) instead of reading a bare instantaneous "collapsed +
            falling" as sufficient.

            READ THIS BEFORE RAISING ``threshold_h_erect`` TO STOP A FALSE
            ALARM: it answers "did a genuine standing-to-collapsed transition
            just happen", not "was this transition a fall". A deliberate
            crouch (tying a shoe) is ALSO standing-then-collapsed — the two
            events have the same shape in H, V and T alike. This guard only
            removes the case where H briefly reads low from noise, or where
            H was never confirmed erect at all (subject already down when
            observation started); it does not and cannot separate a fall
            from a controlled crouch, which is a question of intent that no
            posture trajectory answers by itself. ``<= 0.0`` disables the
            requirement and the fallback reverts to the bare instantaneous
            check above.
        threshold_a_high / threshold_a_low / a_window_s: EXPERIMENTAL
            (phase 8.4), not in §3.4/§3.5. A third, separate condition, OR'd
            like the H fallback and never mixed into the T/V arithmetic:
            Quantity A (head height against gravity, ``quantities.
            GravityCalibrator``) reaches ``threshold_a_low`` or less while a
            reading at/above ``threshold_a_high`` exists in the last
            ``a_window_s`` seconds, AND the body is still descending fast on
            that same frame (V below ``threshold_v_tps``). Held ``hold_s``,
            with its own latch.

            It covers the falls the score cannot see: a collapse with the
            trunk vertical (sitting or kneeling fall) keeps T near 0, so the
            score stays low although the head drops to floor level. Measured
            on PEF-FallDB (notas/TRASPASO-SESION.md, 8.4): A3-S2 has its head
            at 9 % of standing height and a score peak of 2.3.

            Why V is read on the SAME frame, not over the peak window the
            H fallback uses: what separates a fall from lying down on purpose
            is how the body ARRIVES near the floor. A fall is still
            accelerated by gravity when it gets there; a controlled descent
            brakes before (pre-impact vertical velocity, Bourke et al. 2008,
            Med. Eng. Phys. 30:937-946). Measured when the head reaches 0.3:
            V from -0.53 to +0.66 in the lying-down clips (B11-S1, B11-S2,
            B12-S2), -2.8 in A3-S2. With the peak window, B12-S2 (fast kneel,
            then a slow lie-down) raises a false alarm.

            A is NaN until the subject has been seen standing, so this
            condition cannot fire before calibration. ``threshold_a_high
            <= 0`` disables it, the same "off by an impossible threshold"
            convention as ``threshold_h_ratio``.
    """

    def __init__(
        self,
        formulation: str = SCORE,
        threshold_t_deg: float = 45.0,
        threshold_v_tps: float = -1.5,
        hold_s: float = 0.1,
        confirm_window_s: float = 1.0,
        threshold_score: float = 2.2,
        peak_window_s: float = 0.0,
        threshold_h_ratio: float = 0.0,
        threshold_h_erect: float = 0.0,
        h_sequence_window_s: float = 2.0,
        threshold_a_high: float = 0.0,
        threshold_a_low: float = 0.3,
        a_window_s: float = 1.5,
    ) -> None:
        if formulation not in FORMULATIONS:
            raise ValueError(
                f"unknown trigger formulation {formulation!r}; "
                f"expected one of {FORMULATIONS}"
            )
        if hold_s < 0.0:
            raise ValueError("hold_s must be >= 0")
        if peak_window_s < 0.0:
            raise ValueError("peak_window_s must be >= 0")
        if threshold_v_tps >= 0.0:
            # §3.5 requires the trigger to fire while V is negative. A
            # non-negative threshold would make "V < threshold" true for a
            # subject standing still, firing continuously.
            raise ValueError("threshold_v_tps must be negative (downward)")
        if threshold_a_high > 0.0:
            if threshold_a_low >= threshold_a_high:
                # A drop needs somewhere to drop from: with low >= high a
                # subject simply standing at A = 1.0 would satisfy both ends.
                raise ValueError("threshold_a_low must be below threshold_a_high")
            if a_window_s <= 0.0:
                raise ValueError("a_window_s must be > 0")
        self.formulation = formulation
        self.threshold_t_deg = float(threshold_t_deg)
        self.threshold_v_tps = float(threshold_v_tps)
        self.hold_s = float(hold_s)
        self.confirm_window_s = float(confirm_window_s)
        self.threshold_score = float(threshold_score)
        #: Score of THIS frame, or NaN if the formulation computes none or
        #: the frame was not evaluable. The pipeline records it (column
        #: ``trigger_score``): it is what the trigger actually compared.
        self.last_score = float("nan")
        self.peak_window_s = float(peak_window_s)
        self._peaks = PeakWindow(peak_window_s) if peak_window_s > 0.0 else None
        self._condition_since: float | None = None
        self._fired = False
        self._sequential = Sequential(confirm_window_s)
        self.threshold_h_ratio = float(threshold_h_ratio)
        #: Latch state for the H+V fallback, kept SEPARATE from
        #: ``_condition_since``/``_fired`` above so the fallback's own
        #: hysteresis cannot interfere with (or be short-circuited by) the
        #: primary formulation's.
        self._h_condition_since: float | None = None
        self._h_fired = False
        self.threshold_h_erect = float(threshold_h_erect)
        self.h_sequence_window_s = float(h_sequence_window_s)
        #: Trailing (timestamp, h_ratio) samples, kept only while the
        #: erect-sequence requirement is active. A deque, same shape as
        #: PeakWindow's own history, trimmed to h_sequence_window_s.
        self._h_history: deque[tuple[float, float]] = deque()
        self.threshold_a_high = float(threshold_a_high)
        self.threshold_a_low = float(threshold_a_low)
        self.a_window_s = float(a_window_s)
        #: Trailing (timestamp, A) readings for the A-drop condition, only
        #: measured ones (NaN = not calibrated or not measurable), trimmed to
        #: a_window_s. A third latch, independent of the other two for the
        #: same reason ``_held_h`` is.
        self._a_history: deque[tuple[float, float]] = deque()
        self._a_condition_since: float | None = None
        self._a_fired = False
        #: Which condition raised the LAST firing: the formulation's name,
        #: :data:`H_FALLBACK` or :data:`A_FALLBACK`; empty on frames that did
        #: not fire. Read by the state machine on the same frame and stored
        #: in the event, so the record says which rule raised it.
        self.last_source = ""

    def update(self, timestamp: float, t_deg: float, v_tps: float,
               h_ratio: float = float("nan"),
               a_head: float = float("nan")) -> bool:
        """Feed one frame's T, V and (optionally) H and A; True on the frame that fires.

        ``h_ratio`` and ``a_head`` only feed the OR'd fallbacks described on
        ``threshold_h_ratio`` and ``threshold_a_high`` above; every
        formulation's own T/V arithmetic is untouched; see
        :meth:`_evaluate_formulation`. ``last_source`` names the condition
        that fired (the formulation first, then H, then A).
        """
        fired, v_hit = self._evaluate_formulation(timestamp, t_deg, v_tps)
        # Evaluated unconditionally, NOT short-circuited into the h_hit
        # chain below: it must record an erect reading into the history on
        # every frame it is seen, including the frames where H has not (yet)
        # collapsed — which is precisely every frame before a fall.  Folding
        # it into the `and` chain would only ever call it once H was already
        # at or below the collapse threshold, so the erect samples that are
        # supposed to satisfy threshold_h_erect would never be recorded.
        recently_erect = self._recently_erect(timestamp, h_ratio)
        h_hit = (self.threshold_h_ratio > 0.0 and not math.isnan(h_ratio)
                 and h_ratio <= self.threshold_h_ratio
                 and recently_erect)
        fallback_fired = self._held_h(timestamp, h_hit and v_hit)
        # The RAW v_tps of this frame, not v_hit: the peak window's minimum
        # exists only inside _evaluate_formulation, and the A condition must
        # read V on the frame the head arrives (see threshold_a_high above).
        # Each condition keeps its own state, so the order of the three
        # evaluations does not matter.
        a_fired = self._a_drop(timestamp, a_head, v_tps)
        if fired:
            self.last_source = self.formulation
        elif fallback_fired:
            self.last_source = H_FALLBACK
        elif a_fired:
            self.last_source = A_FALLBACK
        else:
            self.last_source = ""
        return fired or fallback_fired or a_fired

    def _a_drop(self, timestamp: float, a_head: float, v_tps: float) -> bool:
        """The A-drop condition of phase 8.4, held; True on the frame it fires.

        Called on every frame the trigger sees, in every stage of the
        machine, so the history and the latch stay coherent while an event
        is being judged (same reason as the other two conditions). Disabled
        entirely, history included, when ``threshold_a_high <= 0``.
        """
        if self.threshold_a_high <= 0.0:
            return False
        if not math.isnan(a_head):
            self._a_history.append((timestamp, a_head))
        # Inclusive window: a reading exactly a_window_s old still counts.
        # The tolerance absorbs the rounding of frame_index / fps, so the
        # boundary frame does not come and go with floating point.
        cutoff = timestamp - self.a_window_s - 1e-9
        while self._a_history and self._a_history[0][0] < cutoff:
            self._a_history.popleft()
        # The isnan guards are defensive, as in _evaluate_formulation: IEEE
        # comparisons are already False for NaN, and removing them changes
        # nothing today (verified by mutation).
        condition = (not math.isnan(a_head) and a_head <= self.threshold_a_low
                     and not math.isnan(v_tps) and v_tps < self.threshold_v_tps
                     and any(a >= self.threshold_a_high for _ts, a in self._a_history))
        return self._held_a(timestamp, condition)

    def _held_a(self, timestamp: float, condition: bool) -> bool:
        """Same edge-triggered hold as :meth:`_held_h`, for the A-drop condition."""
        if not condition:
            self._a_condition_since = None
            self._a_fired = False
            return False
        if self._a_fired:
            return False
        if self._a_condition_since is None:
            self._a_condition_since = timestamp
        if timestamp - self._a_condition_since >= self.hold_s:
            self._a_condition_since = None
            self._a_fired = True
            return True
        return False

    def _recently_erect(self, timestamp: float, h_ratio: float) -> bool:
        """Whether H confirmed the subject standing within the last window.

        EXPERIMENTAL, see ``threshold_h_erect`` above for what this does and
        — more importantly — does NOT establish. Disabled entirely
        (returns True unconditionally, i.e. a no-op) when
        ``threshold_h_erect <= 0``, which reduces the H+V fallback to the
        bare instantaneous check it had before this guard existed.
        """
        if self.threshold_h_erect <= 0.0:
            return True
        self._h_history.append((timestamp, h_ratio))
        cutoff = timestamp - self.h_sequence_window_s
        while self._h_history and self._h_history[0][0] < cutoff:
            self._h_history.popleft()
        return any(not math.isnan(h) and h >= self.threshold_h_erect
                  for _ts, h in self._h_history)

    def _evaluate_formulation(self, timestamp: float, t_deg: float, v_tps: float
                              ) -> tuple[bool, bool]:
        """The §3.5 trigger in the selected formulation. Returns (fired, v_hit).

        Unchanged from before Quantity H existed — ``v_hit`` is returned
        alongside the verdict only so :meth:`update`'s H fallback can reuse
        the exact same "V still falling" reading (after any peak-window
        substitution) instead of recomputing it.

        Feed one frame's T and V; returns True on the frame that fires.

        NaN inputs count as "condition not met": a quantity that could not be
        computed is not evidence of a fall, and letting an unmeasured frame
        vote is how an occlusion turns into an alarm.

        The explicit ``isnan`` guards below are **defensive, not reachable**:
        IEEE comparison already returns False for every NaN operand, so
        removing them changes nothing today and no test can detect their
        removal (verified by mutation). They are kept because the guarantee
        is stated in prose here, and a future rewrite that reaches these
        conditions through ``any()``, a numpy array or an inverted
        comparison would silently lose it. The same reasoning, and the same
        honesty about unreachability, applies to the cosine clamp in
        ``quantities.trunk_inclination_deg``.
        """
        if self._peaks is not None:
            # Widen "the same frame" to "the same short window". Every frame
            # is fed in, NaN included, so the window advances on wall-clock
            # time rather than on the count of measurable frames; what the
            # formulations then see is the window's extremes. Nothing below
            # this line knows the difference, which is the point: the
            # combination rule of §3.5 is untouched, only the span the two
            # quantities may be drawn from.
            t_deg, v_tps = self._peaks.update(timestamp, t_deg, v_tps)

        self.last_score = float("nan")
        t_hit = not math.isnan(t_deg) and t_deg > self.threshold_t_deg
        v_hit = not math.isnan(v_tps) and v_tps < self.threshold_v_tps

        if self.formulation == SCORE:
            # The reading of §3.5 that follows its sentence literally: "an
            # instantaneous combination of the two quantities exceeds a
            # configurable trigger threshold while V remains negative". Each
            # term is the quantity over its own threshold, so it contributes
            # 1.0 exactly when that quantity reaches the value §3.4 calls
            # the fall range; a score of 2.0 therefore means "both at
            # threshold", and anything above it means the two together are
            # more extreme than that. The downward condition is enforced on
            # the same frame, not inherited from an earlier one.
            # The sign check is load-bearing, and not for the obvious
            # reason. A large POSITIVE V suppresses itself: its term goes
            # strongly negative and sinks the score. What it actually guards
            # is the subject already on the floor — T near 180 contributes
            # 4.0 by itself, more than any sensible threshold, so with V
            # drifting upward by a hair the score would sail through on the
            # trunk angle alone and alarm on someone lying still.
            #
            # That T can carry the sum unaided is a property of this
            # formulation worth watching rather than hiding. On the 13 real
            # falls it produced 15 events against the sequential rule's 13 —
            # no runaway — but those clips are 2 to 10 seconds long, and the
            # steady state of a subject lying inverted for a minute has never
            # been observed. Capping the terms would remove the risk and
            # would also be a fresh departure from §3.5, which is what this
            # formulation exists to avoid.
            if math.isnan(t_deg) or math.isnan(v_tps) or v_tps >= 0.0:
                return self._held(timestamp, False), v_hit
            score = (t_deg / self.threshold_t_deg) + (v_tps / self.threshold_v_tps)
            self.last_score = score
            return self._held(timestamp, score >= self.threshold_score), v_hit

        if self.formulation == V_ONLY:
            return self._held(timestamp, v_hit), v_hit

        # The two readings kept only for the §3.7 ablations (ablations.py).
        if self.formulation == SIMULTANEOUS:
            return simultaneous(self._held, timestamp, t_hit, v_hit), v_hit
        return self._sequential.step(self._held, timestamp, t_hit, v_hit), v_hit

    def _held(self, timestamp: float, condition: bool) -> bool:
        """True on the frame where ``condition`` completes its hold time.

        Edge-triggered, not level-triggered: after firing, the condition must
        become false again before it can fire a second time. Without that
        latch a condition that simply stays true — a subject lying with the
        trunk past threshold — would re-arm every ``hold_s`` and emit an
        event per interval, so one fall would arrive as a burst. The cooldown
        in :class:`FallStateMachine` limits the rate of that burst; only the
        latch removes it.
        """
        if not condition:
            self._condition_since = None
            self._fired = False
            return False
        if self._fired:
            return False
        if self._condition_since is None:
            self._condition_since = timestamp
        if timestamp - self._condition_since >= self.hold_s:
            self._condition_since = None
            self._fired = True
            return True
        return False

    def _held_h(self, timestamp: float, condition: bool) -> bool:
        """Same edge-triggered hold as :meth:`_held`, for the H+V fallback.

        A second, independent latch — not a reuse of ``_held`` — so the
        fallback's own hysteresis cannot interact with the primary
        formulation's: the two conditions are evaluated over different
        quantities and must not share a clock.
        """
        if not condition:
            self._h_condition_since = None
            self._h_fired = False
            return False
        if self._h_fired:
            return False
        if self._h_condition_since is None:
            self._h_condition_since = timestamp
        if timestamp - self._h_condition_since >= self.hold_s:
            self._h_condition_since = None
            self._h_fired = True
            return True
        return False

    def reset(self) -> None:
        """Drop all pending state (subject lost, seek, discontinuity).

        Matters most for the ``sequential`` ablation: an armed V event must
        not survive a break in observation and be confirmed by a T measured on
        what may be a different body, or a different point in the timeline.
        """
        self._condition_since = None
        self._sequential.reset()
        self._fired = False
        self._h_condition_since = None
        self._h_fired = False
        self._h_history.clear()
        # The A readings belong to the body seen before the break; after a
        # re-acquisition they may be another person's (same as H above).
        self._a_history.clear()
        self._a_condition_since = None
        self._a_fired = False
        self.last_source = ""
        self.last_score = float("nan")
        if self._peaks is not None:
            self._peaks.reset()
