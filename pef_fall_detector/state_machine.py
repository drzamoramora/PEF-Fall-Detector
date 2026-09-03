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

All four are implemented and selected from ``config.yaml``, which also
supplies the ablation §3.7 requires — "the three-stage logic is replaced
with a single-threshold pipeline and the precision drop is reported" —
instead of it needing separate throwaway scripts. Reproduce the table with
``python notas/replay_disparador.py``.

**Time, never frames.** The hold and cooldown durations are in seconds.
Expressed in frames they would mean different physical durations on the
author's machine and on the §3.6 Raspberry Pi, and the measured sensitivity
of the trigger to that parameter (8/13 against 12/13 above) is precisely
what a frame count would make device-dependent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

#: Trigger formulations. See the module docstring for why there are four.
SCORE = "score"                 # T and V combined into one instantaneous score
SIMULTANEOUS = "simultaneous"   # T and V above threshold in the same frame
SEQUENTIAL = "sequential"       # V fires, T confirms within a later window
V_ONLY = "v_only"               # V alone; the single-threshold ablation
FORMULATIONS = (SCORE, SIMULTANEOUS, SEQUENTIAL, V_ONLY)


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


#: Stage-3 outcomes and the severity vocabulary of §3.3.
CONFIRMED_FALL = "stage3_confirmed"
NULLIFIED = "stage3_nullified"
UNRESOLVED = "stage3_unresolved"
MILD, MODERATE, SEVERE = "mild", "moderate", "severe"


class Stage3Evaluator:
    """Immobility, recovery and severity (§3.5, §3.3).

    §3.5: *"Stage 3 evaluates I across the post-trigger window: if a
    successful get-up sequence is detected within the immobility window as
    evidenced by torso and hip landmark recovery to upright, the alert's
    severity tag is downgraded or the alarm is nullified; if the immobility
    persists, the alert is dispatched with the appropriate severity tag."*

    §3.3 fixes what the tags mean, and they are defined by **recovery, not
    by impact**: the subject recovered on their own (``mild``), partially
    recovered (``moderate``), or remained on the ground (``severe``).

    **When it resolves.** Whichever comes first: a recovery held long enough
    to be real, or ``I`` reaching the confirmation threshold W of §3.4 ("If I
    exceeds a confirmation threshold W, the event is classified as a
    confirmed fall"). Waiting out the full observation window before every
    alarm would add tens of seconds of latency to the one output that is
    time-critical. §3.5's richer behaviour — dispatch, then *downgrade* if
    the subject later gets up — is a refinement this does not implement; the
    observation window only bounds how long the verdict may stay open.

    **What counts as recovery, and its known limit.** §3.5 asks for "torso
    and hip landmark recovery to upright", so two signals are used: the
    trunk angle T returning below ``upright_t_deg`` (torso), and the
    hip-to-ankle extension returning to standing range (hip landmarks).

    The second one deserves a flag. That extension is recorded as
    ``extension_ratio_EXP`` — an experimental column, and this project's own
    rule is that no ``_EXP`` signal enters a decision before the draft says
    so. It is used here because without it the tags cannot be separated at
    all: with T alone, a subject sitting upright on the floor is
    indistinguishable from one standing, which is exactly the
    ``moderate`` / ``mild`` boundary. §3.5's phrase "hip landmark recovery"
    is the closest the draft comes to authorising it, and that is not the
    same as authorising it. ``use_leg_extension=False`` runs without it and
    reports ``moderate`` for any recovery it cannot qualify — never
    ``mild``, because claiming a full recovery one cannot verify is the
    error that matters here.

    **``defer_to_end``: the labelling mode.** With it set, no branch closes
    the event early; the evaluator accumulates and the verdict comes from
    :meth:`finalise`, which reads how the episode ENDED. §3.3 defines
    severity by recovery — "recovered alone", "partially recovered",
    "remained down" — and all three are statements about the end of the
    episode, not about a moment inside it. Greedy resolution answers a
    different question, and on clip A13 of this project's own dataset the two
    answers are opposites: the subject lies still for about six seconds and
    then stands up unaided, so with the immobility radius calibrated to the
    real noise floor the immobility branch closes the event as ``severe``
    three seconds before the recovery it was supposed to be watching for.

    Off by default. On the §3.6 device, waiting for the end of an episode
    that may never end is not an option — there, latency is the whole point,
    and being wrong about A13 is the price of alerting in time.
    """

    def __init__(self, window_s: float = 30.0, threshold_w_s: float = 5.0,
                 upright_t_deg: float = 30.0, recovery_hold_s: float = 1.0,
                 standing_extension: float = 1.1,
                 use_leg_extension: bool = True,
                 defer_to_end: bool = False,
                 final_window_s: float = 1.0,
                 lying_t_deg: float = 60.0,
                 max_extension: float = 3.0) -> None:
        if window_s <= 0.0 or threshold_w_s <= 0.0:
            raise ValueError("window_s and threshold_w_s must be > 0")
        if final_window_s <= 0.0:
            raise ValueError("final_window_s must be > 0")
        if lying_t_deg <= upright_t_deg:
            raise ValueError("lying_t_deg must be above upright_t_deg")
        self.window_s = float(window_s)
        self.threshold_w_s = float(threshold_w_s)
        self.upright_t_deg = float(upright_t_deg)
        self.recovery_hold_s = float(recovery_hold_s)
        self.standing_extension = float(standing_extension)
        self.use_leg_extension = bool(use_leg_extension)
        self.defer_to_end = bool(defer_to_end)
        self.final_window_s = float(final_window_s)
        self.lying_t_deg = float(lying_t_deg)
        self.max_extension = float(max_extension)
        self._started_at: float | None = None
        self._upright_since: float | None = None
        self._legs_extended = False
        self._max_still = 0.0
        self._resolution: tuple[str, str] | None = None
        #: Trailing samples of (timestamp, t_deg, extension_ratio), kept only
        #: for the final read. The LAST frame alone is a coin flip on noisy
        #: landmarks; a short window at the end is the posture the subject
        #: actually finished in.
        self._tail: list[tuple[float, float, float]] = []

    def start(self, timestamp: float) -> None:
        self._started_at = float(timestamp)
        self._upright_since = None
        self._legs_extended = False
        self._max_still = 0.0
        self._resolution = None
        self._tail = []

    def observe(self, timestamp: float, t_deg: float, i_still_s: float,
                extension_ratio: float) -> None:
        """Feed one post-trigger frame. Resolves as soon as it can."""
        if self._started_at is None or self._resolution is not None:
            return

        if not math.isnan(i_still_s):
            self._max_still = max(self._max_still, i_still_s)

        # Keep a trailing window for finalise(); cheap, and only this.
        self._tail.append((float(timestamp), t_deg, extension_ratio))
        cutoff = float(timestamp) - self.final_window_s
        while len(self._tail) > 1 and self._tail[0][0] < cutoff:
            self._tail.pop(0)

        upright = not math.isnan(t_deg) and t_deg < self.upright_t_deg
        if upright:
            if self._upright_since is None:
                self._upright_since = timestamp
            if not math.isnan(extension_ratio):
                # Latched: the legs only need to reach standing range once
                # while the trunk is up. Requiring both in the same frame
                # would lose the common get-up, where the trunk straightens
                # a moment before the knees finish extending.
                self._legs_extended |= extension_ratio >= self.standing_extension
        else:
            self._upright_since = None
            self._legs_extended = False

        if self.defer_to_end:
            # Labelling mode: accumulate, decide in finalise(). Both branches
            # below answer "what happened at some point", and the label needs
            # "how did it end".
            return

        # A recovery that held long enough to be a recovery, not a wobble.
        if (self._upright_since is not None
                and timestamp - self._upright_since >= self.recovery_hold_s):
            if not self.use_leg_extension:
                self._resolution = (NULLIFIED, MODERATE)
            elif self._legs_extended:
                self._resolution = (NULLIFIED, MILD)
            else:
                # Upright but not standing: sitting up, kneeling, propped
                # against furniture. §3.3's "partially recovered".
                self._resolution = (CONFIRMED_FALL, MODERATE)
            return

        # Still down, and immobile for the confirmation threshold of §3.4.
        if self._max_still >= self.threshold_w_s:
            self._resolution = (CONFIRMED_FALL, SEVERE)

    def finalise(self) -> None:
        """Decide from how the episode ENDED. Called when the source runs out.

        §3.3's three tags are three endings, so this reads the posture the
        subject finished in, over ``final_window_s`` rather than one frame —
        a single landmark frame is noise, a second of them is a posture.

        THREE BANDS, NOT TWO. The trunk angle is cut twice, not once::

            T_final >= lying_t_deg              -> remained down    -> severe
            upright_t_deg <= T_final < lying    -> sat up, knelt    -> moderate
            T_final < upright_t_deg, legs out   -> stood up         -> mild
            T_final < upright_t_deg, legs folded-> could not verify -> moderate

        A single cut at ``upright_t_deg`` was the original design, and it sent
        "partially recovered" into the "remained down" bucket: measured on
        this project's dataset, 9 of 24 PartiallyRecovered clips came out
        NotRecovered. Their final trunk angles cluster at 35-55 deg — someone
        who sat up on the floor, knelt, or propped themselves against
        furniture is neither vertical nor lying down, and §3.3's middle tag
        names exactly that posture. A binary cut cannot represent three
        outcomes.

        ``lying_t_deg`` is the same boundary the display labels already use
        for LYING, so the two agree on what "down" means.

        A subject with no usable trunk angle in the final window is left
        UNRESOLVED. Reporting "remained down" for someone the camera stopped
        being able to see would be inventing the one outcome that matters
        most, and the clip label has an honest place for that answer.
        """
        if self._started_at is None or self._resolution is not None:
            return
        angles = [t for _ts, t, _e in self._tail if not math.isnan(t)]
        if not angles:
            return                              # stays UNRESOLVED
        # The median, not the mean: one frame of a collapsed pose estimate —
        # measured at 2 px of trunk on clip A14 — drags a mean across the
        # upright threshold, and that single frame would decide the label.
        angles.sort()
        final_t = angles[len(angles) // 2]
        if final_t >= self.lying_t_deg:
            self._resolution = (CONFIRMED_FALL, SEVERE)
            return
        if final_t >= self.upright_t_deg:
            # Trunk up but not vertical: sat up, knelt, propped against
            # furniture. §3.3's "partially recovered", read directly.
            self._resolution = (CONFIRMED_FALL, MODERATE)
            return
        if not self.use_leg_extension:
            self._resolution = (NULLIFIED, MODERATE)
            return
        # The upper bound is a plausibility guard, not a taste: a hip-to-ankle
        # span of more than a few torso lengths is the ratio dividing by a
        # foreshortened torso, not a leg. Values up to 183 torso lengths were
        # recorded on this dataset. An impossible number must not be evidence
        # of anything, least of all of a recovery.
        extended = any(not math.isnan(e)
                       and self.standing_extension <= e <= self.max_extension
                       for _ts, _t, e in self._tail)
        self._resolution = ((NULLIFIED, MILD) if extended
                            else (CONFIRMED_FALL, MODERATE))

    def ready(self, timestamp: float) -> bool:
        if self._resolution is not None:
            return True
        if self._started_at is None:
            return False
        if self.defer_to_end:
            # Only finalise() closes a deferred event; the observation window
            # must not quietly resolve it as UNRESOLVED behind our back.
            return False
        return timestamp - self._started_at >= self.window_s

    def verdict(self) -> tuple[str, str]:
        """(verdict, severity). ``UNRESOLVED`` when the window ran out."""
        if self._resolution is not None:
            return self._resolution
        # Neither still enough to confirm nor upright enough to clear: the
        # subject moved about on the floor for the whole window. Reporting a
        # severity here would be inventing one.
        return (UNRESOLVED, MODERATE)

    @property
    def max_immobility_s(self) -> float:
        return self._max_still

    def reset(self) -> None:
        self._started_at = None
        self._upright_since = None
        self._legs_extended = False
        self._max_still = 0.0
        self._resolution = None


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


class Stage1Trigger:
    """The kinematic trigger of §3.5, in the selected formulation.

    Holds the state the formulations need: how long the condition has been
    continuously true, and (for ``sequential``) whether a V event is armed
    and waiting for T to confirm.

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
    """

    def __init__(
        self,
        formulation: str = SCORE,
        threshold_t_deg: float = 45.0,
        threshold_v_tps: float = -1.5,
        hold_s: float = 0.1,
        confirm_window_s: float = 1.0,
        threshold_score: float = 2.2,
    ) -> None:
        if formulation not in FORMULATIONS:
            raise ValueError(
                f"unknown trigger formulation {formulation!r}; "
                f"expected one of {FORMULATIONS}"
            )
        if hold_s < 0.0:
            raise ValueError("hold_s must be >= 0")
        if threshold_v_tps >= 0.0:
            # §3.5 requires the trigger to fire while V is negative. A
            # non-negative threshold would make "V < threshold" true for a
            # subject standing still, firing continuously.
            raise ValueError("threshold_v_tps must be negative (downward)")
        self.formulation = formulation
        self.threshold_t_deg = float(threshold_t_deg)
        self.threshold_v_tps = float(threshold_v_tps)
        self.hold_s = float(hold_s)
        self.confirm_window_s = float(confirm_window_s)
        self.threshold_score = float(threshold_score)
        self._condition_since: float | None = None
        self._armed_at: float | None = None
        self._fired = False

    def update(self, timestamp: float, t_deg: float, v_tps: float) -> bool:
        """Feed one frame's T and V; returns True on the frame that fires.

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
                return self._held(timestamp, False)
            score = (t_deg / self.threshold_t_deg) + (v_tps / self.threshold_v_tps)
            return self._held(timestamp, score >= self.threshold_score)

        if self.formulation == SIMULTANEOUS:
            return self._held(timestamp, t_hit and v_hit)
        if self.formulation == V_ONLY:
            return self._held(timestamp, v_hit)

        # SEQUENTIAL: V sustained arms the event, then T confirms it inside
        # the window. Measured on real falls, the peaks of T and V are offset
        # by 0.03 s to 1.8 s, which is why requiring them in the same frame
        # loses events that both quantities clearly witnessed.
        if self._held(timestamp, v_hit):
            self._armed_at = timestamp
        if self._armed_at is None:
            return False
        if timestamp - self._armed_at > self.confirm_window_s:
            self._armed_at = None
            return False
        if t_hit:
            self._armed_at = None
            return True
        return False

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

    def reset(self) -> None:
        """Drop all pending state (subject lost, seek, discontinuity).

        Matters most for ``sequential``: an armed V event must not survive a
        break in observation and be confirmed by a T measured on what may be
        a different body, or a different point in the timeline.
        """
        self._condition_since = None
        self._armed_at = None
        self._fired = False


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
                 cooldown_s: float = 3.0) -> None:
        self.trigger = trigger
        self.stage2 = stage2
        self.stage3 = stage3
        self.cooldown_s = float(cooldown_s)
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
               extension_ratio: float = float("nan")) -> TriggerEvent | None:
        self.just_resolved = None
        """Advance the machine one frame; returns a NEW event, or None.

        Only the frame on which Stage 1 fires returns an event. Later frames
        return None even while that event's verdict is still being decided —
        the object already handed out is updated in place.
        """
        if self.stage is Stage.OBSERVING:
            self._advance_observing(timestamp, t_deg, i_still_s, extension_ratio)
            self.trigger.update(timestamp, t_deg, v_tps)
            return None

        if self.stage is Stage.CONFIRMING:
            self._advance_confirming(timestamp, p_offset, t_deg,
                                     i_still_s, extension_ratio)
            # The trigger keeps seeing frames so its own hold state stays
            # coherent, but a second firing cannot start while one event is
            # still being judged.
            self.trigger.update(timestamp, t_deg, v_tps)
            return None

        if self.stage is Stage.COOLDOWN:
            if self._resolved_at is not None and (
                timestamp - self._resolved_at >= self.cooldown_s
            ):
                self.stage = Stage.MONITORING
                self._resolved_at = None
            else:
                self.trigger.update(timestamp, t_deg, v_tps)
                return None

        if self.stage is not Stage.MONITORING:
            return None

        if not self.trigger.update(timestamp, t_deg, v_tps):
            return None

        event = TriggerEvent(
            frame_index=frame_index,
            timestamp=timestamp,
            t_deg=t_deg,
            v_tps=v_tps,
            formulation=self.trigger.formulation,
        )
        self.events.append(event)

        if self.stage2 is None:
            # No geometric check configured: the event stands on Stage 1
            # alone and says so. This is the §3.7 ablation "remove Stage 2".
            self._resolve(timestamp, event)
            return event

        self._pending = event
        self.stage2.start(timestamp)
        self.stage2.observe(p_offset)
        self.stage = Stage.CONFIRMING
        return event

    def _advance_confirming(self, timestamp: float, p_offset: float,
                            t_deg: float, i_still_s: float,
                            extension_ratio: float) -> None:
        assert self.stage2 is not None
        self.stage2.observe(p_offset)
        if not self.stage2.ready(timestamp):
            return
        verdict = self.stage2.verdict()
        if self._pending is not None:
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
        self.stage3.observe(timestamp, t_deg, i_still_s, extension_ratio)
        self.stage = Stage.OBSERVING

    def _advance_observing(self, timestamp: float, t_deg: float,
                           i_still_s: float, extension_ratio: float) -> None:
        assert self.stage3 is not None
        self.stage3.observe(timestamp, t_deg, i_still_s, extension_ratio)
        if not self.stage3.ready(timestamp):
            return
        verdict, severity = self.stage3.verdict()
        if self._pending is not None:
            self._pending.verdict = verdict
            self._pending.severity = severity
            self._pending.max_immobility_s = self.stage3.max_immobility_s
        self.stage3.reset()
        self._resolve(timestamp)

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
            self.stage3.reset()
        elif self.stage is Stage.CONFIRMING and self.stage2 is not None:
            self._pending.verdict = INCONCLUSIVE
            self._pending.p_outside_fraction = self.stage2.outside_fraction_seen
            self._pending.p_samples = self.stage2.samples
            self.stage2.reset()
        closed = self._pending
        self._resolve(timestamp)
        return closed

    def _resolve(self, timestamp: float, event: TriggerEvent | None = None) -> None:
        self.just_resolved = event if event is not None else self._pending
        self._pending = None
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
        self.trigger.reset()
        if self.stage2 is not None:
            self.stage2.reset()
        if self.stage3 is not None:
            self.stage3.reset()
        self._pending = None
        self.stage = Stage.MONITORING
        self._resolved_at = None
