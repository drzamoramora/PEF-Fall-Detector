"""Stage 3 of §3.5: immobility, recovery and severity (§3.3)."""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Iterable

from .events import (
    CONFIRMED_FALL,
    MILD,
    MODERATE,
    NOT_DOWN,
    NULLIFIED,
    SEVERE,
    UNRESOLVED,
)


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

    **``threshold_h_ratio_recovery``: EXPERIMENTAL, not in §3.4/§3.5.**
    Quantity H (``quantities.body_height_ratio``) at or above this value is
    OR'd into the ``upright`` check alongside T, for the same reason it is
    OR'd into Stage 1's trigger: a subject standing back up while pitched
    toward or away from the camera can leave T's trunk vector too
    foreshortened to read a clean "upright" angle, which — unfixed — reads as
    a recovery never happening and a subject who in fact stood up gets
    dispatched as a confirmed, unrecovered fall. H does not share that
    failure mode (see the ``quantities`` package docstring). ``<= 0.0``
    disables it, same convention as ``Stage1Trigger.threshold_h_ratio``.

    **``threshold_h_ratio_lying``: EXPERIMENTAL, closes a gap this project's
    own review found.** The immobility branch of :meth:`observe` used to
    confirm ``SEVERE`` from stillness ALONE — ``_max_still >= threshold_w_s``
    — without checking how "down" the subject actually was. A subject bent
    over but not upright (kneeling, propped, or — the case that motivated
    this — crouched still for a few seconds tying a shoe) could reach that
    same immobility threshold and be tagged ``SEVERE`` (remained down)
    instead of ``MODERATE`` (ambiguous posture), because nothing re-checked
    the trunk angle at the moment of confirmation. This threshold makes that
    branch use the same "near horizontal" test :meth:`finalise` already uses
    for its top band — ``t_deg >= lying_t_deg``, OR (when this is set) H
    collapsed — instead of treating "immobile" and "lying" as the same fact.

    This narrows WHICH immobile posture gets called ``SEVERE`` (a labelling
    correctness fix); it does not by itself stop a crouch-to-tie-a-shoe from
    reaching this evaluator in the first place, still less prove intent —
    see the caution on ``Stage1Trigger.threshold_h_erect``. ``<= 0.0``
    disables the H half of the OR; the T half (``lying_t_deg``) always
    applies, since it costs nothing new and was already the paper's own
    boundary for "down".
    """

    def __init__(self, window_s: float = 30.0, threshold_w_s: float = 5.0,
                 upright_t_deg: float = 30.0, recovery_hold_s: float = 1.0,
                 standing_extension: float = 1.1,
                 use_leg_extension: bool = True,
                 defer_to_end: bool = False,
                 final_window_s: float = 1.0,
                 lying_t_deg: float = 60.0,
                 threshold_h_ratio_recovery: float = 0.0,
                 threshold_h_ratio_lying: float = 0.0,
                 max_extension: float = 3.0,
                 persistent_still_s: float = 0.5,
                 persistent_fraction: float = 0.0,
                 getup_rise_torsos: float = 0.0,
                 getup_window_s: float = 2.0,
                 a_band_floor: float = 0.0,
                 a_band_standing: float = 0.82,
                 a_reach_floor: float = 0.0,
                 a_hip_raised: float = 0.22,
                 a_reach_before_s: float = 1.0,
                 a_reach_after_s: float = 4.0) -> None:
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
        self.threshold_h_ratio_recovery = float(threshold_h_ratio_recovery)
        self.threshold_h_ratio_lying = float(threshold_h_ratio_lying)
        # Persistent immobility (§3.5), read by finalise(); see there.
        self.persistent_still_s = float(persistent_still_s)
        self.persistent_fraction = float(persistent_fraction)
        # Phase 6a: a get-up in progress when observation ends (§3.5's "torso
        # and hip landmark recovery"): T says down but the hip keeps rising
        # -> moderate, not severe. 0 switches it off. Measurements:
        # notas/CODIGO-HISTORIA.md.
        self.getup_rise_torsos = float(getup_rise_torsos)
        self.getup_window_s = float(getup_window_s)
        # Phase 8 (Quantity A, EXPERIMENTAL), labelling mode only (finalise):
        #   8.2 (a_band_floor > 0): with A calibrated, the final posture comes
        #       from A: <= a_band_floor on the floor, >= a_band_standing
        #       standing, in between sitting or kneeling.
        #   8.3 (a_reach_floor > 0): if the head never went below a_reach_floor
        #       from a_reach_before_s before to a_reach_after_s after the
        #       trigger, and the hip ended above a_hip_raised, there was no
        #       fall. Abstains if A was not calibrated at the trigger.
        # 0 in a_band_floor / a_reach_floor switches each rule off.
        if a_band_floor > 0.0 and a_band_standing <= a_band_floor:
            raise ValueError("a_band_standing must be above a_band_floor")
        if a_reach_before_s < 0.0 or a_reach_after_s <= 0.0:
            raise ValueError("a_reach window must be positive")
        self.a_band_floor = float(a_band_floor)
        self.a_band_standing = float(a_band_standing)
        self.a_reach_floor = float(a_reach_floor)
        self.a_hip_raised = float(a_hip_raised)
        self.a_reach_before_s = float(a_reach_before_s)
        self.a_reach_after_s = float(a_reach_after_s)
        self._a_trigger_t: float | None = None
        self._a_ready = False
        self._a_min = float("nan")
        self._a_tail: deque[tuple[float, float, float]] = deque()
        #: (timestamp, hip_y_px, torso_px) of observed frames, last
        #: getup_window_s seconds only. Image y grows downward.
        self._hip: deque[tuple[float, float, float]] = deque()
        self._n_observed = 0
        self._n_still = 0
        self._reason = ""
        self._started_at: float | None = None
        self._upright_since: float | None = None
        self._legs_extended = False
        self._max_still = 0.0
        self._resolution: tuple[str, str] | None = None
        #: Trailing samples of (timestamp, t_deg, extension_ratio, h_ratio),
        #: kept only for the final read. The LAST frame alone is a coin flip
        #: on noisy landmarks; a short window at the end is the posture the
        #: subject actually finished in.
        self._tail: deque[tuple[float, float, float, float]] = deque()

    def start(self, timestamp: float) -> None:
        self._started_at = float(timestamp)
        self._upright_since = None
        self._legs_extended = False
        self._max_still = 0.0
        self._resolution = None
        self._tail = deque()
        self._hip = deque()
        self._n_observed = 0
        self._n_still = 0
        self._reason = ""
        self._clear_a()

    def _clear_a(self) -> None:
        self._a_trigger_t = None
        self._a_ready = False
        self._a_min = float("nan")
        self._a_tail = deque()

    @property
    def a_armed(self) -> bool:
        """Whether this episode already knows its trigger time (phase 8)."""
        return self._a_trigger_t is not None

    @property
    def a_min_window(self) -> float:
        """Lowest A seen in [trigger - before, trigger + after]; NaN if none."""
        return self._a_min

    def arm_a(self, trigger_t: float, calibrated: bool,
              history: Iterable[tuple[float, float, float]] = ()) -> None:
        """Phase 8: tie this episode to its trigger, seeding the earlier samples.

        ``calibrated``: whether A was calibrated on the trigger frame. If not,
        8.3 abstains for the whole episode — a calibration that arrives later
        comes from the subject already back on their feet (A16-S3 on the Mac
        calibrates 2.6 s after its fall), and would read "never went down".
        ``history``: (timestamp, a_head, a_hip) samples from before Stage 3
        started; the ones inside the window are kept.
        """
        if self._started_at is None:
            return
        self._a_trigger_t = float(trigger_t)
        self._a_ready = bool(calibrated)
        # _feed_a applies the window; older samples only pass through the tail,
        # which trims them to final_window_s.
        for ts, a_head, a_hip in history:
            self._feed_a(float(ts), float(a_head), float(a_hip))

    def observe_a(self, timestamp: float, a_head: float, a_hip: float) -> None:
        """Feed Quantity A and the hip ratio of one observed frame (phase 8)."""
        if self._started_at is None or self._resolution is not None or not self.a_armed:
            return
        self._feed_a(float(timestamp), float(a_head), float(a_hip))

    def _feed_a(self, ts: float, a_head: float, a_hip: float) -> None:
        t0 = self._a_trigger_t
        if (not math.isnan(a_head) and t0 is not None
                and t0 - self.a_reach_before_s <= ts <= t0 + self.a_reach_after_s):
            if math.isnan(self._a_min) or a_head < self._a_min:
                self._a_min = a_head
        self._a_tail.append((ts, a_head, a_hip))
        cutoff = ts - self.final_window_s
        while len(self._a_tail) > 1 and self._a_tail[0][0] < cutoff:
            self._a_tail.popleft()

    def _a_end(self) -> tuple[float, float]:
        """Median A and hip ratio over the final window; NaN when unmeasured."""
        heads = sorted(a for _t, a, _h in self._a_tail if not math.isnan(a))
        hips = sorted(h for _t, _a, h in self._a_tail if not math.isnan(h))
        return (heads[len(heads) // 2] if heads else float("nan"),
                hips[len(hips) // 2] if hips else float("nan"))

    def observe_hip(self, timestamp: float, hip_y_px: float, torso_px: float) -> None:
        """Feed the mid-hip height of one observed frame (phase 6a)."""
        if self._started_at is None or self._resolution is not None:
            return
        if math.isnan(hip_y_px) or math.isnan(torso_px) or torso_px <= 0.0:
            return
        self._hip.append((float(timestamp), float(hip_y_px), float(torso_px)))
        cutoff = float(timestamp) - self.getup_window_s
        while self._hip and self._hip[0][0] < cutoff:
            self._hip.popleft()

    def getup_rise(self) -> float:
        """How far the hip rose over the last window, in torso lengths.

        Median of the first half-second against the median of the last one,
        so a single jittery frame cannot fake a rise. NaN when the window is
        not covered: fewer than 5 samples at either end, or under 1.5 s
        spanned.
        """
        if len(self._hip) < 10:
            return float("nan")
        t0, t1 = self._hip[0][0], self._hip[-1][0]
        if t1 - t0 < 0.75 * self.getup_window_s:
            return float("nan")
        first = [y for t, y, _ in self._hip if t <= t0 + 0.5]
        last = [y for t, y, _ in self._hip if t >= t1 - 0.5]
        if len(first) < 5 or len(last) < 5:
            return float("nan")
        torso = statistics.median(tr for _t, _y, tr in self._hip)
        return (statistics.median(first) - statistics.median(last)) / torso

    def _is_lying(self, t_deg: float, h_ratio: float) -> bool:
        """Near-horizontal by T; by H (EXPERIMENTAL) only when T is unmeasured.

        The same "down" boundary :meth:`finalise` already uses, shared here
        so the live and labelling paths agree on what it means.

        **T decides whenever it was measured; H is a fallback, not a vote.**
        Until 23/09 the two were OR'd, so a low H overruled a valid T. That
        manufactured ``severe`` for subjects sitting on the floor: measured
        on A12-S1 (truth PartiallyRecovered), T final 35.0 deg — the three
        bands' "sat up / knelt" — but H final 0.344 <= 0.35, because sitting
        on the floor legitimately leaves little vertical spread. No H spike
        was involved (minimum shoulder width 13.9 px), so a plausibility
        guard on H could not have caught it. Same on A12-S4 (T 42.5, H 0.213)
        and on B05-S1..S3, NoFall clips whose trunk never passed 49 deg.

        The cost, stated so it is measured rather than forgotten: H was also
        meant to rescue the camera-axis fall (A14), where T reads a VALID
        but falsely low angle. Deferring to a valid T gives that up. Which
        of the two the dataset needs more is an empirical question; see
        notas/HALLAZGO-B05.md.
        """
        if not math.isnan(t_deg):
            return t_deg >= self.lying_t_deg
        return (self.threshold_h_ratio_lying > 0.0 and not math.isnan(h_ratio)
                and h_ratio <= self.threshold_h_ratio_lying)

    def _lying_text(self, t_deg: float, h_ratio: float) -> str:
        """The same test as :meth:`_is_lying`, in words, for the reason."""
        if not math.isnan(t_deg):
            rel = ">=" if t_deg >= self.lying_t_deg else "<"
            return f"T {t_deg:.0f} deg {rel} {self.lying_t_deg:.0f}"
        if math.isnan(h_ratio) or self.threshold_h_ratio_lying <= 0.0:
            return "T not measured"
        rel = "<=" if h_ratio <= self.threshold_h_ratio_lying else ">"
        return f"T not measured, H {h_ratio:.2f} {rel} {self.threshold_h_ratio_lying:.2f}"

    def observe(self, timestamp: float, t_deg: float, i_still_s: float,
                extension_ratio: float, h_ratio: float = float("nan")) -> None:
        """Feed one post-trigger frame. Resolves as soon as it can."""
        if self._started_at is None or self._resolution is not None:
            return

        if not math.isnan(i_still_s):
            self._max_still = max(self._max_still, i_still_s)
            self._n_observed += 1
            self._n_still += int(i_still_s >= self.persistent_still_s)

        # Keep a trailing window for finalise(); cheap, and only this.
        self._tail.append((float(timestamp), t_deg, extension_ratio, h_ratio))
        cutoff = float(timestamp) - self.final_window_s
        while len(self._tail) > 1 and self._tail[0][0] < cutoff:
            self._tail.popleft()

        upright = not math.isnan(t_deg) and t_deg < self.upright_t_deg
        if not upright and self.threshold_h_ratio_recovery > 0.0 and not math.isnan(h_ratio):
            # EXPERIMENTAL fallback (see threshold_h_ratio_recovery above):
            # T missed the recovery, but H shows the body back near its
            # standing height regardless of trunk-vector direction.
            upright = h_ratio >= self.threshold_h_ratio_recovery
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
            held = f"T < {self.upright_t_deg:.0f} deg held {self.recovery_hold_s:.1f} s"
            if not self.use_leg_extension:
                self._resolution = (NULLIFIED, MODERATE)
                self._reason = f"upright: {held}; legs not evaluated"
            elif self._legs_extended:
                self._resolution = (NULLIFIED, MILD)
                self._reason = f"got up: {held}, legs extended"
            else:
                # Upright but not standing: sitting up, kneeling, propped
                # against furniture. §3.3's "partially recovered".
                self._resolution = (CONFIRMED_FALL, MODERATE)
                self._reason = f"upright without extended legs: {held}"
            return

        # Still down, and immobile for the confirmation threshold of §3.4.
        # SEVERE only if the posture backs up "down" (near-horizontal T, or
        # H when T cannot tell) — immobile-but-not-lying (kneeling, bent
        # over) is §3.3's "partially recovered" instead, per _is_lying.
        if self._max_still >= self.threshold_w_s:
            lying = self._is_lying(t_deg, h_ratio)
            self._resolution = (CONFIRMED_FALL, SEVERE if lying else MODERATE)
            self._reason = (f"still {self._max_still:.1f} s >= W {self.threshold_w_s:.1f} s; "
                            f"{'lying' if lying else 'not lying'} "
                            f"({self._lying_text(t_deg, h_ratio)})")

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
        a_end, hip_end = self._a_end()
        # Phase 8.3, "did it reach the floor?": a fall is a loss of height
        # against gravity. Head never low and hip ended raised -> sat on a
        # chair or crouched. The hip condition protects a collapse against a
        # wall. Abstains without A calibrated at the trigger. Measurements:
        # notas/CODIGO-HISTORIA.md.
        if (self.a_reach_floor > 0.0 and self._a_ready and not math.isnan(self._a_min)
                and self._a_min > self.a_reach_floor
                and not math.isnan(hip_end) and hip_end > self.a_hip_raised):
            self._resolution = (NOT_DOWN, "")
            self._reason = (f"minimum A {self._a_min:.2f} > {self.a_reach_floor:.2f} and "
                            f"final hip {hip_end:.2f} > {self.a_hip_raised:.2f}: "
                            f"never reached the floor")
            return
        angles = [t for _ts, t, _e, _h in self._tail if not math.isnan(t)]
        if not angles:
            self._reason = "no skeleton measured in the last second"
            return                              # stays UNRESOLVED
        # The median, not the mean: one frame of a collapsed pose estimate —
        # measured at 2 px of trunk on clip A14 — drags a mean across the
        # upright threshold, and that single frame would decide the label.
        angles.sort()
        final_t = angles[len(angles) // 2]
        # Same rule as observe()'s immobility branch (_is_lying): T decides
        # when measured. Here final_t always is — an empty tail returned
        # above — so H cannot overrule it; final_h is passed for the shared
        # signature and for the day this path handles an unmeasured T.
        h_values = sorted(h for _ts, _t, _e, h in self._tail if not math.isnan(h))
        final_h = h_values[len(h_values) // 2] if h_values else float("nan")
        fin = f"T final {final_t:.0f} deg"
        if self._is_lying(final_t, final_h) and self.getup_rise_torsos > 0.0:
            rise = self.getup_rise()
            if not math.isnan(rise) and rise >= self.getup_rise_torsos:
                self._resolution = (CONFIRMED_FALL, MODERATE)
                self._reason = (f"{fin} >= {self.lying_t_deg:.0f}, but the hip rose "
                                f"{rise:.2f} torsos in {self.getup_window_s:.0f} s "
                                f"(>= {self.getup_rise_torsos:.2f}): getting up")
                return
        # Phase 8.2: with A calibrated, the final posture comes from A, not
        # from the 2D angle, which cannot tell a fall away from the camera
        # from standing. The hip rise (6a, above) and persistence (phase 3)
        # still apply. Measurements: notas/CODIGO-HISTORIA.md.
        if self.a_band_floor > 0.0 and not math.isnan(a_end):
            fin_a = f"A final {a_end:.2f} ({fin})"
            if a_end <= self.a_band_floor:
                self._resolution = (CONFIRMED_FALL, SEVERE)
                self._reason = f"{fin_a} <= {self.a_band_floor:.2f}: ended on the floor"
            elif a_end >= self.a_band_standing:
                self._resolution = (NULLIFIED, MILD)
                self._reason = f"{fin_a} >= {self.a_band_standing:.2f}: standing, got up"
            else:
                sev = self._moderate_or_persistent()
                self._resolution = (CONFIRMED_FALL, sev)
                self._reason = (f"{fin_a} between {self.a_band_floor:.2f} and "
                                f"{self.a_band_standing:.2f}: sitting or kneeling"
                                f"{self._persistence_text(sev)}")
            return
        if self._is_lying(final_t, final_h):
            self._resolution = (CONFIRMED_FALL, SEVERE)
            self._reason = f"{fin} >= {self.lying_t_deg:.0f}: ended on the floor"
            return
        if final_t >= self.upright_t_deg:
            # Trunk up but not vertical: sat up, knelt. §3.3's "partially
            # recovered" — unless the stillness says otherwise, see below.
            sev = self._moderate_or_persistent()
            self._resolution = (CONFIRMED_FALL, sev)
            self._reason = (f"{fin} between {self.upright_t_deg:.0f} and {self.lying_t_deg:.0f}: "
                            f"sitting or kneeling{self._persistence_text(sev)}")
            return
        if not self.use_leg_extension:
            self._resolution = (NULLIFIED, MODERATE)
            self._reason = f"{fin} < {self.upright_t_deg:.0f}: upright; legs not evaluated"
            return
        # The upper bound is a plausibility guard, not a taste: a hip-to-ankle
        # span of more than a few torso lengths is the ratio dividing by a
        # foreshortened torso, not a leg. Values up to 183 torso lengths were
        # recorded on this dataset. An impossible number must not be evidence
        # of anything, least of all of a recovery.
        extended = any(not math.isnan(e)
                       and self.standing_extension <= e <= self.max_extension
                       for _ts, _t, e, _h in self._tail)
        if extended:
            self._resolution = (NULLIFIED, MILD)
            self._reason = (f"{fin} < {self.upright_t_deg:.0f} with extended legs: "
                            f"got up")
            return
        sev = self._moderate_or_persistent()
        self._resolution = (CONFIRMED_FALL, sev)
        self._reason = (f"{fin} < {self.upright_t_deg:.0f} without verifiable legs"
                        f"{self._persistence_text(sev)}")

    def _persistence_text(self, severity: str) -> str:
        """The §3.5 persistence clause, in words, when it is switched on."""
        if self.persistent_fraction <= 0.0 or math.isnan(self.still_fraction):
            return ""
        pct, cut = 100.0 * self.still_fraction, 100.0 * self.persistent_fraction
        if severity == SEVERE:
            return f"; still {pct:.0f}% >= {cut:.0f}%: the immobility persisted"
        return f"; still {pct:.0f}% < {cut:.0f}%"

    @property
    def still_fraction(self) -> float:
        """Share of Stage-3 frames with I >= ``persistent_still_s``; NaN if none."""
        if self._n_observed == 0:
            return float("nan")
        return self._n_still / self._n_observed

    def _moderate_or_persistent(self) -> str:
        """The not-lying, not-standing ending: moderate, or severe if nobody moved.

        §3.5: *"if the immobility persists, the alert is dispatched with the
        appropriate severity tag"*. The three bands read only the final
        POSTURE, and a posture cannot tell someone who sat up on the floor
        (§3.3 lists "recovering by sitting on the floor") from someone who
        was left propped against a wall — which §3.3 lists as an ATTEMPT:
        *"attempting to use a wall for support, recovering by kneeling"*;
        *"support-furniture attempt, optional successful get-up"*. Both end
        seated; what separates them is whether the subject kept moving.

        Measured on PEF-FallDB (share of Stage-3 frames with I >= 0.5 s):
        A17-S2 63.4 % and A17-S4 57.8 % — reclined against a wall, truth
        NotRecovered, trunk final 2.7 and 25.3 deg (a back against a wall
        keeps the trunk VERTICAL, so T alone reads them as upright). Every
        other fall ending below 60 deg: 0-35.3 %. Only two positive clips,
        and the 0.5 cut was chosen looking at all 128, not at the
        calibration split — state both when reporting it.

        ``persistent_fraction <= 0`` disables it and restores plain moderate.
        Only the labelling path uses it; the live path already has I >= W.
        """
        f = self.still_fraction
        if (self.persistent_fraction > 0.0 and not math.isnan(f)
                and f >= self.persistent_fraction):
            return SEVERE
        return MODERATE

    def last_seen_down(self) -> bool:
        """Whether the last observed stretch shows the subject on the floor.

        Median T over the trailing window of frames actually OBSERVED — an
        undetected frame never reaches :meth:`observe`, so during a loss of
        detection this is what was seen just before it.
        """
        angles = sorted(t for _ts, t, _e, _h in self._tail if not math.isnan(t))
        return bool(angles) and angles[len(angles) // 2] >= self.lying_t_deg

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

    @property
    def reason(self) -> str:
        """Why :meth:`verdict` says what it says, in one line."""
        if self._reason:
            return self._reason
        if self._resolution is None:
            return (f"ventana de {self.window_s:.0f} s agotada: "
                    f"ni quietud >= W ni recuperacion sostenida")
        return ""

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
        self._hip = deque()
        self._clear_a()
        self._upright_since = None
        self._legs_extended = False
        self._max_still = 0.0
        self._resolution = None
        self._n_observed = 0
        self._n_still = 0
        self._reason = ""
