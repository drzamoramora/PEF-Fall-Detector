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
import statistics
from collections import deque
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


class DescentTracker:
    """Qué tan profundo y qué tan sostenido fue el descenso (§3.4).

    POR QUÉ EXISTE. El §3.5 le da a la Etapa 2 el trabajo de separar una caída
    de *"a controlled sit-down in which COM stays within the support polygon"*,
    y le da P como única herramienta. Medido sobre PEF-FallDB, esa premisa es
    falsa: en los ADL que producen falso positivo, la Etapa 2 midió P sobre 11
    cuadros y encontró el centro de masa **fuera** del polígono de apoyo en el
    100 % de ellos. No es un fallo de medición — es que sentarse en un sofá
    saca el COM de sobre los pies, porque el peso pasa al mueble. La geometría
    dice "caída" con razón, y P sola no puede resolverlo.

    QUÉ SÍ SEPARA, y el §3.4 ya lo dice: *"a sit-down event produces a
    monotonic negative V over a longer window, and a fall produces a
    large-magnitude negative V within a short window."* La distinción es de
    MAGNITUD. Medido sobre los 40 clips de calibración que dispararon:

        caídas          V_min mediana -3.00, tiempo bajo -1 tps  0.77 s
        falsos positivos V_min  -1.16 y -1.39, tiempo           0.17 y 0.20 s

    Se exigen las DOS condiciones para rechazar — poco profundo Y breve — no
    una. Un descenso puede ser corto y aun así violento (una caída de síncope),
    y puede ser lento y aun así ser una caída. Sólo cuando falla en las dos a la
    vez se trata de un descenso controlado.

    LO QUE NO RESUELVE. Un clip de calibración (B05-S1) llega a V_min = -3.13
    con el COM fuera del apoyo: por profundidad, duración y geometría es
    indistinguible de una caída. Probablemente alguien dejándose caer en un
    sofá. Ninguna de las cuatro cantidades del §3.4 lo separa, y este criterio
    lo deja pasar a propósito en lugar de apretarse hasta perder caídas reales.
    """

    def __init__(self, floor_tps: float = -1.0, window_s: float = 2.0) -> None:
        if floor_tps >= 0.0:
            raise ValueError("floor_tps debe ser negativo (V negativa = hacia abajo)")
        if window_s <= 0.0:
            raise ValueError("window_s debe ser > 0")
        self.floor_tps = float(floor_tps)
        self.window_s = float(window_s)
        self._samples: list[tuple[float, float]] = []

    def update(self, timestamp: float, v_tps: float) -> None:
        if math.isnan(v_tps):
            return
        self._samples.append((float(timestamp), float(v_tps)))
        cutoff = float(timestamp) - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.pop(0)

    @property
    def deepest(self) -> float:
        """V más negativa vista en la ventana. 0.0 si no hay muestras."""
        return min((v for _ts, v in self._samples), default=0.0)

    @property
    def sustained_s(self) -> float:
        """Racha más larga con V por debajo del piso, en segundos."""
        best = 0.0
        run: float | None = None
        for ts, v in self._samples:
            if v < self.floor_tps:
                if run is None:
                    run = ts
                best = max(best, ts - run)
            else:
                run = None
        return best

    def is_controlled(self, min_depth_tps: float, min_sustained_s: float) -> bool:
        """Poco profundo Y breve: un descenso controlado, no una caída."""
        return (self.deepest > min_depth_tps
                and self.sustained_s < min_sustained_s)

    def reset(self) -> None:
        self._samples.clear()


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
    failure mode (see ``quantities.py``'s module docstring). ``<= 0.0``
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
                 getup_window_s: float = 2.0) -> None:
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
        # Fase 6a: levantarse en curso al terminar la observacion. El §3.5
        # baja la severidad ante "torso and hip landmark recovery to
        # upright"; hasta aqui solo se leia el torso (T). Si al final T dice
        # "en el suelo" pero la CADERA viene subiendo de forma sostenida, el
        # sujeto se esta levantando: moderate, no severe. Medido en A06-S14
        # (sube 0.70 torsos en 2 s en la nube, 0.81 en el Mac) contra un
        # maximo de 0.21 / 0.13 en las caidas NotRecovered con T >= 60.
        # 0 lo apaga.
        self.getup_rise_torsos = float(getup_rise_torsos)
        self.getup_window_s = float(getup_window_s)
        #: (timestamp, hip_y_px, torso_px) of observed frames, last
        #: getup_window_s seconds only. Image y grows downward.
        self._hip: list[tuple[float, float, float]] = []
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
        self._tail: list[tuple[float, float, float, float]] = []

    def start(self, timestamp: float) -> None:
        self._started_at = float(timestamp)
        self._upright_since = None
        self._legs_extended = False
        self._max_still = 0.0
        self._resolution = None
        self._tail = []
        self._hip = []
        self._n_observed = 0
        self._n_still = 0
        self._reason = ""

    def observe_hip(self, timestamp: float, hip_y_px: float, torso_px: float) -> None:
        """Feed the mid-hip height of one observed frame (fase 6a)."""
        if self._started_at is None or self._resolution is not None:
            return
        if math.isnan(hip_y_px) or math.isnan(torso_px) or torso_px <= 0.0:
            return
        self._hip.append((float(timestamp), float(hip_y_px), float(torso_px)))
        cutoff = float(timestamp) - self.getup_window_s
        while self._hip and self._hip[0][0] < cutoff:
            self._hip.pop(0)

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
            return "T no medida"
        rel = "<=" if h_ratio <= self.threshold_h_ratio_lying else ">"
        return f"T no medida, H {h_ratio:.2f} {rel} {self.threshold_h_ratio_lying:.2f}"

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
            self._tail.pop(0)

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
            held = f"T < {self.upright_t_deg:.0f} deg sostenido {self.recovery_hold_s:.1f} s"
            if not self.use_leg_extension:
                self._resolution = (NULLIFIED, MODERATE)
                self._reason = f"erguido: {held}; piernas sin evaluar"
            elif self._legs_extended:
                self._resolution = (NULLIFIED, MILD)
                self._reason = f"se levanto: {held}, piernas extendidas"
            else:
                # Upright but not standing: sitting up, kneeling, propped
                # against furniture. §3.3's "partially recovered".
                self._resolution = (CONFIRMED_FALL, MODERATE)
                self._reason = f"erguido sin piernas extendidas: {held}"
            return

        # Still down, and immobile for the confirmation threshold of §3.4.
        # SEVERE only if the posture backs up "down" (near-horizontal T, or
        # H when T cannot tell) — immobile-but-not-lying (kneeling, bent
        # over) is §3.3's "partially recovered" instead, per _is_lying.
        if self._max_still >= self.threshold_w_s:
            lying = self._is_lying(t_deg, h_ratio)
            self._resolution = (CONFIRMED_FALL, SEVERE if lying else MODERATE)
            self._reason = (f"quieto {self._max_still:.1f} s >= W {self.threshold_w_s:.1f} s; "
                            f"{'tumbado' if lying else 'no tumbado'} "
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
        angles = [t for _ts, t, _e, _h in self._tail if not math.isnan(t)]
        if not angles:
            self._reason = "sin esqueleto medido en el ultimo segundo"
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
                self._reason = (f"{fin} >= {self.lying_t_deg:.0f}, pero la cadera subio "
                                f"{rise:.2f} torsos en {self.getup_window_s:.0f} s "
                                f"(>= {self.getup_rise_torsos:.2f}): se esta levantando")
                return
        if self._is_lying(final_t, final_h):
            self._resolution = (CONFIRMED_FALL, SEVERE)
            self._reason = f"{fin} >= {self.lying_t_deg:.0f}: termino en el suelo"
            return
        if final_t >= self.upright_t_deg:
            # Trunk up but not vertical: sat up, knelt. §3.3's "partially
            # recovered" — unless the stillness says otherwise, see below.
            sev = self._moderate_or_persistent()
            self._resolution = (CONFIRMED_FALL, sev)
            self._reason = (f"{fin} entre {self.upright_t_deg:.0f} y {self.lying_t_deg:.0f}: "
                            f"sentado o arrodillado{self._persistence_text(sev)}")
            return
        if not self.use_leg_extension:
            self._resolution = (NULLIFIED, MODERATE)
            self._reason = f"{fin} < {self.upright_t_deg:.0f}: erguido; piernas sin evaluar"
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
            self._reason = (f"{fin} < {self.upright_t_deg:.0f} con piernas extendidas: "
                            f"se levanto")
            return
        sev = self._moderate_or_persistent()
        self._resolution = (CONFIRMED_FALL, sev)
        self._reason = (f"{fin} < {self.upright_t_deg:.0f} sin piernas verificables"
                        f"{self._persistence_text(sev)}")

    def _persistence_text(self, severity: str) -> str:
        """The §3.5 persistence clause, in words, when it is switched on."""
        if self.persistent_fraction <= 0.0 or math.isnan(self.still_fraction):
            return ""
        pct, cut = 100.0 * self.still_fraction, 100.0 * self.persistent_fraction
        if severity == SEVERE:
            return f"; quieto {pct:.0f}% >= {cut:.0f}%: la inmovilidad persistio"
        return f"; quieto {pct:.0f}% < {cut:.0f}%"

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
        self._hip = []
        self._upright_since = None
        self._legs_extended = False
        self._max_still = 0.0
        self._resolution = None
        self._n_observed = 0
        self._n_still = 0
        self._reason = ""


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
    #: Disparo de la banda inferior: la evidencia de la Etapa 1 alcanzo para
    #: sospechar pero no para el umbral pleno. Un evento provisional solo
    #: puede terminar en alarma si la inmovilidad lo confirma; ver
    #: :class:`Stage1Trigger` y :meth:`FallStateMachine._resolve`.
    provisional: bool = False
    #: Quantity H (EXPERIMENTAL, not in §3.4) at the firing frame, kept for
    #: the same auditability reason as t_deg/v_tps — including when it was
    #: NOT what fired the event, so a reviewer can see whether the H+V
    #: fallback (Stage1Trigger.threshold_h_ratio) or T/V's own formulation
    #: was responsible.
    h_ratio: float = float("nan")
    #: Why the verdict is what it is: the rule that decided, with the numbers
    #: it read. §3.5 promises that "a downstream reviewer can reconstruct the
    #: decision"; without this the reviewer has to re-derive which branch of
    #: the funnel closed the event. Empty until something resolves it.
    reason: str = ""


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
        threshold_score_provisional: borde inferior de la banda, para
            ``score``. Entre este valor y ``threshold_score`` el disparador
            sigue disparando, pero marca el evento como **provisional**.
            0.0 desactiva la banda y el disparador vuelve a ser un corte
            unico, que es la ablacion del §3.7.
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
            "off by an impossible threshold" convention as
            ``stage2.descent_min_depth_tps``.
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
    """

    def __init__(
        self,
        formulation: str = SCORE,
        threshold_t_deg: float = 45.0,
        threshold_v_tps: float = -1.5,
        hold_s: float = 0.1,
        confirm_window_s: float = 1.0,
        threshold_score: float = 2.2,
        threshold_score_provisional: float = 0.0,
        peak_window_s: float = 0.0,
        threshold_h_ratio: float = 0.0,
        threshold_h_erect: float = 0.0,
        h_sequence_window_s: float = 2.0,
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
        if threshold_score_provisional < 0.0:
            raise ValueError("threshold_score_provisional must be >= 0")
        if threshold_score_provisional > threshold_score:
            # Al reves la banda no existe: todo disparo sera provisional y el
            # umbral pleno no se alcanzaria nunca. Falla ruidosamente en vez
            # de producir un sistema que nunca confirma nada.
            raise ValueError(
                "threshold_score_provisional must not exceed threshold_score")
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
        self.threshold_score_provisional = float(threshold_score_provisional)
        #: Si el ULTIMO disparo vino de la banda inferior. Lo lee la maquina
        #: de estados en el mismo cuadro; no es historia, es el tramo de la
        #: senal que acaba de cruzar.
        self.last_firing_provisional = False
        #: Puntaje de ESTE cuadro, o NaN si la formulacion no calcula uno o
        #: el cuadro no era evaluable. Lo lee la maquina de estados para
        #: promover un evento que nacio en la banda y despues se fortalecio.
        self.last_score = float("nan")
        self.peak_window_s = float(peak_window_s)
        self._peaks = PeakWindow(peak_window_s) if peak_window_s > 0.0 else None
        self._condition_since: float | None = None
        self._armed_at: float | None = None
        self._fired = False
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

    def update(self, timestamp: float, t_deg: float, v_tps: float,
               h_ratio: float = float("nan")) -> bool:
        """Feed one frame's T, V and (optionally) H; True on the frame that fires.

        ``h_ratio`` only feeds the OR'd fallback described on
        ``threshold_h_ratio`` above; every formulation's own T/V arithmetic
        is untouched; see :meth:`_evaluate_formulation`.
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
        return fired or fallback_fired

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
            # La banda, cuando esta configurada. El enganche y el pestillo
            # trabajan sobre el borde INFERIOR: la condicion de sospecha es
            # una sola y dura lo que dura, y el tramo solo decide de que
            # clase es el evento que sale. Usar dos enganches separados haria
            # que una senal cruzando de gris a plena reiniciara su propio
            # tiempo de sostenimiento y perdiera el evento.
            piso = self.threshold_score_provisional or self.threshold_score
            disparo = self._held(timestamp, score >= piso)
            if disparo:
                # El tramo se lee en el cuadro del disparo, que con la ventana
                # de pico activa ya es un maximo sobre esa ventana y no un
                # instante suelto.
                self.last_firing_provisional = score < self.threshold_score
            return disparo, v_hit

        if self.formulation == SIMULTANEOUS:
            return self._held(timestamp, t_hit and v_hit), v_hit
        if self.formulation == V_ONLY:
            return self._held(timestamp, v_hit), v_hit

        # SEQUENTIAL: V sustained arms the event, then T confirms it inside
        # the window. Measured on real falls, the peaks of T and V are offset
        # by 0.03 s to 1.8 s, which is why requiring them in the same frame
        # loses events that both quantities clearly witnessed.
        if self._held(timestamp, v_hit):
            self._armed_at = timestamp
        if self._armed_at is None:
            return False, v_hit
        if timestamp - self._armed_at > self.confirm_window_s:
            self._armed_at = None
            return False, v_hit
        if t_hit:
            self._armed_at = None
            return True, v_hit
        return False, v_hit

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

        Matters most for ``sequential``: an armed V event must not survive a
        break in observation and be confirmed by a T measured on what may be
        a different body, or a different point in the timeline.
        """
        self._condition_since = None
        self._armed_at = None
        self._fired = False
        self._h_condition_since = None
        self._h_fired = False
        self._h_history.clear()
        self.last_firing_provisional = False
        self.last_score = float("nan")
        if self._peaks is not None:
            self._peaks.reset()


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
                 cooldown_after_rejection: bool = True,
                 descent: "DescentTracker | None" = None,
                 min_depth_tps: float = -1.3,
                 min_sustained_s: float = 0.40) -> None:
        self.trigger = trigger
        #: Alimentado en CADA cuadro, tambien antes del disparo: el descenso
        #: que hay que juzgar empieza antes de que la Etapa 1 se entere.
        self.descent = descent
        self.min_depth_tps = float(min_depth_tps)
        self.min_sustained_s = float(min_sustained_s)
        self.stage2 = stage2
        self.stage3 = stage3
        self.cooldown_s = float(cooldown_s)
        #: Fase 4. El enfriamiento existe para que UNA caida no llegue como
        #: rafaga de eventos. Un candidato que la Etapa 2 rechazo no es una
        #: caida: el §3.5 lo descarta ahi. Dejar al sistema ciego 3 s despues
        #: de descartarlo no protege de ninguna rafaga (el pestillo de
        #: Stage1Trigger ya impide re-disparar mientras la condicion siga
        #: cierta) y si puede tapar la caida real que viene detras (A17-S3:
        #: rechazo a los 2.5 s, caida real a los 4.0-5.0 s, enfriamiento
        #: hasta 5.5 s). False: tras un rechazo se vuelve a MONITORING.
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
               h_ratio: float = float("nan")) -> TriggerEvent | None:
        """Advance the machine one frame; returns a NEW event, or None.

        Only the frame on which Stage 1 fires returns an event. Later frames
        return None even while that event's verdict is still being decided —
        the object already handed out is updated in place.

        ``h_ratio``: Quantity H (EXPERIMENTAL, not in §3.4/§3.5). Threaded
        through to :class:`Stage1Trigger`'s OR'd fallback and to
        :class:`Stage3Evaluator`'s recovery check; NaN (the default) reduces
        both to their pre-H behaviour exactly. See ``quantities.py``'s module
        docstring for what it is proposed to cover.
        """
        self.just_resolved = None
        if self.descent is not None:
            self.descent.update(timestamp, v_tps)
        if self.stage is Stage.OBSERVING:
            self._advance_observing(timestamp, t_deg, i_still_s, extension_ratio, h_ratio)
            self.trigger.update(timestamp, t_deg, v_tps, h_ratio)
            # Sin promocion aqui, a proposito: en OBSERVING el sujeto ya esta
            # en el suelo y T ronda 180, que por si solo da 4.0 de puntaje.
            # Promover ahi ascenderia a pleno a CUALQUIER evento provisional
            # por el mero hecho de estar acostado, que es el mismo agujero
            # que la regla evita al no mirar "stage3_confirmed". La
            # promocion pertenece a la ventana en que la evidencia del
            # DISPARO se esta formando, no a la que observa el desenlace.
            return None

        if self.stage is Stage.CONFIRMING:
            self._advance_confirming(timestamp, p_offset, t_deg,
                                     i_still_s, extension_ratio, h_ratio)
            # The trigger keeps seeing frames so its own hold state stays
            # coherent, but a second firing cannot start while one event is
            # still being judged.
            self.trigger.update(timestamp, t_deg, v_tps, h_ratio)
            self._promote_if_strong()
            return None

        if self.stage is Stage.COOLDOWN:
            if self._resolved_at is not None and (
                timestamp - self._resolved_at >= self.cooldown_s
            ):
                self.stage = Stage.MONITORING
                self._resolved_at = None
            else:
                self.trigger.update(timestamp, t_deg, v_tps, h_ratio)
                return None

        if self.stage is not Stage.MONITORING:
            return None

        if not self.trigger.update(timestamp, t_deg, v_tps, h_ratio):
            return None

        event = TriggerEvent(
            frame_index=frame_index,
            timestamp=timestamp,
            t_deg=t_deg,
            v_tps=v_tps,
            formulation=self.trigger.formulation,
            provisional=self.trigger.last_firing_provisional,
            h_ratio=h_ratio,
        )
        self.events.append(event)

        if self.stage2 is None:
            # No geometric check configured: the event stands on Stage 1
            # alone and says so. This is the §3.7 ablation "remove Stage 2".
            event.reason = "solo Etapa 1 (sin Etapa 2 configurada)"
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
        # El §3.5 encarga a la Etapa 2 separar una caida de un descenso
        # controlado. P sola no puede: medido, los ADL que disparan ponen el
        # COM fuera del apoyo en el 100 % de los cuadros medibles, porque
        # sentarse traslada el peso al mueble. El descenso si separa, y el
        # §3.4 ya lo dice al describir V. Ver DescentTracker.
        # min_depth_tps >= 0 desactiva el rechazo: V negativa es hacia abajo,
        # asi que "mas superficial que 0" no lo cumple ningun descenso real.
        # Ver config.yaml y notas/TRASPASO-DRAFT.md para por que esta apagado.
        by_descent = (self.descent is not None and self.min_depth_tps < 0.0
                      and self.descent.is_controlled(self.min_depth_tps,
                                                     self.min_sustained_s))
        if by_descent:
            verdict = REJECTED
        if self._pending is not None:
            self._pending.reason = self._stage2_text(verdict, by_descent)
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

    def _stage2_text(self, verdict: str, by_descent: bool) -> str:
        """Stage 2's outcome in words. Stage 3 overwrites it if it runs."""
        n = self.stage2.samples
        if by_descent:
            return "Etapa 2: descenso controlado (no es una caida)"
        if verdict == INCONCLUSIVE:
            return (f"Etapa 2: pies no medibles ({n} muestras de P "
                    f"< {self.stage2.min_samples})")
        frac = 100.0 * self.stage2.outside_fraction_seen
        cut = 100.0 * self.stage2.outside_fraction
        rel = ">=" if verdict == CONFIRMED else "<"
        return f"Etapa 2: COM fuera del apoyo {frac:.0f}% {rel} {cut:.0f}% ({n} muestras)"

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
            closed.reason = f"deteccion perdida en el suelo; {closed.reason}"
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
            self._pending.reason = "el clip termino durante la Etapa 2"
            self._pending.verdict = INCONCLUSIVE
            self._pending.p_outside_fraction = self.stage2.outside_fraction_seen
            self._pending.p_samples = self.stage2.samples
            self.stage2.reset()
        closed = self._pending
        self._resolve(timestamp)
        return closed

    #: Veredicto de un evento provisional que no llego a confirmarse por
    #: inmovilidad. No es ``stage2_rejected`` —ninguna etapa dictamino que no
    #: hubo derribo— ni ``stage3_unresolved`` —el embudo si termino—: es que
    #: la evidencia de la Etapa 1 nunca alcanzo el umbral pleno y nada
    #: posterior la respaldo.
    PROVISIONAL_UNCONFIRMED = "provisional_unconfirmed"

    def _promote_if_strong(self) -> None:
        """Un evento que nacio debil y despues se fortalecio deja de ser debil.

        POR QUE HACE FALTA. El enganche corre sobre el borde INFERIOR de la
        banda, asi que el evento se levanta cuando el puntaje lleva
        ``hold_s`` por encima de 2.2 — que en una caida real ocurre mientras
        el puntaje todavia esta subiendo. Medido en A04-S1: cruza 2.2 en
        t=2.60, dispara en t=2.70, cruza 2.6 en t=2.73 y llega a 3.32. Sin
        esta promocion ese evento queda marcado provisional para siempre por
        trece centesimas, y la marca termina dependiendo de una carrera entre
        el reloj del enganche y la pendiente de la señal en vez de la fuerza
        de la evidencia.

        SOLO EN CONFIRMING. En OBSERVING el sujeto ya esta en el suelo y T
        ronda 180, que da 4.0 por si solo: promover ahi ascenderia cualquier
        evento provisional por el mero hecho de estar acostado. La ventana de
        promocion es aquella en que la evidencia del DISPARO todavia se
        forma.

        El puntaje se lee del disparador, que se sigue alimentando en cada
        cuadro mientras el evento se juzga — no se recalcula aqui, para que
        haya un solo sitio donde la formulacion viva.
        """
        if self._pending is None or not self._pending.provisional:
            return
        score = getattr(self.trigger, "last_score", float("nan"))
        if not math.isnan(score) and score >= self.trigger.threshold_score:
            self._pending.provisional = False

    def _resolve(self, timestamp: float, event: TriggerEvent | None = None) -> None:
        resuelto = event if event is not None else self._pending
        # La banda inferior entra al embudo, pero no sale sola. El §3.4
        # condiciona la confirmacion a que I supere W —"if I exceeds a
        # confirmation threshold W, the event is classified as a confirmed
        # fall"— y a un evento que ya venia debil se le exige exactamente esa
        # evidencia, no una mas barata. Un evento pleno sigue resolviendose
        # como hasta ahora.
        #
        # La prueba es I >= W medido, NO el veredicto "stage3_confirmed". Los
        # dos coinciden en linea, pero en modo etiquetado (defer_to_end) la
        # Etapa 3 resuelve por la POSTURA final, asi que ahi
        # "stage3_confirmed" significa "termino en el suelo" y no "estuvo
        # inmovil". Con esa lectura la banda admitiria a cualquiera que
        # termine tumbado —incluida la persona que se acuesta en la cama, que
        # es justo el falso positivo que hay que frenar—. Preguntar por I
        # directamente dice lo mismo en los dos modos.
        if resuelto is not None and resuelto.provisional:
            w = self.stage3.threshold_w_s if self.stage3 is not None else None
            inmovil = (w is not None
                       and not math.isnan(resuelto.max_immobility_s)
                       and resuelto.max_immobility_s >= w)
            if not inmovil:
                resuelto.verdict = self.PROVISIONAL_UNCONFIRMED
                resuelto.severity = ""
                i_max = resuelto.max_immobility_s
                resuelto.reason = (
                    "disparo provisional (banda gris) sin quietud >= W: I max "
                    + ("no medida" if math.isnan(i_max) else f"{i_max:.1f} s"))
        self.just_resolved = resuelto
        self._pending = None
        if (not self.cooldown_after_rejection and resuelto is not None
                and resuelto.verdict == REJECTED):
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
                "abandonado por discontinuidad (deteccion perdida o salto); "
                + (self._pending.reason or "sin veredicto de etapa"))
        self.trigger.reset()
        if self.stage2 is not None:
            self.stage2.reset()
        if self.stage3 is not None:
            self.stage3.reset()
        # El tracker NO se limpia aqui: describe el movimiento del cuerpo, no
        # el estado del embudo, y una laguna corta de deteccion no borra el
        # descenso que ya ocurrio.
        self._pending = None
        self.stage = Stage.MONITORING
        self._resolved_at = None
