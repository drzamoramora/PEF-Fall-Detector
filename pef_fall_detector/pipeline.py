"""Per-frame orchestration, shared by every entry point.

``FramePipeline`` is the single place where a frame turns into results:
pose extraction, the physics-informed quantities (§3.4), reliability
policy, the person-state display label, HUD text and CSV values. PEF-Lab
(GUI), the headless CLI and the future batch evaluator (§3.7) all consume
this class instead of wiring the steps themselves.

Why this class exists (design rule of the project): quantities like V carry
state — centroid history, EMA filter memory. If each entry point orchestrated
its own calls, each would hold its own copy of that state and the lab and the
batch runs could produce *different numbers for the same video*, which would
invalidate calibration. One pipeline object = one state = one truth.

Reliability policy: a frame is ``reliable`` when a person was detected AND
the core landmarks (shoulders, hips) meet the visibility threshold from
``config.yaml``. Quantities are still computed and logged for unreliable
frames — data is never silently discarded — but consumers doing statistics
(the headless summary, calibration plots) must filter on this flag, because
below the threshold MediaPipe extrapolates anatomy that is not in the image
(e.g. hips placed hundreds of pixels below the frame border).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .alerts import Alert, AlertDispatcher
from .audit_log import csv_column
from .config import Config
from .person_state import STANDING, WALKING, classify_state
from .pose_frontend import (
    FOOT_LANDMARKS,
    HEAD_AND_TORSO_LANDMARKS,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseFrame,
    PoseFrontend,
)
from .quantities import (
    ExponentialMovingAverage,
    HeightBaseline,
    ImmobilityTimer,
    VelocityEstimator,
    body_height_ratio,
    body_height_ratio_raw,
    body_vertical_extent,
    centroid,
    com_support_offset,
    feet_in_contact,
    shoulder_width,
    support_polygon,
    support_width,
    trunk_band,
    trunk_inclination_deg,
    wrist_ankle_proximity,
)
from .state_machine import (
    DescentTracker,
    FallStateMachine,
    Stage,
    Stage1Trigger,
    Stage2Evaluator,
    Stage3Evaluator,
)


class Timings:
    """Per-section wall-clock stats, for finding where the frame budget goes.

    Exists because this project has already once optimised from a plausible
    guess and been wrong (see the EMA episode in ``notas/``). When the live
    frame rate collapsed to 5 fps the obvious suspect was MediaPipe — but
    inference had been measured at ~25 ms months earlier, so most of the
    200 ms had to be elsewhere. Guessing which "elsewhere" is exactly the
    mistake worth not repeating.

    Keeps a bounded window of recent samples so a long session cannot grow
    it without limit, and reports medians rather than means: one 400 ms
    stall while a model loads should not colour the whole picture.
    """

    def __init__(self, window: int = 300) -> None:
        self.window = int(window)
        self._samples: dict[str, list[float]] = {}

    def add(self, section: str, seconds: float) -> None:
        bucket = self._samples.setdefault(section, [])
        bucket.append(seconds * 1000.0)
        if len(bucket) > self.window:
            del bucket[: len(bucket) - self.window]

    def median_ms(self, section: str) -> float:
        bucket = self._samples.get(section)
        if not bucket:
            return float("nan")
        ordered = sorted(bucket)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    def sections(self) -> list[str]:
        return sorted(self._samples)

    def summary(self) -> str:
        """One compact line, ready for a status bar."""
        if not self._samples:
            return ""
        parts = [f"{name} {self.median_ms(name):.0f}ms" for name in self.sections()]
        total = sum(self.median_ms(n) for n in self.sections())
        fps = 1000.0 / total if total > 0 else float("nan")
        return " | ".join(parts) + f"  ->  {total:.0f}ms/frame ({fps:.1f} fps)"

    def reset(self) -> None:
        self._samples.clear()


@dataclass
class FrameResult:
    """Everything the pipeline knows about one processed frame.

    Attributes:
        pose: the front-end output (landmarks, Step-0 anchors, visibility).
        reliable: detection present AND core visibility above threshold.
            Statistics and calibration must use only reliable frames.
        quantities: named quantity values as floats (NaN = not computable).
            Keys match audit-log column names (e.g. ``T_deg``, ``V_tps``).
        state: person-state display label (``person_state`` vocabulary),
            or None when nothing was detected.
        centroid_px: smoothed centroid position in pixels (for the overlay),
            or None when nothing was detected.
        com_px: hip-weighted centre of mass in pixels — Quantity P's point,
            which §3.4 defines differently from Quantity V's centroid. None
            when the feet were not visible enough to compute P.
        support_hull_px: (N, 2) vertices of the support polygon in pixels,
            or None on the same condition.
        stage: where this frame sits in the §3.5 funnel (``MONITORING``,
            ``CONFIRMING``, ``COOLDOWN``).
        event: the Stage-1 event raised ON this frame, or None. Only the
            firing frame carries it, so a consumer can count events by
            counting non-None values instead of de-duplicating.
        resolved_event: the event whose verdict landed on THIS frame, or
            None. A different moment from ``event``: a record or an alarm
            belongs to the frame where the funnel decided, not to the frame
            where it started wondering.
        hud_lines: ready-to-draw text lines for the PEF-Lab overlay.
    """

    pose: PoseFrame
    reliable: bool
    quantities: dict[str, float] = field(default_factory=dict)
    state: str | None = None
    centroid_px: np.ndarray | None = None
    com_px: np.ndarray | None = None
    support_hull_px: np.ndarray | None = None
    stage: str = "MONITORING"
    event: object | None = None
    resolved_event: object | None = None
    hud_lines: list[str] = field(default_factory=list)

    def csv_extra(self) -> dict[str, str]:
        """Values formatted for the audit CSV (NaN -> empty cell).

        Names are translated to their record spelling on the way out — the
        ``_EXP`` mark on columns the paper does not define. That mapping
        lives in ``audit_log`` so this pipeline, the GUI and the tests keep
        using one vocabulary regardless of how a column is labelled in the
        record.
        """
        extra = {
            csv_column(name): f"{value:.3f}"
            for name, value in self.quantities.items()
            if not math.isnan(value)
        }
        extra["reliable"] = "1" if self.reliable else "0"
        extra["stage"] = self.stage
        extra["stage1_fired"] = "1" if self.event is not None else "0"
        if self.state is not None:
            extra[csv_column("state")] = self.state
        if self.centroid_px is not None:
            extra["centroid_x_px"] = f"{self.centroid_px[0]:.2f}"
            extra["centroid_y_px"] = f"{self.centroid_px[1]:.2f}"
        return extra


class FramePipeline:
    """One frame in, one :class:`FrameResult` out — same code on every path.

    Holds ALL cross-frame state: EMA filters for the centroid and the torso
    length, and the velocity window. A gap in detection (or a big seek)
    longer than ``stage1.history_max_gap_s`` resets that state, because
    smoothing or differentiating across a gap fabricates motion that never
    happened.
    """

    def __init__(self, cfg: Config, labelling: bool = False) -> None:
        """``labelling``: decide each event from how the episode ENDED.

        For a recorded clip being classified there is an end, and §3.3 defines
        every severity tag in terms of it. For a live camera there is none,
        and waiting for one would mean never alerting. See
        :class:`Stage3Evaluator` for the measurement that separates the two.
        """
        self.cfg = cfg
        self.labelling = bool(labelling)
        self._visibility_threshold = float(cfg.pose.visibility_threshold)
        self._max_gap_s = float(cfg.stage1.history_max_gap_s)
        self._max_extension_ratio = float(cfg.state_display.max_extension_ratio)
        self._min_trunk_ratio = float(cfg.stage1.min_trunk_ratio)
        self._trunk_window_s = float(cfg.stage1.trunk_reference_window_s)
        #: (timestamp, torso_px) of recent frames, for the plausibility guard.
        self._trunk_history: list[tuple[float, float]] = []
        # The pose front-end is created lazily, on the first real frame. This
        # keeps MediaPipe out of the import path of :meth:`analyze`, so the
        # per-frame logic can be unit-tested against synthetic PoseFrames
        # without loading the model (see tests/test_pipeline.py).
        self._frontend: PoseFrontend | None = None
        # Cross-frame state (the reason this class must be shared, not copied).
        tau = float(cfg.stage1.ema_time_constant_s)
        self._centroid_ema = ExponentialMovingAverage(tau)
        self._torso_ema = ExponentialMovingAverage(tau)
        self._ratio_ema = ExponentialMovingAverage(
            float(cfg.state_display.ratio_time_constant_s)
        )
        self._velocity = VelocityEstimator(
            window_seconds=float(cfg.stage1.velocity_window_s),
            max_gap_s=self._max_gap_s,
        )
        # Quantity P (§3.4). Provisional values: nothing here is calibrated
        # yet — the thresholds and the hip weight are chosen for plausibility,
        # not from data, and are due to be fixed against the dataset.
        self._com_hip_weight = float(cfg.stage2.com_hip_weight)
        self._min_foot_visibility = float(cfg.stage2.min_foot_visibility)
        self._contact_band_torso = float(cfg.stage2.contact_band_torso)
        # Quantity I (§3.4), run continuously rather than post-trigger — see
        # ImmobilityTimer for why, and for the one draft sentence it needs.
        self._immobility = ImmobilityTimer(float(cfg.stage3.epsilon))
        # Quantity H (EXPERIMENTAL, not in §3.4) — see quantities.py's module
        # docstring and _quantity_h below for what it is and how it is
        # assembled from the pose each frame. Whether it actually affects a
        # decision is entirely the experimental.trigger_H_* / recovery_H_ratio
        # thresholds below: all disabled at 0.0 reproduces pre-H behaviour
        # exactly (their non-zero shipped defaults are placeholders pending
        # calibration — see config.yaml).
        self._height_baseline = HeightBaseline(
            time_constant_s=float(cfg.experimental.height_baseline_time_constant_s)
        )
        # Stage 1 (§3.5). The formulation is a config choice, not a code
        # decision — see state_machine for the measurements behind that.
        self.machine = FallStateMachine(
            Stage1Trigger(
                formulation=str(cfg.stage1.trigger_formulation),
                threshold_t_deg=float(cfg.stage1.threshold_T_deg),
                threshold_v_tps=float(cfg.stage1.threshold_V),
                hold_s=float(cfg.stage1.trigger_hold_s),
                confirm_window_s=float(cfg.stage1.confirm_window_s),
                threshold_score=float(cfg.stage1.trigger_score),
                threshold_score_provisional=float(
                    cfg.stage1.trigger_score_provisional),
                peak_window_s=float(cfg.stage1.trigger_peak_window_s),
                threshold_h_ratio=float(cfg.experimental.trigger_H_ratio),
                threshold_h_erect=float(cfg.experimental.trigger_H_erect),
                h_sequence_window_s=float(
                    cfg.experimental.trigger_H_sequence_window_s),
            ),
            Stage2Evaluator(
                window_s=float(cfg.stage2.com_eval_window_s),
                outside_fraction=float(cfg.stage2.com_outside_fraction),
                min_samples=int(cfg.stage2.com_min_samples),
            ),
            Stage3Evaluator(
                window_s=float(cfg.stage3.observation_window_seconds),
                threshold_w_s=float(cfg.stage3.threshold_W_seconds),
                upright_t_deg=float(cfg.stage3.upright_T_deg),
                recovery_hold_s=float(cfg.stage3.recovery_hold_seconds),
                standing_extension=float(cfg.state_display.standing_extension),
                use_leg_extension=bool(cfg.stage3.severity_uses_leg_extension),
                defer_to_end=self.labelling,
                lying_t_deg=float(cfg.state_display.lying_T_deg),
                max_extension=float(cfg.state_display.max_extension_ratio),
                threshold_h_ratio_recovery=float(
                    cfg.experimental.recovery_H_ratio),
                threshold_h_ratio_lying=float(cfg.experimental.trigger_H_ratio),
                persistent_still_s=float(cfg.stage3.persistent_still_s),
                persistent_fraction=float(cfg.stage3.persistent_immobility_fraction),
                getup_rise_torsos=float(
                    cfg.stage3.as_dict().get("getup_rise_torsos", 0.0)),
                getup_window_s=float(cfg.stage3.as_dict().get("getup_window_s", 2.0)),
            ),
            cooldown_s=float(cfg.stage3.cooldown_seconds),
            cooldown_after_rejection=bool(
                cfg.stage3.as_dict().get("cooldown_after_rejection", True)),
            descent=DescentTracker(
                floor_tps=float(cfg.stage2.descent_floor_tps),
                window_s=float(cfg.stage2.descent_window_s)),
            min_depth_tps=float(cfg.stage2.descent_min_depth_tps),
            min_sustained_s=float(cfg.stage2.descent_min_sustained_s),
        )
        # The §3.5 output seam. No sink is attached by default: PEF-Lab adds
        # a screen banner, the headless runner a console line, and Phase 7
        # the §3.6 transports. The pipeline itself only builds the alert.
        self.alerts = AlertDispatcher(
            dispatch_verdicts=tuple(cfg.alerts.dispatch_verdicts))
        self.source_name = ""
        self._recent_state = ""
        self.timings = Timings()
        self._last_timestamp: float | None = None
        # Set whenever continuity is broken; the next analysed frame starts a
        # fresh motion history instead of differentiating across the break.
        self._discontinuous = True
        #: When the subject was first missing, or None while visible. Only a
        #: gap LONGER than ``history_max_gap_s`` abandons a pending verdict —
        #: see :meth:`_abandon_pending_event`.
        self._lost_since: float | None = None
        # State-display thresholds, read once (display-only section).
        sd = cfg.state_display
        self._state_kwargs = dict(
            transition_v_tps=float(sd.transition_v_tps),
            lying_t_deg=float(sd.lying_T_deg),
            leaning_t_deg=float(sd.leaning_T_deg),
            standing_extension=float(sd.standing_extension),
            crouch_extension=float(sd.crouch_extension),
            walking_vh_tps=float(sd.walking_vh_tps),
        )

    # ------------------------------------------------------------------ core
    def process(self, frame_bgr: np.ndarray, frame_index: int, timestamp: float) -> FrameResult:
        """Extract pose from a frame and analyse it. The full per-frame path."""
        if self._frontend is None:
            cfg = self.cfg
            self._frontend = PoseFrontend(
                model_complexity=cfg.pose.model_complexity,
                min_detection_confidence=cfg.pose.min_detection_confidence,
                min_tracking_confidence=cfg.pose.min_tracking_confidence,
                smooth_landmarks=cfg.pose.smooth_landmarks,
                visibility_threshold=float(cfg.pose.visibility_threshold),
                reacquire_after_s=float(
                    cfg.pose.as_dict().get("reacquire_after_s", 0.0)),
            )
        t0 = time.perf_counter()
        pf = self._frontend.process(frame_bgr, frame_index, timestamp)
        t1 = time.perf_counter()
        result = self.analyze(pf)
        t2 = time.perf_counter()
        # Where the per-frame budget actually goes. Kept because this project
        # already once optimised the wrong thing from a plausible guess; a
        # measured split is cheap and settles the argument.
        self.timings.add("pose", t1 - t0)
        self.timings.add("quantities", t2 - t1)
        return result

    def analyze(self, pf: PoseFrame) -> FrameResult:
        """Compute every implemented quantity from an already-extracted pose.

        Split out from :meth:`process` so the stateful per-frame logic —
        smoothing, velocity history, discontinuity handling — can be tested
        with synthetic :class:`PoseFrame` objects, without MediaPipe and
        without video.
        """
        timestamp = pf.timestamp
        reliable = pf.detected and pf.core_visibility >= self._visibility_threshold

        if not pf.detected:
            # ANY loss of detection breaks continuity, however brief. When the
            # subject reappears there is no guarantee it is the same body, nor
            # any record of where it went meanwhile — and in multi-person
            # scenes the tracker demonstrably re-attaches to a different
            # person. Measured in production data, a single lost frame
            # (0.033 s, far under the gap threshold) followed by such a jump
            # produced +26 torso/s, a physically impossible value that would
            # fire Stage 1 on a tracking artefact rather than on a fall.
            self._discontinuous = True
            self._last_timestamp = None
            # How long the subject has been missing. A brief flicker leaves a
            # pending verdict alone; a real loss abandons it, because after
            # long enough there is no guarantee the body that reappears is the
            # same one (measured in multi-person footage).
            closed = None
            if self._lost_since is None:
                self._lost_since = timestamp
            elif timestamp - self._lost_since > self._max_gap_s:
                # Seen on the floor and then lost: close on what was seen
                # (FallStateMachine.close_if_last_seen_down); otherwise
                # abandon, as before.
                closed = self.machine.close_if_last_seen_down(timestamp)
                if closed is not None:
                    self._dispatch(closed)
                else:
                    self._abandon_pending_event()
            return FrameResult(pose=pf, reliable=False,
                               stage=self.machine.stage.value,
                               resolved_event=closed)
        self._lost_since = None

        # Fase 4: el rastreador se re-sembro (pose_frontend). Los landmarks de
        # este cuadro vienen de una deteccion nueva, no de seguir al anterior:
        # misma ruptura de continuidad que una perdida breve.
        if pf.reacquired:
            self._discontinuous = True
            # Y la ventana de pico del disparador: lo anterior al reinicio era
            # un esqueleto pegado, no el mismo cuerpo. (Una perdida comun NO
            # la limpia: medido, eso perdio 4 caidas reales cuya deteccion
            # parpadea durante la caida.)
            self.machine.trigger.reset()

        # A jump in time — forward past the gap threshold, or backwards at
        # all — means the operator moved elsewhere in the recording.
        if self._last_timestamp is not None and (
            timestamp - self._last_timestamp > self._max_gap_s
            or timestamp < self._last_timestamp
        ):
            self._discontinuous = True
            # A seek is not an occlusion. The operator jumped somewhere else
            # in the recording, so the pending event belongs to a stretch of
            # time that is no longer being observed; nothing can finish
            # judging it.
            self._abandon_pending_event()

        if self._discontinuous:
            self._reset_motion_state()

        # Real elapsed time since the previous analysed frame. Every filter is
        # driven by this rather than by a per-sample coefficient, so their
        # behaviour does not change with the capture rate or with dropped
        # frames (see ExponentialMovingAverage). It is 0 on the first frame
        # after a discontinuity, which seeds the filters instead of blending.
        dt = (
            timestamp - self._last_timestamp
            if self._last_timestamp is not None and timestamp > self._last_timestamp
            else 0.0
        )
        self._last_timestamp = timestamp
        quantities: dict[str, float] = {}
        hud_lines: list[str] = []

        # --- Quantity T: trunk inclination (§3.4) ---------------------------
        t_deg = trunk_inclination_deg(pf.mid_hip, pf.mid_shoulder)
        if not self._trunk_is_plausible(pf.torso_length, timestamp):
            t_deg = float("nan")
        quantities["T_deg"] = t_deg
        hud_lines.append(f"T (trunk): {t_deg:5.1f} deg  [{trunk_band(t_deg)}]")

        # --- Centroid + EMA smoothing (2.2), then Quantity V (2.3) ----------
        raw_centroid = centroid(pf.mid_hip, pf.mid_shoulder)
        smoothed_centroid = self._centroid_ema.update(raw_centroid, dt)
        smoothed_torso = float(self._torso_ema.update(pf.torso_length, dt))
        v_tps, vh_tps = self._velocity.update(timestamp, smoothed_centroid, smoothed_torso)
        quantities["V_tps"] = v_tps
        quantities["Vh_tps"] = vh_tps  # EXPERIMENTAL: not in the paper (§3.4)
        if math.isnan(v_tps):
            hud_lines.append("V (vert) : filling window...")
        else:
            hud_lines.append(f"V (vert) : {v_tps:+5.2f} torso/s   Vh: {vh_tps:+5.2f}")

        # --- Diagnostic: metric torso length (world landmarks) --------------
        if pf.world_torso_length is not None:
            quantities["world_torso_len_m"] = pf.world_torso_length

        # --- Person-state display label (2.4) -------------------------------
        extension_ratio = self._hip_ankle_extension(pf, dt)
        state = classify_state(
            t_deg, extension_ratio, v_tps, vh_tps, **self._state_kwargs
        )
        quantities["extension_ratio"] = extension_ratio
        hud_lines.append(f"State    : {state}")
        # Kept for the alert's human-readable prelude ("subject was WALKING
        # before"). Only updated while nothing is pending, so the alert
        # reports what the subject was doing BEFORE the event rather than
        # the posture the fall itself produced.
        if state and self.machine.stage is Stage.MONITORING:
            self._recent_state = state

        # --- Quantity H: body height vs. personal baseline (EXPERIMENTAL) ---
        # Not in §3.4. Whether it affects Stage 1/3 at all is entirely the
        # experimental.* thresholds passed into the FallStateMachine above —
        # see _quantity_h and quantities.body_height_ratio.
        h_ratio, h_raw, h_baseline = self._quantity_h(pf, dt, state)
        quantities["H_ratio"] = h_ratio
        quantities["H_raw"] = h_raw
        quantities["H_baseline"] = h_baseline
        if math.isnan(h_ratio):
            hud_lines.append("H (height): -- [EXP]")
        else:
            hud_lines.append(f"H (height): {h_ratio:4.2f}  [EXP]")

        # --- Quantity R: wrist-to-ankle proximity (EXPERIMENTAL) ------------
        # Not in §3.4, not wired into any §3.5 stage yet (pending calibration
        # — see quantities.wrist_ankle_proximity). Logged so it can be
        # evaluated as the discriminator H and T cannot be: intent, not
        # posture (tying a shoe vs. falling look alike to everything above).
        reach = self._quantity_reach(pf)
        quantities["reach_proximity"] = reach
        if math.isnan(reach):
            hud_lines.append("R (reach): -- [EXP]")
        else:
            hud_lines.append(f"R (reach): {reach:4.2f}  [EXP]")

        # --- Quantity P: COM vs. support polygon (§3.4) ---------------------
        com_px, hull_px, p_offset, p_width = self._quantity_p(pf)
        quantities["P_offset"] = p_offset
        quantities["P_support_width"] = p_width
        if math.isnan(p_offset):
            hud_lines.append("P (COM)  : feet not visible")
        else:
            where = "OUTSIDE" if p_offset > 0.0 else "inside"
            hud_lines.append(
                f"P (COM)  : {p_offset:+5.2f} torso [{where}]  support {p_width:4.2f}"
            )

        # --- Quantity I: immobility (§3.4), as a continuous clock -----------
        i_seconds, i_disp = self._quantity_i(pf, timestamp)
        quantities["I_still_s"] = i_seconds
        quantities["I_displacement"] = i_disp
        if math.isnan(i_seconds):
            hud_lines.append("I (still): --")
        else:
            hud_lines.append(
                f"I (still): {i_seconds:5.1f} s   (moved {i_disp:.3f} of {self._immobility.epsilon_torso:.3f})"
            )

        # --- Stage 1: the kinematic trigger (§3.5) --------------------------
        # Evaluated only on reliable frames. Below the visibility threshold
        # the geometry is MediaPipe extrapolating anatomy outside the image,
        # and a decision taken on it would be a decision taken on nothing.
        event = None
        if reliable:
            event = self.machine.update(pf.frame_index, timestamp,
                                        t_deg, v_tps, p_offset,
                                        i_seconds, extension_ratio, h_ratio)
            # Fase 6a: la altura de la cadera, para leer un levantarse en
            # curso al final (Stage3Evaluator.getup_rise).
            if (self.machine.stage is Stage.OBSERVING and self.machine.stage3 is not None
                    and pf.mid_hip is not None and pf.torso_length):
                self.machine.stage3.observe_hip(timestamp, float(pf.mid_hip[1]),
                                                float(pf.torso_length))
        # What Stage 1 actually compared on this frame — the score over the
        # peak window, not T and V separately. A curve crossing the old
        # threshold_T / threshold_V lines says nothing about firing; this does.
        # Only fresh when the machine ran; an unreliable frame has none.
        quantities["trigger_score"] = (
            float(getattr(self.machine.trigger, "last_score", float("nan")))
            if reliable else float("nan"))
        # Phase 3's running measure: share of Stage-3 frames with I >= the
        # still cut. Read from the evaluator itself, so the curve PEF-Lab draws
        # and the rule finalise() applies cannot disagree. Only meaningful
        # while Stage 3 is observing.
        st3 = self.machine.stage3
        quantities["still_fraction"] = (
            float(st3.still_fraction)
            if st3 is not None and self.machine.stage is Stage.OBSERVING
            else float("nan"))
        stage = self.machine.stage
        resolved = self.machine.just_resolved
        if resolved is not None:
            self._dispatch(resolved)
        if event is not None:
            hud_lines.append(
                f"** STAGE 1 FIRED ** T={event.t_deg:.1f} V={event.v_tps:+.2f}"
            )
        elif stage is Stage.OBSERVING:
            hud_lines.append(f"Stage    : {stage.value} (waiting on I / recovery)")
        elif stage is not Stage.MONITORING:
            hud_lines.append(f"Stage    : {stage.value}")

        if not reliable:
            hud_lines.append("LOW VISIBILITY - values unreliable")

        return FrameResult(
            pose=pf,
            reliable=reliable,
            quantities=quantities,
            state=state,
            centroid_px=np.asarray(smoothed_centroid, dtype=float),
            com_px=com_px,
            support_hull_px=hull_px,
            stage=stage.value,
            event=event,
            resolved_event=resolved,
            hud_lines=hud_lines,
        )

    # --------------------------------------------------------------- helpers
    def _dispatch(self, event) -> None:
        """Build the alert for a resolved event and hand it to the sinks."""
        self.alerts.dispatch(Alert(
            timestamp=event.timestamp,
            severity=event.severity,
            verdict=event.verdict,
            frame_index=event.frame_index,
            t_deg=event.t_deg,
            v_tps=event.v_tps,
            p_outside_fraction=event.p_outside_fraction,
            max_immobility_s=event.max_immobility_s,
            prior_state=self._recent_state,
            source=self.source_name,
            reason=getattr(event, "reason", ""),
        ))

    def _quantity_i(self, pf: PoseFrame, timestamp: float) -> tuple[float, float]:
        """Quantity I for one frame: (still_seconds, displacement).

        §3.4 watches "the head and torso landmark positions". Those are not
        equally available: in a face-down posture — the very outcome this
        quantity exists to confirm — the nose is occluded while the torso
        stays visible. Requiring all five would therefore blind the
        measurement exactly where it matters most.

        So the torso four are required and the nose is optional: below the
        visibility threshold it is passed as NaN, which the timer skips
        without ending the episode, and it re-anchors itself if the face
        comes back into view. If the torso itself goes unreliable there is
        nothing trustworthy left to measure, and the episode resets rather
        than accumulating time on extrapolated anatomy.
        """
        if not pf.torso_length or pf.core_visibility < self._visibility_threshold:
            self._immobility.reset()
            return float("nan"), float("nan")
        pts = pf.landmarks[list(HEAD_AND_TORSO_LANDMARKS), :2].astype(float).copy()
        vis = pf.visibility[list(HEAD_AND_TORSO_LANDMARKS)]
        pts[vis < self._visibility_threshold] = np.nan
        return self._immobility.update(timestamp, pts, pf.torso_length)

    def _quantity_p(self, pf: PoseFrame):
        """Quantity P for one frame: (com_px, hull_px, offset, width).

        Two decisions worth reading before trusting the numbers.

        **The COM is hip-weighted, not the centroid of V.** §3.4 approximates
        the centre of mass as a projection "in which the hip cluster carries
        the dominant contribution", which is anatomically where the body's
        mass sits (near the sacrum). Quantity V's centroid is a plain 0.5/0.5
        midpoint by its own definition in the same section. They are two
        different points on purpose, and reusing one for the other would
        quietly redefine a published quantity.

        **Occluded feet make P undefined, never zero.** Without the feet
        there is no support polygon, and an assumed one would decide a fall
        on invented geometry. NaN is the same policy §3.4 states for a
        degenerate trunk in T, and the same one already applied to the
        extension ratio: an unobserved quantity reads as unknown.
        """
        vis = pf.visibility[list(FOOT_LANDMARKS)]
        if float(vis.min()) < self._min_foot_visibility or not pf.torso_length:
            return None, None, float("nan"), float("nan")
        # §3.4 builds the polygon from the foot landmarks *in bipedal
        # contact*, not from every one MediaPipe reports. Skipping that
        # clause is what produced supports 165 torso-lengths wide in this
        # project's own footage when a foot was lifted.
        feet = feet_in_contact(
            pf.landmarks[list(FOOT_LANDMARKS), :2],
            pf.frame_size, pf.torso_length, self._contact_band_torso,
        )
        if len(feet) == 0:
            # Defence in depth rather than the mechanism: an empty hull
            # already makes both P quantities NaN downstream, so a mutation
            # removing this early return passes every test (verified). It
            # stays because "no foot could be located on the floor" is a
            # distinct fact from "the arithmetic happened to yield NaN", and
            # the next person to touch the hull code should not have to
            # rediscover that the two coincide today.
            return None, None, float("nan"), float("nan")
        com = centroid(pf.mid_hip, pf.mid_shoulder, hip_weight=self._com_hip_weight)
        hull = support_polygon(feet)
        return (
            com,
            hull,
            com_support_offset(com, hull, pf.torso_length),
            support_width(hull, pf.torso_length),
        )

    def _quantity_h(self, pf: PoseFrame, dt: float, state: str | None
                    ) -> tuple[float, float, float]:
        """Quantity H for one frame (EXPERIMENTAL, not in §3.4).

        NOT consumed by any §3.5 stage. Proposed to cover two gaps measured
        on this project's own footage that T and P share: P is undefined the
        moment feet are occluded or leave frame, and T reads a falsely low,
        "upright"-looking angle when a fall's rotation axis points toward or
        away from the camera (its trunk vector's 2D projection nearly
        vanishes). See ``quantities.body_height_ratio`` for the two limits it
        does NOT overcome — camera-angle dependence, and confirming posture
        rather than intent — before this number is read as more than a
        diagnostic.

        Assembled here, rather than from ``pf.mid_hip``/``pf.mid_shoulder``
        directly, because those midpoints do not carry per-landmark
        visibility: H needs to know WHICH of nose, shoulders, hips and
        ankles were actually seen this frame, so a point below the
        visibility threshold can be dropped (NaN) instead of silently
        trusted — the same policy already applied to the ankles in
        ``_hip_ankle_extension``.

        The calibration input (``is_standing``) reuses ``classify_state``'s
        own STANDING/WALKING verdict — both are upright with legs extended,
        which is exactly the posture ``HeightBaseline`` needs to calibrate
        against.

        Returns:
            ``(H, raw_ratio, baseline)`` — H is the candidate reading; the
            other two are logged alongside it so a calibration pass can see
            how the baseline converged and whether a given frame's raw ratio
            was itself trustworthy, without having to re-derive them.
        """
        vis = pf.visibility
        threshold = self._visibility_threshold

        def midpoint_or_nan(a: int, b: int) -> np.ndarray:
            if vis[a] < threshold or vis[b] < threshold:
                return np.array([np.nan, np.nan])
            return pf.landmarks[[a, b], :2].mean(axis=0)

        nose = (pf.landmarks[NOSE, :2] if vis[NOSE] >= threshold
                else np.array([np.nan, np.nan]))
        spine = np.array([
            nose,
            midpoint_or_nan(LEFT_SHOULDER, RIGHT_SHOULDER),
            midpoint_or_nan(LEFT_HIP, RIGHT_HIP),
            midpoint_or_nan(LEFT_ANKLE, RIGHT_ANKLE),
        ])
        extent = body_vertical_extent(spine)

        if vis[LEFT_SHOULDER] < threshold or vis[RIGHT_SHOULDER] < threshold:
            width = float("nan")
        else:
            width = shoulder_width(pf.landmarks[LEFT_SHOULDER, :2],
                                   pf.landmarks[RIGHT_SHOULDER, :2])

        raw_ratio = body_height_ratio_raw(extent, width)
        is_standing = state in (STANDING, WALKING)
        baseline = self._height_baseline.update(dt, raw_ratio, is_standing)
        return body_height_ratio(raw_ratio, baseline), raw_ratio, baseline

    def _quantity_reach(self, pf: PoseFrame) -> float:
        """Quantity R for one frame (EXPERIMENTAL, not in §3.4): how close
        the hands are to the feet.

        NOT consumed by any §3.5 stage — logged so it can be evaluated
        against labelled clips before it is. See
        ``quantities.wrist_ankle_proximity`` for what it is proposed to
        cover: T, V, P, I and H all read posture, and a deliberate bend
        (tying a shoe) produces close to the same posture as a fall by every
        one of them. This is the only quantity in the module that looks at
        the hands at all, which is what makes it a candidate for telling the
        two apart instead of just confirming that "something collapsed"
        happened again.

        Per-landmark visibility gating, same policy as every other quantity
        here: a wrist or ankle below the threshold is passed as NaN rather
        than trusted.
        """
        vis = pf.visibility
        threshold = self._visibility_threshold
        wrists = pf.landmarks[[LEFT_WRIST, RIGHT_WRIST], :2].copy()
        wrists[vis[[LEFT_WRIST, RIGHT_WRIST]] < threshold] = np.nan
        ankles = pf.landmarks[[LEFT_ANKLE, RIGHT_ANKLE], :2].copy()
        ankles[vis[[LEFT_ANKLE, RIGHT_ANKLE]] < threshold] = np.nan
        return wrist_ankle_proximity(wrists, ankles, pf.torso_length)

    def _trunk_is_plausible(self, torso_px: float, timestamp: float) -> bool:
        """Whether the projected trunk is long enough for T to mean anything.

        This is the guard §3.2 already promises — *"depth ambiguity from the
        monocular input is mitigated by tracking within-frame and across-frame
        landmark displacement ratios rather than relying on absolute 3D
        distances"* — and which the code did not implement.

        WHY THE §3.4 GUARD IS NOT ENOUGH. That section discards T when
        |trunk| < 0.001 px. No real pose reaches that, so the guard never
        fires. Measured on clip A14: the trunk projected to **2 px** while
        MediaPipe reported 0.99 confidence, and T produced 176.9 deg next to
        26.0 deg on consecutive frames. Two pixels is 2000x the paper's
        threshold, so the frame sailed through and the noise reached Stage 1.

        WHY THE COMPARISON IS TO THE SUBJECT'S OWN RECENT TRUNK, not to the
        frame or to an absolute pixel count. A subject far from the camera
        legitimately has a short trunk in pixels; nothing is wrong with that
        frame. What is not legitimate is a trunk that collapses relative to
        what the SAME subject measured a moment ago. That is an across-frame
        ratio, which is what §3.2 asks for.

        The reference is a **median**, not the existing EMA: the collapse
        itself drags a mean down with it, so a mean would normalise away the
        very event being detected.

        WHAT THIS GUARD CANNOT DO, and it matters. A genuine fall also
        shortens the projected trunk — measured across the 56 falls that
        fired, the trunk drops to a median of 0.40 of its own recent value.
        So this cannot separate a fall from an artefact, and a tight ratio
        would reject real falls. It is set loose on purpose: it removes
        frames whose geometry is impossible (collapses to 0.05-0.07 were
        recorded), not frames that are merely foreshortened. Telling an axial
        fall from a collapsed estimate is not possible from a projected trunk
        at all — that is the limitation documented as D4.
        """
        if self._min_trunk_ratio <= 0.0:
            return True
        if not torso_px or torso_px != torso_px:
            return True                      # nothing to judge; §3.4 handles it
        cutoff = timestamp - self._trunk_window_s
        while self._trunk_history and self._trunk_history[0][0] < cutoff:
            self._trunk_history.pop(0)
        recent = [t for _ts, t in self._trunk_history]
        self._trunk_history.append((timestamp, float(torso_px)))
        if len(recent) < 3:
            return True                      # no reference yet
        recent.sort()
        reference = recent[len(recent) // 2]
        return torso_px >= self._min_trunk_ratio * reference

    def _hip_ankle_extension(self, pf: PoseFrame, dt: float) -> float:
        """Hip-to-ankle DISTANCE in torso units (posture feature, 2.4).

        Was the vertical extent — ``(ankle_y - hip_y) / torso`` — which is a
        signed drop, not a length, and it was wrong in a way the dataset made
        obvious: clip A15 recorded ``extension_ratio_EXP = -0.433``. A ratio
        of two lengths cannot be negative. Worse, the quantity is asked "are
        the legs extended or folded?" and a vertical drop cannot answer that:
        a subject lying prone with legs perfectly straight has ankles level
        with the hips, so the vertical extent reads ~0 — indistinguishable
        from kneeling. Since this ratio is what separates a full recovery
        (mild) from a partial one (moderate) in §3.3, the confusion landed
        straight on the severity.

        The Euclidean distance has neither problem: it is a length, so it is
        positive, and it measures leg extension in any orientation. It is
        read only while the trunk is upright — in the Stage-3 latch and, for
        display labels, after ``classify_state`` has already sent horizontal
        trunks to LYING — so "extended while upright" still means standing and
        "folded while upright" still means sitting or kneeling.

        NOTE: the numeric scale shifts for non-upright postures (a prone
        subject now reads ~1.5 instead of ~0), so ``standing_extension`` and
        ``crouch_extension`` need re-checking against the dataset. For
        upright postures — the only ones that consult it — the two
        definitions nearly coincide, because the ankles sit almost directly
        below the hips.

        Computed entirely from **raw** landmark positions — including a raw
        torso length — and then smoothed as a finished number. Mixing
        smoothing stages (a raw numerator over a filtered denominator, as
        this did originally) is close to useless: the noise is dominated by
        the ankles in the numerator, so filtering only the denominator cuts
        the jitter by ~16 %, while filtering the finished ratio cuts it by
        ~80 % (simulated across realistic jitter levels). Without the
        output filter, a perfectly still
        subject positioned near the standing/sitting boundary flips label
        tens of times per minute even in good conditions.

        Returns NaN when the ankles are below the visibility threshold —
        occluded ankles must read as "unknown", never as a posture (same
        principle as the Stage-2 occlusion rule, implementation-plan §5.3).
        A NaN also resets the smoother: blending across a period where the
        ankles were invisible would carry stale posture into the present.
        """
        ankle_vis = float(pf.visibility[[LEFT_ANKLE, RIGHT_ANKLE]].min())
        torso_raw = pf.torso_length
        if ankle_vis < self._visibility_threshold or not torso_raw:
            self._ratio_ema.reset()
            return float("nan")
        mean_ankle = pf.landmarks[[LEFT_ANKLE, RIGHT_ANKLE], :2].mean(axis=0)
        raw_ratio = float(np.linalg.norm(mean_ankle - pf.mid_hip[:2])) / torso_raw
        if raw_ratio > self._max_extension_ratio:
            # Not a leg: the torso is foreshortened to a few pixels and the
            # ratio is dividing by it. Values up to 183 torso lengths appear
            # in this project's own dataset, in 20 of 128 clips. NaN — the
            # same answer occluded ankles get — rather than a clamped number,
            # because "implausible" is not a measurement and a clamped value
            # would sit in the record looking like one.
            self._ratio_ema.reset()
            return float("nan")
        return float(self._ratio_ema.update(raw_ratio, dt))

    def reset(self) -> None:
        """Declare a discontinuity: drop all motion history.

        Call this whenever the stream of frames stops being a continuous
        observation of one body — the obvious case being the operator
        jumping elsewhere in a recording with the seek bar or the frame
        step. Without it, a jump shorter than the gap threshold would let
        the velocity window mix frames from two different places in the
        timeline and report motion that never happened.

        The pipeline also raises this flag by itself on detection loss and
        on time jumps; this method is the explicit door for callers that
        know something the pipeline cannot infer.
        """
        self._reset_motion_state()

    def _reset_motion_state(self) -> None:
        # Note on the two EMA resets below: clearing ``_last_timestamp`` makes
        # the next frame's dt zero, and a zero dt already reseeds an EMA by
        # definition, so those two calls are redundant *as the code stands
        # today* — a mutation removing them is not observable, and the tests
        # cannot catch it. They are kept as defence in depth: relying on a
        # subtle interaction between two mechanisms is exactly how the
        # discontinuity bugs in this file arose in the first place. The
        # ``_ratio_ema`` reset is NOT redundant — occlusion resets it while
        # the clock keeps running.
        self._centroid_ema.reset()
        self._torso_ema.reset()
        self._ratio_ema.reset()
        self._velocity.reset()
        # The immobility clock resets too, and for a reason worth stating
        # because the tempting alternative is wrong in a subtle way. After a
        # detection loss the subject may well reappear inside the same ε ball,
        # which looks like proof that nobody moved — but it is equally
        # consistent with a *different* person standing where the first one
        # was, which is the failure this project already measured in
        # multi-person footage. Until the plausibility guard exists to tell
        # those apart, resuming the clock across a gap would let the system
        # certify stillness it never observed.
        self._immobility.reset()
        # The height baseline is a calibration of THIS subject; resuming it
        # across a gap risks calibrating "standing" against a different body,
        # the same reasoning as the immobility clock above.
        self._height_baseline.reset()
        # The trunk reference belongs to a continuous observation of one body.
        self._trunk_history.clear()
        self._last_timestamp = None
        self._discontinuous = False

    def _abandon_pending_event(self) -> None:
        """Throw away an evaluation in progress. SEPARATE from motion state.

        These two used to be one call, and conflating them cost this project
        11 accuracy points on its own dataset: 46 events were abandoned
        mid-evaluation across 37 of 128 clips, because ANY single lost frame
        reset the funnel along with the filters.

        The two are not the same problem. Differentiating across a gap
        produces a physically impossible velocity — measured here at
        +26 torso/s from one lost frame — so the motion filters MUST forget.
        But Stage 2 and Stage 3 already treat missing input as "unknown"
        rather than "no fall"; that was the whole point of the occlusion rule.
        A funnel that also forgets is throwing away evidence it was built to
        survive without.

        And the timing is the worst possible: detection is least reliable at
        the moment of impact, which is precisely the moment the funnel exists
        to judge. Measured on this dataset, detection gaps have a median of
        0.10 s and 82 % fall under 0.5 s.

        An armed *sequential* trigger is a different matter and is reset with
        the motion state either way: a V event seen before a gap could
        otherwise be confirmed by a T measured on a different body.
        """
        self.machine.reset()

    def finalise(self, timestamp: float):
        """End of the recording: close any event still being judged.

        Only for a source that HAS an end. Returns the event that was closed,
        or None. Without this a clip that ends while the subject is still on
        the floor leaves its event open, and the clip label would read
        "nothing happened" for the outcome that matters most.
        """
        return self.machine.finalise(timestamp)

    def close(self) -> None:
        """Release the pose front-end, if one was ever created."""
        if self._frontend is not None:
            self._frontend.close()
            self._frontend = None
