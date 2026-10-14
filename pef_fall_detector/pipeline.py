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

import hashlib
import math
import time
from collections import deque

import numpy as np

from .alerts import Alert, AlertDispatcher
from .config import Config
from .person_state import STANDING, WALKING, classify_state
from .pose_frontend import PoseFrame, PoseFrontend
from .quantities import (
    ExponentialMovingAverage,
    GravityCalibrator,
    HeightBaseline,
    ImmobilityTimer,
    LegExtension,
    TrunkGuard,
    VelocityEstimator,
    centroid,
    trunk_band,
    trunk_inclination_deg,
)
from .quantities import a as quantity_a
from .quantities import h as quantity_h
from .quantities import i as quantity_i
from .quantities import p as quantity_p
from .quantities import r as quantity_r
from .results import FrameResult, Timings
from .state_machine import Stage
from .state_machine.build import build_machine


def frame_digest(frame_bgr: np.ndarray) -> str:
    """First 16 hex digits of the SHA-1 of a decoded image's bytes.

    Sixty-four bits are plenty to tell two images apart; the full digest
    would only make the record wider.
    """
    return hashlib.sha1(np.ascontiguousarray(frame_bgr).data).hexdigest()[:16]


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
        # The §3.2 plausibility guard on the trunk (quantities.t.TrunkGuard).
        self._trunk_guard = TrunkGuard(float(cfg.stage1.min_trunk_ratio),
                                       float(cfg.stage1.trunk_reference_window_s))
        # The pose front-end is created lazily, on the first real frame. This
        # keeps MediaPipe out of the import path of :meth:`analyze`, so the
        # per-frame logic can be unit-tested against synthetic PoseFrames
        # without loading the model (see tests/test_pipeline.py).
        self._frontend: PoseFrontend | None = None
        # Cross-frame state (the reason this class must be shared, not copied).
        tau = float(cfg.stage1.ema_time_constant_s)
        self._centroid_ema = ExponentialMovingAverage(tau)
        self._torso_ema = ExponentialMovingAverage(tau)
        self._extension = LegExtension(float(cfg.state_display.max_extension_ratio),
                                       float(cfg.state_display.ratio_time_constant_s))
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
        # Quantity H (EXPERIMENTAL, not in §3.4) — see the ``quantities``
        # package docstring and quantities.h.measure for what it is and how it
        # is assembled from the pose each frame. Whether it affects a decision
        # is entirely the experimental.trigger_H_* / recovery_H_ratio
        # thresholds: all at 0.0 reproduces pre-H behaviour exactly. They ship
        # ON, with values chosen for plausibility and never swept on
        # PEF-FallDB, and H decides: in the phase-8.4 records it raised 25 of
        # the 90 Stage-1 events in the cloud and 21 of 90 on the Mac
        # (trigger_source "H").
        self._height_baseline = HeightBaseline(
            time_constant_s=float(cfg.experimental.height_baseline_time_constant_s)
        )
        # Quantity A (EXPERIMENTAL, phase 8): head height against gravity, 3D.
        # Logged since 8.1; read by Stage 3 since 8.2/8.3 and by Stage 1 since
        # 8.4 (quantity_a switches). See quantities.a.
        qa = cfg.quantity_a
        self._a_standing_t = float(qa.standing_T_deg)
        self._a_standing_knee = float(qa.standing_knee_deg)
        self._gravity = GravityCalibrator(
            min_standing_s=float(qa.min_standing_s),
            max_samples=int(qa.max_samples),
        )
        #: (timestamp, A_head, A_hip) of the last few seconds, so Stage 3 can
        #: read the second BEFORE the trigger, which it was not yet watching.
        self._a_history: deque[tuple[float, float, float]] = deque()
        self._a_history_s = float(qa.reach_before_s) + 4.0
        #: (trigger timestamp, A calibrated on that frame) of the last event.
        self._a_trigger: tuple[float, bool] | None = None
        # The §3.5 funnel, every threshold from the configuration
        # (state_machine.build).
        self.machine = build_machine(cfg, self.labelling)
        # The §3.5 output seam. No sink is attached by default: PEF-Lab adds
        # a screen banner, the headless runner a console line, and Phase 7
        # the §3.6 transports. The pipeline itself only builds the alert.
        self.alerts = AlertDispatcher(
            dispatch_verdicts=tuple(cfg.alerts.dispatch_verdicts))
        self.source_name = ""
        #: Hash every decoded image into ``FrameResult.frame_sha1`` (records
        #: schema 2). About 1 ms a 720p frame; set by whoever records, so the
        #: deployed detector, which records nothing, does not pay for it.
        self.hash_frames = False
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
                reacquire_after_s=float(cfg.pose.reacquire_after_s),
            )
        # Hashed before MediaPipe sees it, from the very array it receives.
        digest = frame_digest(frame_bgr) if self.hash_frames else ""
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
        result.frame_sha1 = digest
        result.timing_ms = {"pose_ms": 1000.0 * (t1 - t0), "analysis_ms": 1000.0 * (t2 - t1)}
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

        # Phase 4: the tracker was re-seeded (pose_frontend). This frame's
        # landmarks come from a fresh detection, not from tracking the last
        # one: the same break in continuity as a brief loss.
        if pf.reacquired:
            self._discontinuous = True
            # And the trigger's peak window: what came before the re-seed was
            # a stuck skeleton, not the same body. (A plain loss does NOT
            # clear it; see notas/CODIGO-HISTORIA.md.)
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
        if not self._trunk_guard.plausible(pf.torso_length, timestamp):
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
        extension_ratio = self._extension.update(pf, dt, self._visibility_threshold)
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
        # see quantities.h.measure and quantities.h.body_height_ratio.
        h_ratio, h_raw, h_baseline = quantity_h.measure(
            self._height_baseline, pf, dt, state in (STANDING, WALKING),
            self._visibility_threshold)
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
        reach = quantity_r.measure(pf, self._visibility_threshold)
        quantities["reach_proximity"] = reach
        if math.isnan(reach):
            hud_lines.append("R (reach): -- [EXP]")
        else:
            hud_lines.append(f"R (reach): {reach:4.2f}  [EXP]")

        # --- Quantity A: head height against gravity, 3D (EXPERIMENTAL) ------
        # Logged since 8.1; Stage 3 reads it in 8.2/8.3 (see below). Its
        # calibration is causal (only frames up to this one), so the record is
        # what a live device would have had.
        a_head, a_hip = quantity_a.measure(self._gravity, pf, t_deg, reliable, timestamp,
                                  self._a_standing_t, self._a_standing_knee)
        quantities["A_head"] = a_head
        quantities["A_hip"] = a_hip
        self._a_history.append((float(timestamp), a_head, a_hip))
        while self._a_history and self._a_history[0][0] < timestamp - self._a_history_s:
            self._a_history.popleft()
        if math.isnan(a_head):
            hud_lines.append("A (3D)   : -- (calibrating) [EXP]")
        else:
            hud_lines.append(f"A (3D)   : {a_head:4.2f}  [EXP]")

        # --- Quantity P: COM vs. support polygon (§3.4) ---------------------
        com_px, hull_px, p_offset, p_width = quantity_p.measure(
            pf, self._min_foot_visibility, self._contact_band_torso, self._com_hip_weight)
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
        i_seconds, i_disp = quantity_i.measure(self._immobility, pf, timestamp,
                                      self._visibility_threshold)
        quantities["I_still_s"] = i_seconds
        quantities["I_displacement"] = i_disp
        if math.isnan(i_seconds):
            hud_lines.append("I (still): --")
        else:
            eps = self._immobility.epsilon_torso
            hud_lines.append(
                f"I (still): {i_seconds:5.1f} s   (moved {i_disp:.3f} of {eps:.3f})"
            )

        # --- Stage 1: the kinematic trigger (§3.5) --------------------------
        # Evaluated only on reliable frames. Below the visibility threshold
        # the geometry is MediaPipe extrapolating anatomy outside the image,
        # and a decision taken on it would be a decision taken on nothing.
        event = None
        if reliable:
            event = self.machine.update(pf.frame_index, timestamp,
                                        t_deg, v_tps, p_offset,
                                        i_seconds, extension_ratio, h_ratio,
                                        a_head)
            # Phase 6a: the hip height, to read a get-up in progress at the
            # end (Stage3Evaluator.getup_rise).
            if (self.machine.stage is Stage.OBSERVING and self.machine.stage3 is not None
                    and pf.mid_hip is not None and pf.torso_length):
                self.machine.stage3.observe_hip(timestamp, float(pf.mid_hip[1]),
                                                float(pf.torso_length))
            # Phases 8.2/8.3: Quantity A. Whether it was calibrated on the
            # trigger frame is noted; when Stage 3 starts it gets the second
            # before, and then every observed frame.
            if event is not None:
                self._a_trigger = (float(event.timestamp), bool(self._gravity.calibrated))
            st3a = self.machine.stage3
            if self.machine.stage is Stage.OBSERVING and st3a is not None:
                if not st3a.a_armed and self._a_trigger is not None:
                    st3a.arm_a(self._a_trigger[0], self._a_trigger[1], self._a_history)
                elif st3a.a_armed:
                    st3a.observe_a(timestamp, a_head, a_hip)
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
        # leg-extension reset is NOT redundant — occlusion resets it while
        # the clock keeps running.
        self._centroid_ema.reset()
        self._torso_ema.reset()
        self._extension.reset()
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
        self._trunk_guard.clear()
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
