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
from dataclasses import dataclass, field

import numpy as np

from .audit_log import csv_column
from .config import Config
from .person_state import classify_state
from .pose_frontend import (
    LEFT_ANKLE,
    RIGHT_ANKLE,
    PoseFrame,
    PoseFrontend,
)
from .quantities import (
    ExponentialMovingAverage,
    VelocityEstimator,
    centroid,
    trunk_band,
    trunk_inclination_deg,
)


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
        hud_lines: ready-to-draw text lines for the PEF-Lab overlay.
    """

    pose: PoseFrame
    reliable: bool
    quantities: dict[str, float] = field(default_factory=dict)
    state: str | None = None
    centroid_px: np.ndarray | None = None
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

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._visibility_threshold = float(cfg.pose.visibility_threshold)
        self._max_gap_s = float(cfg.stage1.history_max_gap_s)
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
        self._last_timestamp: float | None = None
        # Set whenever continuity is broken; the next analysed frame starts a
        # fresh motion history instead of differentiating across the break.
        self._discontinuous = True
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
            )
        return self.analyze(self._frontend.process(frame_bgr, frame_index, timestamp))

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
            return FrameResult(pose=pf, reliable=False)

        # A jump in time — forward past the gap threshold, or backwards at
        # all — means the operator moved elsewhere in the recording.
        if self._last_timestamp is not None and (
            timestamp - self._last_timestamp > self._max_gap_s
            or timestamp < self._last_timestamp
        ):
            self._discontinuous = True

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

        if not reliable:
            hud_lines.append("LOW VISIBILITY - values unreliable")

        return FrameResult(
            pose=pf,
            reliable=reliable,
            quantities=quantities,
            state=state,
            centroid_px=np.asarray(smoothed_centroid, dtype=float),
            hud_lines=hud_lines,
        )

    # --------------------------------------------------------------- helpers
    def _hip_ankle_extension(self, pf: PoseFrame, dt: float) -> float:
        """Vertical hip-to-ankle extent in torso units (posture feature, 2.4).

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
        mean_ankle_y = float(pf.landmarks[[LEFT_ANKLE, RIGHT_ANKLE], 1].mean())
        # y grows downward: ankles below hips give a positive extent.
        raw_ratio = (mean_ankle_y - float(pf.mid_hip[1])) / torso_raw
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
        self._last_timestamp = None
        self._discontinuous = False

    def close(self) -> None:
        """Release the pose front-end, if one was ever created."""
        if self._frontend is not None:
            self._frontend.close()
            self._frontend = None
