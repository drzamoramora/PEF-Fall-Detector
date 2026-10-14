"""The §3.5 funnel assembled from the configuration.

Every threshold the three stages read comes from ``config.yaml`` through
here, and only here; ``pipeline.FramePipeline`` asks for a ready machine.
"""

from __future__ import annotations

from .machine import FallStateMachine
from .stage2 import Stage2Evaluator
from .stage3 import Stage3Evaluator
from .trigger import Stage1Trigger


def build_machine(cfg, labelling: bool) -> FallStateMachine:
    """The funnel for one run.

    ``labelling``: decide each event from how the episode ENDED (a recorded
    clip); otherwise resolve as soon as the evidence allows (a live source).
    """
    qa = cfg.quantity_a
    # Phases 8.2 and 8.3: Stage 3 reads A (see Stage3Evaluator.finalise).
    # Each rule has its own switch; off, A is only recorded (8.1).
    use_a_sev = bool(qa.severity_from_a)
    use_a_reach = bool(qa.reach_check)
    a_band_floor = float(qa.band_floor) if use_a_sev else 0.0
    a_band_standing = float(qa.band_standing)
    a_reach_floor = float(qa.floor_reached) if use_a_reach else 0.0
    a_hip_raised = float(qa.hip_raised)
    a_before = float(qa.reach_before_s)
    a_after = float(qa.reach_after_s)
    # Phase 8.4: A as an additional Stage-1 condition (Stage1Trigger's
    # A drop). Off, the trigger never reads A and 8.3 is reproduced.
    use_a_trigger = bool(qa.trigger_from_a)
    a_trigger_high = float(qa.trigger_high) if use_a_trigger else 0.0
    a_trigger_low = float(qa.trigger_low)
    a_trigger_window = float(qa.trigger_window_s)
    # Stage 1 (§3.5). The formulation is a config choice, not a code
    # decision — see state_machine for the measurements behind that.
    return FallStateMachine(
        Stage1Trigger(
            formulation=str(cfg.stage1.trigger_formulation),
            threshold_t_deg=float(cfg.stage1.threshold_T_deg),
            threshold_v_tps=float(cfg.stage1.threshold_V),
            hold_s=float(cfg.stage1.trigger_hold_s),
            confirm_window_s=float(cfg.stage1.confirm_window_s),
            threshold_score=float(cfg.stage1.trigger_score),
            peak_window_s=float(cfg.stage1.trigger_peak_window_s),
            threshold_h_ratio=float(cfg.experimental.trigger_H_ratio),
            threshold_h_erect=float(cfg.experimental.trigger_H_erect),
            h_sequence_window_s=float(
                cfg.experimental.trigger_H_sequence_window_s),
            threshold_a_high=a_trigger_high,
            threshold_a_low=a_trigger_low,
            a_window_s=a_trigger_window,
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
            defer_to_end=bool(labelling),
            lying_t_deg=float(cfg.state_display.lying_T_deg),
            max_extension=float(cfg.state_display.max_extension_ratio),
            threshold_h_ratio_recovery=float(
                cfg.experimental.recovery_H_ratio),
            threshold_h_ratio_lying=float(cfg.experimental.trigger_H_ratio),
            persistent_still_s=float(cfg.stage3.persistent_still_s),
            persistent_fraction=float(cfg.stage3.persistent_immobility_fraction),
            getup_rise_torsos=float(cfg.stage3.getup_rise_torsos),
            getup_window_s=float(cfg.stage3.getup_window_s),
            a_band_floor=a_band_floor,
            a_band_standing=a_band_standing,
            a_reach_floor=a_reach_floor,
            a_hip_raised=a_hip_raised,
            a_reach_before_s=a_before,
            a_reach_after_s=a_after,
        ),
        cooldown_s=float(cfg.stage3.cooldown_seconds),
        cooldown_after_rejection=bool(cfg.stage3.cooldown_after_rejection),
    )
