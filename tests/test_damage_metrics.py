from safe_diffusion_cl_pilots.evaluation.metrics import (
    calibrate_damage_threshold,
    pilot0_gate,
    safety_specific_forgetting,
)


def test_damage_threshold_stays_at_one_when_noop_is_stable():
    threshold, evidence = calibrate_damage_threshold([0.0] * 20)
    assert threshold == 1.0
    assert evidence["noop_rollouts"] == 20


def test_damage_threshold_increases_above_noop_tail():
    threshold, _ = calibrate_damage_threshold([0.0] * 19 + [1.2])
    assert threshold > 1.0


def test_pilot0_yellow_proceeds_when_pooled_modes_exist():
    safe = {"p_damage_given_success": 0.2, "mean_designated_health_loss": 1.0}
    unsafe = {"p_damage_given_success": 0.25, "mean_designated_health_loss": 1.1}
    pooled = {"safe_success_count": 20, "damaging_success_count": 10}
    gate = pilot0_gate(safe, unsafe, pooled, True, True)
    assert gate["color"] == "YELLOW" and gate["proceed_gpu"]


def test_safety_forgetting_requires_residual_competence():
    pre = {"task_success": 0.8, "p_damage_given_success": 0.05}
    post = {"task_success": 0.5, "p_damage_given_success": 0.2}
    assert safety_specific_forgetting(pre, post, 0.3, 0.6)
    collapsed = {"task_success": 0.2, "p_damage_given_success": 0.8}
    assert not safety_specific_forgetting(pre, collapsed, 0.3, 0.8)

