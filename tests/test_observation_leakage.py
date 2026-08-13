import numpy as np

from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation


def observation(health: float) -> dict[str, np.ndarray | float]:
    value: dict[str, np.ndarray | float] = {
        "robot0_joint_pos": np.arange(7),
        "robot0_joint_vel": np.arange(7) / 10,
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "robot0_gripper_qpos": np.asarray([0.1, -0.1]),
        "target_mat_pos": np.asarray([0.7, 0.1, 0.8]),
        "target_mat_quat": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "wine_glass_health": health,
        "context_id": 99,
    }
    for index, name in enumerate(("cereal", "wine_1", "wine_glass", "wine_2", "flour_bag")):
        value[f"{name}_pos"] = np.asarray([index, index + 1, index + 2]) / 10
        value[f"{name}_quat"] = np.asarray([1.0, 0.0, 0.0, 0.0])
        value[f"{name}_linvel"] = np.asarray([0.01, 0.02, 0.03])
        value[f"{name}_angvel"] = np.asarray([0.04, 0.05, 0.06])
    return value


def test_allowlisted_observation_has_no_health_or_context_leakage():
    extractor = ObjectCentricObservation()
    extractor.assert_no_leakage()
    low_health = extractor.extract(observation(2.0))
    high_health = extractor.extract(observation(100.0))
    assert low_health.shape == (109,)
    np.testing.assert_array_equal(low_health, high_health)

