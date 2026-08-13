import numpy as np

from safe_diffusion_cl_pilots.envs.chunked_gym_wrapper import ChunkedGymWrapper


class PrimitiveEnv:
    def __init__(self, terminate_at: int = 3):
        self.steps = 0
        self.terminate_at = terminate_at

    def reset(self, **kwargs):
        self.steps = 0
        return np.asarray([0.0]), {}

    def step(self, action):
        self.steps += 1
        health = 100.0 - self.steps
        info = {
            "initial_per_object_health": {"wine_glass": 100.0, "flour_bag": 100.0},
            "per_object_health": {"wine_glass": health, "flour_bag": 100.0},
            "task_success": self.steps == self.terminate_at,
            "contacted_objects": ["wine_glass"],
        }
        return np.asarray([self.steps]), float(self.steps), self.steps == self.terminate_at, False, info

    def close(self):
        pass


def test_chunk_sums_reward_damage_and_stops_early():
    primitive = PrimitiveEnv(terminate_at=3)
    wrapper = ChunkedGymWrapper(primitive, horizon=8, epsilon_damage=2.5)
    wrapper.reset()
    _, reward, terminated, truncated, info = wrapper.step(np.zeros((8, 7)))
    assert reward == 1 + 2 + 3
    assert terminated and not truncated
    assert primitive.steps == info["primitive_steps_executed"] == 3
    assert info["designated_fragile_health_loss"] == 3
    assert info["total_environment_damage"] == 3
    assert info["damage_event"]

