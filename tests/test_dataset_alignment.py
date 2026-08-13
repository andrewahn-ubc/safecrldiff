import numpy as np

from safe_diffusion_cl_pilots.data.build_lowdim_dataset import ChunkDataset
from safe_diffusion_cl_pilots.data.replay_oopsie_hdf5 import (
    ReplayIntegrityError,
    aligned_state_actions,
)


def test_official_alignment_discards_first_action_and_final_state():
    states = np.arange(30).reshape(5, 6)
    actions = np.arange(35).reshape(5, 7)
    paired_states, paired_actions = aligned_state_actions(states, actions)
    np.testing.assert_array_equal(paired_states, states[:-1])
    np.testing.assert_array_equal(paired_actions, actions[1:])


def test_alignment_rejects_unequal_counts():
    try:
        aligned_state_actions(np.zeros((4, 3)), np.zeros((5, 7)))
    except ReplayIntegrityError:
        pass
    else:
        raise AssertionError("unequal state/action counts must fail")


def test_chunk_dataset_never_crosses_episode_boundary(tmp_path):
    path = tmp_path / "data.npz"
    states = np.arange(5, dtype=np.float32)[:, None]
    actions = np.repeat(np.arange(5, dtype=np.float32)[:, None], 7, axis=1)
    np.savez(path, states=states, actions=actions, traj_lengths=np.asarray([2, 3]))
    dataset = ChunkDataset(path, horizon=4)
    _, last_first_episode = dataset[1]
    _, first_second_episode = dataset[2]
    np.testing.assert_array_equal(last_first_episode[:, 0], [1, 1, 1, 1])
    np.testing.assert_array_equal(first_second_episode[:, 0], [2, 3, 4, 4])

