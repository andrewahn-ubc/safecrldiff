from safe_diffusion_cl_pilots.data.select_smoke_subset import (
    expanded_smoke_subset,
    select_smoke_subset,
)


def test_smoke_subset_is_stable_and_source_dependent():
    episodes = [f"demo_{index}" for index in range(45)]
    first = select_smoke_subset("safe.hdf5", episodes, 10, seed=0)
    assert first == select_smoke_subset("safe.hdf5", reversed(episodes), 10, seed=0)
    assert first != select_smoke_subset("unsafe.hdf5", episodes, 10, seed=0)
    assert len(first) == len(set(first)) == 10


def test_expansion_is_cumulative():
    candidates = expanded_smoke_subset("safe.hdf5", map(str, range(45)))
    assert [len(value) for value in candidates] == [10, 15, 20]
    assert candidates[0] == candidates[1][:10]
    assert candidates[1] == candidates[2][:15]

