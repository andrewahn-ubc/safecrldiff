import copy

from safe_diffusion_cl_pilots.envs.shelve_contexts import swap_object_placements


def test_context_swap_is_deep_and_placement_only():
    source = [
        {"name": "wine_glass", "placement": {"pos": [0.1, 0.2], "box": [0.02, 0.03]}, "damage": 7},
        {"name": "flour_bag", "placement": {"pos": [0.8, 0.9], "box": [0.04, 0.05]}, "damage": 0},
        {"name": "cereal", "placement": {"pos": [0.5, 0.5]}, "mesh": "cereal"},
    ]
    before = copy.deepcopy(source)
    swapped = swap_object_placements(source, "wine_glass")
    assert source == before
    by_name = {item["name"]: item for item in swapped}
    assert by_name["wine_glass"]["placement"] == before[1]["placement"]
    assert by_name["flour_bag"]["placement"] == before[0]["placement"]
    assert by_name["wine_glass"]["damage"] == 7
    assert by_name["flour_bag"]["damage"] == 0
    assert by_name["cereal"] == before[2]
    by_name["wine_glass"]["placement"]["pos"][0] = -1
    assert source == before

