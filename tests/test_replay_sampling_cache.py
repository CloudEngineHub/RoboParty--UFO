import pytest
import torch
from torch.utils._pytree import tree_leaves

from ufo.agents.common.buffers.storage import DictBuffer
from ufo.agents.common.buffers.trajectory import (
    ExpertTrajectoryBuffer,
    ReplayBuffer,
    get_idxs,
)


@pytest.mark.parametrize("trajectory", [False, True])
def test_replay_fixed_format_preserves_nested_values_and_reads_existing_defaults(tmp_path, trajectory):
    import json

    buffer = ReplayBuffer(capacity=8, n_dim=2) if trajectory else DictBuffer(capacity=8)
    shape = (3, 2, 1) if trajectory else (3, 1)
    values = torch.arange(6 if trajectory else 3, dtype=torch.float32).reshape(shape)
    buffer.extend({"done": torch.zeros(shape, dtype=torch.bool), "next": {"observation": {"state": values}}})
    buffer.save(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    assert "nested_key_separator" not in config
    for explicit_default in (False, True):
        if explicit_default:
            config["nested_key_separator"] = "-"
            config_path.write_text(json.dumps(config))
        loaded = type(buffer).load(tmp_path)
        assert len(loaded) == 3
        torch.testing.assert_close(loaded.storage["next"]["observation"]["state"][:3], values, rtol=0, atol=0)
    config["nested_key_separator"] = ":"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="separator"):
        type(buffer).load(tmp_path)


def uncached_sample(buffer, batch_size, seq_length):
    # Same sampler with a cold eligibility cache on every call.
    buffer._sampling_trajectories.clear()
    return buffer.sample(batch_size, seq_length)


def assert_same_sample(buffer, batch_size=8, seq_length=2):
    torch.manual_seed(321)
    actual = buffer.sample(batch_size, seq_length)
    cached = buffer._sampling_trajectories[seq_length + bool(buffer.output_key_tp1)]
    buffer.sample(batch_size, seq_length)
    assert buffer._sampling_trajectories[seq_length + bool(buffer.output_key_tp1)] is cached
    torch.manual_seed(321)
    expected = uncached_sample(buffer, batch_size, seq_length)
    for a, b in zip(tree_leaves(actual), tree_leaves(expected)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_expert_cache_reads_current_priorities():
    episodes = [dict(observation={'x': torch.arange(n).float().reshape(n, 1) + 100*i},
                     motion_id=torch.full((n,), i)) for i, n in enumerate([3, 7, 12])]
    buffer = ExpertTrajectoryBuffer(episodes, seq_length=2)
    buffer._get_idxs = get_idxs
    assert_same_sample(buffer)
    assert_same_sample(buffer, seq_length=4)
    buffer.update_priorities(torch.tensor([0., 0., 1.]), torch.arange(3))
    assert_same_sample(buffer)
    assert (buffer.sample(8)['observation']['x'] >= 200).all()


def test_replay_cache_invalidated_by_extend_and_wrap(tmp_path):
    buffer = ReplayBuffer(capacity=24, n_dim=2, seq_length=2)
    buffer._get_idxs = get_idxs
    for i in range(4):
        done = torch.zeros(8, 2, 1, dtype=torch.bool)
        done[1] = True
        data = dict(observation={'x': torch.arange(16).float().reshape(8, 2, 1) + 100*i}, done=done)
        buffer.extend(data)
        if i == 2:
            # get_full_buffer also recomputes boundaries before a later sample.
            buffer.get_full_buffer()
        assert_same_sample(buffer)
        assert_same_sample(buffer, seq_length=4)
    buffer.save(tmp_path)
    loaded = ReplayBuffer.load(tmp_path)
    loaded._get_idxs = get_idxs
    torch.manual_seed(712)
    expected = buffer.sample(8)
    torch.manual_seed(712)
    actual = loaded.sample(8)
    for a, b in zip(tree_leaves(actual), tree_leaves(expected)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_no_eligible_trajectory_still_raises():
    buffer = ReplayBuffer(capacity=12, n_dim=2, seq_length=2)
    buffer.extend(dict(observation={'x': torch.zeros(4, 2, 1)}, done=torch.ones(4, 2, 1,dtype=torch.bool)))
    with pytest.raises(ValueError, match='No trajectories'):
        buffer.sample(8)


@pytest.mark.parametrize("next_states", [False, True])
def test_replay_readiness_uses_actual_sequence_length_without_consuming_rng(next_states):
    buffer = ReplayBuffer(capacity=6, n_dim=2, output_key_tp1=["observation"] if next_states else [])
    buffer._get_idxs = get_idxs
    rng = torch.get_rng_state()
    assert not buffer.can_sample()
    buffer.extend(dict(observation={"x": torch.zeros(1, 2, 1)}, done=torch.zeros(1, 2, 1, dtype=torch.bool)))
    assert buffer.can_sample() is (not next_states)
    buffer.extend(dict(observation={"x": torch.ones(1, 2, 1)}, done=torch.zeros(1, 2, 1, dtype=torch.bool)))
    assert buffer.can_sample()
    assert buffer.can_sample(2) is (not next_states)
    assert not buffer.can_sample(64)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert_same_sample(buffer, batch_size=8, seq_length=1)


def test_replay_readiness_rechecks_boundaries_after_wrap():
    buffer = ReplayBuffer(capacity=6, n_dim=2)
    buffer.extend(dict(observation={"x": torch.arange(12).reshape(6, 2, 1)}, done=torch.ones(6, 2, 1, dtype=torch.bool)))
    assert buffer._is_full and buffer._idx == 0
    assert not buffer.can_sample()
    buffer.extend(dict(observation={"x": torch.zeros(2, 2, 1)}, done=torch.zeros(2, 2, 1, dtype=torch.bool)))
    assert buffer.can_sample()
    assert not buffer.can_sample(2)


def test_sealing_restored_replay_tail_prevents_transitions_across_reset(tmp_path):
    buffer = ReplayBuffer(capacity=8, n_dim=2)
    buffer.extend(dict(observation={"x": torch.arange(3).reshape(3, 1, 1).expand(-1, 2, -1)}, done=torch.zeros(3, 2, 1, dtype=torch.bool)))
    buffer.save(tmp_path)
    restored = ReplayBuffer.load(tmp_path)
    restored.storage["done"][(restored._idx - 1) % restored.capacity] = True
    restored.extend(
        dict(observation={"x": (torch.arange(2) + 100).reshape(2, 1, 1).expand(-1, 2, -1)}, done=torch.zeros(2, 2, 1, dtype=torch.bool))
    )
    assert restored.can_sample()
    transitions = restored.get_full_buffer()
    torch.testing.assert_close(
        transitions["next"]["observation"]["x"] - transitions["observation"]["x"], torch.ones(6, 1, dtype=torch.long), rtol=0, atol=0
    )


@pytest.mark.parametrize("field,value", [("_idx", None), ("_is_full", None), ("_idx", -1), ("_is_full", 1)])
def test_replay_requires_current_cursor_metadata(tmp_path, field, value):
    import json

    buffer = ReplayBuffer(capacity=12, n_dim=2, seq_length=2)
    buffer.extend(dict(observation={'x': torch.arange(8).reshape(4, 2, 1)}, done=torch.zeros(4, 2, 1, dtype=torch.bool)))
    buffer.save(tmp_path)
    restored = ReplayBuffer.load(tmp_path)
    assert len(restored) == 4 and not restored._is_full
    torch.testing.assert_close(restored.storage['observation']['x'], buffer.storage['observation']['x'])
    path = tmp_path / 'config.json'
    cfg = json.loads(path.read_text())
    if value is None:
        del cfg[field]
    else:
        cfg[field] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises((KeyError, ValueError)):
        ReplayBuffer.load(tmp_path)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dim", [0, 1])
def test_tree_concat_preserves_values_structure_and_leaf_gradients(nested, dim):
    from ufo.agents.common.buffers.trajectory import tree_concat

    tensors = [torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_(), torch.ones(2, 3, requires_grad=True)]
    trees = [{"nested": {"state": x}, "action": x * 2} for x in tensors] if nested else tensors
    actual = tree_concat(trees, dim=dim)
    expected = torch.cat(tensors, dim=dim)
    if nested:
        torch.testing.assert_close(actual["action"], expected * 2, rtol=0, atol=0)
        actual = actual["nested"]["state"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    for tensor in tensors:
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor), rtol=0, atol=0)


@pytest.mark.parametrize("trajectory", [False, True])
def test_replay_device_override_is_applied_before_allocating_storage(tmp_path, trajectory):
    import json

    buffer_type = ReplayBuffer if trajectory else DictBuffer
    buffer = buffer_type(capacity=8)
    buffer.extend(dict(observation={"x": torch.arange(3).reshape(3, 1)}, done=torch.zeros(3, 1, dtype=torch.bool)))
    buffer.save(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["device"] = "cuda:63"
    path.write_text(json.dumps(config))
    restored = buffer_type.load(tmp_path, device="cpu")
    assert restored.device == "cpu"
    assert all(value.device.type == "cpu" for value in tree_leaves(restored.storage))
    torch.testing.assert_close(restored.storage["observation"]["x"][:3], buffer.storage["observation"]["x"][:3], rtol=0, atol=0)


@pytest.mark.parametrize("n_dim,envs", [(1, 1), (2, 1), (2, 3)])
def test_singleton_dimensions_preserve_transition_shapes_and_boundaries(n_dim, envs):
    buffer = ReplayBuffer(capacity=4, n_dim=n_dim, seq_length=1)
    for step in range(6):
        observations = torch.stack([torch.full((2,), step * 10 + env) for env in range(envs)])[None]
        done = torch.full((1, envs, 1), step == 1, dtype=torch.bool)
        if n_dim == 1:
            observations, done = observations[:, 0], done[:, 0]
        buffer.extend({"observation": observations, "done": done})
        if step == 0:
            assert not buffer.can_sample()
        if step == 3:
            transitions = buffer.get_full_buffer()
            assert set(transitions["observation"][:, 0].tolist()) == {t * 10 + env for t in (0, 2) for env in range(envs)}
    # After wrap, the retained sequence is 2, 3, 4, 5 for every environment.
    full = buffer.get_full_buffer()
    assert full["observation"].shape == full["next"]["observation"].shape == (3 * envs, 2)
    assert set(full["observation"][:, 0].tolist()) == {step * 10 + env for step in (2, 3, 4) for env in range(envs)}
    torch.testing.assert_close(full["next"]["observation"] - full["observation"], torch.full((3 * envs, 2), 10))
    sampled = buffer.sample(64)
    assert sampled["observation"].shape == (64, 2)
    torch.testing.assert_close(sampled["next"]["observation"] - sampled["observation"], torch.full((64, 2), 10))
