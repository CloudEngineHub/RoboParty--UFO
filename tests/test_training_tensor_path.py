from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ufo.agents.common.buffers.storage import DictBuffer
from ufo.training.distributed import average_metrics
from ufo.training.logging import _assert_finite, torso_contact_force_metrics


@pytest.mark.parametrize("as_numpy", [False, True])
def test_replay_owns_nested_data_across_wrap(as_numpy):
    buffer = DictBuffer(capacity=5)
    source = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    convert = lambda t: t.numpy() if as_numpy else t
    buffer.extend({"state": convert(source), "next": {"state": convert(source + 10)}})
    source.fill_(-100)
    torch.testing.assert_close(buffer.storage["state"][:3], torch.arange(6, dtype=torch.float64).reshape(3, 2))
    second = torch.arange(6, 12, dtype=torch.float32).reshape(3, 2)
    buffer.extend({"state": convert(second), "next": {"state": convert(second + 10)}})
    second.fill_(-100)
    expected = torch.tensor([[10, 11], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=torch.float64)
    torch.testing.assert_close(buffer.storage["state"], expected)
    torch.testing.assert_close(buffer.storage["next"]["state"], expected + 10)


def test_packed_metrics_preserve_shapes_values_and_inputs():
    inputs = {"loss": torch.tensor(2.0), "vector": torch.tensor([4.0, 6.0]), "count": torch.tensor(8), "note": "fb"}
    with (
        patch("ufo.training.distributed.is_distributed", return_value=True),
        patch("ufo.training.distributed.dist.get_world_size", return_value=2),
        patch("ufo.training.distributed.dist.all_reduce", side_effect=lambda value, **kw: value.mul_(2)) as reduce,
    ):
        result = average_metrics(inputs)
    assert reduce.call_count == 1
    for key in ("loss", "vector", "count"):
        torch.testing.assert_close(result[key], inputs[key].float())
    assert result["note"] == "fb"
    result["vector"].zero_()
    assert inputs["vector"].tolist() == [4.0, 6.0]


@pytest.mark.parametrize("bad", [torch.tensor(float("nan")), np.array([float("inf")]), float("-inf")])
def test_finite_checks_retain_nested_error_details(bad):
    context = dict(label="update", rank=2, local_time=4, global_time=32, optimizer_steps=1)
    _assert_finite({"ok": [torch.ones(2), np.ones(3), 2.0]}, **context)
    with pytest.raises(FloatingPointError, match=r"update.nested\[0\]"):
        _assert_finite({"ok": torch.ones(2), "nested": [bad]}, **context)


def test_finite_check_keeps_mixed_dtype_range_and_nested_details():
    context = {"label": "update", "rank": 0, "local_time": 0, "global_time": 0, "optimizer_steps": 1}
    values = {"large": torch.tensor(1e300, dtype=torch.float64), "small": torch.ones(4, dtype=torch.float16)}
    _assert_finite(values, **context)
    values["small"][2] = float("inf")
    with pytest.raises(FloatingPointError, match="update.small"):
        _assert_finite(values, **context)


def test_batched_metric_accumulation_preserves_sparse_counts_and_snapshots():
    from ufo.training.trainer import _accumulate_metrics

    total, counts = None, {}
    total, counts = _accumulate_metrics(total, counts, {"a": torch.tensor(2.0), "b": torch.tensor([1, 2])})
    previous = total["a"]
    value = torch.tensor(3.0, dtype=torch.float64)
    total, counts = _accumulate_metrics(total, counts, {"a": value, "b": torch.tensor([2, 3]), "c": torch.tensor(4.0)})
    total, counts = _accumulate_metrics(total, counts, {"b": torch.tensor([3.0, 4.0])})
    torch.testing.assert_close(total["a"], torch.tensor(5.0), rtol=0, atol=0)
    torch.testing.assert_close(total["b"], torch.tensor([6.0, 9.0]), rtol=0, atol=0)
    assert total["a"].dtype == torch.float32 and previous.item() == 2.0 and value.item() == 3.0
    assert counts == {"a": 2, "b": 3, "c": 1}


def test_torso_logging_preserves_signed_force_and_strict_contact_threshold():
    force = torch.tensor([[[3.0, 4.0, 0.0]], [[0.0, 0.0, -2.0]], [[0.0, 0.0, 1.0]]], requires_grad=True)
    position = torch.tensor([[[0.0, 0.0, 0.2]], [[0.0, 0.0, 0.4]], [[0.0, 0.0, 0.6]]])
    env = SimpleNamespace(torso_index=0, contact_forces=force, body_pos=position)
    before = force.clone(), position.clone(), torch.get_rng_state()
    metrics = torso_contact_force_metrics(env, enabled=True, threshold=1.0)
    assert metrics["torso_contact_force_contact_count"] == 2
    assert metrics["torso_contact_force_contact_frac"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["torso_contact_force_z_mean"] == pytest.approx(-1 / 3, abs=1e-6)
    assert metrics["torso_z_contact_mean"] == pytest.approx(0.3)
    assert metrics == torso_contact_force_metrics(SimpleNamespace(_env=env), enabled=True, threshold=1.0)
    no_contact = torso_contact_force_metrics(env, enabled=True, threshold=5.0)
    assert no_contact["torso_contact_force_contact_count"] == 0
    assert all(no_contact[key] == 0 for key in ("torso_z_contact_mean", "torso_z_contact_min", "torso_z_contact_p05"))
    assert torso_contact_force_metrics(object(), enabled=False, threshold=1.0) == {}
    assert torso_contact_force_metrics(object(), enabled=True, threshold=1.0) == {}
    for actual, expected in zip((force, position, torch.get_rng_state()), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert force.grad is None
