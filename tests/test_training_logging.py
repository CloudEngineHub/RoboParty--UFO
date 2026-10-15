"""Logging keeps collective participation and counts actual completed samples."""

import csv
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from ufo.training import trainer as current
from ufo.training.logging import CSVLogger
from ufo.training.trainer import EveryNStepsChecker


@pytest.mark.parametrize("batched", [False, True])
def test_csv_adds_phase_metrics_and_resumes_with_existing_header(tmp_path, batched):
    path = tmp_path / "train.csv"
    rows = [
        {"timestep": 9007199254740993, "tech_te_loss": 0.5, "label": 'dance,一\n"pose"'},
        {"timestep": 9007199254740994, "actor_loss": -1.25, "critic_loss": 0.75},
        {"timestep": 9007199254740995, "tech_te_loss": 0.25, "actor_loss": -2.5},
    ]
    logger = CSVLogger(path)
    for index, row in enumerate(rows):
        if index == 2:
            logger = CSVLogger(path)
        logger.log({key: [value] for key, value in row.items()} if batched else row)
    with path.open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    fields = set().union(*(row.keys() for row in rows))
    assert actual == [{key: str(row[key]) if key in row else "" for key in fields} for row in rows]


def test_csv_resume_respects_old_column_order_and_empty_files(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("z,a\nleft,right\n")
    CSVLogger(path).log({"a": "new-right", "z": "new-left"})
    with path.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"a": "right", "z": "left"}, {"a": "new-right", "z": "new-left"}]
    path.write_text("")
    CSVLogger(path).log({"a": 1})
    assert path.read_text() == "a\n1\n"


def test_csv_failed_schema_rewrite_preserves_old_rows_and_cleans_temporary(tmp_path):
    path = tmp_path / "train.csv"
    logger = CSVLogger(path)
    logger.log({"old": 1})
    before = path.read_bytes()
    with patch.object(csv.DictWriter, "writerows", side_effect=OSError("disk write failed")):
        with pytest.raises(OSError, match="disk write failed"):
            logger.log({"new": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    logger.log({"new": 2})
    with path.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"new": "", "old": "1"}, {"new": "2", "old": ""}]


def test_csv_rejects_mixed_batch_before_changing_file(tmp_path):
    path = tmp_path / "train.csv"
    logger = CSVLogger(path)
    logger.log({"a": 1})
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="mixed types"):
        logger.log({"a": [2, 3], "new": 4})
    assert path.read_bytes() == before


def make_workspace(*, writer, world_size):
    ws = current.Trainer.__new__(current.Trainer)
    ws.cfg = SimpleNamespace(
        distributed_sync=world_size > 1,
        distributed_world_size=world_size,
        online_parallel_envs=1024,
        buffer_size=5120000,
        agent=SimpleNamespace(train=SimpleNamespace(batch_size=1024)),
        log_torso_contact_forces=True,
        torso_contact_force_threshold=1.0,
        use_wandb=True,
    )
    ws.agent = SimpleNamespace(device="cpu", cfg=SimpleNamespace(model=SimpleNamespace(archi=SimpleNamespace())))
    ws.distributed_world_size = world_size
    ws._write_shared_artifacts = writer
    ws._optimizer_steps = 16
    ws.train_env = object()
    ws.train_logger = Mock() if writer else None
    return ws


def test_nonwriter_participates_in_reduce_without_materializing_logs():
    ws = make_workspace(writer=False, world_size=8)
    checker = EveryNStepsChecker(0, 384000)
    # Invalid metric values would fail if this rank tried to format them.
    metrics = {"loss": object()}
    with (
        patch.object(current, "average_metrics", return_value=metrics) as reduce,
        patch.object(current, "torso_contact_force_metrics") as contact,
        patch.object(current.wandb, "log") as wandb_log,
        patch.object(current.time, "time", return_value=20.0),
    ):
        now = ws._log_training(metrics, {}, 48128, 385024, 0.0, 10.0, checker, fps_start_step=8192)
    reduce.assert_called_once_with(metrics)
    contact.assert_not_called()
    wandb_log.assert_not_called()
    assert checker.last_step == 385024 and now == 20.0


@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("resume_step", [0, 192000000])
def test_fps_counts_first_resumed_and_rounded_intervals(world_size, resume_step):
    ws = make_workspace(writer=True, world_size=world_size)
    batch = 1024 * world_size
    checker = EveryNStepsChecker(resume_step, 384000)
    metrics = {"loss": torch.tensor(8.0)}
    started = 10.0
    with (
        patch.object(current, "average_metrics", side_effect=lambda values: values),
        patch.object(current, "torso_contact_force_metrics", return_value={}),
        patch.object(current.wandb, "log") as wandb_log,
        patch.object(current.time, "time", side_effect=[20.0, 35.0]),
    ):
        # Logging runs after the current collection, including on the first interval.
        step = resume_step + 3 * batch
        boundary = ws._log_training(
            metrics, {"loss": 2}, step // world_size, step, started, started, checker,
            fps_start_step=resume_step,
        )
        first = ws.train_logger.log.call_args.args[0]
        assert first["FPS"] == pytest.approx(4 * batch / 10.0)
        assert first["loss"] == 4.0
        assert first["timestep"] == step
        assert boundary == 20.0
        # The second interval includes prior log I/O and uses actual batch-aligned steps.
        previous_completed = step + batch
        step += 47 * batch
        boundary = ws._log_training(
            metrics, {"loss": 2}, step // world_size, step, started, boundary, checker,
            fps_start_step=previous_completed,
        )
        second = ws.train_logger.log.call_args.args[0]
        assert second["FPS"] == pytest.approx(47 * batch / 15.0)
        assert second["duration [minutes]"] == pytest.approx(25.0 / 60.0)
        assert second["timestep"] == checker.last_step == step
        assert boundary == 35.0 and wandb_log.call_count == 2
