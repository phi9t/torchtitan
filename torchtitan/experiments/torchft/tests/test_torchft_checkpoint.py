# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import tempfile
import time
import unittest
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.observability.run_evidence import RunEvidence


@contextmanager
def torchft_checkpoint_evidence(dump_folder: str, test_name: str):
    environment = {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "TORCHTITAN_RUN_ID": f"torchft-checkpoint-{test_name}",
        "TORCHTITAN_ATTEMPT_ID": "attempt-1",
    }
    with mock.patch.dict(os.environ, environment, clear=False):
        with RunEvidence(
            RunEvidence.Config(),
            dump_folder=dump_folder,
            job_config={"training": {"steps": 8}},
            role="trainer",
            actor_id="torchft",
        ) as evidence:
            yield evidence


def torchft_checkpoint_artifact_rows(evidence: RunEvidence) -> list[dict]:
    index_path = next(
        (
            Path(evidence.dump_folder)
            / "run_evidence"
            / evidence.run_id
            / evidence.attempt_id
            / "indexes"
        ).glob("artifacts.*.jsonl")
    )
    return [
        row
        for row in (json.loads(line) for line in index_path.read_text().splitlines())
        if row["kind"] == "torchtitan.checkpoint"
    ]


class FakeOptimizersContainer:
    def __init__(self):
        self._fake_param = torch.tensor([1.0], dtype=torch.float32)

    def state_dict(self):
        return {"fake_param": self._fake_param}

    def load_state_dict(self, sd: dict):
        if "fake_param" in sd:
            self._fake_param = sd["fake_param"]

    def init_cache_state_dict(self):
        pass


class FakeLRSchedulersContainer:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass


class FakeDataLoader(DataLoader):
    def __init__(self):
        super().__init__(dataset=[], batch_size=1)

    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass


class DummyFuture:
    def __new__(cls):
        # Return a Mock that mimics Future instead of an instance of this class
        # That allows isinstance(DummyFuture, Future) to pass
        instance = mock.Mock(spec=Future)
        instance.result = mock.Mock()

        return instance


def fake_async_save(*args, **kwargs):
    return DummyFuture()


class DummyFTManager:
    """Mimics TorchFTManager for testing without requiring torchft."""

    def __init__(self, enabled=True, replica_id=0, participating_rank=0):
        self._enabled = enabled
        self.replica_id = replica_id
        if enabled:
            self.manager = mock.MagicMock()
            self.manager.participating_rank.return_value = participating_rank
        else:
            self.manager = None

    @property
    def enabled(self):
        return self._enabled


class TestFTCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.base_temp_dir, self._testMethodName)
        os.makedirs(self.test_folder, exist_ok=True)
        self.model_parts = [nn.Linear(2, 2)]
        self.states = {"trainer": torch.tensor([1.2347])}
        self.optimizers = FakeOptimizersContainer()
        self.lr_schedulers = FakeLRSchedulersContainer()
        self.data_loader = FakeDataLoader()
        self.ft_manager = DummyFTManager(enabled=True, participating_rank=0)
        self.patcher_group = mock.patch(
            "torch.distributed.new_group", return_value="pg"
        )
        self.patcher_group.start()
        # Patch process group destruction
        self.patcher_destroy = mock.patch("torch.distributed.destroy_process_group")
        self.patcher_destroy.start()

    def tearDown(self):
        self.patcher_group.stop()
        self.patcher_destroy.stop()
        shutil.rmtree(self.base_temp_dir)
        time.sleep(0.1)

    @mock.patch("torch.cuda.Stream")
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_torchft_async_save_calls_maybe_wait_for_saving(
        self,
        mock_async_save,
        mock_cuda_stream,
    ):
        """
        Test that with FT enabled, AsyncMode.ASYNC via FT triggers correct waits.
        """
        config = TorchFTCheckpointManager.Config(
            enable=True,
            async_mode="async",
            folder=self.test_folder,
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
            enable_ft_dataloader_checkpoints=True,
        )
        manager = TorchFTCheckpointManager(
            config,
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            sd_adapter=None,
            base_folder=self.test_folder,
            ft_manager=self.ft_manager,
        )

        # Initially no future
        self.assertIsNone(manager.save_future)
        manager.save(curr_step=5, last_step=False)
        self.assertIsNotNone(manager.save_future)

        manager.save_future.result.assert_not_called()
        prev_future = manager.save_future
        manager.save(curr_step=6, last_step=False)
        prev_future.result.assert_called_once()
        self.assertIsNotNone(manager.save_future)
        manager.save_future.result.assert_not_called()

        manager.close()

    @mock.patch("torchtitan.components.checkpoint.dcp.async_save")
    def test_torchft_async_full_checkpoint_evidence_closes_before_next_save(
        self, mock_async_save
    ):
        def async_save(*args, **kwargs):
            checkpoint_id = kwargs["checkpoint_id"]
            future = DummyFuture()
            future.result.side_effect = lambda: os.makedirs(
                checkpoint_id, exist_ok=True
            )
            return future

        mock_async_save.side_effect = async_save
        config = TorchFTCheckpointManager.Config(
            enable=True,
            async_mode="async",
            folder=self.test_folder,
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
            enable_ft_dataloader_checkpoints=False,
        )
        manager = TorchFTCheckpointManager(
            config,
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            sd_adapter=None,
            base_folder=self.test_folder,
            ft_manager=self.ft_manager,
        )

        with torchft_checkpoint_evidence(
            self.test_folder, self._testMethodName
        ) as evidence:
            manager.save(curr_step=5)
            manager.save(curr_step=6)
            manager.maybe_wait_for_saving()
            rows = torchft_checkpoint_artifact_rows(evidence)

        self.assertEqual(
            [(row["step"], row["state"]) for row in rows],
            [(5, "declared"), (5, "complete"), (6, "declared"), (6, "complete")],
        )
        manager.close()


if __name__ == "__main__":
    unittest.main()
