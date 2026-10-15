# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.glm53_flash.checkpoint import (
    Glm53FlashLoRACheckpointManager,
)
from torchtitan.experiments.glm53_flash.sharded_hf_load import (
    _copy_one_to_one_tensor,
    ShardedLoadStats,
    TensorSliceTarget,
)
from torchtitan.experiments.glm53_flash_4gpu_stream.analysis import (
    estimate_streaming_h2d,
)
from torchtitan.experiments.glm53_flash_4gpu_stream.config_registry import (
    glm53_flash_countdown_lora_stream,
    glm53_flash_countdown_lora_truncated_plain,
    glm53_flash_countdown_lora_truncated_stream,
)
from torchtitan.experiments.glm53_flash_4gpu_stream.grad_probe import (
    compare_probe_dirs,
    max_abs_delta,
    summarize_tensor,
)
from torchtitan.experiments.glm53_flash_4gpu_stream.model import (
    StreamGlm53FlashForCausalLM,
)


class _RecordingSlice:
    def __init__(self) -> None:
        self.slices = []

    @staticmethod
    def get_shape() -> tuple[int, int]:
        return (8, 5)

    def __getitem__(self, item):
        self.slices.append(item)
        row_slice, col_slice = item
        rows = row_slice.stop - row_slice.start
        cols = col_slice.stop - col_slice.start
        return (
            torch.arange(rows * cols, dtype=torch.float32).view(rows, cols).add_(1000)
        )


def test_full_stream_config_uses_cpu_offload_and_fixed_batch_math(monkeypatch):
    monkeypatch.setenv("GLM53_STREAM_LOCAL_BATCH_SIZE", "4")
    monkeypatch.setenv("GLM53_STREAM_STEPS", "12")

    config = glm53_flash_countdown_lora_stream()

    assert config.training.enable_cpu_offload
    assert config.training.max_context_length == 512
    assert config.training.num_tokens_per_train_step == 32768
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 4 * 512
    assert config.training.steps == 12
    assert config.debug.seed == 42
    assert config.debug.distinct_seed_mesh_axes == ["dp_shard"]
    assert isinstance(config.activation_checkpoint, FullAC.Config)
    assert isinstance(config.checkpointer, Glm53FlashLoRACheckpointManager.Config)
    assert config.checkpointer.load_frozen_base_from_hf
    assert config.checkpointer.trainable_model_only
    assert not config.checkpointer.initial_load_in_hf
    assert isinstance(config.model, StreamGlm53FlashForCausalLM.Config)
    assert config.model.glm_config.num_hidden_layers == 45
    assert config.model.glm_config.hidden_size == 4096
    assert config.model.glm_config.lora_rank == 16
    assert config.model.glm_config.attn_impl == "fast"
    assert config.optim.optimizer.optimizers[0].pattern == r".*\.lora_[ab]$"


def test_truncated_stream_and_plain_configs_match_except_offload(monkeypatch):
    monkeypatch.setenv("GLM53_STREAM_NUM_LAYERS", "8")
    monkeypatch.setenv("GLM53_STREAM_LOCAL_BATCH_SIZE", "2")
    monkeypatch.setenv("GLM53_STREAM_STEPS", "3")
    monkeypatch.setenv("GLM53_STREAM_ACTIVATION_CHECKPOINT", "none")

    stream = glm53_flash_countdown_lora_truncated_stream()
    plain = glm53_flash_countdown_lora_truncated_plain()

    assert stream.training.enable_cpu_offload
    assert not plain.training.enable_cpu_offload
    assert isinstance(stream.model, StreamGlm53FlashForCausalLM.Config)
    assert not isinstance(plain.model, StreamGlm53FlashForCausalLM.Config)
    assert stream.activation_checkpoint is None
    assert plain.activation_checkpoint is None
    assert stream.model.glm_config.num_hidden_layers == 8
    assert stream.model.glm_config.sparse_layers == (3, 7)
    assert stream.model.glm_config.mlp_layer_types[:8] == (
        "dense",
        "dense",
        "dense",
        "sparse",
        "sparse",
        "sparse",
        "sparse",
        "sparse",
    )
    assert stream.model.glm_config.hidden_size == plain.model.glm_config.hidden_size
    assert stream.training.num_tokens_per_microbatch_per_dp_rank == (
        plain.training.num_tokens_per_microbatch_per_dp_rank
    )
    assert stream.training.num_tokens_per_train_step == (
        plain.training.num_tokens_per_train_step
    )
    assert stream.debug.seed == plain.debug.seed


def test_stream_config_rejects_batch_math_that_changes_fixed_step(monkeypatch):
    monkeypatch.setenv("GLM53_STREAM_LOCAL_BATCH_SIZE", "3")

    with pytest.raises(ValueError, match="LOCAL_BATCH_SIZE"):
        glm53_flash_countdown_lora_stream()


def test_rank_local_copy_reads_only_requested_cpu_slice():
    hf_slice = _RecordingSlice()
    target = torch.empty((2, 3), dtype=torch.bfloat16, device="cpu")
    stats = ShardedLoadStats(verified_keys=[])
    reader = SimpleNamespace(get_slice=lambda _key: hf_slice)

    _copy_one_to_one_tensor(
        reader,
        "model.language_model.layers.0.weight",
        TensorSliceTarget(
            tensor=target,
            global_shape=(8, 5),
            global_offset=(4, 1),
        ),
        stats,
    )

    assert hf_slice.slices == [(slice(4, 6), slice(1, 4))]
    assert target.device.type == "cpu"
    assert stats.bytes_loaded == (
        target.numel() * torch.empty((), dtype=torch.float32).element_size()
    )
    expected = torch.arange(6, dtype=torch.float32).view(2, 3).add_(1000)
    torch.testing.assert_close(
        target.float(),
        expected.to(torch.bfloat16).float(),
    )


def test_streaming_h2d_estimate_tracks_microbatch_tuning():
    lb1 = estimate_streaming_h2d(
        world_size=4,
        seq_len=512,
        local_batch_size=1,
        tokens_per_step=32768,
    )
    lb4 = estimate_streaming_h2d(
        world_size=4,
        seq_len=512,
        local_batch_size=4,
        tokens_per_step=32768,
    )

    assert lb1.gradient_accumulation_steps == 16
    assert lb4.gradient_accumulation_steps == 4
    assert lb1.frozen_base_shard_gib == pytest.approx(145.904158, rel=1e-6)
    assert lb4.h2d_gib_per_optimizer_step == pytest.approx(
        lb1.h2d_gib_per_optimizer_step / 4,
        rel=1e-12,
    )
    assert lb4.h2d_bandwidth_gib_per_second(10.0) == pytest.approx(
        lb4.h2d_gib_per_optimizer_step / 10.0
    )


def test_grad_probe_tensor_summary_and_delta():
    tensor = torch.tensor([[1.0, -2.0], [0.0, 3.0]])

    summary = summarize_tensor(tensor)

    assert summary["present"]
    assert summary["shape"] == [2, 2]
    assert summary["numel"] == 4
    assert summary["max_abs"] == 3.0
    assert summary["nonzero"] == 3
    assert max_abs_delta(tensor, tensor + 0.25) == pytest.approx(0.25)
    assert max_abs_delta(None, None) == 0.0
    assert max_abs_delta(tensor, None) is None


def test_grad_probe_comparison_reports_per_parameter_deltas(tmp_path):
    plain_dir = tmp_path / "plain"
    stream_dir = tmp_path / "stream"
    plain_dir.mkdir()
    stream_dir.mkdir()
    plain_snapshot = {
        "rank": 0,
        "all_parameter_grad_norm": 2.0,
        "optimizer_parameter_grad_norm": 2.0,
        "lora": {
            "layers.0.self_attn.q_proj.lora_a": {
                "parameter": torch.tensor([1.0, 2.0]),
                "gradient": torch.tensor([0.5, 1.5]),
            },
            "layers.0.self_attn.q_proj.lora_b": {
                "parameter": torch.tensor([0.0, 0.0]),
                "gradient": None,
            },
        },
    }
    stream_snapshot = {
        "rank": 0,
        "all_parameter_grad_norm": 3.0,
        "optimizer_parameter_grad_norm": 3.0,
        "lora": {
            "layers.0.self_attn.q_proj.lora_a": {
                "parameter": torch.tensor([1.0, 3.0]),
                "gradient": torch.tensor([0.5, 2.0]),
            },
            "layers.0.self_attn.q_proj.lora_b": {
                "parameter": torch.tensor([0.0, 0.0]),
                "gradient": None,
            },
        },
    }
    torch.save(plain_snapshot, plain_dir / "rank_0.pt")
    torch.save(stream_snapshot, stream_dir / "rank_0.pt")

    comparison = compare_probe_dirs(plain_dir, stream_dir)

    assert comparison["num_parameters"] == 2
    assert comparison["num_differing_parameters"] == 1
    assert comparison["max_abs_parameter_delta"] == 1.0
    assert comparison["max_abs_gradient_delta"] == 0.5
    row = next(
        row
        for row in comparison["parameters"]
        if row["name"] == "layers.0.self_attn.q_proj.lora_a"
    )
    assert row["parameter_delta_rank"] == 0
    assert row["gradient_delta_rank"] == 0
