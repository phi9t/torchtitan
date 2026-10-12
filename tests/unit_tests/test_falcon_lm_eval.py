# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Behavior tests for Falcon fixed-region LM checkpoint evaluation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import torch

from torchtitan.experiments.falcon.bin_reader import write_nanogpt_bin


def _write_tokens(path: Path, token_count: int = 128) -> str:
    tokens = torch.arange(token_count, dtype=torch.int64) % 101
    write_nanogpt_bin(path, tokens)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_region_registry_validation_rejects_bad_layouts_and_digest_drift(
    tmp_path: Path,
):
    """Overlaps, duplicates, bounds bugs, or source drift must reject a registry."""
    from torchtitan.experiments.falcon.lm_eval import (
        Region,
        RegionRegistry,
        validate_region_registry,
    )

    source = tmp_path / "val.bin"
    source_digest = _write_tokens(source, token_count=128)
    registry = RegionRegistry.from_regions(
        version="test-v1",
        source_path=source,
        source_digest=source_digest,
        source_token_count=128,
        regions=(
            Region("r0", token_offset=0, token_count=8),
            Region("r1", token_offset=32, token_count=8),
            Region("r2", token_offset=96, token_count=8),
        ),
    )
    assert validate_region_registry(registry) == registry

    duplicate = RegionRegistry.from_regions(
        version="test-v1",
        source_path=source,
        source_digest=source_digest,
        source_token_count=128,
        regions=(
            Region("r0", token_offset=0, token_count=8),
            Region("r0", token_offset=32, token_count=8),
            Region("r2", token_offset=96, token_count=8),
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_region_registry(duplicate)

    overlapping = RegionRegistry.from_regions(
        version="test-v1",
        source_path=source,
        source_digest=source_digest,
        source_token_count=128,
        regions=(
            Region("r0", token_offset=0, token_count=16),
            Region("r1", token_offset=8, token_count=8),
            Region("r2", token_offset=96, token_count=8),
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_region_registry(overlapping)

    out_of_range = RegionRegistry.from_regions(
        version="test-v1",
        source_path=source,
        source_digest=source_digest,
        source_token_count=128,
        regions=(
            Region("r0", token_offset=0, token_count=8),
            Region("r1", token_offset=32, token_count=8),
            Region("r2", token_offset=124, token_count=8),
        ),
    )
    with pytest.raises(ValueError, match="range"):
        validate_region_registry(out_of_range)

    missing_label = RegionRegistry.from_regions(
        version="test-v1",
        source_path=source,
        source_digest=source_digest,
        source_token_count=128,
        regions=(
            Region("r0", token_offset=0, token_count=8),
            Region("r1", token_offset=32, token_count=8),
            Region("r2", token_offset=120, token_count=8),
        ),
    )
    with pytest.raises(ValueError, match="shifted label"):
        validate_region_registry(missing_label)

    _write_tokens(source, token_count=129)
    with pytest.raises(ValueError, match="source digest"):
        validate_region_registry(registry)


def test_aggregate_region_ce_is_token_weighted_and_rejects_wrong_registry():
    """Aggregation must weight by token count and bind every result to the registry."""
    from torchtitan.experiments.falcon.lm_eval import (
        aggregate_region_ce,
        Region,
        RegionRegistry,
    )

    registry = RegionRegistry.from_regions(
        version="test-v1",
        source_path="val.bin",
        source_digest="a" * 64,
        source_token_count=80,
        regions=(
            Region("r0", token_offset=0, token_count=10),
            Region("r1", token_offset=20, token_count=20),
            Region("r2", token_offset=40, token_count=30),
        ),
    )
    aggregate = aggregate_region_ce(
        [
            {
                "region_id": "r0",
                "registry_digest": registry.registry_digest,
                "region_loss_sum": 5.0,
                "region_token_count": 10,
            },
            {
                "region_id": "r1",
                "registry_digest": registry.registry_digest,
                "region_loss_sum": 20.0,
                "region_token_count": 20,
            },
            {
                "region_id": "r2",
                "registry_digest": registry.registry_digest,
                "region_loss_sum": 15.0,
                "region_token_count": 30,
            },
        ],
        registry,
    )

    assert aggregate["aggregate_ce"] == pytest.approx(40.0 / 60.0)
    assert aggregate["ppl"] == pytest.approx(math.exp(40.0 / 60.0))
    assert aggregate["total_region_loss_sum"] == pytest.approx(40.0)
    assert aggregate["total_region_token_count"] == 60

    wrong_registry = registry.registry_digest[:-1] + "0"
    with pytest.raises(ValueError, match="registry"):
        aggregate_region_ce(
            [
                {
                    "region_id": "r0",
                    "registry_digest": wrong_registry,
                    "region_loss_sum": 5.0,
                    "region_token_count": 10,
                }
            ],
            registry,
        )


def test_checkpoint_digest_mismatch_and_mutation_guard(tmp_path: Path):
    """Checkpoint eval must fail before and after evaluation if source files change."""
    from experiments.falcon.checkpoint_eval import (
        assert_checkpoint_unchanged,
        checkpoint_tree_digest,
        CheckpointMutationError,
        require_checkpoint_digest,
        snapshot_checkpoint_tree,
    )

    checkpoint = tmp_path / "step-10"
    checkpoint.mkdir()
    (checkpoint / ".metadata").write_text("metadata-v1\n")
    (checkpoint / "__0_0.distcp").write_bytes(b"weights-v1")

    before = snapshot_checkpoint_tree(checkpoint)
    digest = checkpoint_tree_digest(before)
    require_checkpoint_digest(before, digest)
    with pytest.raises(CheckpointMutationError, match="digest mismatch"):
        require_checkpoint_digest(before, "0" * 64)

    (checkpoint / "__0_0.distcp").write_bytes(b"weights-v2")
    after = snapshot_checkpoint_tree(checkpoint)
    with pytest.raises(CheckpointMutationError, match="mutated"):
        assert_checkpoint_unchanged(before, after)


def test_checkpoint_eval_records_same_logical_identity_for_distinct_attempts(
    tmp_path: Path,
):
    """A repeated read-only eval should keep run ID stable while attempts differ."""
    from experiments.falcon.checkpoint_eval import build_attempt_ids

    first = build_attempt_ids(arm="A0", checkpoint_digest="a" * 64)
    second = build_attempt_ids(arm="A0", checkpoint_digest="a" * 64)

    assert first.run_id == "falcon-b11-checkpoint-eval-A0-seed0"
    assert second.run_id == first.run_id
    assert second.attempt_id != first.attempt_id


def test_checkpoint_eval_detects_mutation_after_restore_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The read-only guard must run even when restore/eval raises mid-attempt."""
    import experiments.falcon.checkpoint_eval as checkpoint_eval
    from experiments.falcon.checkpoint_eval import CheckpointMutationError
    from torchtitan.experiments.falcon.lm_eval import Region, RegionRegistry

    checkpoint = tmp_path / "step-10"
    checkpoint.mkdir()
    weights = checkpoint / "__0_0.distcp"
    (checkpoint / ".metadata").write_text("metadata-v1\n")
    weights.write_bytes(b"weights-v1")

    source = tmp_path / "val.bin"
    source_digest = _write_tokens(source, token_count=128)
    registry = RegionRegistry.from_regions(
        version="test-v1",
        source_path=source,
        source_digest=source_digest,
        source_token_count=128,
        regions=(
            Region("r0", token_offset=0, token_count=8),
            Region("r1", token_offset=32, token_count=8),
            Region("r2", token_offset=96, token_count=8),
        ),
    )

    monkeypatch.setattr(checkpoint_eval, "_ATTEMPT_ROOT", tmp_path / "attempts")
    monkeypatch.setattr(
        checkpoint_eval, "build_fixed_region_registry", lambda: registry
    )

    def fail_after_mutation(*args, **kwargs):
        weights.write_bytes(b"weights-v2")
        raise RuntimeError("restore setup failed")

    monkeypatch.setattr(checkpoint_eval, "_build_eval_config", fail_after_mutation)

    with pytest.raises(CheckpointMutationError, match="mutated"):
        checkpoint_eval.run_checkpoint_eval(
            arm="A0",
            checkpoint=checkpoint,
            master_port=29590,
        )


def test_checkpoint_eval_rejects_inherited_multi_rank_env(
    monkeypatch: pytest.MonkeyPatch,
):
    """The single-rank runner must fail loudly on inherited distributed state."""
    from experiments.falcon.checkpoint_eval import _set_single_process_env

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="WORLD_SIZE=1"):
        _set_single_process_env(master_port=29590)
