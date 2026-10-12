# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-region LM evaluation helpers for Falcon checkpoints.

The registry binds three validation spans to one nanogpt validation shard by
path, source digest, and token offsets. Results carry the registry digest so
region-level evidence cannot be accidentally aggregated across incompatible
validation cuts.
"""

from __future__ import annotations

import hashlib
import json
import math

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.experiments.falcon.bin_reader import read_nanogpt_bin


FALCON_LM_EVAL_REGISTRY_VERSION = "falcon-b11-fixed-regions-v1"
FALCON_VAL_BIN = (
    "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/"
    "data/fineweb10B/fineweb_val_000000.bin"
)
_FIXED_REGIONS = (
    ("val_head_0000", 0, 16_384),
    ("val_mid_10m", 10_000_000, 16_384),
    ("val_mid_20m", 20_000_000, 16_384),
)


@dataclass(frozen=True, slots=True)
class Region:
    """One contiguous scored-token span in the validation shard."""

    region_id: str
    token_offset: int
    token_count: int


@dataclass(frozen=True, slots=True)
class RegionRegistry:
    """Immutable identity for Falcon fixed-region validation."""

    version: str
    source_path: str
    source_digest: str
    source_token_count: int
    regions: tuple[Region, ...]
    registry_digest: str

    @classmethod
    def from_regions(
        cls,
        *,
        version: str,
        source_path: str | Path,
        source_digest: str,
        source_token_count: int,
        regions: Sequence[Region],
    ) -> "RegionRegistry":
        payload = {
            "version": version,
            "source_path": str(source_path),
            "source_digest": source_digest,
            "source_token_count": source_token_count,
            "regions": [asdict(region) for region in regions],
        }
        return cls(
            version=version,
            source_path=str(source_path),
            source_digest=source_digest,
            source_token_count=source_token_count,
            regions=tuple(regions),
            registry_digest=_digest_json(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_path": self.source_path,
            "source_digest": self.source_digest,
            "source_token_count": self.source_token_count,
            "regions": [asdict(region) for region in self.regions],
            "registry_digest": self.registry_digest,
        }


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_token_count(path: str | Path) -> int:
    import numpy as np

    header = np.fromfile(Path(path), dtype=np.int32, count=256)
    if header.shape[0] != 256:
        raise ValueError(f"nanogpt bin {path} is smaller than a 256-int32 header")
    return int(header[2])


def build_fixed_region_registry(
    source_path: str | Path = FALCON_VAL_BIN,
) -> RegionRegistry:
    """Return the current fixed Falcon validation registry for ``source_path``."""
    source = Path(source_path)
    regions = tuple(
        Region(region_id, token_offset=offset, token_count=count)
        for region_id, offset, count in _FIXED_REGIONS
    )
    registry = RegionRegistry.from_regions(
        version=FALCON_LM_EVAL_REGISTRY_VERSION,
        source_path=source,
        source_digest=sha256_file(source),
        source_token_count=_source_token_count(source),
        regions=regions,
    )
    return validate_region_registry(registry)


def validate_region_registry(registry: RegionRegistry) -> RegionRegistry:
    """Validate region layout and source binding, returning ``registry``."""
    if not registry.version:
        raise ValueError("registry version must be nonempty")
    if len(registry.regions) != 3:
        raise ValueError("Falcon B11 registry must contain exactly three regions")
    if not registry.source_digest or len(registry.source_digest) != 64:
        raise ValueError("source digest must be a sha256 hex digest")
    if registry.source_token_count < 1:
        raise ValueError("source_token_count must be positive")

    seen: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    for region in registry.regions:
        if not region.region_id:
            raise ValueError("region_id must be nonempty")
        if region.region_id in seen:
            raise ValueError(f"duplicate region_id: {region.region_id}")
        seen.add(region.region_id)
        if region.token_offset < 0 or region.token_count < 1:
            raise ValueError("region token offsets/counts must be positive")
        end = region.token_offset + region.token_count
        if end >= registry.source_token_count:
            raise ValueError(f"region {region.region_id} lacks shifted label range")
        ranges.append((region.token_offset, end, region.region_id))

    for prev, cur in zip(sorted(ranges), sorted(ranges)[1:]):
        if prev[1] > cur[0]:
            raise ValueError(f"region overlap: {prev[2]} and {cur[2]}")

    recomputed = RegionRegistry.from_regions(
        version=registry.version,
        source_path=registry.source_path,
        source_digest=registry.source_digest,
        source_token_count=registry.source_token_count,
        regions=registry.regions,
    )
    if recomputed.registry_digest != registry.registry_digest:
        raise ValueError("registry digest does not match registry contents")

    source = Path(registry.source_path)
    if source.exists() and sha256_file(source) != registry.source_digest:
        raise ValueError(f"source digest drift for {registry.source_path}")
    return registry


def aggregate_region_ce(
    region_results: Iterable[dict[str, Any]],
    registry: RegionRegistry,
) -> dict[str, Any]:
    """Aggregate region CE with token weighting and registry lineage checks."""
    validate_region_registry(registry)
    expected_ids = {region.region_id: region.token_count for region in registry.regions}
    seen: set[str] = set()
    total_loss = 0.0
    total_tokens = 0

    for result in region_results:
        region_id = result.get("region_id")
        if result.get("registry_digest") != registry.registry_digest:
            raise ValueError("region result registry_digest does not match registry")
        if region_id not in expected_ids:
            raise ValueError(f"unknown region result: {region_id}")
        if region_id in seen:
            raise ValueError(f"duplicate region result: {region_id}")
        token_count = result.get("region_token_count")
        if token_count != expected_ids[region_id]:
            raise ValueError(f"region_token_count mismatch for {region_id}")
        loss_sum = float(result.get("region_loss_sum"))
        if not math.isfinite(loss_sum) or loss_sum < 0:
            raise ValueError("region_loss_sum must be finite and nonnegative")
        total_loss += loss_sum
        total_tokens += int(token_count)
        seen.add(region_id)

    missing = set(expected_ids) - seen
    if missing:
        raise ValueError(f"missing region results: {sorted(missing)}")
    aggregate_ce = total_loss / total_tokens
    return {
        "registry_digest": registry.registry_digest,
        "total_region_loss_sum": total_loss,
        "total_region_token_count": total_tokens,
        "aggregate_ce": aggregate_ce,
        "ppl": math.exp(aggregate_ce),
    }


def evaluate_regions(
    model: torch.nn.Module,
    registry: RegionRegistry,
    *,
    seq_len: int,
    vocab_size: int,
) -> list[dict[str, Any]]:
    """Evaluate a model on every fixed region and return loss-sum rows."""
    registry = validate_region_registry(registry)
    tokens = read_nanogpt_bin(registry.source_path)
    if tokens.numel() != registry.source_token_count:
        raise ValueError("registry source_token_count does not match bin contents")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for region in registry.regions:
            start = region.token_offset
            stop = start + region.token_count
            body = tokens[start:stop].to(device)
            labels = tokens[start + 1 : stop + 1].to(device)
            if labels.numel() != region.token_count:
                raise ValueError(f"region {region.region_id} lacks shifted labels")
            region_loss_sum = 0.0
            for chunk_start in range(0, region.token_count, seq_len):
                chunk_stop = min(region.token_count, chunk_start + seq_len)
                chunk_tokens = body[chunk_start:chunk_stop].unsqueeze(0)
                chunk_labels = labels[chunk_start:chunk_stop].unsqueeze(0)
                logits = model(chunk_tokens)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size).float(),
                    chunk_labels.reshape(-1),
                    reduction="sum",
                )
                region_loss_sum += float(loss.item())
            rows.append(
                {
                    "region_id": region.region_id,
                    "registry_digest": registry.registry_digest,
                    "region_loss_sum": region_loss_sum,
                    "region_token_count": region.token_count,
                    "region_ce": region_loss_sum / region.token_count,
                }
            )
    if was_training:
        model.train()
    return rows
