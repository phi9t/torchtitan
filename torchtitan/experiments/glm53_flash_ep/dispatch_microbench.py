# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-shaped all-to-all versus DeepEP dispatch microbenchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.distributed.spmd_types import set_spmd_meshes
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _make_inputs(
    *,
    tokens: int,
    hidden_dim: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(1234 + _rank())
    x_TD = torch.randn(tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    base_T1 = torch.arange(tokens, device=device, dtype=torch.int64).unsqueeze(1)
    offsets_1K = torch.arange(top_k, device=device, dtype=torch.int64).unsqueeze(0)
    topk_expert_ids_TK = (base_T1 * top_k + offsets_1K + _rank()) % num_experts
    topk_scores_TK = torch.rand(tokens, top_k, generator=generator, device=device)
    topk_scores_TK = topk_scores_TK / topk_scores_TK.sum(dim=-1, keepdim=True)
    routing_map_TE = torch.zeros(
        tokens,
        num_experts,
        dtype=torch.bool,
        device=device,
    ).scatter_(-1, topk_expert_ids_TK, True)
    return x_TD, topk_scores_TK, topk_expert_ids_TK, routing_map_TE.sum(dim=0)


def _benchmark_dispatcher(
    name: str,
    dispatcher,
    *,
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    dispatcher.init_buffer()
    measurements = []
    for idx in range(warmup + iters):
        _sync()
        begin = time.perf_counter()
        with torch.no_grad():
            routed_RD, _tokens_per_expert, metadata = dispatcher.dispatch(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )
            dispatch_end = time.perf_counter()
            out_TD = dispatcher.combine(torch.zeros_like(routed_RD), metadata, x_TD)
        _sync()
        end = time.perf_counter()
        if idx >= warmup:
            measurements.append(
                {
                    "dispatch_ms": (dispatch_end - begin) * 1000.0,
                    "combine_ms": (end - dispatch_end) * 1000.0,
                    "roundtrip_ms": (end - begin) * 1000.0,
                    "output_checksum": float(out_TD.float().sum().item()),
                }
            )
    return {
        "name": name,
        "rank": _rank(),
        "measurements": measurements,
        "mean_dispatch_ms": _mean(row["dispatch_ms"] for row in measurements),
        "mean_combine_ms": _mean(row["combine_ms"] for row in measurements),
        "mean_roundtrip_ms": _mean(row["roundtrip_ms"] for row in measurements),
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / float(1024**3),
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _rank0_reduce(records: dict[str, Any]) -> dict[str, Any] | None:
    if not dist.is_initialized():
        return {"world": [records]}
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, records)
    if _rank() != 0:
        return None
    return {"world": gathered}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=288)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")
    mesh = init_device_mesh(
        "cuda",
        (dist.get_world_size(),),
        mesh_dim_names=("ep",),
    )
    set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)

    tokens = args.seq_len * args.local_batch_size
    x_TD, topk_scores_TK, topk_expert_ids_TK, counts_E = _make_inputs(
        tokens=tokens,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
        num_experts=args.num_experts,
        device=device,
    )

    alltoall = AllToAllTokenDispatcher.Config(
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).build()
    deepep = DeepEPTokenDispatcher.Config(
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_max_tokens_per_rank=tokens,
        hidden_dim=args.hidden_dim,
    ).build()
    records = {
        "schema_version": 1,
        "rank": _rank(),
        "world_size": dist.get_world_size(),
        "shape": {
            "tokens_per_rank": tokens,
            "hidden_dim": args.hidden_dim,
            "top_k": args.top_k,
            "num_experts": args.num_experts,
            "ep_size": args.ep_size,
        },
        "benchmarks": [
            _benchmark_dispatcher(
                "alltoall",
                alltoall,
                x_TD=x_TD,
                topk_scores_TK=topk_scores_TK,
                topk_expert_ids_TK=topk_expert_ids_TK,
                num_local_tokens_per_expert_E=counts_E,
                warmup=args.warmup,
                iters=args.iters,
            ),
            _benchmark_dispatcher(
                "deepep",
                deepep,
                x_TD=x_TD,
                topk_scores_TK=topk_scores_TK,
                topk_expert_ids_TK=topk_expert_ids_TK,
                num_local_tokens_per_expert_E=counts_E,
                warmup=args.warmup,
                iters=args.iters,
            ),
        ],
    }
    output = _rank0_reduce(records)
    if output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
