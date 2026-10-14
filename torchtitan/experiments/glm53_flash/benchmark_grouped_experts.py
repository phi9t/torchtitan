# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Micro-benchmark for GLM-5.3-Flash MoE expert implementations."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from torchtitan.experiments.glm53_flash.flavors import glm53_flash
from torchtitan.experiments.glm53_flash.model import GlmMoE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark loop versus grouped GLM MoE experts."
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--impl",
        choices=("loop", "grouped", "both"),
        default="both",
    )
    return parser.parse_args()


def _cuda_memory_gib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**3)


def _run_impl(
    moe: GlmMoE,
    *,
    impl: str,
    x_BLD: torch.Tensor,
    grad_BLD: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    moe.experts.experts_impl = impl
    for _ in range(warmup):
        x_BLD.grad = None
        out_BLD = moe(x_BLD)
        (out_BLD.float() * grad_BLD).sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        x_BLD.grad = None
        out_BLD = moe(x_BLD)
        (out_BLD.float() * grad_BLD).sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "impl": impl,
        "iters": iters,
        "ms_per_iter": elapsed * 1000.0 / iters,
        "peak_memory_gib": _cuda_memory_gib(),
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("benchmark_grouped_experts requires a CUDA device")

    torch.manual_seed(args.seed)
    cfg = glm53_flash()
    cfg.experts_impl = "loop"
    moe = GlmMoE(cfg).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for param in moe.parameters():
            param.normal_(mean=0.0, std=cfg.initializer_range)
        moe.gate.e_score_correction_bias.zero_()
    for param in moe.parameters():
        param.requires_grad_(False)

    x_BLD = torch.randn(
        1,
        args.tokens,
        cfg.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_BLD = torch.randn(
        1,
        args.tokens,
        cfg.hidden_size,
        device=device,
        dtype=torch.float32,
    )

    impls = ("loop", "grouped") if args.impl == "both" else (args.impl,)
    results = [
        _run_impl(
            moe,
            impl=impl,
            x_BLD=x_BLD,
            grad_BLD=grad_BLD,
            warmup=args.warmup,
            iters=args.iters,
        )
        for impl in impls
    ]
    by_impl = {result["impl"]: result for result in results}
    if "loop" in by_impl and "grouped" in by_impl:
        by_impl["grouped"]["speedup_vs_loop"] = (
            by_impl["loop"]["ms_per_iter"] / by_impl["grouped"]["ms_per_iter"]
        )
    print(
        json.dumps(
            {
                "benchmark": "glm53_flash_moe_layer",
                "dtype": "bfloat16",
                "tokens": args.tokens,
                "hidden_size": cfg.hidden_size,
                "moe_intermediate_size": cfg.moe_intermediate_size,
                "num_experts": cfg.num_experts,
                "top_k": cfg.num_experts_per_tok,
                "weights_require_grad": False,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
