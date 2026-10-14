# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerics and microbenchmark harness for sparse top-k attention.

Run inside the TorchTitan rootfs. The numerics mode uses the kernel verifier
gate from ``.claude/skills/kernel-numerics-verifier/SKILL.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from torchtitan.experiments.glm53_arch_speedrun.elements.attention import (
    sparse_causal_attention_masked_dense,
    sparse_causal_attention_reference,
)
from torchtitan.experiments.glm53_kernels.sparse_topk_attention import (
    sparse_topk_attention_blhd,
)


@dataclass(slots=True)
class NumericsRow:
    seq_len: int
    top_k: int
    index_granularity: int
    tensor: str
    reference_error: float
    target_error: float
    rounding_floor: float
    threshold: float
    golden_scale: float
    reference_relative_error: float
    gate_valid: bool
    passed: bool


@dataclass(slots=True)
class BenchmarkRow:
    seq_len: int
    top_k: int
    index_granularity: int
    method: str
    milliseconds: float
    peak_memory_mib: float
    tokens_per_second: float


def _sync() -> None:
    torch.cuda.synchronize()


def _set_memory_cap(device: torch.device, memory_cap_gib: float) -> None:
    if device.type != "cuda" or memory_cap_gib <= 0:
        return
    total_memory = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, memory_cap_gib * 1024**3 / total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)


def _selected_indices(
    batch_size: int,
    seq_len: int,
    top_k: int,
    index_granularity: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if index_granularity == 1:
        width = top_k
        offsets = torch.arange(width, device=device)
        positions = torch.arange(seq_len, device=device)
        selected = positions[:, None] - offsets[None]
        selected = selected.clamp_min(-1)
        selected = selected.masked_fill(selected < 0, -1)
        return selected.view(1, seq_len, width).expand(batch_size, -1, -1).to(torch.int32)
    if index_granularity != 4:
        raise ValueError("index_granularity must be 1 or 4.")
    width = math.ceil(top_k / index_granularity)
    offsets = torch.arange(width, device=device)
    positions = torch.arange(seq_len, device=device)
    pool = torch.div(positions, index_granularity, rounding_mode="floor")
    selected = pool[:, None] - offsets[None]
    selected = selected.masked_fill(selected < 0, -1)
    return selected.view(1, seq_len, width).expand(batch_size, -1, -1).to(torch.int32)


def _selected_tokens_for_oracle(
    selected_BLS: torch.Tensor,
    index_granularity: int,
) -> torch.Tensor:
    if index_granularity == 1:
        return selected_BLS
    offsets = torch.arange(
        index_granularity,
        dtype=selected_BLS.dtype,
        device=selected_BLS.device,
    )
    expanded = selected_BLS[..., None] * index_granularity + offsets
    expanded = expanded.masked_fill(selected_BLS[..., None] < 0, -1)
    return expanded.flatten(-2)


def _attention_mask(batch_size: int, seq_len: int, *, device: torch.device) -> torch.Tensor:
    return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)


def _compute_grads(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = k.detach().clone().requires_grad_(True)
    v_leaf = v.detach().clone().requires_grad_(True)
    out = fn(q_leaf, k_leaf, v_leaf)
    out.backward(grad_out.to(out.dtype))
    assert q_leaf.grad is not None
    assert k_leaf.grad is not None
    assert v_leaf.grad is not None
    return out.detach(), q_leaf.grad.detach(), k_leaf.grad.detach(), v_leaf.grad.detach()


def _row(
    *,
    seq_len: int,
    top_k: int,
    index_granularity: int,
    tensor_name: str,
    golden: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
    project_atol: float,
    fudge_factor: float,
    reference_relative_trigger: float,
) -> NumericsRow:
    golden64 = golden.detach().to(torch.float64)
    reference64 = reference.detach().to(torch.float64)
    target64 = target.detach().to(torch.float64)
    reference_error = float((reference64 - golden64).abs().max().item())
    target_error = float((target64 - golden64).abs().max().item())
    target_dtype = target.dtype
    rounding_floor = float(
        (golden64.to(target_dtype).to(torch.float64) - golden64).abs().max().item()
    )
    absolute_floor = max(project_atol, rounding_floor)
    threshold = fudge_factor * reference_error + absolute_floor
    golden_scale = float(golden64.abs().max().item())
    reference_relative_error = reference_error / max(golden_scale, absolute_floor, 1.0e-300)
    gate_valid = reference_relative_error <= reference_relative_trigger
    passed = bool(gate_valid and target_error <= threshold)
    return NumericsRow(
        seq_len=seq_len,
        top_k=top_k,
        index_granularity=index_granularity,
        tensor=tensor_name,
        reference_error=reference_error,
        target_error=target_error,
        rounding_floor=rounding_floor,
        threshold=threshold,
        golden_scale=golden_scale,
        reference_relative_error=reference_relative_error,
        gate_valid=gate_valid,
        passed=passed,
    )


def run_numerics(args: argparse.Namespace) -> list[NumericsRow]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows: list[NumericsRow] = []
    dtype = getattr(torch, args.dtype)
    for seq_len in args.seq_len:
        batch_size = args.batch_size
        num_heads = args.num_heads
        head_dim = args.head_dim
        q_low = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        ).to(dtype)
        k_low = torch.randn_like(q_low)
        v_low = torch.randn_like(q_low)
        grad_low = torch.randn_like(q_low)
        mask = _attention_mask(batch_size, seq_len, device=device)
        for denom in args.k_denominator:
            top_k = max(1, seq_len // denom)
            for index_granularity in args.index_granularity:
                selected = _selected_indices(
                    batch_size,
                    seq_len,
                    top_k,
                    index_granularity,
                    device=device,
                )
                oracle_selected = _selected_tokens_for_oracle(
                    selected,
                    index_granularity,
                )

                def golden_fn(q, k, v):
                    return sparse_causal_attention_reference(
                        q,
                        k,
                        v,
                        oracle_selected,
                        attention_mask_BL=mask,
                    )

                def reference_fn(q, k, v):
                    return sparse_causal_attention_masked_dense(
                        q,
                        k,
                        v,
                        oracle_selected,
                        attention_mask_BL=mask,
                    )

                def target_fn(q, k, v):
                    return sparse_topk_attention_blhd(
                        q,
                        k,
                        v,
                        selected,
                        attention_mask_BL=mask,
                        index_granularity=index_granularity,
                    )

                golden_out, golden_dq, golden_dk, golden_dv = _compute_grads(
                    golden_fn,
                    q_low.to(torch.float64),
                    k_low.to(torch.float64),
                    v_low.to(torch.float64),
                    grad_low.to(torch.float64),
                )
                ref_out, ref_dq, ref_dk, ref_dv = _compute_grads(
                    reference_fn,
                    q_low,
                    k_low,
                    v_low,
                    grad_low,
                )
                target_out, target_dq, target_dk, target_dv = _compute_grads(
                    target_fn,
                    q_low,
                    k_low,
                    v_low,
                    grad_low,
                )
                for tensor_name, golden, reference, target in (
                    ("out", golden_out, ref_out, target_out),
                    ("dq", golden_dq, ref_dq, target_dq),
                    ("dk", golden_dk, ref_dk, target_dk),
                    ("dv", golden_dv, ref_dv, target_dv),
                ):
                    rows.append(
                        _row(
                            seq_len=seq_len,
                            top_k=top_k,
                            index_granularity=index_granularity,
                            tensor_name=tensor_name,
                            golden=golden,
                            reference=reference,
                            target=target,
                            project_atol=args.project_atol,
                            fudge_factor=args.fudge_factor,
                            reference_relative_trigger=args.reference_relative_trigger,
                        )
                    )
    return rows


def _bench_one(
    fn: Callable[[], torch.Tensor],
    *,
    tokens: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        out = fn()
        if out.requires_grad:
            out.float().square().mean().backward()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        if out.requires_grad:
            out.float().square().mean().backward()
    _sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    tokens_per_second = tokens / (elapsed_ms / 1000.0)
    return elapsed_ms, tokens_per_second if math.isfinite(tokens_per_second) else 0.0, peak_mib


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkRow]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows: list[BenchmarkRow] = []
    dtype = getattr(torch, args.dtype)
    for seq_len in args.seq_len:
        batch_size = args.batch_size
        num_heads = args.num_heads
        head_dim = args.head_dim
        mask = _attention_mask(batch_size, seq_len, device=device)
        for denom in args.k_denominator:
            top_k = max(1, seq_len // denom)
            for index_granularity in args.index_granularity:
                selected = _selected_indices(
                    batch_size,
                    seq_len,
                    top_k,
                    index_granularity,
                    device=device,
                )
                oracle_selected = _selected_tokens_for_oracle(
                    selected,
                    index_granularity,
                )
                methods: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {
                    "triton": lambda q, k, v: sparse_topk_attention_blhd(
                        q,
                        k,
                        v,
                        selected,
                        attention_mask_BL=mask,
                        index_granularity=index_granularity,
                    ),
                    "masked_dense": lambda q, k, v: sparse_causal_attention_masked_dense(
                        q,
                        k,
                        v,
                        oracle_selected,
                        attention_mask_BL=mask,
                    ),
                    "dense_flash_sdpa": lambda q, k, v: F.scaled_dot_product_attention(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        is_causal=True,
                    ).transpose(1, 2),
                }
                for method, method_fn in methods.items():
                    q = torch.randn(
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        dtype=dtype,
                        device=device,
                        requires_grad=True,
                    )
                    k = torch.randn_like(q, requires_grad=True)
                    v = torch.randn_like(q, requires_grad=True)

                    def _call(method_fn=method_fn):
                        q.grad = None
                        k.grad = None
                        v.grad = None
                        return method_fn(q, k, v)

                    elapsed_ms, tokens_per_second, peak_mib = _bench_one(
                        _call,
                        tokens=batch_size * seq_len,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    rows.append(
                        BenchmarkRow(
                            seq_len=seq_len,
                            top_k=top_k,
                            index_granularity=index_granularity,
                            method=method,
                            milliseconds=elapsed_ms,
                            peak_memory_mib=peak_mib,
                            tokens_per_second=tokens_per_second,
                        )
                    )
    return rows


def _print_jsonl(rows: list[NumericsRow] | list[BenchmarkRow]) -> None:
    for row in rows:
        print(json.dumps(asdict(row), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("numerics", "bench"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--k-denominator", type=int, nargs="+", default=[16, 8, 4])
    parser.add_argument("--index-granularity", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fudge-factor", type=float, default=2.0)
    parser.add_argument("--project-atol", type=float, default=0.0)
    parser.add_argument("--reference-relative-trigger", type=float, default=0.10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--memory-cap-gib", type=float, default=24.0)
    args = parser.parse_args()
    _set_memory_cap(torch.device(args.device), args.memory_cap_gib)

    if args.mode == "numerics":
        rows = run_numerics(args)
        _print_jsonl(rows)
        if not all(row.passed for row in rows):
            raise SystemExit(1)
    else:
        _print_jsonl(run_benchmark(args))


if __name__ == "__main__":
    main()

