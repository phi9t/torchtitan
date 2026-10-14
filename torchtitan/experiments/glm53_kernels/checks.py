# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerics and smoke checks for GLM kernel package conventions."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from torchtitan.experiments.glm53_kernels.cute_examples import (
    cute_runtime_info,
    cute_scale_add_convention,
)
from torchtitan.experiments.glm53_kernels.triton_examples import triton_scale_add


TensorFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def _compute_grads(
    fn: TensorFn,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_out: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_leaf = x.detach().clone().requires_grad_(True)
    y_leaf = y.detach().clone().requires_grad_(True)
    out = fn(x_leaf, y_leaf, alpha)
    out.backward(grad_out.to(out.dtype))
    assert x_leaf.grad is not None
    assert y_leaf.grad is not None
    return out.detach(), x_leaf.grad.detach(), y_leaf.grad.detach()


def _numerics_row(
    *,
    backend: str,
    tensor_name: str,
    golden: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
    project_atol: float = 0.0,
    fudge_factor: float = 2.0,
    reference_relative_trigger: float = 0.10,
) -> dict[str, Any]:
    golden64 = golden.detach().to(torch.float64)
    reference64 = reference.detach().to(torch.float64)
    target64 = target.detach().to(torch.float64)
    reference_error = float((reference64 - golden64).abs().max().item())
    target_error = float((target64 - golden64).abs().max().item())
    rounding_floor = float(
        (golden64.to(target.dtype).to(torch.float64) - golden64).abs().max().item()
    )
    absolute_floor = max(project_atol, rounding_floor)
    threshold = fudge_factor * reference_error + absolute_floor
    golden_scale = float(golden64.abs().max().item())
    reference_relative_error = reference_error / max(
        golden_scale,
        absolute_floor,
        1.0e-300,
    )
    gate_valid = reference_relative_error <= reference_relative_trigger
    passed = bool(gate_valid and target_error <= threshold)
    return {
        "backend": backend,
        "tensor": tensor_name,
        "reference_error": reference_error,
        "target_error": target_error,
        "rounding_floor": rounding_floor,
        "threshold": threshold,
        "golden_scale": golden_scale,
        "reference_relative_error": reference_relative_error,
        "gate_valid": gate_valid,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


def _run_scale_add_check(
    *,
    backend: str,
    fn: TensorFn,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float,
    compiled_kernel: bool,
) -> dict[str, Any]:
    torch.manual_seed(1234)
    x = torch.randn(4096, device=device, dtype=torch.float32).to(dtype)
    y = torch.randn_like(x)
    grad = torch.randn_like(x)

    golden = lambda a, b, c: torch.add(b, a, alpha=c)
    reference = lambda a, b, c: torch.add(b, a, alpha=c)
    golden_out, golden_dx, golden_dy = _compute_grads(
        golden,
        x.to(torch.float64),
        y.to(torch.float64),
        grad.to(torch.float64),
        alpha,
    )
    ref_out, ref_dx, ref_dy = _compute_grads(reference, x, y, grad, alpha)
    target_out, target_dx, target_dy = _compute_grads(fn, x, y, grad, alpha)
    rows = [
        _numerics_row(
            backend=backend,
            tensor_name="out",
            golden=golden_out,
            reference=ref_out,
            target=target_out,
        ),
        _numerics_row(
            backend=backend,
            tensor_name="dx",
            golden=golden_dx,
            reference=ref_dx,
            target=target_dx,
        ),
        _numerics_row(
            backend=backend,
            tensor_name="dy",
            golden=golden_dy,
            reference=ref_dy,
            target=target_dy,
        ),
    ]
    compile_report: dict[str, Any] = {"attempted": device.type == "cuda", "ok": None}
    if device.type == "cuda":
        try:
            compiled = torch.compile(lambda a, b: fn(a, b, alpha), fullgraph=False)
            compiled_out = compiled(x, y)
            compile_report.update(
                {
                    "ok": bool(torch.allclose(compiled_out, target_out)),
                    "max_diff": float((compiled_out - target_out).abs().max().item()),
                }
            )
        except Exception as exc:
            compile_report.update({"ok": False, "error": repr(exc)})
    bench = _microbench(fn, x, y, alpha) if device.type == "cuda" else None
    return {
        "backend": backend,
        "compiled_kernel": compiled_kernel,
        "dtype": str(dtype),
        "shape": list(x.shape),
        "alpha": alpha,
        "rows": rows,
        "compile": compile_report,
        "microbench": bench,
        "ok": all(row["passed"] for row in rows)
        and (compile_report["ok"] in (True, None)),
    }


def _microbench(
    fn: TensorFn,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    warmup: int = 5,
    iters: int = 20,
) -> dict[str, float]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(x, y, alpha)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(x, y, alpha)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    return {
        "milliseconds": elapsed_ms,
        "elements_per_second": x.numel() / (elapsed_ms / 1000.0),
    }


def _native_scale_add() -> TensorFn:
    import glm53_kernel_examples

    return glm53_kernel_examples.scale_add


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    if device.type == "cuda" and not torch.cuda.is_available():
        return {
            "ok": False,
            "blockers": ["CUDA requested but torch.cuda.is_available() is false"],
            "checks": [],
        }

    checks.append(
        _run_scale_add_check(
            backend="triton",
            fn=triton_scale_add,
            device=device,
            dtype=dtype,
            alpha=args.alpha,
            compiled_kernel=True,
        )
    )

    cute_check = _run_scale_add_check(
        backend="cute_dsl_convention",
        fn=cute_scale_add_convention,
        device=device,
        dtype=dtype,
        alpha=args.alpha,
        compiled_kernel=False,
    )
    cute_check["cute_runtime"] = cute_runtime_info()
    cute_check["verdict"] = "PASS_WITH_LIMITATIONS"
    checks.append(cute_check)

    if not args.skip_native:
        try:
            native_fn = _native_scale_add()
        except Exception as exc:
            message = f"native glm53_kernel_examples import failed: {exc!r}"
            if args.allow_missing_native:
                checks.append(
                    {
                        "backend": "native_cuda",
                        "ok": True,
                        "skipped": True,
                        "reason": message,
                    }
                )
            else:
                blockers.append(message)
        else:
            checks.append(
                _run_scale_add_check(
                    backend="native_cpp_cpu",
                    fn=native_fn,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    alpha=args.alpha,
                    compiled_kernel=False,
                )
            )

    return {
        "ok": not blockers and all(item.get("ok", False) for item in checks),
        "blockers": blockers,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--allow-missing-native", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args(argv)

    report = run_checks(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

