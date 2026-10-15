# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-step LoRA gradient probe for the GLM CPU-offload stream path."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import ConfigLoader
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer


logger = logging.getLogger(__name__)


def _is_lora_parameter(name: str) -> bool:
    return name.endswith(".lora_a") or name.endswith(".lora_b")


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def _clone_cpu_float(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    local = _local_tensor(tensor.detach())
    return local.to(device="cpu", dtype=torch.float32).clone()


def summarize_tensor(tensor: torch.Tensor | None) -> dict[str, Any]:
    """Return compact stats for a possibly-missing tensor."""

    if tensor is None:
        return {
            "present": False,
            "shape": None,
            "dtype": None,
            "device": None,
            "numel": 0,
            "max_abs": None,
            "l2_norm": None,
            "nonzero": 0,
            "finite": True,
        }
    local = _local_tensor(tensor.detach())
    flat = local.float().reshape(-1)
    finite = bool(torch.isfinite(flat).all().item())
    return {
        "present": True,
        "shape": list(local.shape),
        "dtype": str(local.dtype),
        "device": str(local.device),
        "numel": local.numel(),
        "max_abs": float(flat.abs().max().item()) if flat.numel() else 0.0,
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
        "nonzero": int(flat.ne(0).sum().item()),
        "finite": finite,
    }


def _optimizer_parameters(trainer: Trainer) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for optimizer in trainer.engine.optim.optimizers.optimizers:
        for group in optimizer.param_groups:
            params.extend(group["params"])
    return params


def _total_grad_norm(
    parameters: list[torch.nn.Parameter],
    trainer: Trainer,
) -> torch.Tensor:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return torch.zeros((), device=trainer.engine.device)
    total_norm = torch.nn.utils.get_total_norm(
        grads,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=True,
    )
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    pp_mesh = trainer.engine.parallelism_context.get_optional_mesh("pp")
    if pp_mesh is not None:
        total_norm = total_norm.pow(2.0)
        torch.distributed.all_reduce(
            total_norm,
            op=torch.distributed.ReduceOp.SUM,
            group=pp_mesh.get_group(),
        )
        total_norm = total_norm.pow(0.5)
    return total_norm


def capture_lora_state(trainer: Trainer) -> dict[str, Any]:
    """Capture rank-local LoRA parameter values and gradients."""

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    optimizer_params = _optimizer_parameters(trainer)
    all_norm = _total_grad_norm(trainer.engine.optim.parameters, trainer)
    optimizer_norm = _total_grad_norm(optimizer_params, trainer)
    records: dict[str, dict[str, Any]] = {}
    for model_part in trainer.engine.model_parts:
        for name, param in model_part.named_parameters():
            canonical_name = canonical_fqn(name)
            if not _is_lora_parameter(canonical_name):
                continue
            records[canonical_name] = {
                "requires_grad": bool(param.requires_grad),
                "parameter_stats": summarize_tensor(param),
                "gradient_stats": summarize_tensor(param.grad),
                "parameter": _clone_cpu_float(param),
                "gradient": _clone_cpu_float(param.grad),
            }
    return {
        "schema_version": 1,
        "rank": rank,
        "world_size": world_size,
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "all_parameter_grad_norm": float(all_norm.detach().cpu().item()),
        "optimizer_parameter_grad_norm": float(optimizer_norm.detach().cpu().item()),
        "lora": records,
    }


def save_rank_probe(snapshot: dict[str, Any], output_dir: Path) -> Path:
    """Save one rank's probe snapshot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rank = int(snapshot["rank"])
    path = output_dir / f"rank_{rank}.pt"
    tmp_path = output_dir / f"rank_{rank}.pt.tmp"
    torch.save(snapshot, tmp_path)
    tmp_path.replace(path)
    metadata_path = output_dir / f"rank_{rank}.summary.json"
    summary = {key: value for key, value in snapshot.items() if key not in {"lora"}}
    summary["num_lora_parameters"] = len(snapshot["lora"])
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def compare_probe_dirs(
    plain_dir: Path,
    stream_dir: Path,
) -> dict[str, Any]:
    """Compare rank-local LoRA params and grads from two probe directories."""

    plain = _load_rank_snapshots(plain_dir)
    stream = _load_rank_snapshots(stream_dir)
    ranks = sorted(set(plain) | set(stream))
    rows: dict[str, dict[str, Any]] = {}
    for rank in ranks:
        plain_snapshot = plain.get(rank)
        stream_snapshot = stream.get(rank)
        plain_lora = {} if plain_snapshot is None else plain_snapshot["lora"]
        stream_lora = {} if stream_snapshot is None else stream_snapshot["lora"]
        names = sorted(set(plain_lora) | set(stream_lora))
        for name in names:
            plain_record = plain_lora.get(name)
            stream_record = stream_lora.get(name)
            row = rows.setdefault(
                name,
                {
                    "name": name,
                    "max_abs_parameter_delta": 0.0,
                    "max_abs_gradient_delta": 0.0,
                    "parameter_delta_rank": None,
                    "gradient_delta_rank": None,
                    "plain_gradient_present_ranks": [],
                    "stream_gradient_present_ranks": [],
                    "missing_ranks": [],
                    "shape_mismatches": [],
                },
            )
            if plain_record is None or stream_record is None:
                row["missing_ranks"].append(rank)
                continue
            if plain_record["gradient"] is not None:
                row["plain_gradient_present_ranks"].append(rank)
            if stream_record["gradient"] is not None:
                row["stream_gradient_present_ranks"].append(rank)
            param_delta = max_abs_delta(
                plain_record["parameter"],
                stream_record["parameter"],
            )
            grad_delta = max_abs_delta(
                plain_record["gradient"],
                stream_record["gradient"],
            )
            if param_delta is None or grad_delta is None:
                row["shape_mismatches"].append(rank)
                continue
            if param_delta > row["max_abs_parameter_delta"]:
                row["max_abs_parameter_delta"] = param_delta
                row["parameter_delta_rank"] = rank
            if grad_delta > row["max_abs_gradient_delta"]:
                row["max_abs_gradient_delta"] = grad_delta
                row["gradient_delta_rank"] = rank

    differing = [
        row
        for row in rows.values()
        if row["max_abs_parameter_delta"] != 0.0
        or row["max_abs_gradient_delta"] != 0.0
        or row["missing_ranks"]
        or row["shape_mismatches"]
    ]
    return {
        "schema_version": 1,
        "plain_dir": str(plain_dir),
        "stream_dir": str(stream_dir),
        "ranks": ranks,
        "plain_norms_by_rank": {
            rank: _norm_summary(snapshot) for rank, snapshot in plain.items()
        },
        "stream_norms_by_rank": {
            rank: _norm_summary(snapshot) for rank, snapshot in stream.items()
        },
        "num_parameters": len(rows),
        "num_differing_parameters": len(differing),
        "max_abs_parameter_delta": max(
            (row["max_abs_parameter_delta"] for row in rows.values()),
            default=0.0,
        ),
        "max_abs_gradient_delta": max(
            (row["max_abs_gradient_delta"] for row in rows.values()),
            default=0.0,
        ),
        "parameters": [rows[name] for name in sorted(rows)],
    }


def max_abs_delta(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> float | None:
    """Return the max absolute difference, treating two missing tensors as equal."""

    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return None
    if tuple(left.shape) != tuple(right.shape):
        return None
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().item())


def write_comparison_markdown(comparison: dict[str, Any], output_path: Path) -> None:
    """Write a compact Markdown rendering of a probe comparison."""

    top = sorted(
        comparison["parameters"],
        key=lambda row: (
            row["max_abs_gradient_delta"],
            row["max_abs_parameter_delta"],
        ),
        reverse=True,
    )[:40]
    lines = [
        "# GLM stream LoRA gradient probe comparison",
        "",
        f"Plain: `{comparison['plain_dir']}`",
        f"Stream: `{comparison['stream_dir']}`",
        "",
        f"- parameters: {comparison['num_parameters']}",
        f"- differing parameters: {comparison['num_differing_parameters']}",
        f"- max |parameter delta|: {comparison['max_abs_parameter_delta']:.9g}",
        f"- max |gradient delta|: {comparison['max_abs_gradient_delta']:.9g}",
        "",
        "## Norms",
        "",
        "| Rank | plain all | plain optim | stream all | stream optim |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    ranks = comparison["ranks"]
    for rank in ranks:
        plain_norms = comparison["plain_norms_by_rank"].get(rank, {})
        stream_norms = comparison["stream_norms_by_rank"].get(rank, {})
        lines.append(
            "| "
            f"{rank} | "
            f"{plain_norms.get('all_parameter_grad_norm', 'missing')} | "
            f"{plain_norms.get('optimizer_parameter_grad_norm', 'missing')} | "
            f"{stream_norms.get('all_parameter_grad_norm', 'missing')} | "
            f"{stream_norms.get('optimizer_parameter_grad_norm', 'missing')} |"
        )
    lines.extend(
        [
            "",
            "## Largest Parameter Or Gradient Deltas",
            "",
            "| Parameter | max |param delta| | param rank | max |grad delta| | grad rank | plain grad ranks | stream grad ranks |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in top:
        lines.append(
            "| "
            f"`{row['name']}` | "
            f"{row['max_abs_parameter_delta']:.9g} | "
            f"{row['parameter_delta_rank']} | "
            f"{row['max_abs_gradient_delta']:.9g} | "
            f"{row['gradient_delta_rank']} | "
            f"{row['plain_gradient_present_ranks']} | "
            f"{row['stream_gradient_present_ranks']} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _norm_summary(snapshot: dict[str, Any]) -> dict[str, float]:
    return {
        "all_parameter_grad_norm": snapshot["all_parameter_grad_norm"],
        "optimizer_parameter_grad_norm": snapshot["optimizer_parameter_grad_norm"],
    }


def _load_rank_snapshots(path: Path) -> dict[int, dict[str, Any]]:
    snapshots = {}
    for snapshot_path in sorted(path.glob("rank_*.pt")):
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        snapshots[int(snapshot["rank"])] = snapshot
    return snapshots


def _run_probe(argv: list[str]) -> None:
    init_logger()
    config = cast(Trainer.Config, ConfigLoader().load(argv))
    probe_dir = Path(
        os.environ.get(
            "GLM53_STREAM_GRAD_PROBE_DIR",
            str(Path(config.dump_folder) / "grad_probe"),
        )
    )
    with config.run_evidence.build(
        dump_folder=config.dump_folder,
        job_config=config.to_dict(),
        role="trainer",
        actor_id="grad_probe",
    ):
        trainer: Trainer | None = None
        try:
            sl.init_structured_logger(
                source="training",
                output_dir=config.dump_folder,
                enable=config.debug.enable_structured_logging,
            )
            trainer = config.build()
            trainer.engine.load_checkpoint()
            _patch_probe_step(trainer, probe_dir)
            logger.info("Running one GLM stream gradient-probe optimizer step")
            data_iterator = trainer.microbatch_generator(trainer.dataloader)
            trainer.train_step(data_iterator)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        finally:
            if trainer is not None:
                trainer.close()
            if torch.distributed.is_initialized():
                with sl.log_trace_span("torch_distributed_teardown"):
                    torch.distributed.destroy_process_group()
            sl.close_structured_logger()


def _patch_probe_step(trainer: Trainer, probe_dir: Path) -> None:
    def probe_step(loss: torch.Tensor, *, current_step: int) -> torch.Tensor:
        del loss, current_step
        snapshot = capture_lora_state(trainer)
        path = save_rank_probe(snapshot, probe_dir)
        logger.info("Saved GLM stream gradient probe to %s", path)
        return torch.tensor(
            snapshot["all_parameter_grad_norm"],
            dtype=torch.float32,
            device=trainer.engine.device,
        )

    trainer.engine.optim.step = probe_step


def _compare(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Compare GLM LoRA grad probes.")
    parser.add_argument("--plain-dir", required=True, type=Path)
    parser.add_argument("--stream-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args(argv)
    comparison = compare_probe_dirs(args.plain_dir, args.stream_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    write_comparison_markdown(comparison, args.output_md)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        _compare(sys.argv[2:])
    else:
        argv = sys.argv[1:]
        if argv[:1] == ["run"]:
            argv = argv[1:]
        _run_probe(argv)


if __name__ == "__main__":
    main()
