# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-step full-gradient probes for the GLM EP comparison ladder.

The probe runs one optimizer step up to the post-backward/pre-update point,
gathers trainable parameter gradients to full CPU fp32 tensors, records the
core TorchTitan grad-norm metric without clipping, and exits before mutating
weights. It is intentionally experiment-local: the ticket needs heavyweight
diagnostic evidence, not a new core training feature.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import ConfigLoader
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer


logger = logging.getLogger(__name__)


def is_lora_parameter(name: str) -> bool:
    """Return whether ``name`` is one of the GLM LoRA adapter parameters."""

    return name.endswith(".lora_a") or name.endswith(".lora_b")


def should_probe_parameter(name: str, param: torch.nn.Parameter) -> bool:
    """Probe LoRA params and any parameter left trainable by the model policy."""

    return is_lora_parameter(name) or param.requires_grad or param.grad is not None


def full_cpu_float_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Gather a possibly-DTensor value and return a CPU fp32 clone."""

    if tensor is None:
        return None
    local_or_full = tensor.detach()
    if isinstance(local_or_full, DTensor):
        local_or_full = local_or_full.full_tensor()
    return local_or_full.to(device="cpu", dtype=torch.float32).clone()


def tensor_stats(tensor: torch.Tensor | None) -> dict[str, Any]:
    """Return stable JSON stats for a gathered tensor."""

    if tensor is None:
        return {
            "present": False,
            "shape": None,
            "numel": 0,
            "l2_norm": None,
            "max_abs": None,
            "nonzero": 0,
            "finite": True,
        }
    flat = tensor.reshape(-1)
    finite = bool(torch.isfinite(flat).all().item())
    return {
        "present": True,
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "l2_norm": float(torch.linalg.vector_norm(flat.double()).item()),
        "max_abs": float(flat.abs().max().item()) if flat.numel() else 0.0,
        "nonzero": int(flat.ne(0).sum().item()),
        "finite": finite,
    }


def offline_grad_norm(records: dict[str, dict[str, Any]]) -> float:
    """Compute global L2 grad norm from gathered fp32 gradient dumps."""

    sumsq = torch.zeros((), dtype=torch.float64)
    for record in records.values():
        gradient = record.get("gradient")
        if gradient is None:
            continue
        flat = gradient.reshape(-1).double()
        sumsq += torch.dot(flat, flat)
    return float(torch.sqrt(sumsq).item())


def core_grad_norm(trainer: Trainer) -> torch.Tensor:
    """Compute TorchTitan's logged grad norm without changing gradients."""

    return dist_utils.clip_grad_norm_(
        trainer.engine.optim.parameters,
        max_norm=math.inf,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=True,
        pp_mesh=trainer.engine.parallelism_context.get_optional_mesh("pp"),
        ep_enabled=trainer.engine.parallelism_context.ep_enabled,
    )


def capture_gradients(
    trainer: Trainer,
    *,
    current_step: int,
    loss: torch.Tensor,
    tensor_save_rank: int,
) -> dict[str, Any]:
    """Capture full trainable gradients and norm evidence for one rank."""

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    records: dict[str, dict[str, Any]] = {}
    num_trainable = 0
    num_lora = 0
    for model_part in trainer.engine.model_parts:
        for raw_name, param in model_part.named_parameters():
            name = canonical_fqn(raw_name)
            if not should_probe_parameter(name, param):
                continue
            num_trainable += int(param.requires_grad)
            num_lora += int(is_lora_parameter(name))
            gradient = full_cpu_float_tensor(param.grad)
            parameter = full_cpu_float_tensor(param)
            records[name] = {
                "requires_grad": bool(param.requires_grad),
                "is_lora": is_lora_parameter(name),
                "parameter_stats": tensor_stats(parameter),
                "gradient_stats": tensor_stats(gradient),
                "gradient": gradient if rank == tensor_save_rank else None,
            }

    core_norm = core_grad_norm(trainer).detach().cpu()
    snapshot = {
        "schema_version": 1,
        "rank": rank,
        "world_size": world_size,
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "current_step": int(current_step),
        "loss": float(loss.detach().float().cpu().item()),
        "core_logged_grad_norm_no_clip": float(core_norm.item()),
        "offline_full_grad_norm": offline_grad_norm(records)
        if rank == tensor_save_rank
        else None,
        "num_probe_parameters": len(records),
        "num_trainable_parameters": num_trainable,
        "num_lora_parameters": num_lora,
        "tensor_save_rank": tensor_save_rank,
        "parameters": records,
    }
    return snapshot


def save_probe(snapshot: dict[str, Any], output_dir: Path) -> Path:
    """Save one rank's probe snapshot and compact summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rank = int(snapshot["rank"])
    path = output_dir / f"rank_{rank}.pt"
    tmp_path = output_dir / f"rank_{rank}.pt.tmp"
    torch.save(snapshot, tmp_path)
    tmp_path.replace(path)

    summary = {key: value for key, value in snapshot.items() if key != "parameters"}
    summary_path = output_dir / f"rank_{rank}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def load_probe_snapshot(path: Path) -> dict[str, Any]:
    """Load the rank snapshot that contains full gradient tensors."""

    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)
    candidates = sorted(path.glob("rank_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no rank_*.pt snapshots found under {path}")
    for candidate in candidates:
        snapshot = torch.load(candidate, map_location="cpu", weights_only=False)
        if any(
            record.get("gradient") is not None
            for record in snapshot.get("parameters", {}).values()
        ):
            return snapshot
    return torch.load(candidates[0], map_location="cpu", weights_only=False)


def compare_probe_dirs(
    reference_dir: Path,
    candidate_dir: Path,
) -> dict[str, Any]:
    """Compare two probe directories by parameter gradient tensors."""

    reference = load_probe_snapshot(reference_dir)
    candidate = load_probe_snapshot(candidate_dir)
    rows = compare_snapshots(reference, candidate)
    differing = [
        row
        for row in rows
        if row["max_abs_gradient_delta"] not in (None, 0.0)
        or row["gradient_l2_norm_delta"] not in (None, 0.0)
        or row["reason"] is not None
    ]
    return {
        "schema_version": 1,
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "reference_rank": reference.get("rank"),
        "candidate_rank": candidate.get("rank"),
        "reference_core_logged_grad_norm_no_clip": reference.get(
            "core_logged_grad_norm_no_clip"
        ),
        "candidate_core_logged_grad_norm_no_clip": candidate.get(
            "core_logged_grad_norm_no_clip"
        ),
        "reference_offline_full_grad_norm": reference.get("offline_full_grad_norm"),
        "candidate_offline_full_grad_norm": candidate.get("offline_full_grad_norm"),
        "core_grad_norm_delta": _optional_delta(
            reference.get("core_logged_grad_norm_no_clip"),
            candidate.get("core_logged_grad_norm_no_clip"),
        ),
        "offline_grad_norm_delta": _optional_delta(
            reference.get("offline_full_grad_norm"),
            candidate.get("offline_full_grad_norm"),
        ),
        "reference_core_minus_offline": _optional_delta(
            reference.get("offline_full_grad_norm"),
            reference.get("core_logged_grad_norm_no_clip"),
        ),
        "candidate_core_minus_offline": _optional_delta(
            candidate.get("offline_full_grad_norm"),
            candidate.get("core_logged_grad_norm_no_clip"),
        ),
        "num_parameters": len(rows),
        "num_differing_parameters": len(differing),
        "max_abs_gradient_delta": max(
            (
                row["max_abs_gradient_delta"]
                for row in rows
                if row["max_abs_gradient_delta"] is not None
            ),
            default=None,
        ),
        "max_abs_gradient_l2_norm_delta": max(
            (
                abs(row["gradient_l2_norm_delta"])
                for row in rows
                if row["gradient_l2_norm_delta"] is not None
            ),
            default=None,
        ),
        "parameters": sorted(
            rows,
            key=lambda row: (
                -1.0
                if row["max_abs_gradient_delta"] is None
                else -row["max_abs_gradient_delta"],
                row["name"],
            ),
        ),
    }


def compare_snapshots(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare two loaded probe snapshots."""

    reference_params = reference.get("parameters", {})
    candidate_params = candidate.get("parameters", {})
    rows: list[dict[str, Any]] = []
    for name in sorted(set(reference_params) | set(candidate_params)):
        ref = reference_params.get(name)
        cand = candidate_params.get(name)
        if ref is None or cand is None:
            rows.append(
                {
                    "name": name,
                    "reference_present": ref is not None,
                    "candidate_present": cand is not None,
                    "shape": None,
                    "reference_gradient_l2_norm": None,
                    "candidate_gradient_l2_norm": None,
                    "gradient_l2_norm_delta": None,
                    "max_abs_gradient_delta": None,
                    "reason": "missing parameter",
                }
            )
            continue
        ref_grad = ref.get("gradient")
        cand_grad = cand.get("gradient")
        rows.append(compare_gradient_tensors(name, ref_grad, cand_grad))
    return rows


def compare_gradient_tensors(
    name: str,
    reference: torch.Tensor | None,
    candidate: torch.Tensor | None,
) -> dict[str, Any]:
    """Compare one pair of gathered gradient tensors."""

    if reference is None or candidate is None:
        return {
            "name": name,
            "reference_present": reference is not None,
            "candidate_present": candidate is not None,
            "shape": None,
            "reference_gradient_l2_norm": None,
            "candidate_gradient_l2_norm": None,
            "gradient_l2_norm_delta": None,
            "max_abs_gradient_delta": None,
            "reason": "missing gradient",
        }
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "name": name,
            "reference_present": True,
            "candidate_present": True,
            "shape": [list(reference.shape), list(candidate.shape)],
            "reference_gradient_l2_norm": None,
            "candidate_gradient_l2_norm": None,
            "gradient_l2_norm_delta": None,
            "max_abs_gradient_delta": None,
            "reason": "shape mismatch",
        }
    ref_norm = float(torch.linalg.vector_norm(reference.reshape(-1).double()).item())
    cand_norm = float(torch.linalg.vector_norm(candidate.reshape(-1).double()).item())
    max_delta = (
        0.0
        if reference.numel() == 0
        else float((reference - candidate).abs().max().item())
    )
    return {
        "name": name,
        "reference_present": True,
        "candidate_present": True,
        "shape": list(reference.shape),
        "reference_gradient_l2_norm": ref_norm,
        "candidate_gradient_l2_norm": cand_norm,
        "gradient_l2_norm_delta": cand_norm - ref_norm,
        "max_abs_gradient_delta": max_delta,
        "reason": None,
    }


def write_comparison_markdown(
    comparison: dict[str, Any],
    output_path: Path,
    *,
    reference_label: str,
    candidate_label: str,
) -> None:
    """Write a compact Markdown comparison."""

    top = comparison["parameters"][:40]
    lines = [
        f"# GLM EP gradient probe: {reference_label} vs {candidate_label}",
        "",
        f"Reference: `{comparison['reference_dir']}`",
        f"Candidate: `{comparison['candidate_dir']}`",
        "",
        f"- reference core grad norm: {_fmt(comparison.get('reference_core_logged_grad_norm_no_clip'))}",
        f"- reference offline grad norm: {_fmt(comparison.get('reference_offline_full_grad_norm'))}",
        f"- candidate core grad norm: {_fmt(comparison.get('candidate_core_logged_grad_norm_no_clip'))}",
        f"- candidate offline grad norm: {_fmt(comparison.get('candidate_offline_full_grad_norm'))}",
        f"- core grad norm delta: {_fmt(comparison.get('core_grad_norm_delta'))}",
        f"- offline grad norm delta: {_fmt(comparison.get('offline_grad_norm_delta'))}",
        f"- max |gradient tensor delta|: {_fmt(comparison.get('max_abs_gradient_delta'))}",
        f"- differing parameters: {comparison['num_differing_parameters']} / {comparison['num_parameters']}",
        "",
        "## Largest Gradient Deltas",
        "",
        "| Parameter | shape | reference norm | candidate norm | norm delta | max |grad delta| | note |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in top:
        lines.append(
            "| "
            f"`{row['name']}` | "
            f"`{row.get('shape')}` | "
            f"{_fmt(row.get('reference_gradient_l2_norm'))} | "
            f"{_fmt(row.get('candidate_gradient_l2_norm'))} | "
            f"{_fmt(row.get('gradient_l2_norm_delta'))} | "
            f"{_fmt(row.get('max_abs_gradient_delta'))} | "
            f"{row.get('reason') or ''} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _optional_delta(reference: Any, candidate: Any) -> float | None:
    if reference is None or candidate is None:
        return None
    return float(candidate) - float(reference)


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.10g}"


def _run_probe(argv: list[str]) -> None:
    init_logger()
    config = cast(Trainer.Config, ConfigLoader().load(argv))
    probe_dir = Path(
        os.environ.get(
            "GLM53_EP_GRAD_PROBE_DIR",
            str(Path(config.dump_folder) / "grad_probe"),
        )
    )
    tensor_save_rank = int(os.environ.get("GLM53_EP_GRAD_PROBE_TENSOR_RANK", "0"))
    with config.run_evidence.build(
        dump_folder=config.dump_folder,
        job_config=config.to_dict(),
        role="trainer",
        actor_id="glm53_flash_ep_grad_probe",
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
            _patch_probe_step(
                trainer,
                probe_dir=probe_dir,
                tensor_save_rank=tensor_save_rank,
            )
            logger.info("Running one GLM EP gradient-probe optimizer step")
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


def _patch_probe_step(
    trainer: Trainer,
    *,
    probe_dir: Path,
    tensor_save_rank: int,
) -> None:
    def probe_step(loss: torch.Tensor, *, current_step: int) -> torch.Tensor:
        snapshot = capture_gradients(
            trainer,
            current_step=current_step,
            loss=loss,
            tensor_save_rank=tensor_save_rank,
        )
        path = save_probe(snapshot, probe_dir)
        logger.info("Saved GLM EP gradient probe to %s", path)
        return torch.tensor(
            snapshot["core_logged_grad_norm_no_clip"],
            dtype=torch.float32,
            device=trainer.engine.device,
        )

    trainer.engine.optim.step = probe_step


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("label must be non-empty")
    return label, Path(path)


def _compare(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Compare GLM EP grad probes.")
    parser.add_argument("--reference", required=True, type=_parse_labeled_path)
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        type=_parse_labeled_path,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    reference_label, reference_dir = args.reference
    comparisons: dict[str, Any] = {}
    args.output_md_dir.mkdir(parents=True, exist_ok=True)
    for candidate_label, candidate_dir in args.candidate:
        comparison = compare_probe_dirs(reference_dir, candidate_dir)
        comparisons[candidate_label] = comparison
        write_comparison_markdown(
            comparison,
            args.output_md_dir / f"{reference_label}_vs_{candidate_label}.md",
            reference_label=reference_label,
            candidate_label=candidate_label,
        )

    output = {
        "schema_version": 1,
        "reference_label": reference_label,
        "reference_dir": str(reference_dir),
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        _compare(sys.argv[2:])
        return
    argv = sys.argv[1:]
    if argv[:1] == ["run"]:
        argv = argv[1:]
    _run_probe(argv)


if __name__ == "__main__":
    main()
