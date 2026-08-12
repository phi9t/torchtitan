# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""CLI tools for scaffold-to-policy reasoning tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

from torchtitan.experiments.scaffold_to_policy.arithmetic_words import (
    ArithmeticWordProblem,
    build_report_input,
    build_split_registry,
    evaluate_fixture_rollouts,
    generate_split,
    load_problems,
    prompt_for_problem,
    summarize_evaluations,
    write_json,
    write_jsonl,
)
from torchtitan.experiments.scaffold_to_policy.modular_sequences import (
    ModularSequenceProblem,
)
from torchtitan.experiments.scaffold_to_policy import arc_grid
from torchtitan.experiments.scaffold_to_policy import coding_style
from torchtitan.experiments.scaffold_to_policy import external_harness
from torchtitan.experiments.scaffold_to_policy import gsm_style
from torchtitan.experiments.scaffold_to_policy import math_style
from torchtitan.experiments.scaffold_to_policy import modular_sequences
from torchtitan.experiments.scaffold_to_policy import multiple_choice
from torchtitan.experiments.scaffold_to_policy import report_artifacts


def generate_arithmetic_words(args: argparse.Namespace) -> None:
    problems = generate_split(seed=args.seed, num_problems=args.num_problems)
    write_jsonl(args.output, [problem.to_json() for problem in problems])


def generate_modular_sequences(args: argparse.Namespace) -> None:
    problems = modular_sequences.generate_split(
        seed=args.seed,
        num_problems=args.num_problems,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        min_modulus=args.min_modulus,
        max_modulus=args.max_modulus,
    )
    modular_sequences.write_jsonl(args.output, [problem.to_json() for problem in problems])


def prepare_gsm_style_split(args: argparse.Namespace) -> None:
    problems = gsm_style.load_problems(args.input)
    gsm_style.write_jsonl(args.output, [problem.to_json() for problem in problems])


def _load_public_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], str]:
    if args.offline:
        if args.raw_cache is None:
            raise ValueError("--offline requires --raw-cache")
        return _read_jsonl_rows(args.raw_cache), "raw_cache"
    if args.raw_cache is not None and args.raw_cache.is_file():
        return _read_jsonl_rows(args.raw_cache), "raw_cache"
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            f"datasets is required for {args.command}. Run through the "
            "TorchTitan rootfs, or pass --offline --raw-cache with cached rows."
        ) from exc
    dataset = _load_hf_dataset(args, load_dataset)
    rows = _slice_rows(dataset, limit=args.limit, offset=args.offset)
    if args.raw_cache is not None:
        _write_jsonl_rows(args.raw_cache, rows)
    return rows, "huggingface"


def _load_hf_dataset(
    args: argparse.Namespace,
    load_dataset: Callable[..., object],
) -> object:
    if getattr(args, "subset", None):
        return load_dataset(
            args.dataset,
            args.subset,
            split=args.source_split,
            revision=args.revision,
        )
    return load_dataset(
        args.dataset,
        split=args.source_split,
        revision=args.revision,
    )


def _slice_rows(
    rows: object,
    *,
    limit: int,
    offset: int,
) -> list[dict[str, object]]:
    selected = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if len(selected) >= limit:
            break
        selected.append(dict(row))
    return selected


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid raw cache JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"raw cache row at {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _augment_public_provenance(
    provenance: dict[str, object],
    *,
    row_source: str,
    raw_cache: Path | None,
    offline: bool,
) -> dict[str, object]:
    provenance["row_source"] = row_source
    provenance["offline"] = offline
    if raw_cache is not None:
        provenance["raw_cache"] = str(raw_cache)
        if raw_cache.is_file():
            provenance["raw_cache_artifact"] = report_artifacts.describe_artifact(
                raw_cache,
                run_id=str(provenance.get("source", "")),
            )
    return provenance


def import_gsm8k_split(args: argparse.Namespace) -> None:
    rows, row_source = _load_public_rows(args)
    source = (
        f"{args.dataset}:{args.subset}:{args.revision}:"
        f"{args.source_split}"
    )
    problems = gsm_style.import_public_rows(
        rows,
        source=source,
        limit=None,
        offset=0,
    )
    gsm_style.write_jsonl(args.output, [problem.to_json() for problem in problems])
    if args.provenance is not None:
        provenance = gsm_style.build_public_provenance(
            dataset=args.dataset,
            subset=args.subset,
            revision=args.revision,
            source_split=args.source_split,
            output=args.output,
            limit=args.limit,
            offset=args.offset,
            problems=problems,
        )
        gsm_style.write_json(
            args.provenance,
            _augment_public_provenance(
                provenance,
                row_source=row_source,
                raw_cache=args.raw_cache,
                offline=args.offline,
            ),
        )


def import_math_split(args: argparse.Namespace) -> None:
    rows, row_source = _load_public_rows(args)
    source = (
        f"{args.dataset}:{args.subset}:{args.revision}:"
        f"{args.source_split}"
    )
    problems = math_style.import_public_rows(
        rows,
        source=source,
        limit=None,
        offset=0,
    )
    math_style.write_jsonl(args.output, [problem.to_json() for problem in problems])
    if args.provenance is not None:
        provenance = math_style.build_public_provenance(
            dataset=args.dataset,
            subset=args.subset,
            revision=args.revision,
            source_split=args.source_split,
            output=args.output,
            limit=args.limit,
            offset=args.offset,
            problems=problems,
        )
        math_style.write_json(
            args.provenance,
            _augment_public_provenance(
                provenance,
                row_source=row_source,
                raw_cache=args.raw_cache,
                offline=args.offline,
            ),
        )


def import_aime_split(args: argparse.Namespace) -> None:
    rows, row_source = _load_public_rows(args)
    source = math_style._public_source(
        args.dataset,
        args.subset or "default",
        args.revision,
        args.source_split,
    )
    problems = math_style.import_aime_rows(
        rows,
        source=source,
        limit=None,
        offset=0,
    )
    math_style.write_jsonl(args.output, [problem.to_json() for problem in problems])
    if args.provenance is not None:
        provenance = math_style.build_public_provenance(
            dataset=args.dataset,
            subset=args.subset or "default",
            revision=args.revision,
            source_split=args.source_split,
            output=args.output,
            limit=args.limit,
            offset=args.offset,
            problems=problems,
        )
        math_style.write_json(
            args.provenance,
            _augment_public_provenance(
                provenance,
                row_source=row_source,
                raw_cache=args.raw_cache,
                offline=args.offline,
            ),
        )


def import_gpqa_split(args: argparse.Namespace) -> None:
    rows, row_source = _load_public_rows(args)
    source = multiple_choice._public_source(
        args.dataset,
        args.subset,
        args.revision,
        args.source_split,
    )
    problems = multiple_choice.import_gpqa_rows(
        rows,
        source=source,
        limit=None,
        offset=0,
    )
    multiple_choice.write_jsonl(
        args.output,
        [problem.to_json() for problem in problems],
    )
    if args.provenance is not None:
        provenance = multiple_choice.build_public_provenance(
            dataset=args.dataset,
            subset=args.subset,
            revision=args.revision,
            source_split=args.source_split,
            output=args.output,
            limit=args.limit,
            offset=args.offset,
            problems=problems,
        )
        multiple_choice.write_json(
            args.provenance,
            _augment_public_provenance(
                provenance,
                row_source=row_source,
                raw_cache=args.raw_cache,
                offline=args.offline,
            ),
        )


def import_arc_grid_split(args: argparse.Namespace) -> None:
    task_dir = args.task_dir / args.source_split
    task_paths = sorted(task_dir.glob("*.json"))
    if not task_paths:
        raise ValueError(f"no ARC task JSON files found in {task_dir}")
    source = f"{args.repo_url}:{args.revision}:{args.source_split}"
    problems = arc_grid.import_arc_tasks(
        task_paths,
        source=source,
        limit=args.limit,
        offset=args.offset,
    )
    arc_grid.write_jsonl(args.output, [problem.to_json() for problem in problems])
    if args.provenance is not None:
        provenance = arc_grid.build_public_provenance(
            repo_url=args.repo_url,
            revision=args.revision,
            source_split=args.source_split,
            task_dir=task_dir,
            output=args.output,
            limit=args.limit,
            offset=args.offset,
            problems=problems,
        )
        arc_grid.write_json(args.provenance, provenance)


def import_humaneval_split(args: argparse.Namespace) -> None:
    rows, row_source = _load_public_rows(args)
    source = coding_style._public_source(
        args.dataset,
        args.subset,
        args.revision,
        args.source_split,
    )
    problems = coding_style.import_public_rows(
        rows,
        source=source,
        limit=None,
        offset=0,
    )
    coding_style.write_jsonl(args.output, [problem.to_json() for problem in problems])
    if args.provenance is not None:
        provenance = coding_style.build_public_provenance(
            dataset=args.dataset,
            subset=args.subset,
            revision=args.revision,
            source_split=args.source_split,
            output=args.output,
            limit=args.limit,
            offset=args.offset,
            problems=problems,
        )
        coding_style.write_json(
            args.provenance,
            _augment_public_provenance(
                provenance,
                row_source=row_source,
                raw_cache=args.raw_cache,
                offline=args.offline,
            ),
        )


def import_mbpp_split(args: argparse.Namespace) -> None:
    rows, row_source = _load_public_rows(args)
    source = coding_style._public_source(
        args.dataset,
        args.subset,
        args.revision,
        args.source_split,
    )
    problems = coding_style.import_mbpp_rows(
        rows,
        source=source,
        limit=None,
        offset=0,
    )
    coding_style.write_jsonl(args.output, [problem.to_json() for problem in problems])
    if args.provenance is not None:
        provenance = coding_style.build_public_provenance(
            dataset=args.dataset,
            subset=args.subset,
            revision=args.revision,
            source_split=args.source_split,
            output=args.output,
            limit=args.limit,
            offset=args.offset,
            problems=problems,
        )
        coding_style.write_json(
            args.provenance,
            _augment_public_provenance(
                provenance,
                row_source=row_source,
                raw_cache=args.raw_cache,
                offline=args.offline,
            ),
        )


def import_bigcodebench_split(args: argparse.Namespace) -> None:
    rows, row_source = _load_public_rows(args)
    source = coding_style._public_source(
        args.dataset,
        args.subset,
        args.revision,
        args.source_split,
    )
    problems = coding_style.import_bigcodebench_rows(
        rows,
        source=source,
        limit=None,
        offset=0,
    )
    coding_style.write_jsonl(args.output, [problem.to_json() for problem in problems])
    if args.provenance is not None:
        provenance = coding_style.build_public_provenance(
            dataset=args.dataset,
            subset=args.subset,
            revision=args.revision,
            source_split=args.source_split,
            output=args.output,
            limit=args.limit,
            offset=args.offset,
            problems=problems,
        )
        coding_style.write_json(
            args.provenance,
            _augment_public_provenance(
                provenance,
                row_source=row_source,
                raw_cache=args.raw_cache,
                offline=args.offline,
            ),
        )


def validate_arithmetic_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    registry = build_split_registry(split_paths)
    write_json(args.output, registry)
    if not registry["selected"]:
        raise SystemExit("arithmetic split validation failed")


def validate_modular_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    registry = modular_sequences.build_split_registry(split_paths)
    modular_sequences.write_json(args.output, registry)
    if not registry["selected"]:
        raise SystemExit("modular split validation failed")


def validate_gsm_style_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    registry = gsm_style.build_split_registry(split_paths)
    gsm_style.write_json(args.output, registry)
    if not registry["selected"]:
        raise SystemExit("gsm-style split validation failed")


def validate_math_style_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    registry = math_style.build_split_registry(split_paths)
    math_style.write_json(args.output, registry)
    if not registry["selected"]:
        raise SystemExit("math-style split validation failed")


def validate_coding_style_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    registry = coding_style.build_split_registry(split_paths)
    coding_style.write_json(args.output, registry)
    if not registry["selected"]:
        raise SystemExit("coding-style split validation failed")


def validate_multiple_choice_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    registry = multiple_choice.build_split_registry(split_paths)
    multiple_choice.write_json(args.output, registry)
    if not registry["selected"]:
        raise SystemExit("multiple-choice split validation failed")


def validate_arc_grid_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    registry = arc_grid.build_split_registry(split_paths)
    arc_grid.write_json(args.output, registry)
    if not registry["selected"]:
        raise SystemExit("arc-grid split validation failed")


def evaluate_arithmetic_fixture(args: argparse.Namespace) -> None:
    problems = load_problems(args.problems)
    fixture = _load_fixture(args.rollouts)
    evaluations = []
    for problem in problems:
        if problem.problem_id not in fixture:
            raise ValueError(f"missing rollouts for {problem.problem_id}")
        evaluations.append(
            evaluate_fixture_rollouts(
                problem,
                fixture[problem.problem_id][: args.max_rollouts],
            )
        )
    write_jsonl(args.output, [evaluation.to_json() for evaluation in evaluations])
    write_json(args.summary, summarize_evaluations(evaluations))


def evaluate_modular_fixture(args: argparse.Namespace) -> None:
    problems = modular_sequences.load_problems(args.problems)
    fixture = _load_fixture(args.rollouts)
    evaluations = []
    for problem in problems:
        if problem.problem_id not in fixture:
            raise ValueError(f"missing rollouts for {problem.problem_id}")
        evaluations.append(
            modular_sequences.evaluate_fixture_rollouts(
                problem,
                fixture[problem.problem_id][: args.max_rollouts],
            )
        )
    modular_sequences.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    modular_sequences.write_json(
        args.summary,
        modular_sequences.summarize_evaluations(evaluations),
    )


def evaluate_gsm_style_fixture(args: argparse.Namespace) -> None:
    problems = gsm_style.load_problems(args.problems)
    fixture = _load_fixture(args.rollouts)
    evaluations = []
    for problem in problems:
        if problem.problem_id not in fixture:
            raise ValueError(f"missing rollouts for {problem.problem_id}")
        evaluations.append(
            gsm_style.evaluate_fixture_rollouts(
                problem,
                fixture[problem.problem_id][: args.max_rollouts],
            )
        )
    gsm_style.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    gsm_style.write_json(
        args.summary,
        gsm_style.summarize_evaluations(evaluations),
    )


def evaluate_math_style_fixture(args: argparse.Namespace) -> None:
    problems = math_style.load_problems(args.problems)
    fixture = _load_fixture(args.rollouts)
    evaluations = []
    for problem in problems:
        if problem.problem_id not in fixture:
            raise ValueError(f"missing rollouts for {problem.problem_id}")
        evaluations.append(
            math_style.evaluate_fixture_rollouts(
                problem,
                fixture[problem.problem_id][: args.max_rollouts],
            )
        )
    math_style.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    math_style.write_json(
        args.summary,
        math_style.summarize_evaluations(evaluations),
    )


def evaluate_coding_style_fixture(args: argparse.Namespace) -> None:
    problems = coding_style.load_problems(args.problems)
    fixture = _load_fixture(args.rollouts)
    evaluations = []
    for problem in problems:
        if problem.problem_id not in fixture:
            raise ValueError(f"missing rollouts for {problem.problem_id}")
        evaluations.append(
            coding_style.evaluate_fixture_rollouts(
                problem,
                fixture[problem.problem_id][: args.max_rollouts],
                timeout_seconds=args.timeout_seconds,
            )
        )
    coding_style.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    coding_style.write_json(
        args.summary,
        coding_style.summarize_evaluations(evaluations),
    )


def preflight_coding_style_canonical(args: argparse.Namespace) -> None:
    problems = coding_style.load_problems(args.problems)
    preflight = coding_style.preflight_canonical_solutions(
        problems,
        timeout_seconds=args.timeout_seconds,
    )
    coding_style.write_json(args.output, preflight)
    if args.require_selected and not preflight["selected"]:
        failed = [
            record["problem_id"]
            for record in preflight["records"]
            if not record["success"]
        ]
        raise SystemExit(
            "coding-style canonical preflight failed: " + ", ".join(failed)
        )


def evaluate_multiple_choice_fixture(args: argparse.Namespace) -> None:
    problems = multiple_choice.load_problems(args.problems)
    fixture = _load_fixture(args.rollouts)
    evaluations = []
    for problem in problems:
        if problem.problem_id not in fixture:
            raise ValueError(f"missing rollouts for {problem.problem_id}")
        evaluations.append(
            multiple_choice.evaluate_fixture_rollouts(
                problem,
                fixture[problem.problem_id][: args.max_rollouts],
            )
        )
    multiple_choice.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    multiple_choice.write_json(
        args.summary,
        multiple_choice.summarize_evaluations(evaluations),
    )


def evaluate_arc_grid_fixture(args: argparse.Namespace) -> None:
    problems = arc_grid.load_problems(args.problems)
    fixture = _load_fixture(args.rollouts)
    evaluations = []
    for problem in problems:
        if problem.problem_id not in fixture:
            raise ValueError(f"missing rollouts for {problem.problem_id}")
        evaluations.append(
            arc_grid.evaluate_fixture_rollouts(
                problem,
                fixture[problem.problem_id][: args.max_rollouts],
            )
        )
    arc_grid.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    arc_grid.write_json(
        args.summary,
        arc_grid.summarize_evaluations(evaluations),
    )


def preflight_arc_grid_prompts(args: argparse.Namespace) -> None:
    problems = arc_grid.load_problems(args.problems)
    prompts = _build_arc_grid_vllm_prompts(problems, args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_counts = {
        problem.problem_id: len(tokenizer(prompt, add_special_tokens=False).input_ids)
        for problem, prompt in zip(problems, prompts)
    }
    preflight = arc_grid.build_prompt_preflight(
        problems=problems,
        token_counts=token_counts,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
    )
    arc_grid.write_json(args.output, preflight)
    if args.require_selected and not preflight["selected"]:
        failed = [
            record["problem_id"]
            for record in preflight["records"]
            if not record["selected"]
        ]
        raise SystemExit("arc-grid prompt preflight failed: " + ", ".join(failed))


def preflight_vllm_gpu_memory(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for preflight-vllm-gpu-memory. Run through "
            "the TorchTitan rootfs."
        ) from exc

    if not torch.cuda.is_available():
        preflight = {
            "schema_version": 1,
            "kind": "vllm_gpu_memory_preflight",
            "selected": False,
            "reason": "cuda unavailable",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "devices": [],
        }
    else:
        device_index = args.device_index
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            device_name = torch.cuda.get_device_name(device_index)
        except Exception as exc:
            preflight = {
                "schema_version": 1,
                "kind": "vllm_gpu_memory_preflight",
                "selected": False,
                "reason": "cuda memory query failed",
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "devices": [
                    {
                        "device_index": device_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "selected": False,
                    }
                ],
            }
        else:
            required_bytes = int(total_bytes * args.gpu_memory_utilization)
            preflight = {
                "schema_version": 1,
                "kind": "vllm_gpu_memory_preflight",
                "selected": free_bytes >= required_bytes,
                "reason": "sufficient free memory"
                if free_bytes >= required_bytes
                else "insufficient free memory",
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "devices": [
                    {
                        "device_index": device_index,
                        "name": device_name,
                        "free_bytes": free_bytes,
                        "total_bytes": total_bytes,
                        "required_bytes": required_bytes,
                        "free_gib": free_bytes / (1024**3),
                        "total_gib": total_bytes / (1024**3),
                        "required_gib": required_bytes / (1024**3),
                        "selected": free_bytes >= required_bytes,
                    }
                ],
            }
    arc_grid.write_json(args.output, preflight)
    if args.require_selected and not preflight["selected"]:
        device = preflight["devices"][0] if preflight["devices"] else {}
        raise SystemExit(
            "vllm gpu memory preflight failed: "
            f"{preflight['reason']}; "
            f"free_gib={device.get('free_gib')}; "
            f"required_gib={device.get('required_gib')}"
        )


def write_blocker_report_input(args: argparse.Namespace) -> None:
    artifact_paths = _parse_split_paths(args.artifact)
    blocker_payloads = report_artifacts.load_json_files(artifact_paths)
    artifact_details = report_artifacts.describe_artifacts(
        artifact_paths,
        run_id=args.run_id,
        payloads=blocker_payloads,
    )
    freshness = report_artifacts.summarize_artifact_freshness(artifact_details)
    selected_values = [
        payload.get("selected")
        for payload in blocker_payloads.values()
        if isinstance(payload, dict) and "selected" in payload
    ]
    checks = {
        "blocker_artifacts_present": all(
            path.is_file() for path in artifact_paths.values()
        ),
        "artifact_provenance_labeled": bool(freshness["all_labeled"]),
        "benchmark_execution_completed": False,
    }
    if selected_values:
        checks["blocker_selected"] = all(bool(value) for value in selected_values)
    report = {
        "schema_version": 1,
        "run": {
            "run_id": args.run_id,
            "task": args.task,
            "lane": args.lane,
            "scaffold": {
                "type": args.blocker_type,
                "budget": 0,
            },
        },
        "checks": checks,
        "limitations": args.limitation,
        "artifacts": {
            "results_root": str(args.results_root),
            "blockers": {name: str(path) for name, path in artifact_paths.items()},
            "details": artifact_details,
            "freshness": freshness,
        },
        "blockers": blocker_payloads,
    }
    arc_grid.write_json(args.output, report)


def evaluate_arithmetic_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "VLLM_USE_FLASHINFER_SAMPLER",
        args.use_flashinfer_sampler,
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for evaluate-arithmetic-vllm. Run through the "
            "TorchTitan rootfs or use evaluate-arithmetic-fixture."
        ) from exc

    problems = load_problems(args.problems)
    prompts = _build_arithmetic_vllm_prompts(problems, args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
    )
    llm_kwargs = {
        "model": args.model,
        "attention_backend": args.attention_backend,
        "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)
    evaluations = []
    for problem, output in zip(problems, outputs):
        evaluations.append(
            evaluate_fixture_rollouts(
                problem,
                [candidate.text for candidate in output.outputs],
            )
        )
    write_jsonl(args.output, [evaluation.to_json() for evaluation in evaluations])
    write_json(args.summary, summarize_evaluations(evaluations))


def evaluate_modular_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "VLLM_USE_FLASHINFER_SAMPLER",
        args.use_flashinfer_sampler,
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for evaluate-modular-vllm. Run through the "
            "TorchTitan rootfs or use evaluate-modular-fixture."
        ) from exc
    lora_request = None
    if args.lora_adapter is not None:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError(
                "vLLM LoRA support is required when --lora-adapter is set."
            ) from exc
        lora_request = LoRARequest(
            lora_name=args.lora_name,
            lora_int_id=args.lora_id,
            lora_path=str(args.lora_adapter),
        )

    problems = modular_sequences.load_problems(args.problems)
    prompts = _build_modular_vllm_prompts(problems, args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
    )
    llm_kwargs = {
        "model": args.model,
        "enable_lora": args.lora_adapter is not None,
        "attention_backend": args.attention_backend,
        "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
    }
    if args.lora_adapter is not None:
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )
    evaluations = []
    for problem, output in zip(problems, outputs):
        evaluations.append(
            modular_sequences.evaluate_fixture_rollouts(
                problem,
                [candidate.text for candidate in output.outputs],
            )
        )
    modular_sequences.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    modular_sequences.write_json(
        args.summary,
        modular_sequences.summarize_evaluations(evaluations),
    )


def evaluate_gsm_style_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "VLLM_USE_FLASHINFER_SAMPLER",
        args.use_flashinfer_sampler,
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for evaluate-gsm-style-vllm. Run through the "
            "TorchTitan rootfs or use evaluate-gsm-style-fixture."
        ) from exc

    problems = gsm_style.load_problems(args.problems)
    prompts = _build_gsm_style_vllm_prompts(problems, args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
    )
    llm_kwargs = {
        "model": args.model,
        "attention_backend": args.attention_backend,
        "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)
    evaluations = []
    for problem, output in zip(problems, outputs):
        evaluations.append(
            gsm_style.evaluate_fixture_rollouts(
                problem,
                [candidate.text for candidate in output.outputs],
            )
        )
    gsm_style.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    gsm_style.write_json(
        args.summary,
        gsm_style.summarize_evaluations(evaluations),
    )


def evaluate_math_style_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "VLLM_USE_FLASHINFER_SAMPLER",
        args.use_flashinfer_sampler,
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for evaluate-math-style-vllm. Run through the "
            "TorchTitan rootfs or use evaluate-math-style-fixture."
        ) from exc

    problems = math_style.load_problems(args.problems)
    prompts = _build_math_style_vllm_prompts(problems, args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
    )
    llm_kwargs = {
        "model": args.model,
        "attention_backend": args.attention_backend,
        "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)
    evaluations = []
    for problem, output in zip(problems, outputs):
        evaluations.append(
            math_style.evaluate_fixture_rollouts(
                problem,
                [candidate.text for candidate in output.outputs],
            )
        )
    math_style.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    math_style.write_json(
        args.summary,
        math_style.summarize_evaluations(evaluations),
    )


def evaluate_coding_style_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "VLLM_USE_FLASHINFER_SAMPLER",
        args.use_flashinfer_sampler,
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for evaluate-coding-style-vllm. Run through the "
            "TorchTitan rootfs or use evaluate-coding-style-fixture."
        ) from exc

    problems = coding_style.load_problems(args.problems)
    prompts = _build_coding_style_vllm_prompts(problems, args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
    )
    llm_kwargs = {
        "model": args.model,
        "attention_backend": args.attention_backend,
        "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)
    evaluations = []
    for problem, output in zip(problems, outputs):
        evaluations.append(
            coding_style.evaluate_fixture_rollouts(
                problem,
                [candidate.text for candidate in output.outputs],
                timeout_seconds=args.timeout_seconds,
            )
        )
    coding_style.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    coding_style.write_json(
        args.summary,
        coding_style.summarize_evaluations(evaluations),
    )


def evaluate_multiple_choice_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "VLLM_USE_FLASHINFER_SAMPLER",
        args.use_flashinfer_sampler,
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for evaluate-multiple-choice-vllm. Run through "
            "the TorchTitan rootfs or use evaluate-multiple-choice-fixture."
        ) from exc

    problems = multiple_choice.load_problems(args.problems)
    prompts = _build_multiple_choice_vllm_prompts(problems, args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
    )
    llm_kwargs = {
        "model": args.model,
        "attention_backend": args.attention_backend,
        "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)
    evaluations = []
    for problem, output in zip(problems, outputs):
        evaluations.append(
            multiple_choice.evaluate_fixture_rollouts(
                problem,
                [candidate.text for candidate in output.outputs],
            )
        )
    multiple_choice.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    multiple_choice.write_json(
        args.summary,
        multiple_choice.summarize_evaluations(evaluations),
    )


def evaluate_arc_grid_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "VLLM_USE_FLASHINFER_SAMPLER",
        args.use_flashinfer_sampler,
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for evaluate-arc-grid-vllm. Run through "
            "the TorchTitan rootfs or use evaluate-arc-grid-fixture."
        ) from exc

    problems = arc_grid.load_problems(args.problems)
    prompts = _build_arc_grid_vllm_prompts(problems, args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
    )
    llm_kwargs = {
        "model": args.model,
        "attention_backend": args.attention_backend,
        "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)
    evaluations = []
    for problem, output in zip(problems, outputs):
        evaluations.append(
            arc_grid.evaluate_fixture_rollouts(
                problem,
                [candidate.text for candidate in output.outputs],
            )
        )
    arc_grid.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    arc_grid.write_json(
        args.summary,
        arc_grid.summarize_evaluations(evaluations),
    )


def build_arithmetic_report_input(args: argparse.Namespace) -> None:
    summary_paths = _parse_split_paths(args.summary)
    report_input = build_report_input(
        data_root=args.data_root,
        results_root=args.results_root,
        run_id=args.run_id,
        split_registry=args.split_registry,
        summary_paths=summary_paths,
    )
    write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"arithmetic report input failed: {', '.join(failed)}")


def build_modular_report_input(args: argparse.Namespace) -> None:
    summary_paths = _parse_split_paths(args.summary)
    evaluation_paths = None
    if args.evaluation:
        evaluation_paths = _parse_split_paths(args.evaluation)
    report_input = modular_sequences.build_report_input(
        data_root=args.data_root,
        results_root=args.results_root,
        run_id=args.run_id,
        split_registry=args.split_registry,
        summary_paths=summary_paths,
        evaluation_paths=evaluation_paths,
    )
    modular_sequences.write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"modular report input failed: {', '.join(failed)}")


def build_gsm_style_report_input(args: argparse.Namespace) -> None:
    summary_paths = _parse_split_paths(args.summary)
    report_input = gsm_style.build_report_input(
        data_root=args.data_root,
        results_root=args.results_root,
        run_id=args.run_id,
        split_registry=args.split_registry,
        summary_paths=summary_paths,
        scaffold_budget=args.scaffold_budget,
    )
    gsm_style.write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"gsm-style report input failed: {', '.join(failed)}")


def build_math_style_report_input(args: argparse.Namespace) -> None:
    summary_paths = _parse_split_paths(args.summary)
    report_input = math_style.build_report_input(
        data_root=args.data_root,
        results_root=args.results_root,
        run_id=args.run_id,
        split_registry=args.split_registry,
        summary_paths=summary_paths,
        scaffold_budget=args.scaffold_budget,
    )
    math_style.write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"math-style report input failed: {', '.join(failed)}")


def build_coding_style_report_input(args: argparse.Namespace) -> None:
    summary_paths = _parse_split_paths(args.summary)
    preflight_paths = None
    if args.preflight:
        preflight_paths = _parse_split_paths(args.preflight)
    report_input = coding_style.build_report_input(
        data_root=args.data_root,
        results_root=args.results_root,
        run_id=args.run_id,
        split_registry=args.split_registry,
        summary_paths=summary_paths,
        scaffold_budget=args.scaffold_budget,
        preflight_paths=preflight_paths,
    )
    coding_style.write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"coding-style report input failed: {', '.join(failed)}")


def build_multiple_choice_report_input(args: argparse.Namespace) -> None:
    summary_paths = _parse_split_paths(args.summary)
    report_input = multiple_choice.build_report_input(
        data_root=args.data_root,
        results_root=args.results_root,
        run_id=args.run_id,
        split_registry=args.split_registry,
        summary_paths=summary_paths,
        scaffold_budget=args.scaffold_budget,
    )
    multiple_choice.write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"multiple-choice report input failed: {', '.join(failed)}")


def build_arc_grid_report_input(args: argparse.Namespace) -> None:
    summary_paths = _parse_split_paths(args.summary)
    preflight_paths = None
    if args.preflight:
        preflight_paths = _parse_split_paths(args.preflight)
    report_input = arc_grid.build_report_input(
        data_root=args.data_root,
        results_root=args.results_root,
        run_id=args.run_id,
        split_registry=args.split_registry,
        summary_paths=summary_paths,
        scaffold_budget=args.scaffold_budget,
        preflight_paths=preflight_paths,
    )
    arc_grid.write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"arc-grid report input failed: {', '.join(failed)}")


def write_latest_report_index(args: argparse.Namespace) -> None:
    index = report_artifacts.build_latest_report_index(
        manifests_dir=args.manifests_dir,
        pattern=args.pattern,
        task=args.task,
    )
    write_json(args.output, index)
    if args.require_selected and not index["selected"]:
        raise SystemExit("latest report index failed: no report inputs matched")


def write_external_harness_smoke(args: argparse.Namespace) -> None:
    if args.harness_family == "harbor_terminal":
        pins = external_harness.default_harbor_terminal_pins()
    elif args.harness_family == "tau2":
        pins = external_harness.default_tau2_pins()
    else:
        raise ValueError(f"unknown harness family: {args.harness_family}")
    external_harness.write_harness_smoke(
        output=args.output,
        run_id=args.run_id,
        harness_family=args.harness_family,
        pins=pins,
        dry_run=args.dry_run,
        task_subset=args.task_subset,
    )


def write_external_harness_preflight(args: argparse.Namespace) -> None:
    if args.harness_family == "harbor_terminal":
        pins = external_harness.default_harbor_terminal_pins()
    elif args.harness_family == "tau2":
        pins = external_harness.default_tau2_pins()
    else:
        raise ValueError(f"unknown harness family: {args.harness_family}")
    external_harness.write_installed_preflight(
        output=args.output,
        run_id=args.run_id,
        harness_family=args.harness_family,
        pins=pins,
        task_subset=args.task_subset,
        cli_names=args.cli_name,
    )


def write_tau2_mock_score_smoke(args: argparse.Namespace) -> None:
    external_harness.write_tau2_mock_score_smoke(
        output=args.output,
        run_id=args.run_id,
        task_id=args.task_id,
        evaluation_type=args.evaluation_type,
    )


def write_terminal_bench_result_smoke(args: argparse.Namespace) -> None:
    external_harness.write_terminal_bench_result_smoke(
        output=args.output,
        run_id=args.run_id,
        task_id=args.task_id,
    )


def write_terminal_bench_execution_probe(args: argparse.Namespace) -> None:
    external_harness.write_terminal_bench_execution_probe(
        output=args.output,
        run_id=args.run_id,
        task_id=args.task_id,
        command=args.command,
        cwd=args.cwd,
        timeout_seconds=args.timeout_seconds,
        harbor_result_json=args.harbor_result_json,
        agent_name=args.agent_name,
    )


def write_tau2_execution_probe(args: argparse.Namespace) -> None:
    external_harness.write_tau2_execution_probe(
        output=args.output,
        run_id=args.run_id,
        task_id=args.task_id,
        command=args.command,
        cwd=args.cwd,
        timeout_seconds=args.timeout_seconds,
        results_json=args.results_json,
    )


def ingest_external_harness_smoke(args: argparse.Namespace) -> None:
    external_harness.ingest_harness_smoke(
        raw_result=args.raw_result,
        output=args.output,
        results_root=args.results_root,
    )


def build_external_harness_report_input(args: argparse.Namespace) -> None:
    ingested_paths = _parse_split_paths(args.ingested)
    report_input = external_harness.build_report_input(
        results_root=args.results_root,
        run_id=args.run_id,
        ingested_paths=ingested_paths,
    )
    external_harness.write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(
            f"external-harness report input failed: {', '.join(failed)}"
        )


def rescore_math_style_evaluations(args: argparse.Namespace) -> None:
    evaluations = math_style.load_evaluations(args.evaluations)
    math_style.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    math_style.write_json(
        args.summary,
        math_style.summarize_evaluations(evaluations),
    )


def rescore_coding_style_evaluations(args: argparse.Namespace) -> None:
    evaluations = coding_style.load_evaluations(args.evaluations)
    coding_style.write_jsonl(
        args.output,
        [evaluation.to_json() for evaluation in evaluations],
    )
    coding_style.write_json(
        args.summary,
        coding_style.summarize_evaluations(evaluations),
    )


def build_modular_dataset(args: argparse.Namespace) -> None:
    evaluations = modular_sequences.load_evaluations(args.evaluations)
    examples = modular_sequences.build_training_examples(
        evaluations,
        condition=args.condition,
    )
    if len(examples) < args.min_examples:
        raise SystemExit(
            f"only {len(examples)} modular training examples, "
            f"need at least {args.min_examples}"
        )
    modular_sequences.write_jsonl(
        args.output,
        [example.to_json() for example in examples],
    )


def write_arithmetic_fixture(args: argparse.Namespace) -> None:
    problems = load_problems(args.problems)
    rows = []
    for problem in problems:
        rollouts = [
            "\n".join(problem.rationale),
            f"FINAL: {problem.answer + 1}",
            "I cannot solve this.",
        ]
        rows.append({"problem_id": problem.problem_id, "rollouts": rollouts})
    write_jsonl(args.output, rows)


def write_modular_fixture(args: argparse.Namespace) -> None:
    problems = modular_sequences.load_problems(args.problems)
    rows = []
    for problem in problems:
        rollouts = [
            "\n".join(problem.rationale),
            f"FINAL: {problem.answer + 1}",
            "I cannot solve this.",
        ]
        rows.append({"problem_id": problem.problem_id, "rollouts": rollouts})
    modular_sequences.write_jsonl(args.output, rows)


def write_gsm_style_fixture(args: argparse.Namespace) -> None:
    problems = gsm_style.load_problems(args.problems)
    rows = []
    for problem in problems:
        correct_answer = problem.rationale or f"FINAL: {problem.answer}"
        rollouts = [
            correct_answer,
            f"FINAL: {problem.normalized_answer} dollars",
            f"FINAL: {int(problem.normalized_answer) + 1}"
            if problem.normalized_answer.lstrip("-").isdigit()
            else "FINAL: 0",
            "I cannot solve this.",
        ]
        rows.append({"problem_id": problem.problem_id, "rollouts": rollouts})
    gsm_style.write_jsonl(args.output, rows)


def _build_arithmetic_vllm_prompts(
    problems: list[ArithmeticWordProblem],
    args: argparse.Namespace,
) -> list[str]:
    if args.prompt_variant == "plain":
        return [prompt_for_problem(problem) for problem in problems]
    if args.prompt_variant != "chat":
        raise ValueError(f"unknown prompt variant: {args.prompt_variant}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You solve arithmetic word problems. Return a short "
                        "calculation trace and end with FINAL: <integer>."
                    ),
                },
                {"role": "user", "content": prompt_for_problem(problem)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]


def _build_modular_vllm_prompts(
    problems: list[ModularSequenceProblem],
    args: argparse.Namespace,
) -> list[str]:
    if args.prompt_variant == "plain":
        return [modular_sequences.prompt_for_problem(problem) for problem in problems]
    if args.prompt_variant not in {"chat", "concise_chat"}:
        raise ValueError(f"unknown prompt variant: {args.prompt_variant}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.prompt_variant == "concise_chat":
        return [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "Compute the recurrence exactly. Keep working brief. "
                            "End with one line: FINAL: <integer>."
                        ),
                    },
                    {"role": "user", "content": _concise_modular_prompt(problem)},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for problem in problems
        ]
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You solve modular arithmetic recurrences exactly. "
                        "Return a short calculation trace and end with "
                        "FINAL: <integer>."
                    ),
                },
                {"role": "user", "content": modular_sequences.prompt_for_problem(problem)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]


def _build_gsm_style_vllm_prompts(
    problems: list[gsm_style.GSMStyleProblem],
    args: argparse.Namespace,
) -> list[str]:
    if args.prompt_variant == "plain":
        return [gsm_style.prompt_for_problem(problem) for problem in problems]
    if args.prompt_variant != "chat":
        raise ValueError(f"unknown prompt variant: {args.prompt_variant}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You solve grade-school math problems. Return a short "
                        "calculation trace and end with FINAL: <answer>."
                    ),
                },
                {"role": "user", "content": gsm_style.prompt_for_problem(problem)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]


def _build_math_style_vllm_prompts(
    problems: list[math_style.MathStyleProblem],
    args: argparse.Namespace,
) -> list[str]:
    if args.prompt_variant == "plain":
        return [math_style.prompt_for_problem(problem) for problem in problems]
    if args.prompt_variant != "chat":
        raise ValueError(f"unknown prompt variant: {args.prompt_variant}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You solve MATH benchmark problems. Return a short "
                        "calculation trace and end with FINAL: <answer>."
                    ),
                },
                {"role": "user", "content": math_style.prompt_for_problem(problem)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]


def _build_coding_style_vllm_prompts(
    problems: list[coding_style.CodingStyleProblem],
    args: argparse.Namespace,
) -> list[str]:
    if args.prompt_variant == "plain":
        return [coding_style.prompt_for_problem(problem) for problem in problems]
    if args.prompt_variant not in {"chat", "contract_chat"}:
        raise ValueError(f"unknown prompt variant: {args.prompt_variant}")
    from transformers import AutoTokenizer

    if args.prompt_variant == "contract_chat":
        system_content = (
            "You solve Python programming tasks. Return only Python code, with "
            "no Markdown. Preserve the requested function name and signature. "
            "Use the imports and globals from the prompt. Implement the exact "
            "side effects, return types, error behavior, and library calls that "
            "the task description requires."
        )
    else:
        system_content = (
            "You solve Python programming tasks. Return only Python "
            "code for the requested function, with no Markdown."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": coding_style.prompt_for_problem(problem)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]


def _build_multiple_choice_vllm_prompts(
    problems: list[multiple_choice.MultipleChoiceProblem],
    args: argparse.Namespace,
) -> list[str]:
    if args.prompt_variant == "plain":
        return [multiple_choice.prompt_for_problem(problem) for problem in problems]
    if args.prompt_variant != "chat":
        raise ValueError(f"unknown prompt variant: {args.prompt_variant}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You answer difficult multiple-choice reasoning questions. "
                        "Return a short reasoning trace and end with exactly "
                        "FINAL: <A|B|C|D>."
                    ),
                },
                {
                    "role": "user",
                    "content": multiple_choice.prompt_for_problem(problem),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]


def _build_arc_grid_vllm_prompts(
    problems: list[arc_grid.ARCGridProblem],
    args: argparse.Namespace,
) -> list[str]:
    if args.prompt_variant == "plain":
        return [arc_grid.prompt_for_problem(problem) for problem in problems]
    if args.prompt_variant not in {"chat", "strict_chat"}:
        raise ValueError(f"unknown prompt variant: {args.prompt_variant}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.prompt_variant == "strict_chat":
        system_prompt = (
            "You solve ARC grid transformation tasks. Return only one line "
            "matching FINAL: <json-grid>. Do not include reasoning, markdown, "
            "labels, or code fences."
        )
        return [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": arc_grid.strict_prompt_for_problem(problem),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for problem in problems
        ]
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You solve ARC grid transformation tasks. Infer the "
                        "rule from examples and end with exactly "
                        "FINAL: <json-grid>."
                    ),
                },
                {
                    "role": "user",
                    "content": arc_grid.prompt_for_problem(problem),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]


def _concise_modular_prompt(problem: ModularSequenceProblem) -> str:
    return (
        f"x0={problem.start}; for i=1..{problem.steps}, "
        f"x_i=({problem.multiplier}*x_(i-1)+{problem.step_coeff}*i+"
        f"{problem.offset}) mod {problem.modulus}. Return x_{problem.steps}. "
        "Last line exactly FINAL: <integer>."
    )


def _load_fixture(path: Path) -> dict[str, list[str]]:
    fixture = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rollouts = row["rollouts"]
            if not isinstance(rollouts, list):
                raise ValueError("rollouts must be a list")
            fixture[str(row["problem_id"])] = [str(rollout) for rollout in rollouts]
        except Exception as exc:
            raise ValueError(f"invalid fixture at {path}:{line_number}: {exc}") from exc
    return fixture


def _parse_split_paths(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        split, sep, path = value.partition("=")
        if not sep:
            raise ValueError(f"expected SPLIT=PATH, got {value}")
        parsed[split] = Path(path)
    return parsed


def _add_public_import_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-cache", type=Path)
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold-to-policy task tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-arithmetic-words")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--seed", type=int, default=1)
    generate_parser.add_argument("--num-problems", type=int, default=100)
    generate_parser.set_defaults(func=generate_arithmetic_words)

    modular_generate_parser = subparsers.add_parser("generate-modular-sequences")
    modular_generate_parser.add_argument("--output", type=Path, required=True)
    modular_generate_parser.add_argument("--seed", type=int, default=1)
    modular_generate_parser.add_argument("--num-problems", type=int, default=100)
    modular_generate_parser.add_argument("--min-steps", type=int, default=7)
    modular_generate_parser.add_argument("--max-steps", type=int, default=13)
    modular_generate_parser.add_argument("--min-modulus", type=int, default=97)
    modular_generate_parser.add_argument("--max-modulus", type=int, default=997)
    modular_generate_parser.set_defaults(func=generate_modular_sequences)

    gsm_prepare_parser = subparsers.add_parser("prepare-gsm-style-split")
    gsm_prepare_parser.add_argument("--input", type=Path, required=True)
    gsm_prepare_parser.add_argument("--output", type=Path, required=True)
    gsm_prepare_parser.set_defaults(func=prepare_gsm_style_split)

    gsm8k_import_parser = subparsers.add_parser("import-gsm8k-split")
    gsm8k_import_parser.add_argument("--output", type=Path, required=True)
    gsm8k_import_parser.add_argument("--provenance", type=Path)
    gsm8k_import_parser.add_argument("--dataset", default="openai/gsm8k")
    gsm8k_import_parser.add_argument("--subset", default="main")
    gsm8k_import_parser.add_argument("--source-split", default="test")
    gsm8k_import_parser.add_argument("--revision", required=True)
    gsm8k_import_parser.add_argument("--limit", type=int, required=True)
    gsm8k_import_parser.add_argument("--offset", type=int, default=0)
    _add_public_import_cache_args(gsm8k_import_parser)
    gsm8k_import_parser.set_defaults(func=import_gsm8k_split)

    math_import_parser = subparsers.add_parser("import-math-split")
    math_import_parser.add_argument("--output", type=Path, required=True)
    math_import_parser.add_argument("--provenance", type=Path)
    math_import_parser.add_argument("--dataset", default="EleutherAI/hendrycks_math")
    math_import_parser.add_argument("--subset", default="algebra")
    math_import_parser.add_argument("--source-split", default="test")
    math_import_parser.add_argument("--revision", required=True)
    math_import_parser.add_argument("--limit", type=int, required=True)
    math_import_parser.add_argument("--offset", type=int, default=0)
    _add_public_import_cache_args(math_import_parser)
    math_import_parser.set_defaults(func=import_math_split)

    aime_import_parser = subparsers.add_parser("import-aime-split")
    aime_import_parser.add_argument("--output", type=Path, required=True)
    aime_import_parser.add_argument("--provenance", type=Path)
    aime_import_parser.add_argument("--dataset", default="HuggingFaceH4/aime_2024")
    aime_import_parser.add_argument("--subset")
    aime_import_parser.add_argument("--source-split", default="train")
    aime_import_parser.add_argument("--revision", required=True)
    aime_import_parser.add_argument("--limit", type=int, required=True)
    aime_import_parser.add_argument("--offset", type=int, default=0)
    _add_public_import_cache_args(aime_import_parser)
    aime_import_parser.set_defaults(func=import_aime_split)

    gpqa_import_parser = subparsers.add_parser("import-gpqa-split")
    gpqa_import_parser.add_argument("--output", type=Path, required=True)
    gpqa_import_parser.add_argument("--provenance", type=Path)
    gpqa_import_parser.add_argument("--dataset", default="Idavidrein/gpqa")
    gpqa_import_parser.add_argument("--subset", default="gpqa_diamond")
    gpqa_import_parser.add_argument("--source-split", default="train")
    gpqa_import_parser.add_argument("--revision", required=True)
    gpqa_import_parser.add_argument("--limit", type=int, required=True)
    gpqa_import_parser.add_argument("--offset", type=int, default=0)
    _add_public_import_cache_args(gpqa_import_parser)
    gpqa_import_parser.set_defaults(func=import_gpqa_split)

    arc_import_parser = subparsers.add_parser("import-arc-grid-split")
    arc_import_parser.add_argument("--task-dir", type=Path, required=True)
    arc_import_parser.add_argument("--output", type=Path, required=True)
    arc_import_parser.add_argument("--provenance", type=Path)
    arc_import_parser.add_argument(
        "--repo-url",
        default="https://github.com/arcprize/ARC-AGI-2.git",
    )
    arc_import_parser.add_argument("--source-split", default="training")
    arc_import_parser.add_argument("--revision", required=True)
    arc_import_parser.add_argument("--limit", type=int, required=True)
    arc_import_parser.add_argument("--offset", type=int, default=0)
    arc_import_parser.set_defaults(func=import_arc_grid_split)

    humaneval_import_parser = subparsers.add_parser("import-humaneval-split")
    humaneval_import_parser.add_argument("--output", type=Path, required=True)
    humaneval_import_parser.add_argument("--provenance", type=Path)
    humaneval_import_parser.add_argument("--dataset", default="openai/openai_humaneval")
    humaneval_import_parser.add_argument("--subset")
    humaneval_import_parser.add_argument("--source-split", default="test")
    humaneval_import_parser.add_argument("--revision", required=True)
    humaneval_import_parser.add_argument("--limit", type=int, required=True)
    humaneval_import_parser.add_argument("--offset", type=int, default=0)
    _add_public_import_cache_args(humaneval_import_parser)
    humaneval_import_parser.set_defaults(func=import_humaneval_split)

    mbpp_import_parser = subparsers.add_parser("import-mbpp-split")
    mbpp_import_parser.add_argument("--output", type=Path, required=True)
    mbpp_import_parser.add_argument("--provenance", type=Path)
    mbpp_import_parser.add_argument(
        "--dataset",
        default="google-research-datasets/mbpp",
    )
    mbpp_import_parser.add_argument("--subset", default="sanitized")
    mbpp_import_parser.add_argument("--source-split", default="test")
    mbpp_import_parser.add_argument("--revision", required=True)
    mbpp_import_parser.add_argument("--limit", type=int, required=True)
    mbpp_import_parser.add_argument("--offset", type=int, default=0)
    _add_public_import_cache_args(mbpp_import_parser)
    mbpp_import_parser.set_defaults(func=import_mbpp_split)

    bigcodebench_import_parser = subparsers.add_parser(
        "import-bigcodebench-split"
    )
    bigcodebench_import_parser.add_argument("--output", type=Path, required=True)
    bigcodebench_import_parser.add_argument("--provenance", type=Path)
    bigcodebench_import_parser.add_argument(
        "--dataset",
        default="bigcode/bigcodebench-hard",
    )
    bigcodebench_import_parser.add_argument("--subset")
    bigcodebench_import_parser.add_argument("--source-split", default="v0.1.4")
    bigcodebench_import_parser.add_argument("--revision", required=True)
    bigcodebench_import_parser.add_argument("--limit", type=int, required=True)
    bigcodebench_import_parser.add_argument("--offset", type=int, default=0)
    _add_public_import_cache_args(bigcodebench_import_parser)
    bigcodebench_import_parser.set_defaults(func=import_bigcodebench_split)

    fixture_writer = subparsers.add_parser("write-arithmetic-fixture")
    fixture_writer.add_argument("--problems", type=Path, required=True)
    fixture_writer.add_argument("--output", type=Path, required=True)
    fixture_writer.set_defaults(func=write_arithmetic_fixture)

    modular_fixture_writer = subparsers.add_parser("write-modular-fixture")
    modular_fixture_writer.add_argument("--problems", type=Path, required=True)
    modular_fixture_writer.add_argument("--output", type=Path, required=True)
    modular_fixture_writer.set_defaults(func=write_modular_fixture)

    gsm_fixture_writer = subparsers.add_parser("write-gsm-style-fixture")
    gsm_fixture_writer.add_argument("--problems", type=Path, required=True)
    gsm_fixture_writer.add_argument("--output", type=Path, required=True)
    gsm_fixture_writer.set_defaults(func=write_gsm_style_fixture)

    eval_parser = subparsers.add_parser("evaluate-arithmetic-fixture")
    eval_parser.add_argument("--problems", type=Path, required=True)
    eval_parser.add_argument("--rollouts", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--summary", type=Path, required=True)
    eval_parser.add_argument("--max-rollouts", type=int, default=32)
    eval_parser.set_defaults(func=evaluate_arithmetic_fixture)

    modular_eval_parser = subparsers.add_parser("evaluate-modular-fixture")
    modular_eval_parser.add_argument("--problems", type=Path, required=True)
    modular_eval_parser.add_argument("--rollouts", type=Path, required=True)
    modular_eval_parser.add_argument("--output", type=Path, required=True)
    modular_eval_parser.add_argument("--summary", type=Path, required=True)
    modular_eval_parser.add_argument("--max-rollouts", type=int, default=32)
    modular_eval_parser.set_defaults(func=evaluate_modular_fixture)

    gsm_eval_parser = subparsers.add_parser("evaluate-gsm-style-fixture")
    gsm_eval_parser.add_argument("--problems", type=Path, required=True)
    gsm_eval_parser.add_argument("--rollouts", type=Path, required=True)
    gsm_eval_parser.add_argument("--output", type=Path, required=True)
    gsm_eval_parser.add_argument("--summary", type=Path, required=True)
    gsm_eval_parser.add_argument("--max-rollouts", type=int, default=32)
    gsm_eval_parser.set_defaults(func=evaluate_gsm_style_fixture)

    math_eval_parser = subparsers.add_parser("evaluate-math-style-fixture")
    math_eval_parser.add_argument("--problems", type=Path, required=True)
    math_eval_parser.add_argument("--rollouts", type=Path, required=True)
    math_eval_parser.add_argument("--output", type=Path, required=True)
    math_eval_parser.add_argument("--summary", type=Path, required=True)
    math_eval_parser.add_argument("--max-rollouts", type=int, default=32)
    math_eval_parser.set_defaults(func=evaluate_math_style_fixture)

    coding_eval_parser = subparsers.add_parser("evaluate-coding-style-fixture")
    coding_eval_parser.add_argument("--problems", type=Path, required=True)
    coding_eval_parser.add_argument("--rollouts", type=Path, required=True)
    coding_eval_parser.add_argument("--output", type=Path, required=True)
    coding_eval_parser.add_argument("--summary", type=Path, required=True)
    coding_eval_parser.add_argument("--max-rollouts", type=int, default=32)
    coding_eval_parser.add_argument("--timeout-seconds", type=float, default=5.0)
    coding_eval_parser.set_defaults(func=evaluate_coding_style_fixture)

    coding_preflight_parser = subparsers.add_parser(
        "preflight-coding-style-canonical"
    )
    coding_preflight_parser.add_argument("--problems", type=Path, required=True)
    coding_preflight_parser.add_argument("--output", type=Path, required=True)
    coding_preflight_parser.add_argument("--timeout-seconds", type=float, default=5.0)
    coding_preflight_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    coding_preflight_parser.set_defaults(func=preflight_coding_style_canonical)

    multiple_choice_eval_parser = subparsers.add_parser(
        "evaluate-multiple-choice-fixture"
    )
    multiple_choice_eval_parser.add_argument("--problems", type=Path, required=True)
    multiple_choice_eval_parser.add_argument("--rollouts", type=Path, required=True)
    multiple_choice_eval_parser.add_argument("--output", type=Path, required=True)
    multiple_choice_eval_parser.add_argument("--summary", type=Path, required=True)
    multiple_choice_eval_parser.add_argument("--max-rollouts", type=int, default=32)
    multiple_choice_eval_parser.set_defaults(func=evaluate_multiple_choice_fixture)

    arc_eval_parser = subparsers.add_parser("evaluate-arc-grid-fixture")
    arc_eval_parser.add_argument("--problems", type=Path, required=True)
    arc_eval_parser.add_argument("--rollouts", type=Path, required=True)
    arc_eval_parser.add_argument("--output", type=Path, required=True)
    arc_eval_parser.add_argument("--summary", type=Path, required=True)
    arc_eval_parser.add_argument("--max-rollouts", type=int, default=32)
    arc_eval_parser.set_defaults(func=evaluate_arc_grid_fixture)

    arc_preflight_parser = subparsers.add_parser("preflight-arc-grid-prompts")
    arc_preflight_parser.add_argument("--problems", type=Path, required=True)
    arc_preflight_parser.add_argument("--model", required=True)
    arc_preflight_parser.add_argument("--output", type=Path, required=True)
    arc_preflight_parser.add_argument("--max-new-tokens", type=int, default=768)
    arc_preflight_parser.add_argument(
        "--prompt-variant",
        choices=["plain", "chat", "strict_chat"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_PROMPT_VARIANT", "chat"),
    )
    arc_preflight_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN", "4096")),
    )
    arc_preflight_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    arc_preflight_parser.set_defaults(func=preflight_arc_grid_prompts)

    vllm_memory_preflight_parser = subparsers.add_parser(
        "preflight-vllm-gpu-memory"
    )
    vllm_memory_preflight_parser.add_argument("--output", type=Path, required=True)
    vllm_memory_preflight_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(
            os.environ.get("SCAFFOLD_TO_POLICY_VLLM_GPU_MEMORY_UTILIZATION", "0.9")
        ),
    )
    vllm_memory_preflight_parser.add_argument("--device-index", type=int, default=0)
    vllm_memory_preflight_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    vllm_memory_preflight_parser.set_defaults(func=preflight_vllm_gpu_memory)

    vllm_parser = subparsers.add_parser("evaluate-arithmetic-vllm")
    vllm_parser.add_argument("--problems", type=Path, required=True)
    vllm_parser.add_argument("--model", required=True)
    vllm_parser.add_argument("--output", type=Path, required=True)
    vllm_parser.add_argument("--summary", type=Path, required=True)
    vllm_parser.add_argument("--num-rollouts", type=int, default=32)
    vllm_parser.add_argument("--temperature", type=float, default=0.8)
    vllm_parser.add_argument("--top-p", type=float, default=0.95)
    vllm_parser.add_argument("--max-new-tokens", type=int, default=192)
    vllm_parser.add_argument(
        "--prompt-variant",
        choices=["plain", "chat"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_PROMPT_VARIANT", "chat"),
    )
    vllm_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN", "2048")),
    )
    vllm_parser.add_argument(
        "--attention-backend",
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
    )
    vllm_parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=bool(
            int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_FLASHINFER_AUTOTUNE", "0"))
        ),
    )
    vllm_parser.add_argument(
        "--use-flashinfer-sampler",
        choices=["0", "1"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER", "0"),
    )
    vllm_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
    )
    vllm_parser.set_defaults(func=evaluate_arithmetic_vllm)

    modular_vllm_parser = subparsers.add_parser("evaluate-modular-vllm")
    modular_vllm_parser.add_argument("--problems", type=Path, required=True)
    modular_vllm_parser.add_argument("--model", required=True)
    modular_vllm_parser.add_argument("--output", type=Path, required=True)
    modular_vllm_parser.add_argument("--summary", type=Path, required=True)
    modular_vllm_parser.add_argument("--num-rollouts", type=int, default=32)
    modular_vllm_parser.add_argument("--temperature", type=float, default=0.8)
    modular_vllm_parser.add_argument("--top-p", type=float, default=0.95)
    modular_vllm_parser.add_argument("--max-new-tokens", type=int, default=384)
    modular_vllm_parser.add_argument(
        "--prompt-variant",
        choices=["plain", "chat", "concise_chat"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_PROMPT_VARIANT", "chat"),
    )
    modular_vllm_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN", "2048")),
    )
    modular_vllm_parser.add_argument(
        "--attention-backend",
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
    )
    modular_vllm_parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=bool(
            int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_FLASHINFER_AUTOTUNE", "0"))
        ),
    )
    modular_vllm_parser.add_argument(
        "--use-flashinfer-sampler",
        choices=["0", "1"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER", "0"),
    )
    modular_vllm_parser.add_argument("--lora-adapter", type=Path)
    modular_vllm_parser.add_argument("--lora-name", default="modular_adapter")
    modular_vllm_parser.add_argument("--lora-id", type=int, default=1)
    modular_vllm_parser.add_argument("--max-lora-rank", type=int, default=32)
    modular_vllm_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
    )
    modular_vllm_parser.set_defaults(func=evaluate_modular_vllm)

    gsm_vllm_parser = subparsers.add_parser("evaluate-gsm-style-vllm")
    gsm_vllm_parser.add_argument("--problems", type=Path, required=True)
    gsm_vllm_parser.add_argument("--model", required=True)
    gsm_vllm_parser.add_argument("--output", type=Path, required=True)
    gsm_vllm_parser.add_argument("--summary", type=Path, required=True)
    gsm_vllm_parser.add_argument("--num-rollouts", type=int, default=32)
    gsm_vllm_parser.add_argument("--temperature", type=float, default=0.8)
    gsm_vllm_parser.add_argument("--top-p", type=float, default=0.95)
    gsm_vllm_parser.add_argument("--max-new-tokens", type=int, default=256)
    gsm_vllm_parser.add_argument(
        "--prompt-variant",
        choices=["plain", "chat"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_PROMPT_VARIANT", "chat"),
    )
    gsm_vllm_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN", "2048")),
    )
    gsm_vllm_parser.add_argument(
        "--attention-backend",
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
    )
    gsm_vllm_parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=bool(
            int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_FLASHINFER_AUTOTUNE", "0"))
        ),
    )
    gsm_vllm_parser.add_argument(
        "--use-flashinfer-sampler",
        choices=["0", "1"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER", "0"),
    )
    gsm_vllm_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
    )
    gsm_vllm_parser.set_defaults(func=evaluate_gsm_style_vllm)

    math_vllm_parser = subparsers.add_parser("evaluate-math-style-vllm")
    math_vllm_parser.add_argument("--problems", type=Path, required=True)
    math_vllm_parser.add_argument("--model", required=True)
    math_vllm_parser.add_argument("--output", type=Path, required=True)
    math_vllm_parser.add_argument("--summary", type=Path, required=True)
    math_vllm_parser.add_argument("--num-rollouts", type=int, default=32)
    math_vllm_parser.add_argument("--temperature", type=float, default=0.8)
    math_vllm_parser.add_argument("--top-p", type=float, default=0.95)
    math_vllm_parser.add_argument("--max-new-tokens", type=int, default=512)
    math_vllm_parser.add_argument(
        "--prompt-variant",
        choices=["plain", "chat"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_PROMPT_VARIANT", "chat"),
    )
    math_vllm_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN", "2048")),
    )
    math_vllm_parser.add_argument(
        "--attention-backend",
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
    )
    math_vllm_parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=bool(
            int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_FLASHINFER_AUTOTUNE", "0"))
        ),
    )
    math_vllm_parser.add_argument(
        "--use-flashinfer-sampler",
        choices=["0", "1"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER", "0"),
    )
    math_vllm_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
    )
    math_vllm_parser.set_defaults(func=evaluate_math_style_vllm)

    coding_vllm_parser = subparsers.add_parser("evaluate-coding-style-vllm")
    coding_vllm_parser.add_argument("--problems", type=Path, required=True)
    coding_vllm_parser.add_argument("--model", required=True)
    coding_vllm_parser.add_argument("--output", type=Path, required=True)
    coding_vllm_parser.add_argument("--summary", type=Path, required=True)
    coding_vllm_parser.add_argument("--num-rollouts", type=int, default=4)
    coding_vllm_parser.add_argument("--temperature", type=float, default=0.2)
    coding_vllm_parser.add_argument("--top-p", type=float, default=0.95)
    coding_vllm_parser.add_argument("--max-new-tokens", type=int, default=512)
    coding_vllm_parser.add_argument("--timeout-seconds", type=float, default=5.0)
    coding_vllm_parser.add_argument(
        "--prompt-variant",
        choices=["plain", "chat", "contract_chat"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_PROMPT_VARIANT", "chat"),
    )
    coding_vllm_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN", "2048")),
    )
    coding_vllm_parser.add_argument(
        "--attention-backend",
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
    )
    coding_vllm_parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=bool(
            int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_FLASHINFER_AUTOTUNE", "0"))
        ),
    )
    coding_vllm_parser.add_argument(
        "--use-flashinfer-sampler",
        choices=["0", "1"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER", "0"),
    )
    coding_vllm_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
    )
    coding_vllm_parser.set_defaults(func=evaluate_coding_style_vllm)

    multiple_choice_vllm_parser = subparsers.add_parser("evaluate-multiple-choice-vllm")
    multiple_choice_vllm_parser.add_argument("--problems", type=Path, required=True)
    multiple_choice_vllm_parser.add_argument("--model", required=True)
    multiple_choice_vllm_parser.add_argument("--output", type=Path, required=True)
    multiple_choice_vllm_parser.add_argument("--summary", type=Path, required=True)
    multiple_choice_vllm_parser.add_argument("--num-rollouts", type=int, default=4)
    multiple_choice_vllm_parser.add_argument("--temperature", type=float, default=0.2)
    multiple_choice_vllm_parser.add_argument("--top-p", type=float, default=0.95)
    multiple_choice_vllm_parser.add_argument("--max-new-tokens", type=int, default=512)
    multiple_choice_vllm_parser.add_argument(
        "--prompt-variant",
        choices=["plain", "chat"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_PROMPT_VARIANT", "chat"),
    )
    multiple_choice_vllm_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN", "2048")),
    )
    multiple_choice_vllm_parser.add_argument(
        "--attention-backend",
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
    )
    multiple_choice_vllm_parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=bool(
            int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_FLASHINFER_AUTOTUNE", "0"))
        ),
    )
    multiple_choice_vllm_parser.add_argument(
        "--use-flashinfer-sampler",
        choices=["0", "1"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER", "0"),
    )
    multiple_choice_vllm_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
    )
    multiple_choice_vllm_parser.set_defaults(func=evaluate_multiple_choice_vllm)

    arc_vllm_parser = subparsers.add_parser("evaluate-arc-grid-vllm")
    arc_vllm_parser.add_argument("--problems", type=Path, required=True)
    arc_vllm_parser.add_argument("--model", required=True)
    arc_vllm_parser.add_argument("--output", type=Path, required=True)
    arc_vllm_parser.add_argument("--summary", type=Path, required=True)
    arc_vllm_parser.add_argument("--num-rollouts", type=int, default=2)
    arc_vllm_parser.add_argument("--temperature", type=float, default=0.2)
    arc_vllm_parser.add_argument("--top-p", type=float, default=0.95)
    arc_vllm_parser.add_argument("--max-new-tokens", type=int, default=768)
    arc_vllm_parser.add_argument(
        "--prompt-variant",
        choices=["plain", "chat", "strict_chat"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_PROMPT_VARIANT", "chat"),
    )
    arc_vllm_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN", "4096")),
    )
    arc_vllm_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=(
            None
            if os.environ.get("SCAFFOLD_TO_POLICY_VLLM_GPU_MEMORY_UTILIZATION") is None
            else float(os.environ["SCAFFOLD_TO_POLICY_VLLM_GPU_MEMORY_UTILIZATION"])
        ),
    )
    arc_vllm_parser.add_argument(
        "--attention-backend",
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
    )
    arc_vllm_parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=bool(
            int(os.environ.get("SCAFFOLD_TO_POLICY_VLLM_FLASHINFER_AUTOTUNE", "0"))
        ),
    )
    arc_vllm_parser.add_argument(
        "--use-flashinfer-sampler",
        choices=["0", "1"],
        default=os.environ.get("SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER", "0"),
    )
    arc_vllm_parser.set_defaults(func=evaluate_arc_grid_vllm)

    split_parser = subparsers.add_parser("validate-arithmetic-splits")
    split_parser.add_argument("--split", nargs="+", required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.set_defaults(func=validate_arithmetic_splits)

    modular_split_parser = subparsers.add_parser("validate-modular-splits")
    modular_split_parser.add_argument("--split", nargs="+", required=True)
    modular_split_parser.add_argument("--output", type=Path, required=True)
    modular_split_parser.set_defaults(func=validate_modular_splits)

    gsm_split_parser = subparsers.add_parser("validate-gsm-style-splits")
    gsm_split_parser.add_argument("--split", nargs="+", required=True)
    gsm_split_parser.add_argument("--output", type=Path, required=True)
    gsm_split_parser.set_defaults(func=validate_gsm_style_splits)

    math_split_parser = subparsers.add_parser("validate-math-style-splits")
    math_split_parser.add_argument("--split", nargs="+", required=True)
    math_split_parser.add_argument("--output", type=Path, required=True)
    math_split_parser.set_defaults(func=validate_math_style_splits)

    coding_split_parser = subparsers.add_parser("validate-coding-style-splits")
    coding_split_parser.add_argument("--split", nargs="+", required=True)
    coding_split_parser.add_argument("--output", type=Path, required=True)
    coding_split_parser.set_defaults(func=validate_coding_style_splits)

    multiple_choice_split_parser = subparsers.add_parser(
        "validate-multiple-choice-splits"
    )
    multiple_choice_split_parser.add_argument("--split", nargs="+", required=True)
    multiple_choice_split_parser.add_argument("--output", type=Path, required=True)
    multiple_choice_split_parser.set_defaults(func=validate_multiple_choice_splits)

    arc_split_parser = subparsers.add_parser("validate-arc-grid-splits")
    arc_split_parser.add_argument("--split", nargs="+", required=True)
    arc_split_parser.add_argument("--output", type=Path, required=True)
    arc_split_parser.set_defaults(func=validate_arc_grid_splits)

    report_parser = subparsers.add_parser("build-arithmetic-report-input")
    report_parser.add_argument("--data-root", type=Path, required=True)
    report_parser.add_argument("--results-root", type=Path, required=True)
    report_parser.add_argument("--run-id", required=True)
    report_parser.add_argument("--split-registry", type=Path, required=True)
    report_parser.add_argument("--summary", nargs="+", required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    report_parser.set_defaults(func=build_arithmetic_report_input)

    modular_report_parser = subparsers.add_parser("build-modular-report-input")
    modular_report_parser.add_argument("--data-root", type=Path, required=True)
    modular_report_parser.add_argument("--results-root", type=Path, required=True)
    modular_report_parser.add_argument("--run-id", required=True)
    modular_report_parser.add_argument("--split-registry", type=Path, required=True)
    modular_report_parser.add_argument("--summary", nargs="+", required=True)
    modular_report_parser.add_argument("--evaluation", nargs="*")
    modular_report_parser.add_argument("--output", type=Path, required=True)
    modular_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    modular_report_parser.set_defaults(func=build_modular_report_input)

    gsm_report_parser = subparsers.add_parser("build-gsm-style-report-input")
    gsm_report_parser.add_argument("--data-root", type=Path, required=True)
    gsm_report_parser.add_argument("--results-root", type=Path, required=True)
    gsm_report_parser.add_argument("--run-id", required=True)
    gsm_report_parser.add_argument("--split-registry", type=Path, required=True)
    gsm_report_parser.add_argument("--summary", nargs="+", required=True)
    gsm_report_parser.add_argument("--output", type=Path, required=True)
    gsm_report_parser.add_argument("--scaffold-budget", type=int, default=32)
    gsm_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    gsm_report_parser.set_defaults(func=build_gsm_style_report_input)

    math_report_parser = subparsers.add_parser("build-math-style-report-input")
    math_report_parser.add_argument("--data-root", type=Path, required=True)
    math_report_parser.add_argument("--results-root", type=Path, required=True)
    math_report_parser.add_argument("--run-id", required=True)
    math_report_parser.add_argument("--split-registry", type=Path, required=True)
    math_report_parser.add_argument("--summary", nargs="+", required=True)
    math_report_parser.add_argument("--output", type=Path, required=True)
    math_report_parser.add_argument("--scaffold-budget", type=int, default=32)
    math_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    math_report_parser.set_defaults(func=build_math_style_report_input)

    coding_report_parser = subparsers.add_parser("build-coding-style-report-input")
    coding_report_parser.add_argument("--data-root", type=Path, required=True)
    coding_report_parser.add_argument("--results-root", type=Path, required=True)
    coding_report_parser.add_argument("--run-id", required=True)
    coding_report_parser.add_argument("--split-registry", type=Path, required=True)
    coding_report_parser.add_argument("--summary", nargs="+", required=True)
    coding_report_parser.add_argument("--preflight", nargs="*")
    coding_report_parser.add_argument("--output", type=Path, required=True)
    coding_report_parser.add_argument("--scaffold-budget", type=int, default=4)
    coding_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    coding_report_parser.set_defaults(func=build_coding_style_report_input)

    multiple_choice_report_parser = subparsers.add_parser(
        "build-multiple-choice-report-input"
    )
    multiple_choice_report_parser.add_argument("--data-root", type=Path, required=True)
    multiple_choice_report_parser.add_argument("--results-root", type=Path, required=True)
    multiple_choice_report_parser.add_argument("--run-id", required=True)
    multiple_choice_report_parser.add_argument(
        "--split-registry",
        type=Path,
        required=True,
    )
    multiple_choice_report_parser.add_argument("--summary", nargs="+", required=True)
    multiple_choice_report_parser.add_argument("--output", type=Path, required=True)
    multiple_choice_report_parser.add_argument("--scaffold-budget", type=int, default=4)
    multiple_choice_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    multiple_choice_report_parser.set_defaults(
        func=build_multiple_choice_report_input
    )

    arc_report_parser = subparsers.add_parser("build-arc-grid-report-input")
    arc_report_parser.add_argument("--data-root", type=Path, required=True)
    arc_report_parser.add_argument("--results-root", type=Path, required=True)
    arc_report_parser.add_argument("--run-id", required=True)
    arc_report_parser.add_argument("--split-registry", type=Path, required=True)
    arc_report_parser.add_argument("--summary", nargs="+", required=True)
    arc_report_parser.add_argument("--preflight", nargs="*")
    arc_report_parser.add_argument("--output", type=Path, required=True)
    arc_report_parser.add_argument("--scaffold-budget", type=int, default=2)
    arc_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    arc_report_parser.set_defaults(func=build_arc_grid_report_input)

    latest_report_parser = subparsers.add_parser("write-latest-report-index")
    latest_report_parser.add_argument("--manifests-dir", type=Path, required=True)
    latest_report_parser.add_argument("--output", type=Path, required=True)
    latest_report_parser.add_argument("--pattern", default="report_input_*.json")
    latest_report_parser.add_argument("--task")
    latest_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    latest_report_parser.set_defaults(func=write_latest_report_index)

    blocker_report_parser = subparsers.add_parser("write-blocker-report-input")
    blocker_report_parser.add_argument("--results-root", type=Path, required=True)
    blocker_report_parser.add_argument("--run-id", required=True)
    blocker_report_parser.add_argument("--task", required=True)
    blocker_report_parser.add_argument(
        "--lane",
        choices=["reasoning", "coding", "agentic", "external_harness"],
        required=True,
    )
    blocker_report_parser.add_argument("--blocker-type", required=True)
    blocker_report_parser.add_argument("--artifact", nargs="+", required=True)
    blocker_report_parser.add_argument("--limitation", action="append", default=[])
    blocker_report_parser.add_argument("--output", type=Path, required=True)
    blocker_report_parser.set_defaults(func=write_blocker_report_input)

    math_rescore_parser = subparsers.add_parser("rescore-math-style-evaluations")
    math_rescore_parser.add_argument("--evaluations", type=Path, required=True)
    math_rescore_parser.add_argument("--output", type=Path, required=True)
    math_rescore_parser.add_argument("--summary", type=Path, required=True)
    math_rescore_parser.set_defaults(func=rescore_math_style_evaluations)

    coding_rescore_parser = subparsers.add_parser("rescore-coding-style-evaluations")
    coding_rescore_parser.add_argument("--evaluations", type=Path, required=True)
    coding_rescore_parser.add_argument("--output", type=Path, required=True)
    coding_rescore_parser.add_argument("--summary", type=Path, required=True)
    coding_rescore_parser.set_defaults(func=rescore_coding_style_evaluations)

    external_write_parser = subparsers.add_parser("write-external-harness-smoke")
    external_write_parser.add_argument(
        "--harness-family",
        choices=["harbor_terminal", "tau2"],
        required=True,
    )
    external_write_parser.add_argument("--run-id", required=True)
    external_write_parser.add_argument("--task-subset", required=True)
    external_write_parser.add_argument("--output", type=Path, required=True)
    external_write_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    external_write_parser.set_defaults(func=write_external_harness_smoke)

    external_preflight_parser = subparsers.add_parser(
        "write-external-harness-preflight"
    )
    external_preflight_parser.add_argument(
        "--harness-family",
        choices=["harbor_terminal", "tau2"],
        required=True,
    )
    external_preflight_parser.add_argument("--run-id", required=True)
    external_preflight_parser.add_argument("--task-subset", required=True)
    external_preflight_parser.add_argument("--output", type=Path, required=True)
    external_preflight_parser.add_argument("--cli-name", action="append", default=[])
    external_preflight_parser.set_defaults(func=write_external_harness_preflight)

    tau2_score_parser = subparsers.add_parser("write-tau2-mock-score-smoke")
    tau2_score_parser.add_argument("--run-id", required=True)
    tau2_score_parser.add_argument("--task-id", default="create_task_1")
    tau2_score_parser.add_argument(
        "--evaluation-type",
        default="all_ignore_basis",
        choices=[
            "env",
            "action",
            "communicate",
            "all",
            "all_ignore_basis",
        ],
    )
    tau2_score_parser.add_argument("--output", type=Path, required=True)
    tau2_score_parser.set_defaults(func=write_tau2_mock_score_smoke)

    terminal_result_parser = subparsers.add_parser(
        "write-terminal-bench-result-smoke"
    )
    terminal_result_parser.add_argument("--run-id", required=True)
    terminal_result_parser.add_argument("--task-id", default="headless-terminal")
    terminal_result_parser.add_argument("--output", type=Path, required=True)
    terminal_result_parser.set_defaults(func=write_terminal_bench_result_smoke)

    terminal_probe_parser = subparsers.add_parser(
        "write-terminal-bench-execution-probe"
    )
    terminal_probe_parser.add_argument("--run-id", required=True)
    terminal_probe_parser.add_argument("--task-id", required=True)
    terminal_probe_parser.add_argument("--cwd", type=Path, required=True)
    terminal_probe_parser.add_argument("--harbor-result-json", type=Path, required=True)
    terminal_probe_parser.add_argument("--agent-name", default="oracle")
    terminal_probe_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    terminal_probe_parser.add_argument("--output", type=Path, required=True)
    terminal_probe_parser.add_argument("command", nargs=argparse.REMAINDER)
    terminal_probe_parser.set_defaults(func=write_terminal_bench_execution_probe)

    tau2_probe_parser = subparsers.add_parser("write-tau2-execution-probe")
    tau2_probe_parser.add_argument("--run-id", required=True)
    tau2_probe_parser.add_argument("--task-id", required=True)
    tau2_probe_parser.add_argument("--cwd", type=Path, required=True)
    tau2_probe_parser.add_argument("--results-json", type=Path, required=True)
    tau2_probe_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    tau2_probe_parser.add_argument("--output", type=Path, required=True)
    tau2_probe_parser.add_argument("command", nargs=argparse.REMAINDER)
    tau2_probe_parser.set_defaults(func=write_tau2_execution_probe)

    external_ingest_parser = subparsers.add_parser("ingest-external-harness-smoke")
    external_ingest_parser.add_argument("--raw-result", type=Path, required=True)
    external_ingest_parser.add_argument("--results-root", type=Path, required=True)
    external_ingest_parser.add_argument("--output", type=Path, required=True)
    external_ingest_parser.set_defaults(func=ingest_external_harness_smoke)

    external_report_parser = subparsers.add_parser(
        "build-external-harness-report-input"
    )
    external_report_parser.add_argument("--results-root", type=Path, required=True)
    external_report_parser.add_argument("--run-id", required=True)
    external_report_parser.add_argument("--ingested", nargs="+", required=True)
    external_report_parser.add_argument("--output", type=Path, required=True)
    external_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    external_report_parser.set_defaults(func=build_external_harness_report_input)

    modular_dataset_parser = subparsers.add_parser("build-modular-dataset")
    modular_dataset_parser.add_argument("--evaluations", type=Path, required=True)
    modular_dataset_parser.add_argument("--output", type=Path, required=True)
    modular_dataset_parser.add_argument("--condition", default="raw")
    modular_dataset_parser.add_argument("--min-examples", type=int, default=1)
    modular_dataset_parser.set_defaults(func=build_modular_dataset)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
