# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Command line tools for the Countdown search-distillation pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path

from torchtitan.experiments.countdown_search_distill.countdown import (
    CountdownProblem,
    ParsedOperation,
    ProblemFilters,
    VerificationResult,
    generate_problem_pool,
    problem_key,
)
from torchtitan.experiments.countdown_search_distill.datasets import (
    build_canonical_raw_examples,
    build_training_examples,
    write_training_sets,
)
from torchtitan.experiments.countdown_search_distill.evaluate import (
    ProblemEvaluation,
    evaluate_rollouts,
    stable_problem_id,
    write_annotations_jsonl,
    write_evaluations_jsonl,
    write_rollouts_jsonl,
    write_summary_csv,
    write_summary_json,
    write_summary_md,
)
from torchtitan.experiments.countdown_search_distill.experiment_registry import (
    build_countdown_report_input,
    write_countdown_report_input,
)
from torchtitan.experiments.countdown_search_distill.lora_export import (
    Qwen3LoRAExportConfig,
    export_lora_adapter,
)
from torchtitan.experiments.countdown_search_distill.split_registry import (
    build_split_registry,
    write_split_registry,
)


def _prompt_for_problem(problem: CountdownProblem, variant: str) -> str:
    if variant == "default":
        return problem.prompt()
    if variant == "terse":
        numbers = ", ".join(str(number) for number in problem.numbers)
        return "\n".join(
            [
                "Solve this Countdown arithmetic puzzle.",
                f"Numbers: {numbers}",
                f"Target: {problem.target}",
                "Use each listed number exactly once.",
                "Allowed operations: +, -, *, /.",
                "Division must be exact and every intermediate result must be a positive integer.",
                "Return only equations, one per line, then FINAL.",
                "Example format:",
                "2 + 3 = 5",
                "5 * 4 = 20",
                f"FINAL: {problem.target}",
            ]
        )
    raise ValueError(f"unknown prompt variant: {variant}")


def _build_vllm_prompts(problems: list[CountdownProblem], args: argparse.Namespace):
    if args.prompt_variant != "chat":
        return [_prompt_for_problem(problem, args.prompt_variant) for problem in problems]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You solve Countdown arithmetic puzzles. Return only "
                        "valid equations, one per line, followed by FINAL."
                    ),
                },
                {"role": "user", "content": _prompt_for_problem(problem, "terse")},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]


def _load_problems(path: Path) -> list[CountdownProblem]:
    problems: list[CountdownProblem] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            problems.append(_problem_from_row(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid problem at {path}:{line_number}: {exc}") from exc
    return problems


def _load_excluded_problem_keys(paths: list[Path]) -> set[tuple[tuple[int, ...], int]]:
    keys: set[tuple[tuple[int, ...], int]] = set()
    for path in paths:
        for problem in _load_problems(path):
            keys.add(problem_key(problem))
    return keys


def _problem_from_row(row: dict[str, object]) -> CountdownProblem:
    if "canonical_solution" in row:
        generation_config = row.get("generation_config", {})
        if not isinstance(generation_config, dict):
            raise ValueError("generation_config must be a dict")
        numbers = row["numbers"]
        if not isinstance(numbers, list):
            raise ValueError("numbers must be a list")
        canonical_solution = row["canonical_solution"]
        if not isinstance(canonical_solution, list):
            raise ValueError("canonical_solution must be a list")
        return CountdownProblem(
            numbers=tuple(int(number) for number in numbers),
            target=int(row["target"]),
            min_solution_depth=(
                None
                if generation_config.get("min_solution_depth") is None
                else int(generation_config.get("min_solution_depth"))
            ),
            require_all_numbers=bool(
                generation_config.get("require_all_numbers", False)
            ),
            solution=tuple(str(step) for step in canonical_solution),
        )
    return CountdownProblem.from_json(row)


def _load_evaluations(path: Path) -> list[ProblemEvaluation]:
    evaluations: list[ProblemEvaluation] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            problem = _problem_from_row(row["problem"])
            rollouts = [
                _rollout_evaluation_from_row(
                    rollout,
                    problem_id=str(row.get("problem_id") or stable_problem_id(problem)),
                )
                for rollout in row["rollouts"]
            ]
            evaluations.append(
                ProblemEvaluation(
                    problem_id=str(row.get("problem_id") or stable_problem_id(problem)),
                    problem=problem,
                    rollouts=tuple(rollouts),
                )
            )
        except Exception as exc:
            raise ValueError(f"invalid evaluation at {path}:{line_number}: {exc}") from exc
    return evaluations


def _rollout_evaluation_from_row(
    row: dict[str, object],
    *,
    problem_id: str,
) -> "RolloutEvaluation":
    from torchtitan.experiments.countdown_search_distill.evaluate import (
        RolloutEvaluation,
    )

    operations = row.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")
    states = row.get("states", [])
    if not isinstance(states, list):
        raise ValueError("states must be a list")
    available_numbers = row.get("available_numbers", [])
    if not isinstance(available_numbers, list):
        raise ValueError("available_numbers must be a list")
    verification = VerificationResult(
        success=bool(row.get("success", False)),
        final_value=(
            None if row.get("final_value") is None else int(row["final_value"])
        ),
        steps_consumed=int(row.get("steps_consumed", 0)),
        available_numbers=tuple(int(number) for number in available_numbers),
        operations=tuple(_parsed_operation_from_row(operation) for operation in operations),
        states=tuple(
            tuple(int(number) for number in state)
            for state in states
            if isinstance(state, list)
        ),
        valid_prefix_length=int(row.get("valid_prefix_length", 0)),
        first_invalid=_optional_dict(row.get("first_invalid")),
        first_unreachable=_optional_dict(row.get("first_unreachable")),
        distance_to_target=(
            None
            if row.get("distance_to_target") is None
            else int(row["distance_to_target"])
        ),
        solution_depth=(
            None if row.get("solution_depth") is None else int(row["solution_depth"])
        ),
        recoverable=bool(row.get("recoverable", False)),
        error=None if row.get("error") is None else str(row["error"]),
    )
    return RolloutEvaluation(
        problem_id=str(row.get("problem_id") or problem_id),
        sample_index=int(row.get("sample_index", 0)),
        text=str(row.get("text", "")),
        verification=verification,
        model_id=str(row.get("model_id", "fixture")),
        condition=str(row.get("condition", "base")),
        prompt_variant=str(row.get("prompt_variant", "default")),
        token_count=None if row.get("token_count") is None else int(row["token_count"]),
        logprob_sum=(
            None if row.get("logprob_sum") is None else float(row["logprob_sum"])
        ),
    )


def _parsed_operation_from_row(row: object) -> ParsedOperation:
    if not isinstance(row, dict):
        raise ValueError("operation must be a dict")
    return ParsedOperation(
        raw=str(row["raw"]),
        left=int(row["left"]),
        op=str(row["op"]),
        right=int(row["right"]),
        result=int(row["result"]),
        line_number=int(row["line_number"]),
    )


def _optional_dict(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("optional verification field must be a dict or null")
    return value


def _load_rollout_fixture(path: Path) -> dict[tuple[tuple[int, ...], int], list[str]]:
    by_problem: dict[tuple[tuple[int, ...], int], list[str]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problem = _problem_from_row(row["problem"])
            rollouts = row["rollouts"]
            if not isinstance(rollouts, list):
                raise ValueError("rollouts must be a list")
            by_problem[problem_key(problem)] = [str(rollout) for rollout in rollouts]
        except Exception as exc:
            raise ValueError(f"invalid fixture at {path}:{line_number}: {exc}") from exc
    return by_problem


def _write_problem_jsonl(problems: Iterable[CountdownProblem], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for problem in problems:
            row = {
                "problem_id": stable_problem_id(problem),
                "numbers": list(problem.numbers),
                "target": problem.target,
                "prompt": problem.prompt(),
                "generation_config": {
                    "min_solution_depth": problem.min_solution_depth,
                    "require_all_numbers": problem.require_all_numbers,
                },
                "solver_metadata": {
                    "solution_depth": max(len(problem.solution) - 1, 0),
                },
                "canonical_solution": list(problem.solution),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")


def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    excluded_problem_keys = _load_excluded_problem_keys(args.exclude_problems)
    problems = generate_problem_pool(
        rng=rng,
        num_problems=args.num_problems,
        num_numbers=args.num_numbers,
        filters=ProblemFilters(
            require_all_numbers=args.require_all_numbers,
            division_required=args.division_required,
            target_min=args.target_min,
            target_max=args.target_max,
            number_min=args.number_min,
            number_max=args.number_max,
            min_solution_depth=args.min_solution_depth,
            min_distinct_solutions=args.min_distinct_solutions,
        ),
        max_duplicate_attempts=args.max_duplicate_attempts,
        excluded_problem_keys=excluded_problem_keys,
    )
    _write_problem_jsonl(problems, args.output)


def evaluate_fixture(args: argparse.Namespace) -> None:
    problems = _load_problems(args.problems)
    fixture = _load_rollout_fixture(args.rollouts)
    evaluations = []
    for problem in problems:
        key = problem_key(problem)
        if key not in fixture:
            raise ValueError(f"missing rollouts for problem {key}")
        evaluations.append(
            evaluate_rollouts(
                problem,
                fixture[key][: args.max_rollouts],
                problem_id=stable_problem_id(problem),
                model_id=args.model_id,
            )
        )
    write_evaluations_jsonl(evaluations, args.output)
    if args.rollouts_output is not None:
        write_rollouts_jsonl(evaluations, args.rollouts_output)
    if args.annotations is not None:
        write_annotations_jsonl(evaluations, args.annotations)
    write_summary_json(evaluations, args.summary)
    _write_summary_sidecars(args.summary)


def evaluate_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "VLLM_USE_FLASHINFER_SAMPLER",
        args.use_flashinfer_sampler,
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for evaluate-vllm. Install it in the active "
            "environment or use evaluate-fixture with precomputed rollouts."
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

    problems = _load_problems(args.problems)
    prompts = _build_vllm_prompts(problems, args)
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
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )
    evaluations = []
    for problem, output in zip(problems, outputs):
        rollouts = [candidate.text for candidate in output.outputs]
        evaluations.append(
            evaluate_rollouts(
                problem,
                rollouts,
                problem_id=stable_problem_id(problem),
                model_id=args.model,
                prompt_variant=args.prompt_variant,
                compute_recoverability=args.compute_recoverability,
            )
        )
    write_evaluations_jsonl(evaluations, args.output)
    if args.rollouts_output is not None:
        write_rollouts_jsonl(evaluations, args.rollouts_output)
    if args.annotations is not None:
        write_annotations_jsonl(evaluations, args.annotations)
    write_summary_json(evaluations, args.summary)
    _write_summary_sidecars(args.summary)


def build_datasets(args: argparse.Namespace) -> None:
    evaluations = _load_evaluations(args.evaluations)
    examples = build_training_examples(
        evaluations,
        conditions=args.conditions,
        matched_only=args.matched_only,
    )
    if args.canonical_raw_fallback and not examples.get("raw"):
        examples["raw"] = build_canonical_raw_examples(evaluations)
    write_training_sets(examples, args.output_dir)


def summarize(args: argparse.Namespace) -> None:
    evaluations = _load_evaluations(args.evaluations)
    write_summary_json(evaluations, args.summary)
    _write_summary_sidecars(args.summary)


def calibrate(args: argparse.Namespace) -> None:
    evaluations = _load_evaluations(args.evaluations)
    write_summary_json(evaluations, args.summary)
    _write_summary_sidecars(args.summary)
    summary = json.loads(args.summary.read_text())
    pass_at_1 = summary["pass_at_k"].get("1", 0.0)
    pass_at_32 = summary["pass_at_k"].get("32", 0.0)
    selected = (
        args.pass1_min <= pass_at_1 <= args.pass1_max
        and args.pass32_min <= pass_at_32 <= args.pass32_max
    )
    decision = {
        "selected": selected,
        "pass_at_1": pass_at_1,
        "pass_at_32": pass_at_32,
        "target_pass_at_1": [args.pass1_min, args.pass1_max],
        "target_pass_at_32": [args.pass32_min, args.pass32_max],
    }
    args.decision.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")


def select_sweep(args: argparse.Namespace) -> None:
    candidates = []
    for line_number, line in enumerate(args.candidates.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"candidate at {args.candidates}:{line_number} must be an object")
        summary_path = Path(str(row["summary"]))
        summary = json.loads(summary_path.read_text())
        pass_at_1 = float(summary["pass_at_k"].get("1", 0.0))
        pass_at_32 = float(summary["pass_at_k"].get("32", 0.0))
        selected = (
            args.pass1_min <= pass_at_1 <= args.pass1_max
            and args.pass32_min <= pass_at_32 <= args.pass32_max
        )
        candidates.append(
            {
                **row,
                "selected": selected,
                "pass_at_1": pass_at_1,
                "pass_at_32": pass_at_32,
                "num_problems": int(summary["num_problems"]),
                "bucket_counts": summary["bucket_counts"],
            }
        )
    if not candidates:
        raise ValueError("no sweep candidates were provided")

    selected_candidates = [candidate for candidate in candidates if candidate["selected"]]
    target_pass_at_32 = (args.pass32_min + args.pass32_max) / 2.0
    target_pass_at_1 = (args.pass1_min + args.pass1_max) / 2.0
    selected_candidate = None
    if selected_candidates:
        selected_candidate = min(
            selected_candidates,
            key=lambda candidate: (
                abs(candidate["pass_at_32"] - target_pass_at_32),
                abs(candidate["pass_at_1"] - target_pass_at_1),
                str(candidate["name"]),
            ),
        )

    decision = {
        "selected": selected_candidate is not None,
        "selected_name": None if selected_candidate is None else selected_candidate["name"],
        "target_pass_at_1": [args.pass1_min, args.pass1_max],
        "target_pass_at_32": [args.pass32_min, args.pass32_max],
        "candidates": candidates,
    }
    args.decision.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    if selected_candidate is None:
        if args.require_selection:
            raise SystemExit("calibration sweep failed to select a regime")
        return

    args.output.write_text(
        json.dumps(selected_candidate, indent=2, sort_keys=True) + "\n"
    )


def preflight_reduced(args: argparse.Namespace) -> None:
    calibration = json.loads(args.calibration_decision.read_text())
    train_evaluations = _load_evaluations(args.train_evaluations)
    dev_evaluations = _load_evaluations(args.dev_evaluations)
    train_matched = _num_solved_problems(train_evaluations)
    dev_matched = _num_solved_problems(dev_evaluations)
    checks = {
        "calibration_selected": bool(calibration.get("selected", False)),
        "train_matched_coverage": train_matched >= args.min_train_matched,
        "dev_matched_coverage": dev_matched >= args.min_dev_matched,
    }
    decision = {
        "selected": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_train_matched": args.min_train_matched,
            "min_dev_matched": args.min_dev_matched,
        },
        "observed": {
            "train_matched": train_matched,
            "dev_matched": dev_matched,
            "calibration": calibration,
        },
    }
    args.decision.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    if not decision["selected"]:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise SystemExit(f"reduced preflight failed: {failed}")


def export_lora(args: argparse.Namespace) -> None:
    config = Qwen3LoRAExportConfig(
        rank=args.rank,
        alpha=args.alpha,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        base_model_name_or_path=args.base_model_name_or_path,
        torch_dtype=args.torch_dtype,
    )
    summary = export_lora_adapter(args.checkpoint, args.output, config)
    print(json.dumps(summary, indent=2, sort_keys=True))


def preflight_runtime(args: argparse.Namespace) -> None:
    module_checks = {
        module: importlib.util.find_spec(module) is not None
        for module in args.required_modules
    }
    asset_checks = _check_model_assets(args.model)
    gpu_checks = _check_gpu_runtime(args.require_gpu)
    rootfs_active = os.environ.get("TORCHTITAN_IN_ROOTFS") == "1"
    env_checks = {
        "rootfs_active": (not args.require_rootfs) or rootfs_active,
        "flashinfer_sampler_disabled": os.environ.get(
            "COUNTDOWN_VLLM_USE_FLASHINFER_SAMPLER",
            os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0"),
        )
        == "0",
        "flashinfer_autotune_disabled": os.environ.get(
            "COUNTDOWN_VLLM_FLASHINFER_AUTOTUNE",
            "0",
        )
        == "0",
        "attention_backend_set": bool(
            os.environ.get("COUNTDOWN_VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
        ),
    }
    checks = {
        "environment": env_checks,
        "modules": module_checks,
        "model_assets": asset_checks,
        "gpu": gpu_checks,
    }
    decision = {
        "selected": all(
            passed
            for group in checks.values()
            for passed in group.values()
            if isinstance(passed, bool)
        ),
        "checks": checks,
        "model": str(args.model),
        "required_modules": args.required_modules,
    }
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    args.decision.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    if not decision["selected"]:
        failed = _failed_check_names(checks)
        raise SystemExit(f"runtime preflight failed: {', '.join(failed)}")


def validate_eval_matrix(args: argparse.Namespace) -> None:
    expected_problems = _parse_expected_problems(args.expected_problems)
    rows = []
    checks = {}
    for split in args.splits:
        for arm in args.arms:
            summary_path = args.eval_root / split / arm / "summary.json"
            row, arm_checks = _validate_summary(
                summary_path,
                expected_num_problems=expected_problems[split],
                expected_num_rollouts=args.num_rollouts,
            )
            row.update({"split": split, "arm": arm, "summary": str(summary_path)})
            rows.append(row)
            checks[f"{split}/{arm}"] = arm_checks

    decision = {
        "selected": all(all(values.values()) for values in checks.values()),
        "eval_root": str(args.eval_root),
        "num_rollouts": args.num_rollouts,
        "expected_problems": expected_problems,
        "checks": checks,
        "rows": rows,
    }
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    args.decision.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    if not decision["selected"]:
        failed = [
            f"{name}:{check}"
            for name, values in checks.items()
            for check, passed in values.items()
            if not passed
        ]
        raise SystemExit(f"eval matrix validation failed: {', '.join(failed)}")


def validate_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    validation = build_split_registry(split_paths, load_problems=_load_problems)
    write_split_registry(validation, args.output)
    if not validation.selected:
        raise SystemExit(f"split validation failed: {len(validation.overlaps)} overlaps")


def build_report_input(args: argparse.Namespace) -> None:
    report_input = build_countdown_report_input(
        experiment_root=args.experiment_root,
        mode=args.mode,
        run_id=args.run_id,
        manifest=args.manifest,
    )
    write_countdown_report_input(report_input, args.output)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"report input validation failed: {', '.join(failed)}")


def _num_solved_problems(evaluations: list[ProblemEvaluation]) -> int:
    return sum(1 for evaluation in evaluations if evaluation.solved_at() is not None)


def _write_summary_sidecars(summary_path: Path) -> None:
    write_summary_csv(summary_path, summary_path.with_suffix(".csv"))
    write_summary_md(summary_path, summary_path.with_suffix(".md"))


def _check_model_assets(model_dir: Path) -> dict[str, bool]:
    return {
        "directory_exists": model_dir.is_dir(),
        "config_json": (model_dir / "config.json").is_file(),
        "tokenizer_config_json": (model_dir / "tokenizer_config.json").is_file(),
        "tokenizer_json": (model_dir / "tokenizer.json").is_file(),
        "safetensors": any(model_dir.glob("*.safetensors")),
    }


def _check_gpu_runtime(require_gpu: bool) -> dict[str, bool]:
    try:
        import torch
    except ImportError:
        return {
            "torch_imported": False,
            "cuda_available": not require_gpu,
            "cuda_device_count_positive": not require_gpu,
        }
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    return {
        "torch_imported": True,
        "cuda_available": (not require_gpu) or cuda_available,
        "cuda_device_count_positive": (not require_gpu) or cuda_device_count > 0,
    }


def _failed_check_names(checks: dict[str, dict[str, bool]]) -> list[str]:
    return [
        f"{group}.{name}"
        for group, values in checks.items()
        for name, passed in values.items()
        if not passed
    ]


def _parse_expected_problems(values: list[str]) -> dict[str, int]:
    parsed = {}
    for value in values:
        split, sep, count = value.partition("=")
        if not sep:
            raise ValueError(f"expected SPLIT=COUNT, got {value}")
        parsed[split] = int(count)
    return parsed


def _parse_split_paths(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        split, sep, path = value.partition("=")
        if not sep:
            raise ValueError(f"expected SPLIT=PATH, got {value}")
        parsed[split] = Path(path)
    return parsed


def _validate_summary(
    summary_path: Path,
    *,
    expected_num_problems: int,
    expected_num_rollouts: int,
) -> tuple[dict[str, object], dict[str, bool]]:
    if not summary_path.is_file():
        return (
            {
                "num_problems": None,
                "total_rollouts": None,
                "pass_at_1": None,
                "pass_at_32": None,
                "bucket_counts": {},
            },
            {
                "summary_exists": False,
                "num_problems": False,
                "rollout_count": False,
                "pass_curve": False,
                "bucket_total": False,
            },
        )

    summary = json.loads(summary_path.read_text())
    validity_breakdown = summary.get("validity_breakdown", {})
    bucket_counts = summary.get("bucket_counts", {})
    pass_at_k = summary.get("pass_at_k", {})
    total_rollouts = sum(int(count) for count in validity_breakdown.values())
    num_problems = int(summary.get("num_problems", -1))
    checks = {
        "summary_exists": True,
        "num_problems": num_problems == expected_num_problems,
        "rollout_count": total_rollouts
        == expected_num_problems * expected_num_rollouts,
        "pass_curve": all(str(k) in pass_at_k for k in (1, 2, 4, 8, 16, 32)),
        "bucket_total": sum(int(count) for count in bucket_counts.values())
        == expected_num_problems,
    }
    row = {
        "num_problems": num_problems,
        "total_rollouts": total_rollouts,
        "pass_at_1": pass_at_k.get("1"),
        "pass_at_32": pass_at_k.get("32"),
        "bucket_counts": bucket_counts,
    }
    return row, checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Countdown best-of-N search-gap experiment tools."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-pool")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--num-problems", type=int, default=500)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--num-numbers", type=int, default=6)
    generate_parser.add_argument("--number-min", type=int, default=1)
    generate_parser.add_argument("--number-max", type=int, default=100)
    generate_parser.add_argument("--target-min", type=int, default=100)
    generate_parser.add_argument("--target-max", type=int, default=999)
    generate_parser.add_argument("--min-solution-depth", type=int, default=3)
    generate_parser.add_argument(
        "--require-all-numbers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    generate_parser.add_argument(
        "--division-required",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    generate_parser.add_argument("--min-distinct-solutions", type=int, default=1)
    generate_parser.add_argument("--max-duplicate-attempts", type=int, default=100)
    generate_parser.add_argument("--exclude-problems", nargs="*", type=Path, default=[])
    generate_parser.set_defaults(func=generate)

    legacy_generate_parser = subparsers.add_parser("generate")
    legacy_generate_parser.add_argument("--output", type=Path, required=True)
    legacy_generate_parser.add_argument("--num-problems", type=int, default=500)
    legacy_generate_parser.add_argument("--seed", type=int, default=42)
    legacy_generate_parser.add_argument("--num-numbers", type=int, default=6)
    legacy_generate_parser.add_argument("--number-min", type=int, default=1)
    legacy_generate_parser.add_argument("--number-max", type=int, default=100)
    legacy_generate_parser.add_argument("--target-min", type=int, default=100)
    legacy_generate_parser.add_argument("--target-max", type=int, default=999)
    legacy_generate_parser.add_argument("--min-solution-depth", type=int, default=3)
    legacy_generate_parser.add_argument(
        "--require-all-numbers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    legacy_generate_parser.add_argument(
        "--division-required",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    legacy_generate_parser.add_argument("--min-distinct-solutions", type=int, default=1)
    legacy_generate_parser.add_argument("--max-duplicate-attempts", type=int, default=100)
    legacy_generate_parser.add_argument(
        "--exclude-problems",
        nargs="*",
        type=Path,
        default=[],
    )
    legacy_generate_parser.set_defaults(func=generate)

    fixture_parser = subparsers.add_parser("evaluate-fixture")
    fixture_parser.add_argument("--problems", type=Path, required=True)
    fixture_parser.add_argument("--rollouts", type=Path, required=True)
    fixture_parser.add_argument("--output", type=Path, required=True)
    fixture_parser.add_argument("--summary", type=Path, required=True)
    fixture_parser.add_argument("--rollouts-output", type=Path)
    fixture_parser.add_argument("--annotations", type=Path)
    fixture_parser.add_argument("--max-rollouts", type=int, default=32)
    fixture_parser.add_argument("--model-id", default="fixture")
    fixture_parser.set_defaults(func=evaluate_fixture)

    vllm_parser = subparsers.add_parser("evaluate-vllm")
    vllm_parser.add_argument("--problems", type=Path, required=True)
    vllm_parser.add_argument("--model", required=True)
    vllm_parser.add_argument("--output", type=Path, required=True)
    vllm_parser.add_argument("--summary", type=Path, required=True)
    vllm_parser.add_argument("--rollouts-output", type=Path)
    vllm_parser.add_argument("--annotations", type=Path)
    vllm_parser.add_argument("--num-rollouts", type=int, default=32)
    vllm_parser.add_argument("--temperature", type=float, default=0.8)
    vllm_parser.add_argument("--top-p", type=float, default=0.95)
    vllm_parser.add_argument("--max-new-tokens", type=int, default=192)
    vllm_parser.add_argument(
        "--prompt-variant",
        choices=["default", "terse", "chat"],
        default=os.environ.get("COUNTDOWN_PROMPT_VARIANT", "default"),
    )
    vllm_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("COUNTDOWN_VLLM_MAX_MODEL_LEN", "2048")),
    )
    vllm_parser.add_argument(
        "--attention-backend",
        default=os.environ.get("COUNTDOWN_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
    )
    vllm_parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=bool(int(os.environ.get("COUNTDOWN_VLLM_FLASHINFER_AUTOTUNE", "0"))),
    )
    vllm_parser.add_argument(
        "--use-flashinfer-sampler",
        choices=["0", "1"],
        default=os.environ.get("COUNTDOWN_VLLM_USE_FLASHINFER_SAMPLER", "0"),
    )
    vllm_parser.add_argument(
        "--compute-recoverability",
        action=argparse.BooleanOptionalAction,
        default=bool(int(os.environ.get("COUNTDOWN_COMPUTE_RECOVERABILITY", "0"))),
    )
    vllm_parser.add_argument("--lora-adapter", type=Path)
    vllm_parser.add_argument("--lora-name", default="countdown_adapter")
    vllm_parser.add_argument("--lora-id", type=int, default=1)
    vllm_parser.add_argument("--max-lora-rank", type=int, default=32)
    vllm_parser.set_defaults(func=evaluate_vllm)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--evaluations", type=Path, required=True)
    calibrate_parser.add_argument("--summary", type=Path, required=True)
    calibrate_parser.add_argument("--decision", type=Path, required=True)
    calibrate_parser.add_argument("--pass1-min", type=float, default=0.05)
    calibrate_parser.add_argument("--pass1-max", type=float, default=0.20)
    calibrate_parser.add_argument("--pass32-min", type=float, default=0.35)
    calibrate_parser.add_argument("--pass32-max", type=float, default=0.70)
    calibrate_parser.set_defaults(func=calibrate)

    select_sweep_parser = subparsers.add_parser("select-sweep")
    select_sweep_parser.add_argument("--candidates", type=Path, required=True)
    select_sweep_parser.add_argument("--output", type=Path, required=True)
    select_sweep_parser.add_argument("--decision", type=Path, required=True)
    select_sweep_parser.add_argument("--pass1-min", type=float, default=0.05)
    select_sweep_parser.add_argument("--pass1-max", type=float, default=0.20)
    select_sweep_parser.add_argument("--pass32-min", type=float, default=0.35)
    select_sweep_parser.add_argument("--pass32-max", type=float, default=0.70)
    select_sweep_parser.add_argument(
        "--require-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    select_sweep_parser.set_defaults(func=select_sweep)

    preflight_parser = subparsers.add_parser("preflight-reduced")
    preflight_parser.add_argument("--calibration-decision", type=Path, required=True)
    preflight_parser.add_argument("--train-evaluations", type=Path, required=True)
    preflight_parser.add_argument("--dev-evaluations", type=Path, required=True)
    preflight_parser.add_argument("--decision", type=Path, required=True)
    preflight_parser.add_argument("--min-train-matched", type=int, default=300)
    preflight_parser.add_argument("--min-dev-matched", type=int, default=100)
    preflight_parser.set_defaults(func=preflight_reduced)

    dataset_parser = subparsers.add_parser("build-datasets")
    dataset_parser.add_argument("--evaluations", type=Path, required=True)
    dataset_parser.add_argument("--output-dir", type=Path, required=True)
    dataset_parser.add_argument(
        "--conditions",
        nargs="+",
        default=["raw", "clean", "formatting", "hindsight", "curriculum"],
    )
    dataset_parser.add_argument(
        "--matched-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    dataset_parser.add_argument(
        "--canonical-raw-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    dataset_parser.set_defaults(func=build_datasets)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--evaluations", type=Path, required=True)
    summarize_parser.add_argument("--summary", type=Path, required=True)
    summarize_parser.set_defaults(func=summarize)

    export_parser = subparsers.add_parser("export-lora")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--rank", type=int, default=32)
    export_parser.add_argument("--alpha", type=int, default=64)
    export_parser.add_argument("--num-attention-heads", type=int, default=16)
    export_parser.add_argument("--num-key-value-heads", type=int, default=8)
    export_parser.add_argument("--head-dim", type=int, default=128)
    export_parser.add_argument(
        "--base-model-name-or-path",
        default="./assets/hf/Qwen3-1.7B",
    )
    export_parser.add_argument("--torch-dtype", default="bfloat16")
    export_parser.set_defaults(func=export_lora)

    runtime_preflight_parser = subparsers.add_parser("preflight-runtime")
    runtime_preflight_parser.add_argument("--model", type=Path, required=True)
    runtime_preflight_parser.add_argument("--decision", type=Path, required=True)
    runtime_preflight_parser.add_argument(
        "--required-modules",
        nargs="+",
        default=["torch", "vllm", "datasets", "transformers", "spmd_types"],
    )
    runtime_preflight_parser.add_argument(
        "--require-rootfs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    runtime_preflight_parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    runtime_preflight_parser.set_defaults(func=preflight_runtime)

    matrix_parser = subparsers.add_parser("validate-eval-matrix")
    matrix_parser.add_argument("--eval-root", type=Path, required=True)
    matrix_parser.add_argument("--decision", type=Path, required=True)
    matrix_parser.add_argument("--splits", nargs="+", required=True)
    matrix_parser.add_argument("--arms", nargs="+", required=True)
    matrix_parser.add_argument(
        "--expected-problems",
        nargs="+",
        required=True,
        help="Expected split sizes as SPLIT=COUNT.",
    )
    matrix_parser.add_argument("--num-rollouts", type=int, default=32)
    matrix_parser.set_defaults(func=validate_eval_matrix)

    split_registry_parser = subparsers.add_parser("validate-splits")
    split_registry_parser.add_argument(
        "--split",
        nargs="+",
        required=True,
        help="Problem JSONL paths as SPLIT=PATH.",
    )
    split_registry_parser.add_argument("--output", type=Path, required=True)
    split_registry_parser.set_defaults(func=validate_splits)

    report_input_parser = subparsers.add_parser("build-report-input")
    report_input_parser.add_argument("--experiment-root", type=Path, required=True)
    report_input_parser.add_argument("--mode", choices=["smoke", "reduced", "full"], required=True)
    report_input_parser.add_argument("--run-id", required=True)
    report_input_parser.add_argument("--manifest", type=Path, required=True)
    report_input_parser.add_argument("--output", type=Path, required=True)
    report_input_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    report_input_parser.set_defaults(func=build_report_input)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

