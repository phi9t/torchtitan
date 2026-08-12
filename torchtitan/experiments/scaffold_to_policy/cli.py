# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""CLI tools for scaffold-to-policy reasoning tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
from torchtitan.experiments.scaffold_to_policy import modular_sequences


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
    report_input = modular_sequences.build_report_input(
        data_root=args.data_root,
        results_root=args.results_root,
        run_id=args.run_id,
        split_registry=args.split_registry,
        summary_paths=summary_paths,
    )
    modular_sequences.write_json(args.output, report_input)
    if args.require_selected and not all(report_input["checks"].values()):
        failed = [
            name for name, passed in report_input["checks"].items() if not passed
        ]
        raise SystemExit(f"modular report input failed: {', '.join(failed)}")


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

    fixture_writer = subparsers.add_parser("write-arithmetic-fixture")
    fixture_writer.add_argument("--problems", type=Path, required=True)
    fixture_writer.add_argument("--output", type=Path, required=True)
    fixture_writer.set_defaults(func=write_arithmetic_fixture)

    modular_fixture_writer = subparsers.add_parser("write-modular-fixture")
    modular_fixture_writer.add_argument("--problems", type=Path, required=True)
    modular_fixture_writer.add_argument("--output", type=Path, required=True)
    modular_fixture_writer.set_defaults(func=write_modular_fixture)

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
    modular_vllm_parser.set_defaults(func=evaluate_modular_vllm)

    split_parser = subparsers.add_parser("validate-arithmetic-splits")
    split_parser.add_argument("--split", nargs="+", required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.set_defaults(func=validate_arithmetic_splits)

    modular_split_parser = subparsers.add_parser("validate-modular-splits")
    modular_split_parser.add_argument("--split", nargs="+", required=True)
    modular_split_parser.add_argument("--output", type=Path, required=True)
    modular_split_parser.set_defaults(func=validate_modular_splits)

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
    modular_report_parser.add_argument("--output", type=Path, required=True)
    modular_report_parser.add_argument(
        "--require-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    modular_report_parser.set_defaults(func=build_modular_report_input)

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
