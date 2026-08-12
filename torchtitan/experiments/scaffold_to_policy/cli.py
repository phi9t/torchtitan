# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""CLI tools for scaffold-to-policy reasoning tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchtitan.experiments.scaffold_to_policy.arithmetic_words import (
    build_report_input,
    build_split_registry,
    evaluate_fixture_rollouts,
    generate_split,
    load_problems,
    summarize_evaluations,
    write_json,
    write_jsonl,
)


def generate_arithmetic_words(args: argparse.Namespace) -> None:
    problems = generate_split(seed=args.seed, num_problems=args.num_problems)
    write_jsonl(args.output, [problem.to_json() for problem in problems])


def validate_arithmetic_splits(args: argparse.Namespace) -> None:
    split_paths = _parse_split_paths(args.split)
    registry = build_split_registry(split_paths)
    write_json(args.output, registry)
    if not registry["selected"]:
        raise SystemExit("arithmetic split validation failed")


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

    fixture_writer = subparsers.add_parser("write-arithmetic-fixture")
    fixture_writer.add_argument("--problems", type=Path, required=True)
    fixture_writer.add_argument("--output", type=Path, required=True)
    fixture_writer.set_defaults(func=write_arithmetic_fixture)

    eval_parser = subparsers.add_parser("evaluate-arithmetic-fixture")
    eval_parser.add_argument("--problems", type=Path, required=True)
    eval_parser.add_argument("--rollouts", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--summary", type=Path, required=True)
    eval_parser.add_argument("--max-rollouts", type=int, default=32)
    eval_parser.set_defaults(func=evaluate_arithmetic_fixture)

    split_parser = subparsers.add_parser("validate-arithmetic-splits")
    split_parser.add_argument("--split", nargs="+", required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.set_defaults(func=validate_arithmetic_splits)

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
