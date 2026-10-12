# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation helpers for measuring Countdown best-of-N search gaps."""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.countdown_search_distill.countdown import (
    CountdownProblem,
    VerificationResult,
    verify_solution,
)


@dataclass(frozen=True)
class RolloutEvaluation:
    problem_id: str
    sample_index: int
    text: str
    verification: VerificationResult
    model_id: str = "fixture"
    condition: str = "base"
    prompt_variant: str = "default"
    token_count: int | None = None
    logprob_sum: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "model_id": self.model_id,
            "condition": self.condition,
            "sample_index": self.sample_index,
            "prompt_variant": self.prompt_variant,
            "text": self.text,
            "token_count": self.token_count,
            "logprob_sum": self.logprob_sum,
            "success": self.verification.success,
            "final_value": self.verification.final_value,
            "steps_consumed": self.verification.steps_consumed,
            "available_numbers": list(self.verification.available_numbers),
            "operations": [
                operation.to_json() for operation in self.verification.operations
            ],
            "states": [list(state) for state in self.verification.states],
            "valid_prefix_length": self.verification.valid_prefix_length,
            "first_invalid": self.verification.first_invalid,
            "first_unreachable": self.verification.first_unreachable,
            "distance_to_target": self.verification.distance_to_target,
            "solution_depth": self.verification.solution_depth,
            "recoverable": self.verification.recoverable,
            "error": self.verification.error,
        }


@dataclass(frozen=True)
class ProblemEvaluation:
    problem_id: str
    problem: CountdownProblem
    rollouts: tuple[RolloutEvaluation, ...]

    def solved_at(self) -> int | None:
        for index, rollout in enumerate(self.rollouts, start=1):
            if rollout.verification.success:
                return index
        return None

    def strict_solved_at(self) -> int | None:
        for index, rollout in enumerate(self.rollouts, start=1):
            if rollout.verification.success and has_strict_final_line(
                rollout.text,
                self.problem.target,
            ):
                return index
        return None

    def bucket(self) -> str:
        solved_at = self.solved_at()
        if solved_at == 1:
            return "easy"
        if solved_at is None:
            return "unreached"
        return "elicitable"

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "problem": self.problem.to_json(),
            "solved_at": self.solved_at(),
            "bucket": self.bucket(),
            "rollouts": [rollout.to_json() for rollout in self.rollouts],
        }


def evaluate_rollouts(
    problem: CountdownProblem,
    rollouts: Sequence[str],
    *,
    problem_id: str | None = None,
    model_id: str = "fixture",
    condition: str = "base",
    prompt_variant: str = "default",
    compute_recoverability: bool = True,
) -> ProblemEvaluation:
    resolved_problem_id = problem_id or stable_problem_id(problem)
    return ProblemEvaluation(
        problem_id=resolved_problem_id,
        problem=problem,
        rollouts=tuple(
            RolloutEvaluation(
                problem_id=resolved_problem_id,
                sample_index=index,
                text=rollout,
                verification=verify_solution(
                    problem,
                    rollout,
                    compute_recoverability=compute_recoverability,
                ),
                model_id=model_id,
                condition=condition,
                prompt_variant=prompt_variant,
                token_count=len(rollout.split()),
            )
            for index, rollout in enumerate(rollouts)
        ),
    )


def stable_problem_id(problem: CountdownProblem) -> str:
    numbers = "-".join(str(number) for number in problem.numbers)
    return f"cd-{numbers}-t{problem.target}"


def pass_at_k(
    evaluations: Sequence[ProblemEvaluation], ks: Sequence[int]
) -> dict[int, float]:
    if not evaluations:
        return {k: 0.0 for k in ks}
    results: dict[int, float] = {}
    for k in ks:
        solved = 0
        for evaluation in evaluations:
            solved_at = evaluation.solved_at()
            if solved_at is not None and solved_at <= k:
                solved += 1
        results[k] = solved / len(evaluations)
    return results


def strict_pass_at_k(
    evaluations: Sequence[ProblemEvaluation], ks: Sequence[int]
) -> dict[int, float]:
    if not evaluations:
        return {k: 0.0 for k in ks}
    results: dict[int, float] = {}
    for k in ks:
        solved = 0
        for evaluation in evaluations:
            solved_at = evaluation.strict_solved_at()
            if solved_at is not None and solved_at <= k:
                solved += 1
        results[k] = solved / len(evaluations)
    return results


def has_strict_final_line(text: str, target: int) -> bool:
    return any(line.strip() == f"FINAL: {target}" for line in text.splitlines())


def bucket_counts(evaluations: Sequence[ProblemEvaluation]) -> dict[str, int]:
    counts = {"easy": 0, "elicitable": 0, "unreached": 0}
    for evaluation in evaluations:
        counts[evaluation.bucket()] += 1
    return counts


def bucketed_pass_at_k(
    evaluations: Sequence[ProblemEvaluation], ks: Sequence[int]
) -> dict[str, dict[int, float]]:
    by_bucket: dict[str, list[ProblemEvaluation]] = {
        "easy": [],
        "elicitable": [],
        "unreached": [],
    }
    for evaluation in evaluations:
        by_bucket[evaluation.bucket()].append(evaluation)
    return {
        bucket: pass_at_k(bucket_evaluations, ks)
        for bucket, bucket_evaluations in by_bucket.items()
    }


def retained_search_lift(
    *,
    base_pass_at_1: float,
    base_pass_at_32: float,
    distilled_pass_at_1: float,
) -> float | None:
    denominator = base_pass_at_32 - base_pass_at_1
    if denominator <= 0:
        return None
    return (distilled_pass_at_1 - base_pass_at_1) / denominator


def compute_compression(
    *,
    base_curve: dict[int, float],
    distilled_curve: dict[int, float],
    target_success: float,
) -> float | None:
    base_budget = _minimum_budget(base_curve, target_success)
    distilled_budget = _minimum_budget(distilled_curve, target_success)
    if base_budget is None or distilled_budget is None:
        return None
    return base_budget / distilled_budget


def _minimum_budget(curve: dict[int, float], target_success: float) -> int | None:
    for k, value in sorted(curve.items()):
        if value >= target_success:
            return k
    return None


def bootstrap_pass_at_k(
    evaluations: Sequence[ProblemEvaluation],
    ks: Sequence[int],
    *,
    num_resamples: int = 1000,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    if not evaluations:
        return {k: {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0} for k in ks}
    rng = random.Random(seed)
    samples_by_k: dict[int, list[float]] = {k: [] for k in ks}
    for _ in range(num_resamples):
        sample = [rng.choice(evaluations) for _ in evaluations]
        for k, value in pass_at_k(sample, ks).items():
            samples_by_k[k].append(value)
    intervals: dict[int, dict[str, float]] = {}
    for k, samples in samples_by_k.items():
        ordered = sorted(samples)
        low_index = int(0.025 * (len(ordered) - 1))
        high_index = int(0.975 * (len(ordered) - 1))
        intervals[k] = {
            "mean": statistics.fmean(samples),
            "ci_low": ordered[low_index],
            "ci_high": ordered[high_index],
        }
    return intervals


def validity_breakdown(evaluations: Sequence[ProblemEvaluation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evaluation in evaluations:
        for rollout in evaluation.rollouts:
            key = (
                "success"
                if rollout.verification.success
                else (rollout.verification.first_invalid or {}).get("reason", "unknown")
            )
            counts[str(key)] = counts.get(str(key), 0) + 1
    return counts


def format_breakdown(evaluations: Sequence[ProblemEvaluation]) -> dict[str, int]:
    counts = {
        "success_with_strict_final": 0,
        "success_missing_strict_final": 0,
        "failure_with_strict_final": 0,
        "failure_missing_strict_final": 0,
    }
    for evaluation in evaluations:
        for rollout in evaluation.rollouts:
            has_strict_final = has_strict_final_line(
                rollout.text,
                evaluation.problem.target,
            )
            if rollout.verification.success and has_strict_final:
                counts["success_with_strict_final"] += 1
            elif rollout.verification.success:
                counts["success_missing_strict_final"] += 1
            elif has_strict_final:
                counts["failure_with_strict_final"] += 1
            else:
                counts["failure_missing_strict_final"] += 1
    return counts


def length_stats(evaluations: Sequence[ProblemEvaluation]) -> dict[str, float]:
    lengths = [
        rollout.token_count or len(rollout.text.split())
        for evaluation in evaluations
        for rollout in evaluation.rollouts
    ]
    if not lengths:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(lengths),
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }


def solution_diversity(evaluations: Sequence[ProblemEvaluation]) -> dict[str, float]:
    ratios: list[float] = []
    for evaluation in evaluations:
        successful = [
            rollout.text.strip()
            for rollout in evaluation.rollouts
            if rollout.verification.success
        ]
        if successful:
            ratios.append(len(set(successful)) / len(successful))
    if not ratios:
        return {"mean_unique_success_ratio": 0.0}
    return {"mean_unique_success_ratio": statistics.fmean(ratios)}


def write_evaluations_jsonl(
    evaluations: Sequence[ProblemEvaluation], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for evaluation in evaluations:
            f.write(json.dumps(evaluation.to_json(), sort_keys=True) + "\n")


def write_rollouts_jsonl(
    evaluations: Sequence[ProblemEvaluation], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for evaluation in evaluations:
            for rollout in evaluation.rollouts:
                row = rollout.to_json()
                f.write(json.dumps(row, sort_keys=True) + "\n")


def write_annotations_jsonl(
    evaluations: Sequence[ProblemEvaluation], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for evaluation in evaluations:
            for rollout in evaluation.rollouts:
                verification = rollout.verification
                row = {
                    "problem_id": evaluation.problem_id,
                    "sample_index": rollout.sample_index,
                    "success": verification.success,
                    "bucket": evaluation.bucket(),
                    "steps_consumed": verification.steps_consumed,
                    "states": [list(state) for state in verification.states],
                    "first_invalid": verification.first_invalid,
                    "first_unreachable": verification.first_unreachable,
                    "final_value": verification.final_value,
                    "distance_to_target": verification.distance_to_target,
                    "error": verification.error,
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary_json(
    evaluations: Sequence[ProblemEvaluation],
    output_path: Path,
    *,
    ks: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> None:
    pass_curve = pass_at_k(evaluations, ks)
    strict_pass_curve = strict_pass_at_k(evaluations, ks)
    summary = {
        "num_problems": len(evaluations),
        "pass_at_k": {str(k): value for k, value in pass_curve.items()},
        "strict_format_pass_at_k": {
            str(k): value for k, value in strict_pass_curve.items()
        },
        "bucket_counts": bucket_counts(evaluations),
        "bucketed_pass_at_k": {
            bucket: {str(k): value for k, value in curve.items()}
            for bucket, curve in bucketed_pass_at_k(evaluations, ks).items()
        },
        "validity_breakdown": validity_breakdown(evaluations),
        "format_breakdown": format_breakdown(evaluations),
        "length_stats": length_stats(evaluations),
        "solution_diversity": solution_diversity(evaluations),
        "bootstrap_pass_at_k": {
            str(k): interval
            for k, interval in bootstrap_pass_at_k(
                evaluations, ks, num_resamples=200
            ).items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def write_summary_csv(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    lines = ["metric,k,value"]
    for k, value in summary["pass_at_k"].items():
        lines.append(f"pass_at_k,{k},{value}")
    for bucket, count in summary["bucket_counts"].items():
        lines.append(f"bucket_count,{bucket},{count}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def write_summary_md(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    lines = [
        "# Countdown Search Distillation Summary",
        "",
        f"- Problems: {summary['num_problems']}",
        "",
        "## pass@k",
        "",
    ]
    for k, value in summary["pass_at_k"].items():
        lines.append(f"- pass@{k}: {value:.4f}")
    lines.extend(["", "## Buckets", ""])
    for bucket, count in summary["bucket_counts"].items():
        lines.append(f"- {bucket}: {count}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
