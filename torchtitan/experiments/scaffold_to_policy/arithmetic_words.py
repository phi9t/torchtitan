# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Synthetic arithmetic word problems for scaffold-to-policy transfer."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.scaffold_to_policy import report_artifacts


FINAL_RE = re.compile(r"^\s*FINAL:\s*(-?\d+)\s*$")


@dataclass(frozen=True)
class ArithmeticWordProblem:
    problem_id: str
    seed: int
    template: str
    prompt: str
    answer: int
    rationale: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "seed": self.seed,
            "template": self.template,
            "prompt": self.prompt,
            "answer": self.answer,
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class ArithmeticVerification:
    success: bool
    final_value: int | None
    strict_final: bool
    error: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "final_value": self.final_value,
            "strict_final": self.strict_final,
            "error": self.error,
        }


@dataclass(frozen=True)
class ArithmeticRollout:
    sample_index: int
    text: str
    verification: ArithmeticVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class ArithmeticProblemEvaluation:
    problem: ArithmeticWordProblem
    rollouts: tuple[ArithmeticRollout, ...]

    def solved_at(self) -> int | None:
        for index, rollout in enumerate(self.rollouts, start=1):
            if rollout.verification.success:
                return index
        return None

    def strict_solved_at(self) -> int | None:
        for index, rollout in enumerate(self.rollouts, start=1):
            if rollout.verification.success and rollout.verification.strict_final:
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
            "problem_id": self.problem.problem_id,
            "problem": self.problem.to_json(),
            "solved_at": self.solved_at(),
            "strict_solved_at": self.strict_solved_at(),
            "bucket": self.bucket(),
            "rollouts": [rollout.to_json() for rollout in self.rollouts],
        }


def generate_problem(seed: int) -> ArithmeticWordProblem:
    rng = random.Random(seed)
    template_index = rng.randrange(3)
    if template_index == 0:
        start = rng.randint(12, 80)
        added = rng.randint(5, 40)
        removed = rng.randint(1, added + start - 1)
        answer = start + added - removed
        prompt = (
            "A warehouse starts with "
            f"{start} crates. A truck adds {added} crates, then workers ship "
            f"{removed} crates. How many crates remain?"
        )
        rationale = (
            f"{start} + {added} = {start + added}",
            f"{start + added} - {removed} = {answer}",
            f"FINAL: {answer}",
        )
        template = "add_then_subtract"
    elif template_index == 1:
        groups = rng.randint(3, 12)
        per_group = rng.randint(4, 25)
        bonus = rng.randint(2, 50)
        answer = groups * per_group + bonus
        prompt = (
            "A teacher packs "
            f"{groups} boxes with {per_group} pencils in each box, then finds "
            f"{bonus} extra pencils. How many pencils are there in total?"
        )
        rationale = (
            f"{groups} * {per_group} = {groups * per_group}",
            f"{groups * per_group} + {bonus} = {answer}",
            f"FINAL: {answer}",
        )
        template = "multiply_then_add"
    else:
        total = rng.randint(80, 240)
        divisor = rng.randint(2, 12)
        total -= total % divisor
        extra = rng.randint(3, 60)
        answer = total // divisor - extra
        while answer <= 0:
            extra = rng.randint(1, max(total // divisor - 1, 1))
            answer = total // divisor - extra
        prompt = (
            "A lab divides "
            f"{total} samples equally across {divisor} trays, then discards "
            f"{extra} samples from one tray count. What number is left?"
        )
        rationale = (
            f"{total} / {divisor} = {total // divisor}",
            f"{total // divisor} - {extra} = {answer}",
            f"FINAL: {answer}",
        )
        template = "divide_then_subtract"
    return ArithmeticWordProblem(
        problem_id=_problem_id(seed, template, prompt, answer),
        seed=seed,
        template=template,
        prompt=prompt,
        answer=answer,
        rationale=rationale,
    )


def generate_split(*, seed: int, num_problems: int) -> list[ArithmeticWordProblem]:
    problems = []
    seen_ids = set()
    candidate_seed = seed
    while len(problems) < num_problems:
        problem = generate_problem(candidate_seed)
        candidate_seed += 1
        if problem.problem_id in seen_ids:
            continue
        seen_ids.add(problem.problem_id)
        problems.append(problem)
    return problems


def prompt_for_problem(problem: ArithmeticWordProblem) -> str:
    return "\n".join(
        [
            "Solve this arithmetic word problem.",
            problem.prompt,
            "Return a short calculation trace.",
            "The last line must be exactly FINAL: <integer>.",
        ]
    )


def verify_answer(problem: ArithmeticWordProblem, text: str) -> ArithmeticVerification:
    matches = [
        int(match.group(1))
        for line in text.splitlines()
        if (match := FINAL_RE.match(line)) is not None
    ]
    if not matches:
        return ArithmeticVerification(
            success=False,
            final_value=None,
            strict_final=False,
            error="missing FINAL line",
        )
    final_value = matches[-1]
    if final_value != problem.answer:
        return ArithmeticVerification(
            success=False,
            final_value=final_value,
            strict_final=True,
            error=f"final value {final_value} does not match answer {problem.answer}",
        )
    return ArithmeticVerification(
        success=True,
        final_value=final_value,
        strict_final=True,
        error=None,
    )


def evaluate_fixture_rollouts(
    problem: ArithmeticWordProblem,
    rollouts: Sequence[str],
) -> ArithmeticProblemEvaluation:
    return ArithmeticProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            ArithmeticRollout(
                sample_index=index,
                text=text,
                verification=verify_answer(problem, text),
            )
            for index, text in enumerate(rollouts)
        ),
    )


def pass_at_k(
    evaluations: Sequence[ArithmeticProblemEvaluation],
    ks: Sequence[int],
    *,
    strict: bool = False,
) -> dict[int, float]:
    if not evaluations:
        return {k: 0.0 for k in ks}
    results = {}
    for k in ks:
        solved = 0
        for evaluation in evaluations:
            solved_at = evaluation.strict_solved_at() if strict else evaluation.solved_at()
            if solved_at is not None and solved_at <= k:
                solved += 1
        results[k] = solved / len(evaluations)
    return results


def summarize_evaluations(
    evaluations: Sequence[ArithmeticProblemEvaluation],
    *,
    ks: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> dict[str, object]:
    buckets = {"easy": 0, "elicitable": 0, "unreached": 0}
    errors: dict[str, int] = {}
    total_rollouts = 0
    for evaluation in evaluations:
        buckets[evaluation.bucket()] += 1
        for rollout in evaluation.rollouts:
            total_rollouts += 1
            key = rollout.verification.error or "success"
            errors[key] = errors.get(key, 0) + 1
    return {
        "num_problems": len(evaluations),
        "total_rollouts": total_rollouts,
        "pass_at_k": {str(k): value for k, value in pass_at_k(evaluations, ks).items()},
        "strict_format_pass_at_k": {
            str(k): value for k, value in pass_at_k(evaluations, ks, strict=True).items()
        },
        "bucket_counts": buckets,
        "failure_breakdown": errors,
    }


def build_split_registry(split_paths: dict[str, Path]) -> dict[str, object]:
    seen: dict[str, str] = {}
    overlaps = []
    splits = {}
    for split, path in split_paths.items():
        problems = load_problems(path)
        split_ids = [problem.problem_id for problem in problems]
        for problem_id in split_ids:
            if problem_id in seen:
                overlaps.append(
                    {
                        "problem_id": problem_id,
                        "first_split": seen[problem_id],
                        "second_split": split,
                    }
                )
            else:
                seen[problem_id] = split
        splits[split] = {
            "path": str(path),
            "num_problems": len(problems),
            "problem_id_hash": _hash_lines(split_ids),
        }
    return {
        "selected": not overlaps,
        "task": "arithmetic_words",
        "verifier": "strict_final_integer_v1",
        "splits": splits,
        "overlaps": overlaps,
    }


def build_report_input(
    *,
    data_root: Path,
    results_root: Path,
    run_id: str,
    split_registry: Path,
    summary_paths: dict[str, Path],
    runtime_path: Path | None = None,
) -> dict[str, object]:
    return report_artifacts.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id=run_id,
        task="arithmetic_words",
        lane="reasoning",
        scaffold={"type": "fixture_or_best_of_n", "budget": 32},
        split_registry=split_registry,
        summary_paths=summary_paths,
        runtime_path=runtime_path,
        verifier={
            "kind": "exact",
            "name": "strict_final_integer_v1",
            "output_contract": "A line exactly matching FINAL: <integer>.",
        },
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_problems(path: Path) -> list[ArithmeticWordProblem]:
    problems = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problems.append(
                ArithmeticWordProblem(
                    problem_id=str(row["problem_id"]),
                    seed=int(row["seed"]),
                    template=str(row["template"]),
                    prompt=str(row["prompt"]),
                    answer=int(row["answer"]),
                    rationale=tuple(str(step) for step in row["rationale"]),
                )
            )
        except Exception as exc:
            raise ValueError(f"invalid problem at {path}:{line_number}: {exc}") from exc
    return problems


def _problem_id(seed: int, template: str, prompt: str, answer: int) -> str:
    digest = hashlib.sha256(f"{seed}|{template}|{prompt}|{answer}".encode()).hexdigest()
    return f"aw-{digest[:16]}"


def _hash_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()
