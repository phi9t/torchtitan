# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Synthetic modular recurrence problems for scaffold-to-policy transfer."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


FINAL_RE = re.compile(r"^\s*FINAL:\s*(-?\d+)\s*$")


@dataclass(frozen=True)
class ModularSequenceProblem:
    problem_id: str
    seed: int
    start: int
    multiplier: int
    step_coeff: int
    offset: int
    modulus: int
    steps: int
    answer: int
    rationale: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "seed": self.seed,
            "start": self.start,
            "multiplier": self.multiplier,
            "step_coeff": self.step_coeff,
            "offset": self.offset,
            "modulus": self.modulus,
            "steps": self.steps,
            "answer": self.answer,
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class ModularVerification:
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
class ModularRollout:
    sample_index: int
    text: str
    verification: ModularVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class ModularProblemEvaluation:
    problem: ModularSequenceProblem
    rollouts: tuple[ModularRollout, ...]

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


def generate_problem(
    seed: int,
    *,
    min_steps: int = 7,
    max_steps: int = 13,
    min_modulus: int = 97,
    max_modulus: int = 997,
) -> ModularSequenceProblem:
    rng = random.Random(seed)
    modulus = rng.randint(min_modulus, max_modulus)
    start = rng.randint(0, modulus - 1)
    multiplier = rng.randint(2, min(19, modulus - 1))
    step_coeff = rng.randint(1, min(17, modulus - 1))
    offset = rng.randint(0, modulus - 1)
    steps = rng.randint(min_steps, max_steps)
    values = [start]
    rationale = []
    current = start
    for step in range(1, steps + 1):
        raw = multiplier * current + step_coeff * step + offset
        current = raw % modulus
        values.append(current)
        rationale.append(
            f"x_{step} = ({multiplier} * {values[step - 1]} + "
            f"{step_coeff} * {step} + {offset}) mod {modulus} = {current}"
        )
    rationale.append(f"FINAL: {current}")
    return ModularSequenceProblem(
        problem_id=_problem_id(
            seed,
            start,
            multiplier,
            step_coeff,
            offset,
            modulus,
            steps,
            current,
        ),
        seed=seed,
        start=start,
        multiplier=multiplier,
        step_coeff=step_coeff,
        offset=offset,
        modulus=modulus,
        steps=steps,
        answer=current,
        rationale=tuple(rationale),
    )


def generate_split(
    *,
    seed: int,
    num_problems: int,
    min_steps: int = 7,
    max_steps: int = 13,
    min_modulus: int = 97,
    max_modulus: int = 997,
) -> list[ModularSequenceProblem]:
    problems = []
    seen_ids = set()
    candidate_seed = seed
    while len(problems) < num_problems:
        problem = generate_problem(
            candidate_seed,
            min_steps=min_steps,
            max_steps=max_steps,
            min_modulus=min_modulus,
            max_modulus=max_modulus,
        )
        candidate_seed += 1
        if problem.problem_id in seen_ids:
            continue
        seen_ids.add(problem.problem_id)
        problems.append(problem)
    return problems


def prompt_for_problem(problem: ModularSequenceProblem) -> str:
    return "\n".join(
        [
            "Solve this modular recurrence exactly.",
            f"x_0 = {problem.start}",
            (
                "For each step i = 1 through "
                f"{problem.steps}, compute x_i = "
                f"({problem.multiplier} * x_(i-1) + "
                f"{problem.step_coeff} * i + {problem.offset}) mod "
                f"{problem.modulus}."
            ),
            f"What is x_{problem.steps}?",
            "Return a short calculation trace.",
            "The last line must be exactly FINAL: <integer>.",
        ]
    )


def verify_answer(
    problem: ModularSequenceProblem,
    text: str,
) -> ModularVerification:
    matches = [
        int(match.group(1))
        for line in text.splitlines()
        if (match := FINAL_RE.match(line)) is not None
    ]
    if not matches:
        return ModularVerification(
            success=False,
            final_value=None,
            strict_final=False,
            error="missing FINAL line",
        )
    final_value = matches[-1]
    if final_value != problem.answer:
        return ModularVerification(
            success=False,
            final_value=final_value,
            strict_final=True,
            error=f"final value {final_value} does not match answer {problem.answer}",
        )
    return ModularVerification(
        success=True,
        final_value=final_value,
        strict_final=True,
        error=None,
    )


def evaluate_fixture_rollouts(
    problem: ModularSequenceProblem,
    rollouts: Sequence[str],
) -> ModularProblemEvaluation:
    return ModularProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            ModularRollout(
                sample_index=index,
                text=text,
                verification=verify_answer(problem, text),
            )
            for index, text in enumerate(rollouts)
        ),
    )


def pass_at_k(
    evaluations: Sequence[ModularProblemEvaluation],
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
    evaluations: Sequence[ModularProblemEvaluation],
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
        "task": "modular_sequences",
        "verifier": "strict_final_modular_integer_v1",
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
) -> dict[str, object]:
    summaries = {
        split: json.loads(path.read_text()) for split, path in summary_paths.items()
    }
    registry = json.loads(split_registry.read_text())
    checks = {
        "split_registry_selected": bool(registry.get("selected", False)),
        "summaries_present": all(path.is_file() for path in summary_paths.values()),
        "summary_split_counts_match": all(
            summaries[split]["num_problems"]
            == registry["splits"][split]["num_problems"]
            for split in summary_paths
        ),
    }
    return {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "task": "modular_sequences",
            "lane": "reasoning",
            "scaffold": {"type": "fixture_or_best_of_n", "budget": 32},
        },
        "artifacts": {
            "data_root": str(data_root),
            "results_root": str(results_root),
            "split_registry": str(split_registry),
            "summaries": {split: str(path) for split, path in summary_paths.items()},
        },
        "verifier": {
            "kind": "exact",
            "name": "strict_final_modular_integer_v1",
            "output_contract": "A line exactly matching FINAL: <integer>.",
        },
        "checks": checks,
        "metrics": {"splits": summaries},
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_problems(path: Path) -> list[ModularSequenceProblem]:
    problems = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problems.append(
                ModularSequenceProblem(
                    problem_id=str(row["problem_id"]),
                    seed=int(row["seed"]),
                    start=int(row["start"]),
                    multiplier=int(row["multiplier"]),
                    step_coeff=int(row["step_coeff"]),
                    offset=int(row["offset"]),
                    modulus=int(row["modulus"]),
                    steps=int(row["steps"]),
                    answer=int(row["answer"]),
                    rationale=tuple(str(step) for step in row["rationale"]),
                )
            )
        except Exception as exc:
            raise ValueError(f"invalid problem at {path}:{line_number}: {exc}") from exc
    return problems


def _problem_id(
    seed: int,
    start: int,
    multiplier: int,
    step_coeff: int,
    offset: int,
    modulus: int,
    steps: int,
    answer: int,
) -> str:
    digest = hashlib.sha256(
        (
            f"{seed}|{start}|{multiplier}|{step_coeff}|{offset}|"
            f"{modulus}|{steps}|{answer}"
        ).encode()
    ).hexdigest()
    return f"ms-{digest[:16]}"


def _hash_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()
