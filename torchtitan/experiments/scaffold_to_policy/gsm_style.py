# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""GSM-style final-answer normalization for reasoning evaluations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


FINAL_PATTERNS = (
    re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"####\s*(.+?)\s*$"),
)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(
    r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"
)


@dataclass(frozen=True)
class GSMStyleProblem:
    problem_id: str
    source: str
    question: str
    answer: str
    normalized_answer: str
    rationale: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "question": self.question,
            "answer": self.answer,
            "normalized_answer": self.normalized_answer,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class GSMStyleVerification:
    success: bool
    final_value: str | None
    normalized_value: str | None
    strict_final: bool
    error: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "final_value": self.final_value,
            "normalized_value": self.normalized_value,
            "strict_final": self.strict_final,
            "error": self.error,
        }


@dataclass(frozen=True)
class GSMStyleRollout:
    sample_index: int
    text: str
    verification: GSMStyleVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class GSMStyleProblemEvaluation:
    problem: GSMStyleProblem
    rollouts: tuple[GSMStyleRollout, ...]

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


def normalize_answer(value: str) -> str | None:
    value = value.strip()
    boxed = BOXED_RE.findall(value)
    if boxed:
        value = boxed[-1]
    value = value.replace("$", "").replace(",", "").strip()
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", value)
    matches = NUMBER_RE.findall(value)
    if not matches:
        return None
    token = matches[-1].replace(",", "").replace(" ", "")
    try:
        number = Fraction(token)
    except ValueError:
        return None
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def prompt_for_problem(problem: GSMStyleProblem) -> str:
    return "\n".join(
        [
            "Solve this grade-school math problem.",
            problem.question,
            "Return a short calculation trace.",
            "The last line must be exactly FINAL: <answer>.",
        ]
    )


def verify_answer(problem: GSMStyleProblem, text: str) -> GSMStyleVerification:
    final_value = _extract_final_value(text)
    if final_value is None:
        return GSMStyleVerification(
            success=False,
            final_value=None,
            normalized_value=None,
            strict_final=False,
            error="missing final answer",
        )
    normalized = normalize_answer(final_value)
    if normalized is None:
        return GSMStyleVerification(
            success=False,
            final_value=final_value,
            normalized_value=None,
            strict_final=True,
            error=f"could not normalize final answer {final_value!r}",
        )
    if normalized != problem.normalized_answer:
        return GSMStyleVerification(
            success=False,
            final_value=final_value,
            normalized_value=normalized,
            strict_final=True,
            error=(
                f"final value {normalized} does not match answer "
                f"{problem.normalized_answer}"
            ),
        )
    return GSMStyleVerification(
        success=True,
        final_value=final_value,
        normalized_value=normalized,
        strict_final=True,
        error=None,
    )


def evaluate_fixture_rollouts(
    problem: GSMStyleProblem,
    rollouts: Sequence[str],
) -> GSMStyleProblemEvaluation:
    return GSMStyleProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            GSMStyleRollout(
                sample_index=index,
                text=text,
                verification=verify_answer(problem, text),
            )
            for index, text in enumerate(rollouts)
        ),
    )


def pass_at_k(
    evaluations: Sequence[GSMStyleProblemEvaluation],
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
    evaluations: Sequence[GSMStyleProblemEvaluation],
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
        "task": "gsm_style",
        "verifier": "gsm_style_normalized_final_v1",
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
            "task": "gsm_style",
            "lane": "reasoning",
            "scaffold": {"type": "fixture_or_no_tool_sampling", "budget": 32},
        },
        "artifacts": {
            "data_root": str(data_root),
            "results_root": str(results_root),
            "split_registry": str(split_registry),
            "summaries": {split: str(path) for split, path in summary_paths.items()},
        },
        "verifier": {
            "kind": "exact",
            "name": "gsm_style_normalized_final_v1",
            "output_contract": "A line exactly matching FINAL: <answer>.",
            "normalization": [
                "FINAL and GSM8K #### answer markers",
                "commas, currency markers, boxed answers",
                "integers, decimals, and simple fractions",
            ],
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


def load_problems(path: Path) -> list[GSMStyleProblem]:
    problems = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            question = str(row["question"])
            answer = str(row["answer"])
            normalized_answer = normalize_answer(answer)
            if normalized_answer is None:
                raise ValueError(f"could not normalize answer {answer!r}")
            problem_id = str(
                row.get("problem_id")
                or _problem_id(str(row.get("source", "gsm_style")), question, answer)
            )
            problems.append(
                GSMStyleProblem(
                    problem_id=problem_id,
                    source=str(row.get("source", "gsm_style")),
                    question=question,
                    answer=answer,
                    normalized_answer=normalized_answer,
                    rationale=(
                        None
                        if row.get("rationale") is None
                        else str(row.get("rationale"))
                    ),
                )
            )
        except Exception as exc:
            raise ValueError(f"invalid problem at {path}:{line_number}: {exc}") from exc
    return problems


def _extract_final_value(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        for pattern in FINAL_PATTERNS:
            match = pattern.search(line)
            if match is not None:
                return match.group(1).strip()
    return None


def _problem_id(source: str, question: str, answer: str) -> str:
    digest = hashlib.sha256(f"{source}|{question}|{answer}".encode()).hexdigest()
    return f"gsm-{digest[:16]}"


def _hash_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()
