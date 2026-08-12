# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Executable-code evaluation helpers for scaffold-to-policy coding smokes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
ASSERT_CALL_RE = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ASSERT_CALL_ARGS_RE = re.compile(
    r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:==|!=|is|in|<=|>=|<|>)",
    re.DOTALL,
)


@dataclass(frozen=True)
class CodingStyleProblem:
    problem_id: str
    source: str
    prompt: str
    test: str
    entry_point: str
    canonical_solution: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "prompt": self.prompt,
            "test": self.test,
            "entry_point": self.entry_point,
            "canonical_solution": self.canonical_solution,
        }


@dataclass(frozen=True)
class CodingStyleVerification:
    success: bool
    error: str | None
    extracted_code: str
    returncode: int | None
    stdout: str
    stderr: str

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "error": self.error,
            "extracted_code": self.extracted_code,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class CodingStyleRollout:
    sample_index: int
    text: str
    verification: CodingStyleVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class CodingStyleProblemEvaluation:
    problem: CodingStyleProblem
    rollouts: tuple[CodingStyleRollout, ...]

    def solved_at(self) -> int | None:
        for index, rollout in enumerate(self.rollouts, start=1):
            if rollout.verification.success:
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
            "bucket": self.bucket(),
            "rollouts": [rollout.to_json() for rollout in self.rollouts],
        }


def prompt_for_problem(problem: CodingStyleProblem) -> str:
    return "\n".join(
        [
            "Complete this Python function.",
            "Return only Python code for the function implementation.",
            "Do not include Markdown fences or explanatory prose.",
            "",
            problem.prompt.rstrip(),
        ]
    )


def verify_solution(
    problem: CodingStyleProblem,
    text: str,
    *,
    timeout_seconds: float = 5.0,
) -> CodingStyleVerification:
    extracted = extract_candidate_code(problem, text)
    program = "\n".join(
        [
            extracted.rstrip(),
            "",
            problem.test.rstrip(),
            "",
            f"check({problem.entry_point})",
            "",
        ]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="coding-style-") as tmpdir:
            completed = subprocess.run(
                [sys.executable, "-I", "-"],
                input=program,
                text=True,
                capture_output=True,
                cwd=tmpdir,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        return CodingStyleVerification(
            success=False,
            error="timeout",
            extracted_code=extracted,
            returncode=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
    if completed.returncode == 0:
        return CodingStyleVerification(
            success=True,
            error=None,
            extracted_code=extracted,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return CodingStyleVerification(
        success=False,
        error=_classify_failure(completed.stderr),
        extracted_code=extracted,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def extract_candidate_code(problem: CodingStyleProblem, text: str) -> str:
    candidate = text.strip()
    fences = FENCED_CODE_RE.findall(candidate)
    if fences:
        candidate = fences[-1].strip()
    candidate = _strip_leading_chat_text(candidate)
    if f"def {problem.entry_point}" in candidate:
        return _prompt_preamble(problem) + candidate
    return problem.prompt + candidate.lstrip()


def evaluate_fixture_rollouts(
    problem: CodingStyleProblem,
    rollouts: Sequence[str],
    *,
    timeout_seconds: float = 5.0,
) -> CodingStyleProblemEvaluation:
    return CodingStyleProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            CodingStyleRollout(
                sample_index=index,
                text=text,
                verification=verify_solution(
                    problem,
                    text,
                    timeout_seconds=timeout_seconds,
                ),
            )
            for index, text in enumerate(rollouts)
        ),
    )


def pass_at_k(
    evaluations: Sequence[CodingStyleProblemEvaluation],
    ks: Sequence[int],
) -> dict[int, float]:
    if not evaluations:
        return {k: 0.0 for k in ks}
    results = {}
    for k in ks:
        solved = 0
        for evaluation in evaluations:
            solved_at = evaluation.solved_at()
            if solved_at is not None and solved_at <= k:
                solved += 1
        results[k] = solved / len(evaluations)
    return results


def summarize_evaluations(
    evaluations: Sequence[CodingStyleProblemEvaluation],
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
        "task": "coding_style",
        "verifier": "python_executable_tests_v1",
        "splits": splits,
        "overlaps": overlaps,
    }


def import_public_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[CodingStyleProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        prompt = str(row["prompt"])
        test = str(row["test"])
        entry_point = str(row["entry_point"])
        task_id = str(row.get("task_id") or _problem_id(source, prompt, entry_point))
        problems.append(
            CodingStyleProblem(
                problem_id=task_id,
                source=source,
                prompt=prompt,
                test=test,
                entry_point=entry_point,
                canonical_solution=(
                    None
                    if row.get("canonical_solution") is None
                    else str(row.get("canonical_solution"))
                ),
            )
        )
    return problems


def import_mbpp_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[CodingStyleProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        prompt = str(row["prompt"]).strip()
        test_imports = [str(value) for value in row.get("test_imports", [])]
        test_list = [str(value) for value in row["test_list"]]
        if not test_list:
            raise ValueError("MBPP row has no tests")
        entry_point = _entry_point_from_asserts(test_list)
        test = _mbpp_check_source(test_imports, test_list, entry_point)
        task_id = f"MBPP/{row['task_id']}"
        canonical = None if row.get("code") is None else str(row.get("code"))
        signature = _signature_from_first_assert(test_list[0], entry_point)
        problems.append(
            CodingStyleProblem(
                problem_id=task_id,
                source=source,
                prompt=_mbpp_prompt(prompt, entry_point, signature),
                test=test,
                entry_point=entry_point,
                canonical_solution=canonical,
            )
        )
    return problems


def build_public_provenance(
    *,
    dataset: str,
    subset: str | None,
    revision: str,
    source_split: str,
    output: Path,
    limit: int,
    offset: int,
    problems: Sequence[CodingStyleProblem],
) -> dict[str, object]:
    problem_ids = [problem.problem_id for problem in problems]
    return {
        "dataset": dataset,
        "subset": subset,
        "revision": revision,
        "source_split": source_split,
        "output": str(output),
        "limit": limit,
        "offset": offset,
        "num_problems": len(problems),
        "problem_id_hash": _hash_lines(problem_ids),
        "source": _public_source(dataset, subset, revision, source_split),
    }


def build_report_input(
    *,
    data_root: Path,
    results_root: Path,
    run_id: str,
    split_registry: Path,
    summary_paths: dict[str, Path],
    scaffold_budget: int,
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
            "task": "coding_style",
            "lane": "coding",
            "scaffold": {
                "type": "fixture_or_no_tool_sampling",
                "budget": scaffold_budget,
            },
        },
        "artifacts": {
            "data_root": str(data_root),
            "results_root": str(results_root),
            "split_registry": str(split_registry),
            "summaries": {split: str(path) for split, path in summary_paths.items()},
        },
        "verifier": {
            "kind": "executable",
            "name": "python_executable_tests_v1",
            "output_contract": "Python function code that satisfies public tests.",
            "execution": [
                "candidate code is extracted from text or Markdown fences",
                "tests run in a separate isolated Python subprocess",
                "each rollout has a wall-clock timeout",
            ],
            "limitations": [
                "no third-party package installation inside candidate tests",
                "not a secure sandbox for malicious code",
                "HumanEval smoke only; not a LiveCodeBench or Terminal-Bench claim",
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


def load_problems(path: Path) -> list[CodingStyleProblem]:
    problems = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problems.append(
                CodingStyleProblem(
                    problem_id=str(row["problem_id"]),
                    source=str(row["source"]),
                    prompt=str(row["prompt"]),
                    test=str(row["test"]),
                    entry_point=str(row["entry_point"]),
                    canonical_solution=(
                        None
                        if row.get("canonical_solution") is None
                        else str(row.get("canonical_solution"))
                    ),
                )
            )
        except Exception as exc:
            raise ValueError(f"invalid problem at {path}:{line_number}: {exc}") from exc
    return problems


def load_evaluations(path: Path) -> list[CodingStyleProblemEvaluation]:
    evaluations = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problem = _problem_from_json(row["problem"])
            rollouts = [str(rollout["text"]) for rollout in row["rollouts"]]
            evaluations.append(evaluate_fixture_rollouts(problem, rollouts))
        except Exception as exc:
            raise ValueError(
                f"invalid evaluation at {path}:{line_number}: {exc}"
            ) from exc
    return evaluations


def _problem_from_json(row: dict[str, object]) -> CodingStyleProblem:
    return CodingStyleProblem(
        problem_id=str(row["problem_id"]),
        source=str(row["source"]),
        prompt=str(row["prompt"]),
        test=str(row["test"]),
        entry_point=str(row["entry_point"]),
        canonical_solution=(
            None
            if row.get("canonical_solution") is None
            else str(row.get("canonical_solution"))
        ),
    )


def _strip_leading_chat_text(candidate: str) -> str:
    lines = candidate.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "from ", "import ", "@")):
            return "\n".join(lines[index:]).strip()
    return candidate


def _prompt_preamble(problem: CodingStyleProblem) -> str:
    marker = f"def {problem.entry_point}"
    index = problem.prompt.find(marker)
    if index == -1:
        return ""
    return problem.prompt[:index]


def _classify_failure(stderr: str) -> str:
    if "AssertionError" in stderr:
        return "assertion failure"
    if "SyntaxError" in stderr:
        return "syntax error"
    if "NameError" in stderr:
        return "name error"
    if "IndentationError" in stderr:
        return "indentation error"
    if "TypeError" in stderr:
        return "type error"
    if "RecursionError" in stderr:
        return "recursion error"
    if stderr.strip():
        return stderr.strip().splitlines()[-1][:160]
    return "nonzero exit"


def _entry_point_from_asserts(test_list: Sequence[str]) -> str:
    names = []
    for test in test_list:
        match = ASSERT_CALL_RE.search(test)
        if match is None:
            raise ValueError(f"could not infer MBPP entry point from test: {test}")
        names.append(match.group(1))
    if len(set(names)) != 1:
        raise ValueError(f"MBPP tests reference multiple entry points: {sorted(set(names))}")
    return names[0]


def _signature_from_first_assert(test: str, entry_point: str) -> str:
    match = ASSERT_CALL_ARGS_RE.search(test)
    if match is None or match.group(1) != entry_point:
        return "*args, **kwargs"
    args = match.group(2).strip()
    if not args:
        return ""
    return ", ".join(f"arg{index}" for index, _ in enumerate(_split_call_args(args), start=1))


def _split_call_args(args: str) -> list[str]:
    parts = []
    depth = 0
    in_string: str | None = None
    escaped = False
    start = 0
    for index, char in enumerate(args):
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(args[start:index].strip())
            start = index + 1
    parts.append(args[start:].strip())
    return [part for part in parts if part]


def _mbpp_prompt(prompt: str, entry_point: str, signature: str) -> str:
    return "\n".join(
        [
            f"# {prompt}",
            f"def {entry_point}({signature}):",
            "    ",
        ]
    )


def _mbpp_check_source(
    test_imports: Sequence[str],
    test_list: Sequence[str],
    entry_point: str,
) -> str:
    rewritten_tests = [
        ASSERT_CALL_RE.sub("assert candidate(", test, count=1) for test in test_list
    ]
    body = "\n    ".join(rewritten_tests)
    imports = "\n".join(test_imports)
    return "\n".join(
        line
        for line in [
            imports,
            f"def check(candidate):\n    {body}",
        ]
        if line
    )


def _problem_id(source: str, prompt: str, entry_point: str) -> str:
    digest = hashlib.sha256(
        "\n".join([source, prompt, entry_point]).encode()
    ).hexdigest()[:16]
    return f"coding-{digest}"


def _hash_lines(lines: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def _public_source(
    dataset: str,
    subset: str | None,
    revision: str,
    source_split: str,
) -> str:
    if subset:
        return f"{dataset}:{subset}:{revision}:{source_split}"
    return f"{dataset}:{revision}:{source_split}"
