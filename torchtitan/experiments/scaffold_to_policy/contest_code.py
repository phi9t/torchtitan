# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contest-style stdin/stdout code evaluation helpers."""

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

from torchtitan.experiments.scaffold_to_policy import report_artifacts


FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ContestTestCase:
    stdin: str
    stdout: str

    def to_json(self) -> dict[str, object]:
        return {"stdin": self.stdin, "stdout": self.stdout}


@dataclass(frozen=True)
class ContestCodeProblem:
    problem_id: str
    source: str
    title: str
    prompt: str
    starter_code: str
    public_tests: tuple[ContestTestCase, ...]
    difficulty: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "title": self.title,
            "prompt": self.prompt,
            "starter_code": self.starter_code,
            "public_tests": [test.to_json() for test in self.public_tests],
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class ContestCodeVerification:
    success: bool
    error: str | None
    extracted_code: str
    passed_tests: int
    total_tests: int
    failures: tuple[dict[str, object], ...]

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "error": self.error,
            "extracted_code": self.extracted_code,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class ContestCodeRollout:
    sample_index: int
    text: str
    verification: ContestCodeVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class ContestCodeProblemEvaluation:
    problem: ContestCodeProblem
    rollouts: tuple[ContestCodeRollout, ...]

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


def prompt_for_problem(problem: ContestCodeProblem) -> str:
    sections = [
        "Solve this programming contest problem in Python 3.",
        "Return only complete Python source code. Do not include Markdown fences.",
        "",
        f"Title: {problem.title}",
        "",
        problem.prompt.rstrip(),
    ]
    if problem.starter_code.strip():
        sections.extend(["", "Starter code:", problem.starter_code.rstrip()])
    return "\n".join(sections)


def verify_solution(
    problem: ContestCodeProblem,
    text: str,
    *,
    timeout_seconds: float = 5.0,
) -> ContestCodeVerification:
    extracted = extract_candidate_code(text)
    failures = []
    passed = 0
    for test_index, test_case in enumerate(problem.public_tests):
        try:
            with tempfile.TemporaryDirectory(prefix="contest-code-") as tmpdir:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", extracted],
                    input=test_case.stdin,
                    text=True,
                    capture_output=True,
                    cwd=tmpdir,
                    timeout=timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            failures.append(
                {
                    "test_index": test_index,
                    "error": "timeout",
                    "stdout": _subprocess_text(exc.stdout),
                    "stderr": _subprocess_text(exc.stderr),
                }
            )
            continue
        if completed.returncode != 0:
            failures.append(
                {
                    "test_index": test_index,
                    "error": _classify_failure(completed.stderr),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            continue
        if _normalize_output(completed.stdout) != _normalize_output(test_case.stdout):
            failures.append(
                {
                    "test_index": test_index,
                    "error": "wrong answer",
                    "stdout": completed.stdout,
                    "expected_stdout": test_case.stdout,
                }
            )
            continue
        passed += 1
    success = passed == len(problem.public_tests)
    return ContestCodeVerification(
        success=success,
        error=None if success else _first_failure_error(failures),
        extracted_code=extracted,
        passed_tests=passed,
        total_tests=len(problem.public_tests),
        failures=tuple(failures),
    )


def extract_candidate_code(text: str) -> str:
    candidate = text.strip()
    fences = FENCED_CODE_RE.findall(candidate)
    if fences:
        candidate = fences[-1].strip()
    return _strip_leading_chat_text(candidate)


def evaluate_fixture_rollouts(
    problem: ContestCodeProblem,
    rollouts: Sequence[str],
    *,
    timeout_seconds: float = 5.0,
) -> ContestCodeProblemEvaluation:
    return ContestCodeProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            ContestCodeRollout(
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
    evaluations: Sequence[ContestCodeProblemEvaluation],
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
    evaluations: Sequence[ContestCodeProblemEvaluation],
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
        "task": "contest_code",
        "verifier": "python_stdin_stdout_public_tests_v1",
        "splits": splits,
        "overlaps": overlaps,
    }


def import_livecodebench_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[ContestCodeProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        tests = _livecodebench_tests(row.get("public_test_cases"))
        if not tests:
            raise ValueError("LiveCodeBench row has no public stdin/stdout tests")
        question_id = str(
            row.get("question_id") or _problem_id(source, str(row["question_content"]))
        )
        problems.append(
            ContestCodeProblem(
                problem_id=f"LiveCodeBench/{question_id}",
                source=source,
                title=str(row.get("question_title") or question_id),
                prompt=str(row["question_content"]),
                starter_code=""
                if row.get("starter_code") is None
                else str(row["starter_code"]),
                public_tests=tuple(tests),
                difficulty=None
                if row.get("difficulty") is None
                else str(row.get("difficulty")),
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
    problems: Sequence[ContestCodeProblem],
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
    runtime_path: Path | None = None,
) -> dict[str, object]:
    return report_artifacts.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id=run_id,
        task="contest_code",
        lane="coding",
        scaffold={
            "type": "no_tool_sampling",
            "budget": scaffold_budget,
        },
        split_registry=split_registry,
        summary_paths=summary_paths,
        runtime_path=runtime_path,
        verifier={
            "kind": "executable",
            "name": "python_stdin_stdout_public_tests_v1",
            "output_contract": "Complete Python 3 source code that reads stdin and writes stdout.",
            "execution": [
                "candidate code is extracted from text or Markdown fences",
                "each released public test runs in a separate isolated Python subprocess",
                "stdout is compared after trailing whitespace normalization",
            ],
            "limitations": [
                "public-test smoke only; private tests are not used",
                "not a secure sandbox for malicious code",
                "not an official LiveCodeBench leaderboard protocol",
            ],
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


def load_problems(path: Path) -> list[ContestCodeProblem]:
    problems = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            problems.append(_problem_from_json(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid problem at {path}:{line_number}: {exc}") from exc
    return problems


def _problem_from_json(row: dict[str, object]) -> ContestCodeProblem:
    tests = tuple(
        ContestTestCase(
            stdin=str(test["stdin"]),
            stdout=str(test["stdout"]),
        )
        for test in row["public_tests"]
    )
    return ContestCodeProblem(
        problem_id=str(row["problem_id"]),
        source=str(row["source"]),
        title=str(row["title"]),
        prompt=str(row["prompt"]),
        starter_code=str(row.get("starter_code") or ""),
        public_tests=tests,
        difficulty=None
        if row.get("difficulty") is None
        else str(row.get("difficulty")),
    )


def _livecodebench_tests(value: object) -> list[ContestTestCase]:
    raw_tests = json.loads(str(value))
    tests = []
    for test in raw_tests:
        if test.get("testtype") != "stdin":
            continue
        tests.append(
            ContestTestCase(
                stdin=str(test["input"]),
                stdout=str(test["output"]),
            )
        )
    return tests


def _strip_leading_chat_text(candidate: str) -> str:
    lines = candidate.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "#")):
            return "\n".join(lines[index:]).strip()
    return candidate


def _normalize_output(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _first_failure_error(failures: Sequence[dict[str, object]]) -> str:
    if not failures:
        return "unknown failure"
    return str(failures[0].get("error") or "unknown failure")


def _classify_failure(stderr: str) -> str:
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


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _problem_id(source: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{source}|{prompt}".encode()).hexdigest()
    return f"contest-{digest[:16]}"


def _hash_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _public_source(
    dataset: str,
    subset: str | None,
    revision: str,
    source_split: str,
) -> str:
    if subset:
        return f"{dataset}:{subset}:{revision}:{source_split}"
    return f"{dataset}:{revision}:{source_split}"
