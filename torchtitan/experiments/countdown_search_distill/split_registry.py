# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split registry helpers for Countdown split hygiene."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.countdown_search_distill.countdown import (
    CountdownProblem,
    problem_key,
)


@dataclass(frozen=True)
class SplitRegistryEntry:
    split: str
    path: str
    num_problems: int
    problem_key_hash: str
    problem_keys: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "split": self.split,
            "path": self.path,
            "num_problems": self.num_problems,
            "problem_key_hash": self.problem_key_hash,
            "problem_keys": list(self.problem_keys),
        }


@dataclass(frozen=True)
class SplitOverlap:
    problem_key: str
    occurrences: tuple[tuple[str, str], ...]

    def to_json(self) -> dict[str, object]:
        return {
            "problem_key": self.problem_key,
            "occurrences": [
                {"split": split, "problem_id": problem_id}
                for split, problem_id in self.occurrences
            ],
        }


@dataclass(frozen=True)
class SplitRegistryValidation:
    selected: bool
    entries: tuple[SplitRegistryEntry, ...]
    overlaps: tuple[SplitOverlap, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "selected": self.selected,
            "entries": [entry.to_json() for entry in self.entries],
            "overlaps": [overlap.to_json() for overlap in self.overlaps],
            "checks": {
                "no_problem_key_overlap": not self.overlaps,
            },
        }


def build_split_registry(
    split_paths: dict[str, Path],
    *,
    load_problems,
) -> SplitRegistryValidation:
    entries: list[SplitRegistryEntry] = []
    occurrences: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, path in split_paths.items():
        problems = load_problems(path)
        keys = tuple(_problem_key_to_string(problem_key(problem)) for problem in problems)
        entries.append(
            SplitRegistryEntry(
                split=split,
                path=str(path),
                num_problems=len(problems),
                problem_key_hash=_hash_problem_keys(keys),
                problem_keys=keys,
            )
        )
        for problem, key in zip(problems, keys):
            occurrences[key].append((split, _problem_id(problem)))

    overlaps = tuple(
        SplitOverlap(problem_key=key, occurrences=tuple(values))
        for key, values in sorted(occurrences.items())
        if len(values) > 1
    )
    return SplitRegistryValidation(
        selected=not overlaps,
        entries=tuple(entries),
        overlaps=overlaps,
    )


def write_split_registry(
    validation: SplitRegistryValidation,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(validation.to_json(), indent=2, sort_keys=True) + "\n")


def _problem_key_to_string(key: tuple[tuple[int, ...], int]) -> str:
    numbers, target = key
    return f"{','.join(str(number) for number in numbers)}|{target}"


def _hash_problem_keys(keys: tuple[str, ...]) -> str:
    payload = "\n".join(sorted(keys)).encode()
    return hashlib.sha256(payload).hexdigest()


def _problem_id(problem: CountdownProblem) -> str:
    numbers = "-".join(str(number) for number in problem.numbers)
    return f"cd-{numbers}-t{problem.target}"

