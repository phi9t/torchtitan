# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Variable-digit addition task for the Falcon transfer gate (paper section 5.2).

Protocol:

- Prompt ``digits_n(a) + digits_n(b) =`` where ``a`` and ``b`` are ``n``-digit
  operands.
- Target: the reversed ``(n + 1)``-digit sum, least-significant digit first.
  Reversing makes generation left-to-right causal in carry order, which is the
  paper's setup.
- Loss is computed on the answer suffix only; the prompt is context.

A character-level vocabulary is used deliberately: GPT-2 BPE fragments digit
runs unpredictably, so a char encoder keeps digit positions aligned. Train
widths are uniform in ``{1..N}`` (in-distribution) and OOD is ``{N+1..M}``.
This ticket locks ``N=16, M=24`` and declares that as the OOD gap.

Legend for tensors below: B batch, T sequence (prompt + answer).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Fixed character alphabet: digits, plus, and equals. Index 0 is a pad symbol
# so collated batches have a distinguishable padding id.
_PAD = "\x00"
_ALPHABET = _PAD + "0123456789+="
ADDITION_SPLIT_SCHEMA_VERSION = 1
ADDITION_DRAW_IDS = ("id", "ood", "challenge")
ADDITION_EVAL_WIDTHS = range(1, 25)
ADDITION_ID_WIDTHS = range(1, 17)
ADDITION_OOD_WIDTHS = range(17, 25)
ADDITION_EXAMPLES_PER_WIDTH = 8


class AdditionVocab:
    """Character-level vocabulary for the addition task."""

    def __init__(self) -> None:
        self._stoi = {ch: i for i, ch in enumerate(_ALPHABET)}
        self._itos = {i: ch for ch, i in self._stoi.items()}

    @property
    def pad_id(self) -> int:
        return self._stoi[_PAD]

    @property
    def size(self) -> int:
        return len(self._stoi)

    def encode(self, text: str) -> list[int]:
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self._itos[int(i)] for i in ids)


def reversed_sum_digits(a: int, b: int, *, width: int) -> list[int]:
    """Return the ``a + b`` digits, least-significant first, over ``width+1``.

    ``width`` is the operand digit count ``n``; the sum needs at most ``n + 1``
    digits, so the answer is zero-padded on the most-significant end to exactly
    ``width + 1`` digits before reversing.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    total = a + b
    text = str(total).rjust(width + 1, "0")
    # LSD first == reverse of the MSD-first string.
    return [int(ch) for ch in text[::-1]]


@dataclass(frozen=True)
class AdditionExample:
    a: int
    b: int
    width: int
    tokens: list[int]
    loss_mask: list[int]


def encode_addition(
    a: int, b: int, *, width: int, vocab: AdditionVocab
) -> AdditionExample:
    """Encode one addition example with a suffix-only loss mask."""
    prompt = f"{a}+{b}="
    answer_digits = reversed_sum_digits(a, b, width=width)
    answer = "".join(str(d) for d in answer_digits)
    tokens = vocab.encode(prompt + answer)
    # Mask marks the answer positions only; the prompt is context.
    loss_mask = [0] * len(vocab.encode(prompt)) + [1] * len(answer)
    return AdditionExample(a=a, b=b, width=width, tokens=tokens, loss_mask=loss_mask)


def problem_identity(width: int, a: int, b: int) -> tuple[int, int, int]:
    """Commutation-aware identity for one addition problem."""
    lo, hi = sorted((a, b))
    return (width, lo, hi)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _manifest_digest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stripped_draws: dict[str, list[dict[str, Any]]] = {}
    for draw_id, rows in payload["draws"].items():
        stripped_draws[draw_id] = [
            {key: value for key, value in row.items() if key != "digest"}
            for row in rows
        ]
    return {
        "schema_version": payload["schema_version"],
        "seed": payload["seed"],
        "draws": stripped_draws,
    }


def _digest_manifest_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(_manifest_digest_payload(payload))
    ).hexdigest()


def _validate_operand_width(width: int, a: int, b: int) -> None:
    if width < 1 or width not in ADDITION_EVAL_WIDTHS:
        raise ValueError(f"wrong width: {width}")
    low = 0 if width == 1 else 10 ** (width - 1)
    high = 10**width - 1
    if not (low <= a <= high and low <= b <= high):
        raise ValueError(f"wrong width for operands at width {width}: {a}+{b}")


@dataclass(frozen=True)
class AdditionSplitManifest:
    """Immutable held-out addition identities for B10 evaluation."""

    schema_version: int
    seed: int
    draws: dict[str, list[dict[str, Any]]]
    digest: str

    @classmethod
    def generate(cls, *, seed: int) -> "AdditionSplitManifest":
        rng = random.Random(seed)
        vocab = AdditionVocab()
        seen: set[tuple[int, int, int]] = set()
        draws: dict[str, list[dict[str, Any]]] = {}
        for draw_id in ADDITION_DRAW_IDS:
            rows: list[dict[str, Any]] = []
            for width in ADDITION_EVAL_WIDTHS:
                for _ in range(ADDITION_EXAMPLES_PER_WIDTH):
                    while True:
                        x = _sample_operand(width, rng)
                        y = _sample_operand(width, rng)
                        identity = problem_identity(width, x, y)
                        if identity not in seen:
                            seen.add(identity)
                            break
                    if rng.randrange(2) == 0:
                        a, b, orientation = x, y, "ab"
                    else:
                        a, b, orientation = y, x, "ba"
                    encoded = encode_addition(a, b, width=width, vocab=vocab)
                    text = vocab.decode(encoded.tokens)
                    prompt = f"{a}+{b}="
                    rows.append(
                        {
                            "schema_version": ADDITION_SPLIT_SCHEMA_VERSION,
                            "draw_id": draw_id,
                            "width": width,
                            "a": a,
                            "b": b,
                            "orientation": orientation,
                            "prompt": prompt,
                            "target": text[len(prompt) :],
                            "digest": "",
                        }
                    )
            draws[draw_id] = rows
        payload = {
            "schema_version": ADDITION_SPLIT_SCHEMA_VERSION,
            "seed": seed,
            "draws": draws,
        }
        digest = _digest_manifest_payload(payload)
        for rows in draws.values():
            for row in rows:
                row["digest"] = digest
        return cls(
            schema_version=ADDITION_SPLIT_SCHEMA_VERSION,
            seed=seed,
            draws=draws,
            digest=digest,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdditionSplitManifest":
        if not isinstance(payload, dict):
            raise ValueError("manifest must be an object")
        if payload.get("schema_version") != ADDITION_SPLIT_SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
        seed = payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        digest = payload.get("digest")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("digest must be a SHA-256 hex string")
        draws = payload.get("draws")
        if not isinstance(draws, dict) or set(draws) != set(ADDITION_DRAW_IDS):
            raise ValueError(f"draws must contain {ADDITION_DRAW_IDS}")
        recomputed = _digest_manifest_payload(payload)
        if recomputed != digest:
            raise ValueError("manifest digest mismatch")

        seen: set[tuple[int, int, int]] = set()
        normalized_draws: dict[str, list[dict[str, Any]]] = {}
        for draw_id in ADDITION_DRAW_IDS:
            rows = draws[draw_id]
            if not isinstance(rows, list):
                raise ValueError(f"draw {draw_id} must be a list")
            by_width: dict[int, int] = defaultdict(int)
            normalized_rows: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("manifest rows must be objects")
                if row.get("schema_version") != ADDITION_SPLIT_SCHEMA_VERSION:
                    raise ValueError("row schema version mismatch")
                if row.get("draw_id") != draw_id:
                    raise ValueError("row draw_id mismatch")
                if row.get("digest") != digest:
                    raise ValueError("row digest mismatch")
                width = row.get("width")
                a = row.get("a")
                b = row.get("b")
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (width, a, b)
                ):
                    raise ValueError("width and operands must be integers")
                _validate_operand_width(width, a, b)
                identity = problem_identity(width, a, b)
                if identity in seen:
                    raise ValueError("duplicate identity across manifest draws")
                seen.add(identity)
                by_width[width] += 1
                expected_prompt = f"{a}+{b}="
                if row.get("prompt") != expected_prompt:
                    raise ValueError("prompt does not match operands")
                expected_target = str(a + b).rjust(width + 1, "0")[::-1]
                if row.get("target") != expected_target:
                    raise ValueError("target does not match operands")
                if row.get("orientation") not in {"ab", "ba"}:
                    raise ValueError("orientation must be ab or ba")
                normalized_rows.append(dict(row))
            for width in ADDITION_EVAL_WIDTHS:
                if by_width[width] != ADDITION_EXAMPLES_PER_WIDTH:
                    raise ValueError(f"wrong width count for draw {draw_id}: {width}")
            normalized_draws[draw_id] = normalized_rows
        return cls(
            schema_version=ADDITION_SPLIT_SCHEMA_VERSION,
            seed=seed,
            draws=normalized_draws,
            digest=digest,
        )

    @classmethod
    def read_json(cls, path: Path) -> "AdditionSplitManifest":
        return cls.from_dict(json.loads(path.read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "digest": self.digest,
            "draws": {
                draw_id: [dict(row) for row in self.draws[draw_id]]
                for draw_id in ADDITION_DRAW_IDS
            },
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_json(self.to_dict()))
        return path

    def identities(self) -> set[tuple[int, int, int]]:
        return {
            problem_identity(row["width"], row["a"], row["b"])
            for rows in self.draws.values()
            for row in rows
        }

    def examples(self, draw_id: str, *, vocab: AdditionVocab) -> list[AdditionExample]:
        if draw_id not in self.draws:
            raise ValueError(f"unknown addition draw: {draw_id}")
        return [
            encode_addition(row["a"], row["b"], width=row["width"], vocab=vocab)
            for row in self.draws[draw_id]
        ]


class OnlineAdditionStream:
    """Deterministic train stream that excludes all held-out eval identities."""

    def __init__(
        self,
        *,
        seed: int,
        manifest: AdditionSplitManifest,
        vocab: AdditionVocab | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.manifest = manifest
        self.vocab = vocab or AdditionVocab()
        self.excluded = manifest.identities()

    def next_example(self) -> AdditionExample:
        while True:
            width = self.rng.choice(list(ADDITION_ID_WIDTHS))
            a = _sample_operand(width, self.rng)
            b = _sample_operand(width, self.rng)
            if problem_identity(width, a, b) not in self.excluded:
                return encode_addition(a, b, width=width, vocab=self.vocab)

    def __iter__(self) -> "OnlineAdditionStream":
        return self

    def __next__(self) -> AdditionExample:
        return self.next_example()


def _sample_operand(width: int, rng: random.Random) -> int:
    """Sample an exactly ``width``-digit operand (no leading zero for width>1)."""
    if width == 1:
        return rng.randint(0, 9)
    low = 10 ** (width - 1)
    high = 10**width - 1
    return rng.randint(low, high)


@dataclass
class AdditionDataset:
    n: int
    m: int
    id_examples: list[AdditionExample]
    ood_examples: list[AdditionExample]

    def collate(
        self, examples: list[AdditionExample], *, vocab: AdditionVocab
    ) -> dict[str, torch.Tensor]:
        """Right-pad examples into input/labels/loss_mask tensors.

        Labels are the next-token shift; loss_mask marks answer positions in the
        label frame. Padding positions are masked out.
        """
        max_len = max(len(ex.tokens) for ex in examples)
        pad = vocab.pad_id
        inputs = []
        labels = []
        masks = []
        widths = []
        operands_a = []
        operands_b = []
        for ex in examples:
            toks = ex.tokens
            # input predicts labels[t] from input[t]; standard shift.
            inp = toks[:-1]
            lbl = toks[1:]
            # loss_mask in the label frame: mask entry i is on if label i is an
            # answer token (loss_mask[i+1] in the token frame).
            m = ex.loss_mask[1:]
            pad_len = max_len - 1 - len(inp)
            inputs.append(inp + [pad] * pad_len)
            labels.append(lbl + [pad] * pad_len)
            masks.append([bool(x) for x in m] + [False] * pad_len)
            widths.append(ex.width)
            operands_a.append(ex.a)
            operands_b.append(ex.b)
        return {
            "input": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(masks, dtype=torch.bool),
            "width": torch.tensor(widths, dtype=torch.long),
            "a": operands_a,
            "b": operands_b,
        }


def build_addition_dataset(
    *,
    n: int,
    m: int,
    num_id: int,
    num_ood: int,
    seed: int,
) -> AdditionDataset:
    """Build an ID/OOD addition split. ID widths ``{1..n}``, OOD ``{n+1..m}``."""
    if not (1 <= n < m):
        raise ValueError("require 1 <= n < m for a declared OOD gap")
    rng = random.Random(seed)
    vocab = AdditionVocab()

    def draw(widths: range, count: int) -> list[AdditionExample]:
        out: list[AdditionExample] = []
        widths = list(widths)
        for _ in range(count):
            width = rng.choice(widths)
            a = _sample_operand(width, rng)
            b = _sample_operand(width, rng)
            out.append(encode_addition(a, b, width=width, vocab=vocab))
        return out

    id_examples = draw(range(1, n + 1), num_id)
    ood_examples = draw(range(n + 1, m + 1), num_ood)
    return AdditionDataset(n=n, m=m, id_examples=id_examples, ood_examples=ood_examples)


@dataclass(frozen=True)
class AdditionMetrics:
    exact_suffix_micro: float
    token_accuracy_micro: float
    exact_suffix_width_macro: float
    token_accuracy_width_macro: float
    output_position_accuracy: dict[int, float]
    carry_count_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_suffix_micro": self.exact_suffix_micro,
            "token_accuracy_micro": self.token_accuracy_micro,
            "exact_suffix_width_macro": self.exact_suffix_width_macro,
            "token_accuracy_width_macro": self.token_accuracy_width_macro,
            "output_position_accuracy": {
                str(position): value
                for position, value in self.output_position_accuracy.items()
            },
            "carry_count_accuracy": self.carry_count_accuracy,
        }


def _carry_count(a: int, b: int, width: int) -> int:
    carry = 0
    count = 0
    for column in range(width):
        a_digit = (a // (10**column)) % 10
        b_digit = (b // (10**column)) % 10
        carry = 1 if a_digit + b_digit + carry >= 10 else 0
        count += carry
    return count


def _predicted_carry_count(
    predicted_digits: list[int], *, a: int, b: int, width: int
) -> int | None:
    if len(predicted_digits) < width:
        return None
    carry = 0
    count = 0
    for column in range(width):
        a_digit = (a // (10**column)) % 10
        b_digit = (b // (10**column)) % 10
        total = a_digit + b_digit + carry
        if predicted_digits[column] not in range(10):
            return None
        carry_delta = total - predicted_digits[column]
        if carry_delta not in (0, 10):
            return None
        carry = carry_delta // 10
        count += carry
    return count


def evaluate_addition_suffixes(
    logits: torch.Tensor, batch: dict[str, torch.Tensor], *, vocab: AdditionVocab
) -> AdditionMetrics:
    """Evaluate suffix-only addition accuracy on teacher-forced logits."""
    labels = batch["labels"]
    mask = batch["loss_mask"].bool()
    preds = logits.argmax(dim=-1).detach().cpu()
    labels = labels.detach().cpu()
    mask = mask.detach().cpu()
    widths = batch["width"].detach().cpu().tolist()
    operands_a_raw = batch["a"]
    operands_b_raw = batch["b"]
    operands_a = (
        operands_a_raw.detach().cpu().tolist()
        if isinstance(operands_a_raw, torch.Tensor)
        else list(operands_a_raw)
    )
    operands_b = (
        operands_b_raw.detach().cpu().tolist()
        if isinstance(operands_b_raw, torch.Tensor)
        else list(operands_b_raw)
    )

    total_tokens = int(mask.sum().item())
    correct_tokens = int(((preds == labels) & mask).sum().item())

    exact_by_width: dict[int, list[float]] = defaultdict(list)
    token_by_width_num: dict[int, int] = defaultdict(int)
    token_by_width_den: dict[int, int] = defaultdict(int)
    position_num: dict[int, int] = defaultdict(int)
    position_den: dict[int, int] = defaultdict(int)
    carry_correct = 0

    digit_ids = {vocab.encode(str(digit))[0]: digit for digit in range(10)}
    for row_idx, width in enumerate(widths):
        positions = [idx for idx, on in enumerate(mask[row_idx].tolist()) if on]
        row_correct = [
            bool(preds[row_idx, pos].item() == labels[row_idx, pos].item())
            for pos in positions
        ]
        exact = all(row_correct)
        exact_by_width[width].append(1.0 if exact else 0.0)
        token_by_width_num[width] += sum(1 for value in row_correct if value)
        token_by_width_den[width] += len(row_correct)
        for suffix_position, is_correct in enumerate(row_correct):
            position_den[suffix_position] += 1
            position_num[suffix_position] += 1 if is_correct else 0

        predicted_digits = [
            digit_ids.get(int(preds[row_idx, pos].item()), -1) for pos in positions
        ]
        predicted_count = _predicted_carry_count(
            predicted_digits,
            a=int(operands_a[row_idx]),
            b=int(operands_b[row_idx]),
            width=int(width),
        )
        expected_count = _carry_count(
            int(operands_a[row_idx]), int(operands_b[row_idx]), int(width)
        )
        if predicted_count == expected_count:
            carry_correct += 1

    example_count = len(widths)
    exact_micro = (
        sum(sum(values) for values in exact_by_width.values()) / example_count
        if example_count
        else 0.0
    )
    token_micro = correct_tokens / total_tokens if total_tokens else 0.0
    exact_width_macro = (
        sum(sum(values) / len(values) for values in exact_by_width.values())
        / len(exact_by_width)
        if exact_by_width
        else 0.0
    )
    token_width_macro = (
        sum(
            token_by_width_num[width] / token_by_width_den[width]
            for width in token_by_width_den
        )
        / len(token_by_width_den)
        if token_by_width_den
        else 0.0
    )
    return AdditionMetrics(
        exact_suffix_micro=exact_micro,
        token_accuracy_micro=token_micro,
        exact_suffix_width_macro=exact_width_macro,
        token_accuracy_width_macro=token_width_macro,
        output_position_accuracy={
            position: position_num[position] / position_den[position]
            for position in sorted(position_den)
        },
        carry_count_accuracy=carry_correct / example_count if example_count else 0.0,
    )
