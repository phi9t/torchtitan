# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-native Countdown generation for GLM-5.3-Flash."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import torch

from torchtitan.config import ConfigLoader
from torchtitan.experiments.countdown_search_distill.countdown import CountdownProblem
from torchtitan.experiments.countdown_search_distill.evaluate import (
    evaluate_rollouts,
    write_annotations_jsonl,
    write_evaluations_jsonl,
    write_rollouts_jsonl,
    write_summary_csv,
    write_summary_json,
    write_summary_md,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer


logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"</?think>|<\|[^>]+?\|>")


def _problem_from_row(row: dict[str, Any]) -> tuple[str, CountdownProblem]:
    generation_config = row.get("generation_config", {})
    if not isinstance(generation_config, dict):
        generation_config = {}
    solution = row.get("canonical_solution", row.get("solution", []))
    if not isinstance(solution, list):
        solution = []
    problem = CountdownProblem(
        numbers=tuple(int(number) for number in row["numbers"]),
        target=int(row["target"]),
        min_solution_depth=(
            None
            if generation_config.get("min_solution_depth") is None
            else int(generation_config["min_solution_depth"])
        ),
        require_all_numbers=bool(generation_config.get("require_all_numbers", False)),
        solution=tuple(str(step) for step in solution),
    )
    return (
        str(
            row.get("problem_id")
            or f"cd-{'-'.join(map(str, problem.numbers))}-t{problem.target}"
        ),
        problem,
    )


def _load_problems(path: Path, limit: int) -> list[tuple[str, CountdownProblem]]:
    problems: list[tuple[str, CountdownProblem]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("problem row must be a JSON object")
            problems.append(_problem_from_row(row))
        except Exception as exc:
            raise ValueError(
                f"invalid problem row at {path}:{line_number}: {exc}"
            ) from exc
        if len(problems) >= limit:
            break
    if not problems:
        raise ValueError(f"no problems loaded from {path}")
    return problems


def _checkpoint_step(checkpoint_step_dir: Path | None) -> int:
    if checkpoint_step_dir is None:
        return -1
    match = re.fullmatch(r"step-(\d+)", checkpoint_step_dir.name)
    if match is None:
        raise ValueError(
            "--checkpoint-step-dir must end with a step directory such as step-94"
        )
    return int(match.group(1))


def _config_args(args: argparse.Namespace) -> list[str]:
    return [
        "--module",
        args.module,
        "--config",
        args.config,
        "--output-dir",
        str(args.output_dir),
    ]


def _build_trainer(args: argparse.Namespace) -> Trainer:
    checkpoint_step = _checkpoint_step(args.checkpoint_step_dir)
    config = cast(Trainer.Config, ConfigLoader().load(_config_args(args)))
    config.dump_folder = str(args.output_dir)
    config.checkpointer.load_only = True
    config.checkpointer.keep_latest_k = 0
    if args.checkpoint_step_dir is not None:
        config.checkpointer.folder = str(args.checkpoint_step_dir.parent)
        config.checkpointer.load_step = checkpoint_step
    else:
        config.checkpointer.load_step = -1
    trainer = config.build()
    trainer.engine.load_checkpoint()
    for model_part in trainer.engine.model_parts:
        model_part.eval()
        # The engine sets _skip_lm_head for ChunkedLossWrapper, which makes forward
        # return hidden states [T, D]. Generation needs vocabulary logits [T, V].
        if getattr(model_part, "_skip_lm_head", False):
            model_part._skip_lm_head = False
    return trainer


def _render_prompt(tokenizer: Any, problem: CountdownProblem) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem.prompt()}],
        add_generation_prompt=True,
    )
    return tokenizer.encode(text, add_bos=True, add_eos=False)


def _cleanup_completion(text: str) -> str:
    text = _TAG_RE.sub("", text)
    return text.strip()


def _pad_batch(
    sequences: list[list[int]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], device=device)
    max_length = int(lengths.max().item())
    tokens = torch.full(
        (len(sequences), max_length),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        width = len(sequence)
        tokens[index, :width] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :width] = True
    return tokens, attention_mask, lengths


def _sample_top_p(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    return sorted_indices.gather(-1, sampled).squeeze(-1)


@torch.no_grad()
def _generate_rollouts(
    trainer: Trainer,
    problems: list[tuple[str, CountdownProblem]],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    model = trainer.engine.model_parts[0]
    device = trainer.engine.device
    tokenizer = trainer.tokenizer
    pad_id = tokenizer.eos_id
    if pad_id is None:
        pad_id = 0

    problem_keys: list[str] = []
    prompt_lengths: list[int] = []
    sequences: list[list[int]] = []
    for problem_id, problem in problems:
        prompt = _render_prompt(tokenizer, problem)
        for _ in range(args.num_samples):
            problem_keys.append(problem_id)
            prompt_lengths.append(len(prompt))
            sequences.append(prompt.copy())

    finished = [False] * len(sequences)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    eos_id = tokenizer.eos_id

    for step in range(args.max_new_tokens):
        active_indices = [index for index, done in enumerate(finished) if not done]
        for chunk_start in range(0, len(active_indices), args.decode_batch_size):
            chunk_indices = active_indices[
                chunk_start : chunk_start + args.decode_batch_size
            ]
            chunk_sequences = [sequences[index] for index in chunk_indices]
            tokens, attention_mask, lengths = _pad_batch(
                chunk_sequences,
                pad_id=pad_id,
                device=device,
            )
            logits = model(tokens, attention_mask=attention_mask)
            logits = logits.view(tokens.shape[0], tokens.shape[1], -1)
            vocab_size = model.lm_head.out_features
            assert (
                logits.shape[-1] == vocab_size
            ), f"expected vocabulary logits of width {vocab_size}, got {logits.shape[-1]}"
            last_positions = lengths - 1
            next_logits = logits[
                torch.arange(tokens.shape[0], device=device), last_positions
            ]
            next_tokens = _sample_top_p(
                next_logits,
                temperature=args.temperature,
                top_p=args.top_p,
                generator=generator,
            )
            for index, token in zip(chunk_indices, next_tokens.tolist()):
                if eos_id is not None and token == eos_id:
                    finished[index] = True
                    continue
                sequences[index].append(int(token))
        if all(finished):
            logger.info("All generations stopped after %d decode steps.", step + 1)
            break

    by_problem: dict[str, list[str]] = {problem_id: [] for problem_id, _ in problems}
    for problem_id, prompt_length, sequence in zip(
        problem_keys, prompt_lengths, sequences
    ):
        completion_tokens = sequence[prompt_length:]
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        by_problem[problem_id].append(_cleanup_completion(text))
    return by_problem


def _write_outputs(
    *,
    problems: list[tuple[str, CountdownProblem]],
    rollouts: dict[str, list[str]],
    args: argparse.Namespace,
) -> None:
    evaluations = [
        evaluate_rollouts(
            problem,
            rollouts[problem_id],
            problem_id=problem_id,
            model_id=args.model_id,
            condition=args.condition,
            prompt_variant="glm53_chat",
        )
        for problem_id, problem in problems
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_evaluations_jsonl(evaluations, args.output_dir / "evaluations.jsonl")
    write_rollouts_jsonl(evaluations, args.output_dir / "rollouts.jsonl")
    write_annotations_jsonl(evaluations, args.output_dir / "annotations.jsonl")
    summary_path = args.output_dir / "summary.json"
    write_summary_json(evaluations, summary_path)
    write_summary_csv(summary_path, summary_path.with_suffix(".csv"))
    write_summary_md(summary_path, summary_path.with_suffix(".md"))
    manifest = {
        "model_id": args.model_id,
        "condition": args.condition,
        "problems": str(args.problems),
        "num_problems": len(problems),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "decode_batch_size": args.decode_batch_size,
        "seed": args.seed,
        "checkpoint_step_dir": (
            None if args.checkpoint_step_dir is None else str(args.checkpoint_step_dir)
        ),
        "module": args.module,
        "config": args.config,
        "decode_postprocess": "skip_special_tokens plus GLM control tag trim",
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--module",
        default="torchtitan.experiments.glm53_flash.config_registry",
    )
    parser.add_argument("--config", default="glm53_flash_countdown_lora")
    parser.add_argument("--checkpoint-step-dir", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--condition", default="base")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=804)
    args = parser.parse_args()
    if args.decode_batch_size < 1:
        parser.error("--decode-batch-size must be at least 1")

    init_logger()
    trainer: Trainer | None = None
    try:
        problems = _load_problems(args.problems, args.limit)
        trainer = _build_trainer(args)
        rollouts = _generate_rollouts(trainer, problems, args)
        if torch.distributed.get_rank() == 0:
            _write_outputs(problems=problems, rollouts=rollouts, args=args)
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        sl.close_structured_logger()


if __name__ == "__main__":
    main()
