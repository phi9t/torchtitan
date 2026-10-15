# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reference Countdown eval for GLM-5.3-Flash from plain model code.

No Trainer, EP, FSDP, activation checkpointing or chunked loss: the plain
TorchTitan GLM model is built as contiguous layer slices (one slice per visible
GPU), loaded directly from HF safetensors, optionally with a PEFT LoRA adapter
merged into the base weights (W += alpha / r * B @ A, merged in fp32 then cast to
bf16). Decoding recomputes the full prefix each step (no KV cache). This is the
sanity reference that the EP generator and any serving engine are checked
against; it is slow by design.

Run inside the rootfs as a single process (not torchrun).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

from torchtitan.experiments.countdown_search_distill.evaluate import (
    evaluate_rollouts,
    write_annotations_jsonl,
    write_evaluations_jsonl,
    write_rollouts_jsonl,
    write_summary_csv,
    write_summary_json,
    write_summary_md,
)
from torchtitan.experiments.glm53_flash import generate as gen, hf_parity

logger = logging.getLogger(__name__)

_PEFT_PREFIX = "base_model.model."


def _load_adapter_deltas(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Return {hf_weight_key: fp32 delta} from a PEFT LoRA adapter directory."""
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = float(config["lora_alpha"]) / float(config["r"])
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(
        adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as sf:
        for key in sf.keys():
            tensors[key] = sf.get_tensor(key)
    deltas: dict[str, torch.Tensor] = {}
    for key, lora_a in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key[: -len(".lora_A.weight")]
        lora_b = tensors[f"{module}.lora_B.weight"]
        if not module.startswith(_PEFT_PREFIX):
            raise ValueError(f"unexpected adapter key {key}")
        hf_key = f"{module[len(_PEFT_PREFIX):]}.weight"
        deltas[hf_key] = (lora_b.float() @ lora_a.float()) * scale
    if len(deltas) * 2 != len(tensors):
        raise ValueError(
            f"adapter has {len(tensors)} tensors but {len(deltas)} lora_A/lora_B pairs"
        )
    return deltas


def _split_layers(num_layers: int, num_gpus: int, split: str | None) -> list[int]:
    if split:
        counts = [int(x) for x in split.split(",")]
        if sum(counts) != num_layers or len(counts) > num_gpus:
            raise ValueError(
                f"--layer-split {split} must sum to {num_layers} over <= {num_gpus} GPUs"
            )
        return counts
    per = math.ceil(num_layers / num_gpus)
    counts = []
    remaining = num_layers
    while remaining > 0:
        counts.append(min(per, remaining))
        remaining -= counts[-1]
    return counts


def build_slices(
    checkpoint: Path,
    *,
    adapter_dir: Path | None,
    num_layers: int,
    layer_split: str | None,
) -> tuple[list[tuple[torch.device, torch.nn.Module]], dict[str, int]]:
    index = hf_parity._read_index(checkpoint)
    deltas = {} if adapter_dir is None else _load_adapter_deltas(adapter_dir)
    merged = 0
    counts = _split_layers(num_layers, torch.cuda.device_count(), layer_split)
    slices = []
    start = 0
    torch.set_default_dtype(torch.bfloat16)
    for gpu, count in enumerate(counts):
        device = torch.device(f"cuda:{gpu}")
        keys = hf_parity._slice_hf_keys(
            index["weight_map"], range(start, start + count)
        )
        state = hf_parity._load_safetensors_subset(
            checkpoint, index["weight_map"], keys, device="cpu", dtype=torch.bfloat16
        )
        for key, delta in deltas.items():
            if key not in state:
                continue
            if state[key].shape != delta.shape:
                raise ValueError(
                    f"adapter delta {tuple(delta.shape)} does not match {key} "
                    f"{tuple(state[key].shape)}"
                )
            state[key] = (state[key].float() + delta).to(torch.bfloat16)
            merged += 1
        with torch.device(device):
            model = hf_parity._build_tt_model(
                state, layer_start=start, num_layers=count, dtype=torch.bfloat16
            )
        model.to(device).eval()
        del state
        torch.cuda.empty_cache()
        logger.info(
            "slice cuda:%d layers %d..%d allocated %.1f GiB",
            gpu,
            start,
            start + count - 1,
            torch.cuda.memory_allocated(device) / 2**30,
        )
        slices.append((device, model))
        start += count
    if merged != len(deltas):
        raise ValueError(f"merged {merged} of {len(deltas)} adapter deltas")
    return slices, {"adapter_deltas": len(deltas), "merged": merged}


@torch.no_grad()
def forward_logits(
    slices: list[tuple[torch.device, torch.nn.Module]],
    tokens_BL: torch.Tensor,
    attention_mask_BL: torch.Tensor,
) -> torch.Tensor:
    """Return fp32 logits [B, L, V] from a layer-sliced forward."""
    device0, model0 = slices[0]
    h_BLD = model0.tok_embeddings(tokens_BL.to(device0))
    h_BLND = (
        h_BLD.unsqueeze(2).expand(-1, -1, model0.glm_config.hc_mult, -1).contiguous()
    )
    for device, model in slices:
        h_BLND = h_BLND.to(device)
        mask_BL = attention_mask_BL.to(device)
        for layer in model.layers.values():
            h_BLND = layer(h_BLND, attention_mask_BL=mask_BL)
    _, last = slices[-1]
    # Final mHC collapse is a plain stream mean, matching the HF hc_head.
    return last.lm_head(last.norm(h_BLND.mean(dim=2))).float()


@torch.no_grad()
def generate(
    slices,
    prompts: list[list[int]],
    *,
    eos_ids: set[int],
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[list[int]]:
    device = slices[-1][0]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sequences = [list(p) for p in prompts]
    finished = [False] * len(sequences)
    for _ in range(max_new_tokens):
        active = [i for i, done in enumerate(finished) if not done]
        if not active:
            break
        for chunk_start in range(0, len(active), batch_size):
            chunk = active[chunk_start : chunk_start + batch_size]
            tokens, mask, lengths = gen._pad_batch(
                [sequences[i] for i in chunk], pad_id=0, device=slices[0][0]
            )
            logits = forward_logits(slices, tokens, mask)
            rows = torch.arange(len(chunk), device=logits.device)
            next_logits = logits[rows, lengths.to(logits.device) - 1]
            next_tokens = gen._sample_top_p(
                next_logits, temperature=temperature, top_p=top_p, generator=generator
            )
            for i, token in zip(chunk, next_tokens.tolist()):
                if token in eos_ids:
                    finished[i] = True
                else:
                    sequences[i].append(int(token))
    return [seq[len(p) :] for seq, p in zip(sequences, prompts)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--condition", default="base")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=804)
    parser.add_argument("--num-layers", type=int, default=45)
    parser.add_argument("--layer-split", help="comma list of layers per GPU")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    problems = gen._load_problems(args.problems, args.limit)
    started = time.time()
    slices, merge_stats = build_slices(
        args.checkpoint,
        adapter_dir=args.adapter,
        num_layers=args.num_layers,
        layer_split=args.layer_split,
    )
    load_seconds = time.time() - started

    prompts, keys = [], []
    for problem_id, problem in problems:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem.prompt()}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        for _ in range(args.num_samples):
            prompts.append(ids)
            keys.append(problem_id)
    eos_ids = {tokenizer.eos_token_id}
    generation_config = args.checkpoint / "generation_config.json"
    if generation_config.exists():
        eos = json.loads(generation_config.read_text()).get("eos_token_id")
        eos_ids.update(eos if isinstance(eos, list) else [eos] if eos else [])
    eos_ids.discard(None)

    started = time.time()
    completions = generate(
        slices,
        prompts,
        eos_ids=eos_ids,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    generate_seconds = time.time() - started

    rollouts: dict[str, list[str]] = {problem_id: [] for problem_id, _ in problems}
    for problem_id, tokens in zip(keys, completions):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        rollouts[problem_id].append(gen._cleanup_completion(text))
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
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_evaluations_jsonl(evaluations, out / "evaluations.jsonl")
    write_rollouts_jsonl(evaluations, out / "rollouts.jsonl")
    write_annotations_jsonl(evaluations, out / "annotations.jsonl")
    write_summary_json(evaluations, out / "summary.json")
    write_summary_csv(out / "summary.json", out / "summary.csv")
    write_summary_md(out / "summary.json", out / "summary.md")
    manifest = {
        "generator": "reference_eval (plain model, layer-sliced, no KV cache)",
        "checkpoint": str(args.checkpoint),
        "adapter": None if args.adapter is None else str(args.adapter),
        "adapter_merge": merge_stats,
        "model_id": args.model_id,
        "condition": args.condition,
        "problems": str(args.problems),
        "num_problems": len(problems),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eos_ids": sorted(eos_ids),
        "load_seconds": round(load_seconds, 1),
        "generate_seconds": round(generate_seconds, 1),
        "decode_postprocess": "skip_special_tokens plus GLM control tag trim",
    }
    (out / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    logger.info("wrote %s (%s)", out, json.dumps(manifest))


if __name__ == "__main__":
    main()
