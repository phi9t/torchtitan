# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collect Qwen3-MoE expert-load histograms on Countdown dev prompts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


def _load_prompts(path: Path, max_prompts: int) -> list[str]:
    prompts = []
    with path.open() as handle:
        for line in handle:
            if len(prompts) >= max_prompts:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(str(row["prompt"]))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _summarize_histogram(histogram: torch.Tensor) -> dict[str, object]:
    total = int(histogram.sum().item())
    if total == 0:
        return {
            "total_assignments": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "max_fraction": 0.0,
            "zero_experts": int(histogram.numel()),
            "entropy": 0.0,
            "normalized_entropy": 0.0,
        }
    probs = histogram.float() / total
    nonzero = probs[probs > 0]
    entropy = float((-(nonzero * nonzero.log())).sum().item())
    return {
        "total_assignments": total,
        "min": int(histogram.min().item()),
        "max": int(histogram.max().item()),
        "mean": float(histogram.float().mean().item()),
        "max_fraction": float(histogram.max().item() / total),
        "zero_experts": int((histogram == 0).sum().item()),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(histogram.numel()),
    }


def collect_router_stats(
    *,
    hf_assets_path: Path,
    dev_problems: Path,
    output: Path,
    model_id: str,
    adapter_path: Path | None,
    max_prompts: int,
    max_length: int,
    collapse_threshold: float,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Router statistics require a CUDA device")
    if adapter_path is not None and PeftModel is None:
        raise RuntimeError("PEFT is required to collect adapter router statistics")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(hf_assets_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        hf_assets_path,
        dtype=torch.bfloat16,
    ).to(device)
    if adapter_path is not None:
        assert PeftModel is not None
        model = PeftModel.from_pretrained(model, str(adapter_path)).to(device)
    model.eval()

    config = (
        model.config.get_text_config()
        if hasattr(model.config, "get_text_config")
        else model.config
    )
    top_k = int(getattr(config, "num_experts_per_tok", 1))
    num_experts = int(getattr(config, "num_experts", 0))
    if num_experts <= 0:
        raise ValueError("Model config does not expose a positive num_experts")

    prompts = _load_prompts(dev_problems, max_prompts)
    per_layer_histograms: list[torch.Tensor] | None = None
    num_sequences = 0
    num_valid_tokens = 0

    with torch.no_grad():
        for prompt in prompts:
            batch = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_router_logits=True,
                use_cache=False,
                return_dict=True,
            )
            router_logits = getattr(outputs, "router_logits", None)
            if router_logits is None:
                raise RuntimeError("HF model output did not include router_logits")

            if per_layer_histograms is None:
                per_layer_histograms = [
                    torch.zeros(num_experts, dtype=torch.int64)
                    for _ in range(len(router_logits))
                ]

            valid_mask = attention_mask.reshape(-1).bool()
            for layer_idx, logits in enumerate(router_logits):
                logits_TE = logits.detach().float()
                if logits_TE.ndim == 3:
                    logits_TE = logits_TE.reshape(-1, logits_TE.shape[-1])
                if logits_TE.shape[0] == valid_mask.numel():
                    logits_TE = logits_TE[valid_mask]
                expert_ids = torch.topk(logits_TE, k=top_k, dim=-1).indices.reshape(-1)
                counts = torch.bincount(
                    expert_ids.cpu(),
                    minlength=num_experts,
                )
                per_layer_histograms[layer_idx].add_(counts)

            num_sequences += int(input_ids.shape[0])
            num_valid_tokens += int(attention_mask.sum().item())

    assert per_layer_histograms is not None
    overall = torch.stack(per_layer_histograms).sum(dim=0)
    per_layer = [
        {
            "layer": layer_idx,
            "histogram": histogram.tolist(),
            "summary": _summarize_histogram(histogram),
        }
        for layer_idx, histogram in enumerate(per_layer_histograms)
    ]
    overall_summary = _summarize_histogram(overall)
    result = {
        "model_id": model_id,
        "hf_assets_path": str(hf_assets_path),
        "adapter_path": None if adapter_path is None else str(adapter_path),
        "dev_problems": str(dev_problems),
        "num_prompts": len(prompts),
        "num_sequences": num_sequences,
        "num_valid_tokens": num_valid_tokens,
        "num_layers": len(per_layer_histograms),
        "num_experts": num_experts,
        "top_k": top_k,
        "router_dtype_for_stats": "float32",
        "collapse_threshold": collapse_threshold,
        "collapse_detected": overall_summary["max_fraction"] > collapse_threshold,
        "overall_histogram": overall.tolist(),
        "overall_summary": overall_summary,
        "per_layer": per_layer,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-assets-path", type=Path, required=True)
    parser.add_argument("--dev-problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--collapse-threshold", type=float, default=0.25)
    args = parser.parse_args()
    collect_router_stats(
        hf_assets_path=args.hf_assets_path,
        dev_problems=args.dev_problems,
        output=args.output,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        max_prompts=args.max_prompts,
        max_length=args.max_length,
        collapse_threshold=args.collapse_threshold,
    )


if __name__ == "__main__":
    main()
