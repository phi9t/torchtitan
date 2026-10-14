# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF-vs-TorchTitan Qwen3 logits parity for Countdown prompts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchtitan.models.qwen3 import build_model_config


_PROMPTS = (
    "Numbers: 2, 3, 4\nTarget: 20\n\nReturn equations then FINAL.",
    "Numbers: 8, 7, 6, 3\nTarget: 81\n\nReturn equations then FINAL.",
    "Numbers: 75, 50, 8, 7, 6, 3\nTarget: 952\n\nReturn equations then FINAL.",
    "Numbers: 10, 9, 5, 4, 3, 2\nTarget: 372\n\nReturn equations then FINAL.",
)
_TORCHVISION_LIBRARY: torch.library.Library | None = None


def _ensure_torchvision_nms_schema() -> None:
    global _TORCHVISION_LIBRARY
    try:
        _TORCHVISION_LIBRARY = torch.library.Library("torchvision", "DEF")
        for op_name in ("nms", "qnms"):
            _TORCHVISION_LIBRARY.define(
                f"{op_name}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
            )
    except RuntimeError:
        pass


def _build_tt_model(
    *,
    model_flavor: str,
    hf_assets_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model_config = build_model_config(model_flavor, seq_len=512, attn_backend="varlen")
    model = model_config.build().to(device="cpu", dtype=torch.bfloat16)
    adapter_cls = type(model).state_dict_adapter_cls
    if adapter_cls is None:
        raise ValueError("Qwen3 model has no state_dict_adapter_cls")
    adapter = adapter_cls(model_config, str(hf_assets_path))
    hf_state = adapter.to_hf(model.state_dict())
    dcp.load(
        hf_state, storage_reader=adapter.get_hf_storage_reader(str(hf_assets_path))
    )
    incompatible_keys = model.load_state_dict(adapter.from_hf(hf_state), strict=False)
    unexpected = list(incompatible_keys.unexpected_keys)
    missing = [
        key
        for key in incompatible_keys.missing_keys
        if not key.endswith(".moe.expert_bias_E")
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Unexpected HF checkpoint load mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    del hf_state
    model.to(device)
    model.eval()
    return model


def compare_logits(
    *,
    model_flavor: str,
    hf_assets_path: Path,
    output: Path,
    max_length: int,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("HF logits parity requires a CUDA device")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(hf_assets_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    batch = tokenizer(
        list(_PROMPTS),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    attention_mask_cpu = batch["attention_mask"]

    _ensure_torchvision_nms_schema()
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_assets_path,
        dtype=torch.bfloat16,
    ).to(device)
    hf_model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = attention_mask_cpu.to(device)
        hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask).logits
        hf_valid = hf_logits[attention_mask.bool()].float().cpu()
    del hf_logits, hf_model, input_ids, attention_mask
    torch.cuda.empty_cache()

    tt_model = _build_tt_model(
        model_flavor=model_flavor,
        hf_assets_path=hf_assets_path,
        device=device,
    )
    packed_ids = []
    packed_positions = []
    for row_ids, row_mask in zip(batch["input_ids"], attention_mask_cpu):
        length = int(row_mask.sum().item())
        packed_ids.append(row_ids[:length].to(device))
        packed_positions.append(torch.arange(length, device=device))
    tt_input_ids = torch.cat(packed_ids, dim=0)
    tt_positions = torch.cat(packed_positions, dim=0)

    with torch.no_grad():
        attention_metadata = tt_model._get_attention_metadata(
            positions=tt_positions,
            max_context_length=max_length,
        )
        tt_logits = tt_model(
            tt_input_ids,
            positions=tt_positions,
            attention_metadata=attention_metadata,
        )
        tt_valid = tt_logits.float().cpu()

    max_abs_diff = (hf_valid.float() - tt_valid.float()).abs().max().item()
    top1_agreement = (
        hf_valid.argmax(dim=-1).eq(tt_valid.argmax(dim=-1)).float().mean().item()
    )
    result = {
        "model_flavor": model_flavor,
        "hf_assets_path": str(hf_assets_path),
        "num_prompts": len(_PROMPTS),
        "num_valid_tokens": int(attention_mask_cpu.sum().item()),
        "dtype": "bfloat16",
        "max_abs_logit_diff": max_abs_diff,
        "top1_agreement": top1_agreement,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-flavor", default="0.6B")
    parser.add_argument(
        "--hf-assets-path",
        type=Path,
        default=Path(
            "/data05",
            os.environ.get("USER", "user"),
            "glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/glm53_flash_sft_ladder/results/rung0/parity/qwen3_0_6b.json"
        ),
    )
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    compare_logits(
        model_flavor=args.model_flavor,
        hf_assets_path=args.hf_assets_path,
        output=args.output,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
