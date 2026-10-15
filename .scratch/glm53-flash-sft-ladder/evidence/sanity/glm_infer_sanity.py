# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone GLM-5.3-Flash inference sanity check (no Trainer, no EP, no FSDP).

Builds the plain TorchTitan GLM model as layer slices (one slice per GPU),
loads HF safetensors directly through hf_parity, and runs a pipelined forward:
plain-text CE, next-token top-5, greedy decode, and padded-batch consistency.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from torchtitan.experiments.glm53_flash import hf_parity

CKPT = Path("/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16")
DEV = Path("/project/formal-cache/datasets/countdown-sft/v1/dev/problems.jsonl")
NUM_LAYERS = 45


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_slices(num_gpus: int):
    index = hf_parity._read_index(CKPT)
    per = math.ceil(NUM_LAYERS / num_gpus)
    slices = []
    for gpu in range(num_gpus):
        start = gpu * per
        n = min(per, NUM_LAYERS - start)
        if n <= 0:
            break
        dev = torch.device(f"cuda:{gpu}")
        keys = hf_parity._slice_hf_keys(index["weight_map"], range(start, start + n))
        state = hf_parity._load_safetensors_subset(
            CKPT, index["weight_map"], keys, device="cpu", dtype=torch.bfloat16
        )
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(dev):
            model = hf_parity._build_tt_model(
                state, layer_start=start, num_layers=n, dtype=torch.bfloat16
            )
        model.to(dev).eval()
        del state
        torch.cuda.empty_cache()
        log(
            f"slice gpu{gpu} layers {start}..{start + n - 1} mem "
            f"{torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB"
        )
        slices.append((dev, model))
    return slices


@torch.no_grad()
def forward(slices, tokens_BL: torch.Tensor, mask_BL: torch.Tensor | None = None):
    dev0, m0 = slices[0]
    tokens_BL = tokens_BL.to(dev0)
    if mask_BL is None:
        mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)
    h_BLD = m0.tok_embeddings(tokens_BL)
    h = h_BLD.unsqueeze(2).expand(-1, -1, m0.glm_config.hc_mult, -1).contiguous()
    for dev, model in slices:
        h = h.to(dev)
        mask = mask_BL.to(dev)
        for layer in model.layers.values():
            h = layer(h, attention_mask_BL=mask)
    devl, ml = slices[-1]
    out = ml.norm(h.mean(dim=2))
    return ml.lm_head(out).float()  # [B, L, V]


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    slices = build_slices(torch.cuda.device_count())
    report = {}

    # A. plain-text CE
    text = (
        "The quick brown fox jumps over the lazy dog. The capital of France is Paris, "
        "and the capital of Germany is Berlin. Water boils at 100 degrees Celsius at sea level."
    )
    ids = torch.tensor([tok.encode(text)])
    logits = forward(slices, ids)
    ce = F.cross_entropy(logits[0, :-1], ids[0, 1:].to(logits.device)).item()
    report["plain_text_ce_nats"] = ce
    log(f"A. plain-text CE = {ce:.3f} nats/token over {ids.shape[1] - 1} tokens")

    # B. top-5 next token
    ids = torch.tensor([tok.encode("The capital of France is")])
    logits = forward(slices, ids)
    top = torch.topk(logits[0, -1].softmax(-1), 5)
    top5 = [
        (tok.decode([int(i)]), round(float(p), 4))
        for p, i in zip(top.values, top.indices)
    ]
    report["france_top5"] = top5
    log("B. top-5 after 'The capital of France is':", top5)

    # C. greedy decode
    def greedy(prompt_ids, n):
        seq = list(prompt_ids)
        for _ in range(n):
            lg = forward(slices, torch.tensor([seq]))
            nxt = int(lg[0, -1].argmax())
            seq.append(nxt)
            if nxt == tok.eos_token_id:
                break
        return tok.decode(seq[len(prompt_ids) :])

    plain = greedy(tok.encode("The capital of France is"), 24)
    report["greedy_plain"] = plain
    log("C1. greedy plain:", repr(plain))

    row = json.loads(DEV.open().readline())
    chat = tok.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    chat_ids = tok.encode(chat, add_special_tokens=False)
    report["chat_prompt_tail"] = chat[-200:]
    cd = greedy(chat_ids, 96)
    report["greedy_countdown"] = cd
    log("C2. greedy countdown:", repr(cd))

    # D. padded batch consistency: short prompt right-padded next to a longer one
    a = tok.encode("The capital of France is")
    b = tok.encode("Once upon a time, in a small village by the sea, there lived")
    L = max(len(a), len(b))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    toks = torch.tensor([a + [pad] * (L - len(a)), b + [pad] * (L - len(b))])
    mask = torch.tensor(
        [[1] * len(a) + [0] * (L - len(a)), [1] * len(b) + [0] * (L - len(b))]
    ).bool()
    lb = forward(slices, toks, mask)
    l1 = forward(slices, torch.tensor([a]))
    diff = (lb[0, len(a) - 1] - l1[0, -1].to(lb.device)).abs().max().item()
    same_top1 = int(lb[0, len(a) - 1].argmax()) == int(l1[0, -1].argmax())
    report["padded_vs_single_max_abs_logit_diff"] = diff
    report["padded_vs_single_same_top1"] = same_top1
    log(f"D. padded-batch vs single: max|dlogit|={diff:.4f} same_top1={same_top1}")

    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("result.json")
    )
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log("wrote", out)


if __name__ == "__main__":
    main()
