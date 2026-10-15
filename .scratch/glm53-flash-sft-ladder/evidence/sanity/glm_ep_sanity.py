# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Same sanity checks as glm_infer_sanity.py, but through the EP8 Trainer build used by generate.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from torchtitan.experiments.glm53_flash import generate as gen
from torchtitan.observability.logging import init_logger

CKPT = Path("/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16")


def main():
    init_logger()
    out_dir = Path(sys.argv[1])
    args = argparse.Namespace(
        module="torchtitan.experiments.glm53_flash_ep.config_registry",
        config="glm53_flash_ep_countdown_lora_truncated",
        output_dir=out_dir,
        checkpoint_step_dir=None,
    )
    trainer = gen._build_trainer(args)
    model = trainer.engine.model_parts[0]
    device = trainer.engine.device
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    rank = torch.distributed.get_rank()
    report = {}

    def fwd(ids, mask=None):
        t = torch.tensor(ids, device=device)
        with torch.no_grad():
            lg = model(t, attention_mask=mask)
        return lg.view(t.shape[0], t.shape[1], -1).float()

    text = (
        "The quick brown fox jumps over the lazy dog. The capital of France is Paris, "
        "and the capital of Germany is Berlin. Water boils at 100 degrees Celsius at sea level."
    )
    ids = tok.encode(text)
    lg = fwd([ids])
    report["logits_shape"] = list(lg.shape)
    report["plain_text_ce_nats"] = F.cross_entropy(
        lg[0, :-1], torch.tensor(ids[1:], device=device)
    ).item()

    ids = tok.encode("The capital of France is")
    lg = fwd([ids])
    top = torch.topk(lg[0, -1].softmax(-1), 5)
    report["france_top5"] = [
        (tok.decode([int(i)]), round(float(p), 4))
        for p, i in zip(top.values, top.indices)
    ]

    # Same prompt through the torchtitan tokenizer chat path used by generate.py.
    tt_tok = trainer.tokenizer
    row = json.loads(
        Path("/project/formal-cache/datasets/countdown-sft/v1/dev/problems.jsonl")
        .open()
        .readline()
    )
    pid, problem = gen._problem_from_row(row)
    tt_ids = gen._render_prompt(tt_tok, problem)
    hf_chat = tok.apply_chat_template(
        [{"role": "user", "content": problem.prompt()}],
        add_generation_prompt=True,
        tokenize=False,
    )
    hf_ids = tok.encode(hf_chat, add_special_tokens=False)
    report["tt_prompt_ids_head"] = tt_ids[:12]
    report["hf_prompt_ids_head"] = hf_ids[:12]
    report["tt_vs_hf_prompt_equal"] = tt_ids == hf_ids
    report["tt_prompt_decoded_head"] = tok.decode(tt_ids[:40])
    report["tt_eos_id"] = tt_tok.eos_id
    if rank == 0:
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        (out_dir / "ep_result.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
    trainer.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
