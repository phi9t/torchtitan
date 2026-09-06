# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixture-scale Mini Kimi K3 training smoke."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchtitan.experiments.mini_kimi_k3.model import MiniK3ForCausalLM  # noqa: E402
from torchtitan.experiments.mini_kimi_k3.model_contract import (  # noqa: E402
    MiniK3Config,
)
from torchtitan.experiments.mini_kimi_k3.token_shards import (  # noqa: E402
    load_manifest,
    TokenShardLoader,
)


SCHEMA_VERSION = 1


def run_training_smoke(
    *,
    token_manifest: Path,
    tokens_dir: Path,
    report_path: Path,
    seq_len: int = 16,
    batch_size: int = 2,
    steps: int = 1,
    seed: int = 1,
    learning_rate: float = 1.0e-3,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError(
            "Mini-K3 training smoke must run inside the TorchTitan rootfs"
        )
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2 for next-token loss, got {seq_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    _seed_everything(seed)
    config = _tiny_training_config()
    manifest = load_manifest(token_manifest)
    loader = TokenShardLoader(
        manifest,
        tokens_dir,
        seq_len=seq_len,
        rank=0,
        world_size=1,
    )
    model = MiniK3ForCausalLM(config)
    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(model),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )

    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(steps):
        batch = _next_batch(loader, batch_size=batch_size, vocab_size=config.vocab_size)
        positions = torch.arange(
            batch.shape[1] - 1,
            device=batch.device,
        ).expand(batch.shape[0], -1)
        logits = model(batch[:, :-1], positions=positions)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            batch[:, 1:].reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _global_grad_norm(model)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(grad_norm.detach().cpu()))

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_training_smoke",
        "status": "pass",
        "rootfs": {
            "marker": os.environ.get("TORCHTITAN_IN_ROOTFS"),
            "cwd": str(Path.cwd()),
        },
        "data": {
            "manifest": str(token_manifest),
            "tokens_dir": str(tokens_dir),
            "seq_len": seq_len,
            "batch_size": batch_size,
            "tokens_emitted": loader.tokens_emitted,
            "assigned_tokens": loader.assigned_token_count,
            "sources": sorted(loader.assigned_shards),
        },
        "model": {
            "variant": config.name,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size,
        },
        "optimization": {
            "steps": steps,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "adam_betas": [0.9, 0.95],
            "adam_eps": 1.0e-8,
            "losses": losses,
            "max_grad_norm": max(grad_norms),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixture-scale Mini Kimi K3 forward/backward smoke."
    )
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--tokens-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/results/tiny_smoke.json"),
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_training_smoke(
            token_manifest=args.token_manifest,
            tokens_dir=args.tokens_dir,
            report_path=args.report,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
    except Exception as exc:
        print(f"Mini Kimi K3 tiny training smoke failed: {exc}", file=sys.stderr)
        return 21
    print(f"wrote tiny training smoke report: {args.report}", file=sys.stderr)
    return 0


def _tiny_training_config() -> MiniK3Config:
    return MiniK3Config(
        name="tiny",
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=4,
        routed_expert_hidden_size=4,
        linear_attn_num_heads=2,
        full_attn_layers=(2,),
        kda_layers=(1, 3),
        vocab_size=32,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        kv_lora_rank=4,
        q_lora_rank=4,
        num_shared_experts=1,
        first_k_dense_replace=1,
        intermediate_size=16,
        linear_attn_head_dim=4,
        short_conv_kernel_size=1,
    )


def _next_batch(
    loader: TokenShardLoader,
    *,
    batch_size: int,
    vocab_size: int,
) -> torch.Tensor:
    sequences = [loader.next_sequence() for _ in range(batch_size)]
    batch = torch.from_numpy(np.stack(sequences).astype(np.int64, copy=False)).long()
    max_token = int(batch.max().item())
    if max_token >= vocab_size:
        raise ValueError(f"token id {max_token} exceeds tiny vocab size {vocab_size}")
    return batch


def _optimizer_param_groups(model: torch.nn.Module) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "router.bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _global_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32)
    for param in model.parameters():
        if param.grad is None:
            continue
        total = total + param.grad.detach().float().pow(2).sum()
    return total.sqrt()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
