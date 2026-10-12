#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only end-to-end smoke for the modded-nanogpt harness.

This is not a speedrun reproduction. It verifies that the harness can execute a
small GPT-like training and validation loop, write attempt artifacts, and parse
the result without requiring CUDA availability.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard


SCHEMA_VERSION = 1
WRAPPER = "experiments/modded_nanogpt_b200/run_cpu_smoke.sh"
DEFAULT_RESULT_ROOT = Path("experiments/modded_nanogpt_b200/results")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_cpu_smoke(
    *,
    result_dir: Path,
    run_id: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    mlp_dim: int,
    seed: int,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    import torch
    import torch.nn.functional as F
    from torch import nn

    class TinyGPT(nn.Module):
        def __init__(
            self,
            *,
            vocab_size: int,
            seq_len: int,
            embed_dim: int,
            num_heads: int,
            num_layers: int,
            mlp_dim: int,
        ) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.position_embedding = nn.Embedding(seq_len, embed_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, vocab_size)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            batch, current_seq_len = tokens.shape
            positions = torch.arange(current_seq_len, device=tokens.device).expand(
                batch, current_seq_len
            )
            x = self.token_embedding(tokens) + self.position_embedding(positions)
            mask = torch.triu(
                torch.ones(current_seq_len, current_seq_len, device=tokens.device),
                diagonal=1,
            )
            mask = mask.masked_fill(mask == 1, float("-inf"))
            x = self.blocks(x, mask=mask)
            return self.head(self.norm(x))

    def batch_for_step(
        *,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        step: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.arange(batch_size * (seq_len + 1), device=device)
        tokens = (base.reshape(batch_size, seq_len + 1) + step * 7) % vocab_size
        return tokens[:, :-1].long(), tokens[:, 1:].long()

    result_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = TinyGPT(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses: list[float] = []
    start = time.time()
    run_log = result_dir / "run.log"
    with run_log.open("w") as log:
        for step in range(steps):
            model.train()
            x, y = batch_for_step(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                step=step,
                device=device,
            )
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
            losses.append(loss_value)
            log.write(
                f"cpu_smoke step={step + 1}/{steps} train_loss={loss_value:.6f}\n"
            )

        model.eval()
        with torch.no_grad():
            x_val, y_val = batch_for_step(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                step=10_000,
                device=device,
            )
            val_logits = model(x_val)
            val_loss = F.cross_entropy(
                val_logits.reshape(-1, vocab_size), y_val.reshape(-1)
            )
        validation_loss = float(val_loss.detach())
        elapsed = time.time() - start
        log.write(f"cpu_smoke validation_loss={validation_loss:.6f}\n")
        log.write(f"cpu_smoke elapsed_seconds={elapsed:.6f}\n")

    classification = {
        "lane": "CPU",
        "mode": "cpu",
        "arm": "CPU0",
        "claim_label": "CPU-only harness smoke",
        "evidence_tier": "cpu-smoke",
        "run_id": run_id,
        "attempt_id": f"{run_id}_attempt_001",
        "environment_class": "torchtitan-rootfs-cpu",
        "claim_eligible": False,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "classification": classification,
        "device": "cpu",
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_dim": mlp_dim,
        "seed": seed,
        "initial_train_loss": losses[0],
        "final_train_loss": losses[-1],
        "validation_loss": validation_loss,
        "elapsed_seconds": elapsed,
    }
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "command": ["cpu_smoke"],
        "result_dir": str(result_dir),
        "environment": {"torch": torch.__version__, "python": sys.version},
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "classification": classification,
        "training_launched": True,
        "final_validation_reached": True,
        "included_in_baseline_stats": False,
        "blocker": None,
        "environment": {
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "final_metrics": {
            "val_loss": validation_loss,
            "train_time": elapsed,
            "step_avg": elapsed / steps,
        },
        "cpu_smoke": result,
    }
    _write_json(result_dir / "cpu_smoke.json", result)
    _write_json(result_dir / "attempt.json", attempt)
    _write_json(result_dir / "summary.json", summary)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    guard = cli_guard.guard_rootfs_cli(WRAPPER)
    if guard is not None:
        return guard
    args = parse_args()
    run_id = args.run_id or time.strftime(
        "nanogpt_cpu_smoke_%Y%m%dT%H%M%SZ", time.gmtime()
    )
    result_dir = args.result_dir or DEFAULT_RESULT_ROOT / run_id
    result = run_cpu_smoke(
        result_dir=result_dir,
        run_id=run_id,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_dim=args.mlp_dim,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "summary": str(result_dir / "summary.json"),
                "validation_loss": result["validation_loss"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
