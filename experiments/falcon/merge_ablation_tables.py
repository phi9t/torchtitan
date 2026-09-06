# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Merge the fast-arm and slow-arm ablation tables into one combined table.

The Step-4 ablation (ticket 07) split arms across separate runner invocations
because the recurrent neighbor control (A1, ``mixer=gdn``) and the Falcon-1
residual arm (A3, ``variant=falcon1``) use O(L) python scans that are far
slower than the masked-parallel arms (A0/A2/A4/A5). This script joins the
per-tag ``table.json`` files into ``combined/table.{json,md}`` under the
ablation results root so the report reads one table.

Read-only over inputs; writes only under ``combined/``. Rootfs not required
(pure json + stdlib), but this lives with the runner so evidence stays
co-located and gitignored.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path("experiments/falcon/results/ablations")
# Arm display order A0..A5.
_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5"]
_TAGS = [
    "full_2seed_8000",  # A0 A2 A4 A5
    "full_A1_2seed_8000",  # A1
    "full_A3_2seed_8000",  # A3 (may be absent / floored)
]


def _load(tag: str) -> dict:
    p = _ROOT / tag / "table.json"
    if not p.exists():
        return {"arms": [], "addition": {}}
    return json.loads(p.read_text())


def main() -> None:
    arms: list[dict] = []
    addition: dict[str, dict] = {}
    claim = "representative_small + replicated_eval"
    for tag in _TAGS:
        d = _load(tag)
        arms.extend(d.get("arms", []))
        addition.update(d.get("addition", {}))

    def key(rec: dict) -> tuple[int, int]:
        arm = rec["arm"]
        return (_ORDER.index(arm) if arm in _ORDER else 99, rec["seed"])

    arms.sort(key=key)

    def add_key(item: tuple[str, dict]) -> tuple[int, int]:
        k, v = item
        arm = v["arm"]
        return (_ORDER.index(arm) if arm in _ORDER else 99, v["seed"])

    addition = dict(sorted(addition.items(), key=add_key))

    out_dir = _ROOT / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"claim_label": claim, "arms": arms, "addition": addition}
    (out_dir / "table.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"# Falcon Step 4 ablation table (combined; claim_label={claim})",
        "",
        "## LM arms (held-out val on fineweb_val_000000.bin, 8000 steps unless noted)",
        "",
        "| arm | mixer | variant | alignment | phi | seed | steps | final_loss | val_ce | val_ppl | finite |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in arms:
        lines.append(
            f"| {r['arm']} | {r['mixer']} | {r['variant']} | {r['alignment']} | "
            f"{r['phi']} | {r['seed']} | {r['completed_steps']} | "
            f"{r['final_loss']:.4f} | {r['val_ce']:.4f} | {r['val_ppl']:.2f} | "
            f"{r['all_finite']} |"
        )
    lines += [
        "",
        "## Addition transfer (separate small train per mixer, teacher-forced)",
        "",
        "| arm | mixer | variant | alignment | phi | seed | ID acc | OOD acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, v in addition.items():
        lines.append(
            f"| {v['arm']} | {v['mixer']} | {v['variant']} | {v['alignment']} | "
            f"{v['phi']} | {v['seed']} | {v['id_acc']:.3f} | {v['ood_acc']:.3f} |"
        )
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    print(
        f"wrote {out_dir}/table.json and table.md; arms={len(arms)} add={len(addition)}"
    )


if __name__ == "__main__":
    main()
