#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the GLM-5.3 architecture speedrun ablation queue.

This script is intentionally deterministic. It writes the runnable 4a/4b/4c
cells, the deferred 4d/C1 templates, and the design note consumed by the host
scheduler. Run it inside the TorchTitan rootfs:

    TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- \
      bash -lc 'unset CC CXX; cd /workspace/torchtitan && \
      python experiments/glm53_arch_speedrun/queue/generate_cells.py'
"""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


QUEUE_DIR = Path("experiments/glm53_arch_speedrun/queue")
TOKEN_BUDGET = 500_000_000
SEED_PAIR = (0, 1)
SEED_TRIPLE = (0, 1, 2)
SCIENCE_SEQ_LEN = 4096
SHORT_REPLICATE_SEQ_LEN = 1024
FRACTIONAL_SEED = 5303
FOUR_C_RUNS = 24

SPARSE_LEVELS = ("dense", "indexer_topk", "indexer_topk_indexpool")
BINARY_FACTORS = ("mhc", "positions", "mixer", "mtp")
FACTOR_ORDER = ("mhc", "positions", "mixer", "sparse", "mtp", "moe")
SLICE_ORDER = {"4a": 10, "4b": 20, "4c": 30, "4d": 40, "c1": 50}

B0 = {
    "mhc": "off",
    "positions": "rope",
    "mixer": "softmax",
    "sparse": "dense",
    "mtp": "off",
    "moe": "off",
}

G = {
    "mhc": "on",
    "positions": "nope",
    "mixer": "kda_3_to_1",
    "sparse": "indexer_topk_indexpool",
    "mtp": "on",
    "moe": "off",
}

ADD_ONE_ARMS = {
    "b0": {},
    "a1_mhc": {"mhc": "on"},
    "a2_nope": {"positions": "nope"},
    "a3_kda": {"mixer": "kda_3_to_1"},
    "a4_indexer": {"sparse": "indexer_topk"},
    "a5_indexpool": {"sparse": "indexer_topk_indexpool"},
    "a6_mtp": {"mtp": "on"},
}

ARM_ORDER = {arm: index for index, arm in enumerate(ADD_ONE_ARMS)}

LEAVE_ONE_OUT = {
    "g": {},
    "loo_mhc": {"mhc": "off"},
    "loo_nope": {"positions": "rope"},
    "loo_kda": {"mixer": "softmax"},
    "loo_sparse": {"sparse": "dense"},
    "loo_indexpool": {"sparse": "indexer_topk"},
    "loo_mtp": {"mtp": "off"},
}

ARM_ORDER.update({arm: index for index, arm in enumerate(LEAVE_ONE_OUT, start=100)})


@dataclass(frozen=True)
class FourCDesign:
    rows: list[dict[str, str]]
    model_rank: int
    model_terms: list[str]
    max_abs_column_sum: int
    attempts: int


def _copy_levels(base: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    levels = dict(base)
    levels.update(updates)
    return levels


def _top_k_for(seq_len: int, fraction: str) -> int:
    numerator, denominator = fraction.split("/", maxsplit=1)
    return max(1, int(seq_len) * int(numerator) // int(denominator))


def _dose_knobs(levels: dict[str, str], seq_len: int) -> dict[str, object]:
    sparse = levels["sparse"]
    return {
        "mhc_num_streams": 4 if levels["mhc"] == "on" else 1,
        "mhc_sinkhorn_iters": 20 if levels["mhc"] == "on" else 0,
        "kda_softmax_period": 4 if levels["mixer"] == "kda_3_to_1" else 1,
        "sparse_top_k_fraction": "1/8" if sparse != "dense" else None,
        "sparse_top_k": _top_k_for(seq_len, "1/8") if sparse != "dense" else seq_len,
        "index_pool_size": 4 if sparse == "indexer_topk_indexpool" else 1,
        "nope_scope": "all_layers" if levels["positions"] == "nope" else "none",
        "mtp_depth": 1 if levels["mtp"] == "on" else 0,
        "mtp_aux_loss_weight": 0.1 if levels["mtp"] == "on" else 0.0,
        "moe_active_params_matched": levels["moe"] == "on",
    }


def _config_overrides(
    levels: dict[str, str],
    dose_knobs: dict[str, object],
    *,
    seed: int,
    seq_len: int,
    token_budget: int,
) -> dict[str, object]:
    return {
        "debug.seed": seed,
        "training.seq_len": seq_len,
        "training.token_budget": token_budget,
        "model.elements.rope_enabled": levels["positions"] == "rope",
        "model.elements.mhc_enabled": levels["mhc"] == "on",
        "model.elements.mhc_streams": dose_knobs["mhc_num_streams"],
        "model.elements.mhc_sinkhorn_iters": dose_knobs["mhc_sinkhorn_iters"],
        "model.elements.kda_enabled": levels["mixer"] == "kda_3_to_1",
        "model.elements.kda_hybrid_3_to_1": levels["mixer"] == "kda_3_to_1",
        "model.elements.sparse_attention_enabled": levels["sparse"] != "dense",
        "model.elements.sparse_top_k": dose_knobs["sparse_top_k"],
        "model.elements.index_pool_size": dose_knobs["index_pool_size"],
        "model.elements.mtp_enabled": levels["mtp"] == "on",
        "model.elements.mtp_aux_loss_weight": dose_knobs["mtp_aux_loss_weight"],
    }


def _cell(
    *,
    cell_id: str,
    slice_name: str,
    arm: str,
    levels: dict[str, str],
    seed: int,
    seq_len: int,
    token_budget: int = TOKEN_BUDGET,
    runnable: bool = True,
    notes: list[str] | None = None,
    resources: dict[str, object] | None = None,
    context_rule: str | None = None,
) -> dict[str, object]:
    dose_knobs = _dose_knobs(levels, seq_len)
    cell_resources = {
        "num_gpus": 1,
        "gpu_memory_cap_gib": 24,
        "min_free_gib_per_gpu": 30,
        "environment_class": "shared_gpu",
    }
    if resources:
        cell_resources.update(resources)
    output = {
        "cell_id": cell_id,
        "slice": slice_name,
        "slice_order": SLICE_ORDER[slice_name],
        "arm": arm,
        "factor_levels": {name: levels[name] for name in FACTOR_ORDER},
        "dose_knobs": dose_knobs,
        "seed": seed,
        "seq_len": seq_len,
        "token_budget": token_budget,
        "config_overrides": _config_overrides(
            levels,
            dose_knobs,
            seed=seed,
            seq_len=seq_len,
            token_budget=token_budget,
        ),
        "resources": cell_resources,
        "claim_label": "representative_small+replicated_eval",
        "evidence_tier": "ablation_matrix",
        "runnable": runnable,
    }
    if context_rule is not None:
        output["context_rule"] = context_rule
    if notes:
        output["notes"] = notes
    return output


def _seeds_for_arm(arm: str) -> tuple[int, ...]:
    if arm in {"b0", "g"}:
        return SEED_TRIPLE
    return SEED_PAIR


def _slice_4a() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for seq_len in (SCIENCE_SEQ_LEN, SHORT_REPLICATE_SEQ_LEN):
        seq_label = f"s{seq_len}"
        for arm, updates in ADD_ONE_ARMS.items():
            levels = _copy_levels(B0, updates)
            for seed in _seeds_for_arm(arm):
                cells.append(
                    _cell(
                        cell_id=f"4a_{seq_label}_{arm}_seed{seed}",
                        slice_name="4a",
                        arm=arm,
                        levels=levels,
                        seed=seed,
                        seq_len=seq_len,
                        notes=(
                            ["seq-1024 replicate of the add-one slice"]
                            if seq_len == SHORT_REPLICATE_SEQ_LEN
                            else None
                        ),
                    )
                )
    return cells


def _slice_4b() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for arm, updates in LEAVE_ONE_OUT.items():
        levels = _copy_levels(G, updates)
        for seed in _seeds_for_arm(arm):
            cells.append(
                _cell(
                    cell_id=f"4b_s{SCIENCE_SEQ_LEN}_{arm}_seed{seed}",
                    slice_name="4b",
                    arm=arm,
                    levels=levels,
                    seed=seed,
                    seq_len=SCIENCE_SEQ_LEN,
                )
            )
    return cells


def _binary_code(levels: dict[str, str], factor: str) -> int:
    if factor == "mhc":
        return 1 if levels[factor] == "on" else -1
    if factor == "positions":
        return 1 if levels[factor] == "nope" else -1
    if factor == "mixer":
        return 1 if levels[factor] == "kda_3_to_1" else -1
    if factor == "mtp":
        return 1 if levels[factor] == "on" else -1
    raise ValueError(f"unknown binary factor: {factor}")


def _sparse_contrasts(levels: dict[str, str]) -> tuple[int, int]:
    sparse = levels["sparse"]
    if sparse == "dense":
        return 1, 0
    if sparse == "indexer_topk":
        return -1, 1
    if sparse == "indexer_topk_indexpool":
        return 0, -1
    raise ValueError(f"unknown sparse level: {sparse}")


def _model_terms() -> list[str]:
    terms = ["I"]
    terms.extend(BINARY_FACTORS)
    terms.extend(("sparse_c1", "sparse_c2"))
    for left, right in itertools.combinations(BINARY_FACTORS, 2):
        terms.append(f"{left}:{right}")
    for factor in BINARY_FACTORS:
        terms.append(f"{factor}:sparse_c1")
        terms.append(f"{factor}:sparse_c2")
    return terms


def _model_row(levels: dict[str, str]) -> list[Fraction]:
    binary = {factor: _binary_code(levels, factor) for factor in BINARY_FACTORS}
    sparse_c1, sparse_c2 = _sparse_contrasts(levels)
    row = [Fraction(1)]
    row.extend(Fraction(binary[factor]) for factor in BINARY_FACTORS)
    row.extend((Fraction(sparse_c1), Fraction(sparse_c2)))
    for left, right in itertools.combinations(BINARY_FACTORS, 2):
        row.append(Fraction(binary[left] * binary[right]))
    for factor in BINARY_FACTORS:
        row.append(Fraction(binary[factor] * sparse_c1))
        row.append(Fraction(binary[factor] * sparse_c2))
    return row


def _rank(rows: list[list[Fraction]]) -> int:
    if not rows:
        return 0
    matrix = [list(row) for row in rows]
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    rank = 0
    for col in range(n_cols):
        pivot = None
        for row_idx in range(rank, n_rows):
            if matrix[row_idx][col] != 0:
                pivot = row_idx
                break
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][col]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        for row_idx in range(n_rows):
            if row_idx == rank or matrix[row_idx][col] == 0:
                continue
            factor = matrix[row_idx][col]
            matrix[row_idx] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(matrix[row_idx], matrix[rank])
            ]
        rank += 1
        if rank == n_cols:
            break
    return rank


def _balanced_subset(candidate: list[dict[str, str]]) -> bool:
    for factor in BINARY_FACTORS:
        if sum(1 for row in candidate if _binary_code(row, factor) == 1) != 12:
            return False
    for sparse in SPARSE_LEVELS:
        if sum(1 for row in candidate if row["sparse"] == sparse) != 8:
            return False
    return True


def _all_factorial_rows() -> list[dict[str, str]]:
    rows = []
    for mhc_on, nope, kda, sparse, mtp_on in itertools.product(
        (False, True),
        (False, True),
        (False, True),
        SPARSE_LEVELS,
        (False, True),
    ):
        rows.append(
            {
                "mhc": "on" if mhc_on else "off",
                "positions": "nope" if nope else "rope",
                "mixer": "kda_3_to_1" if kda else "softmax",
                "sparse": sparse,
                "mtp": "on" if mtp_on else "off",
                "moe": "off",
            }
        )
    return rows


def _four_c_design() -> FourCDesign:
    universe = _all_factorial_rows()
    terms = _model_terms()
    target_rank = len(terms)
    rng = random.Random(FRACTIONAL_SEED)
    best: tuple[int, list[dict[str, str]]] | None = None
    for attempt in range(1, 200_001):
        # Balance the three-level sparse factor exactly, then require exact
        # balance on every two-level factor. The fixed seed makes this a
        # reproducible nonregular fraction.
        candidate: list[dict[str, str]] = []
        for sparse in SPARSE_LEVELS:
            pool = [row for row in universe if row["sparse"] == sparse]
            candidate.extend(rng.sample(pool, FOUR_C_RUNS // len(SPARSE_LEVELS)))
        if not _balanced_subset(candidate):
            continue
        rank = _rank([_model_row(row) for row in candidate])
        if best is None or rank > best[0]:
            best = (rank, list(candidate))
        if rank == target_rank:
            max_abs_column_sum = max(
                abs(sum(int(value) for value in column))
                for column in zip(*_integer_model_rows(candidate))
            )
            return FourCDesign(
                rows=sorted(
                    candidate,
                    key=lambda row: (
                        row["sparse"],
                        row["mhc"],
                        row["positions"],
                        row["mixer"],
                        row["mtp"],
                    ),
                ),
                model_rank=rank,
                model_terms=terms,
                max_abs_column_sum=max_abs_column_sum,
                attempts=attempt,
            )
    assert best is not None
    raise RuntimeError(
        "could not find a full-rank 4c fraction; "
        f"best rank was {best[0]} of {target_rank}."
    )


def _integer_model_rows(rows: Iterable[dict[str, str]]) -> list[list[int]]:
    return [[int(value) for value in _model_row(row)] for row in rows]


def _slice_4c() -> tuple[list[dict[str, object]], FourCDesign]:
    design = _four_c_design()
    cells = []
    for index, levels in enumerate(design.rows, start=1):
        arm = f"ff{index:02d}"
        for seed in SEED_PAIR:
            cells.append(
                _cell(
                    cell_id=f"4c_s{SCIENCE_SEQ_LEN}_{arm}_seed{seed}",
                    slice_name="4c",
                    arm=arm,
                    levels=levels,
                    seed=seed,
                    seq_len=SCIENCE_SEQ_LEN,
                    notes=["4c nonregular fractional factorial row"],
                )
            )
    return cells, design


def _deferred_4d_templates() -> list[dict[str, object]]:
    templates = []
    sweeps = {
        "mhc_streams": [2, 4, 8],
        "mhc_sinkhorn_iters": [5, 20],
        "kda_softmax_ratio": ["1:1", "3:1", "7:1"],
        "sparse_top_k_fraction": ["1/16", "1/8", "1/4"],
        "index_pool_size": [2, 4, 8],
        "nope_scope": ["all_layers", "softmax_sparse_layers_only"],
        "mtp_aux_loss_weight": [0.1, 0.3],
    }
    for knob, values in sweeps.items():
        for value in values:
            for seed in SEED_PAIR:
                levels = dict(B0)
                templates.append(
                    _cell(
                        cell_id=f"4d_pending_{knob}_{str(value).replace('/', '_')}_seed{seed}",
                        slice_name="4d",
                        arm=f"dose_{knob}",
                        levels=levels,
                        seed=seed,
                        seq_len=SCIENCE_SEQ_LEN,
                        runnable=False,
                        context_rule=(
                            "Resolve after 4a/4b: choose the lower mean validation "
                            "CE between the add-one context and the leave-one-out "
                            "context for this element before running any 4d cell."
                        ),
                        notes=[
                            f"deferred 4d template; set dose knob {knob}={value}",
                            "not runnable until queue/frozen_4d_contexts.json is written",
                        ],
                    )
                )
                templates[-1]["deferred_dose_override"] = {knob: value}
    return templates


def _deferred_c1_templates() -> list[dict[str, object]]:
    templates = []
    for seed in SEED_TRIPLE:
        templates.append(
            _cell(
                cell_id=f"c1_pending_combined_winners_seed{seed}",
                slice_name="c1",
                arm="combined_winners",
                levels=dict(B0),
                seed=seed,
                seq_len=SCIENCE_SEQ_LEN,
                runnable=False,
                context_rule=(
                    "Resolve after 4a-4d: include every element whose mean "
                    "validation CE improves over its declared comparison by more "
                    "than 2x the B0 seed standard deviation."
                ),
                notes=[
                    "deferred C1 template; not runnable until winners are frozen",
                ],
            )
        )
    return templates


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_design_doc(
    *,
    runnable_cells: list[dict[str, object]],
    deferred_4d: list[dict[str, object]],
    deferred_c1: list[dict[str, object]],
    design: FourCDesign,
) -> None:
    slice_counts: dict[str, int] = {}
    for cell in runnable_cells:
        slice_counts[cell["slice"]] = slice_counts.get(cell["slice"], 0) + 1
    lines = [
        "# GLM-5.3 Architecture Speedrun Queue Design",
        "",
        "Generated by `experiments/glm53_arch_speedrun/queue/generate_cells.py`.",
        "All Python generation and analysis is intended to run inside the `glm53_train` rootfs.",
        "",
        "## Runnable Queue",
        "",
        f"- Token budget per run: `{TOKEN_BUDGET}` tokens.",
        "- Science sequence length: `4096`; 4a also has the required `1024` replicate.",
        "- Seeds: B0 and G use seeds `0,1,2`; all other runnable cells use seeds `0,1`.",
        "- Each run requests one GPU, with the scheduler enforcing a 24 GiB "
        "process cap through the runner contract and requiring at least 30 GiB "
        "free before launch.",
        "",
        "| Slice | Runnable runs | Notes |",
        "| --- | ---: | --- |",
    ]
    for slice_name in ("4a", "4b", "4c"):
        if slice_name in slice_counts:
            note = {
                "4a": "add-one at seq 4096 plus seq-1024 replicate",
                "4b": "G plus leave-one-out contexts",
                "4c": "24-row nonregular fraction, two seeds per row",
            }[slice_name]
            lines.append(f"| {slice_name} | {slice_counts[slice_name]} | {note} |")
    lines.extend(
        [
            f"| 4d | 0 runnable / {len(deferred_4d)} templates | frozen after 4a/4b by the declared context rule |",
            f"| C1 | 0 runnable / {len(deferred_c1)} templates | frozen after 4a-4d winner rule |",
            "",
            "## 4c Fractional Factorial",
            "",
            "The 4c generator is deterministic and nonregular:",
            "",
            "1. Enumerate the full `2^4 x 3` design over F1 mHC, F2 positions, F3 mixer, F4 sparse, and F5 MTP.",
            f"2. With random seed `{FRACTIONAL_SEED}`, sample exactly 8 rows from each F4 sparse level.",
            "3. Keep only samples balanced 12/12 on each two-level factor.",
            "4. Accept the first candidate whose model matrix has full rank for "
            "the registered model: intercept, all main effects, all "
            "binary-binary two-way interactions, and all binary-by-sparse "
            "two-way interactions.",
            "",
            "Declared resolution rule for this mixed-level nonregular fraction: "
            "the registered main-effect and two-way-interaction model must have "
            "no exact aliases among its columns. Under the effect coding below, "
            "the accepted design has:",
            "",
            f"- model rank `{design.model_rank}/{len(design.model_terms)}`;",
            "- operational resolution V for the preregistered analysis: no exact "
            "alias among the registered main and two-way terms;",
            "- higher-order effects intentionally pooled into residual error, so "
            "effects are interpreted through the preregistered model only.",
            "",
            "Effect coding:",
            "",
            "- F1 mHC: off=-1, on=+1.",
            "- F2 positions: RoPE=-1, NoPE=+1.",
            "- F3 mixer: softmax=-1, KDA 3:1=+1.",
            "- F5 MTP: off=-1, on=+1.",
            "- F4 sparse uses two contrasts: dense=(+1,0), indexer=(-1,+1), indexer+IndexPool=(0,-1).",
            "",
            "Registered model terms:",
            "",
            "```text",
            ", ".join(design.model_terms),
            "```",
            "",
            "Accepted 4c design rows:",
            "",
            "| Row | F1 mHC | F2 positions | F3 mixer | F4 sparse | F5 MTP |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for index, row in enumerate(design.rows, start=1):
        lines.append(
            f"| {index} | {row['mhc']} | {row['positions']} | {row['mixer']} | {row['sparse']} | {row['mtp']} |"
        )
    lines.extend(
        [
            "",
            "## Deferred Freeze Rules",
            "",
            "4d is not runnable at queue-generation time because the spec "
            "requires each dose sweep to run in the best context observed in "
            "4a/4b. The analysis step must write `queue/frozen_4d_contexts.json` "
            "before any 4d state is initialized.",
            "",
            "C1 is not runnable at queue-generation time because winners must be "
            "frozen after the scalar and behavior reads are available. The rule "
            "is predeclared: include every element whose mean validation CE "
            "improves over its declared comparison by more than "
            "`2 * std(B0 seed CE)` at the matched token budget.",
            "",
            "MoE/F6 cells are not emitted by default. F6 is optional in the spec "
            "and requires a matched-active-parameter implementation and green "
            "S02 gate before it can be added with a separate generator revision.",
            "",
        ]
    )
    (QUEUE_DIR / "DESIGN.md").write_text("\n".join(lines), encoding="utf-8")


def _write_alias_json(design: FourCDesign) -> None:
    payload = {
        "fractional_seed": FRACTIONAL_SEED,
        "rows": design.rows,
        "effect_coding": {
            "mhc": {"off": -1, "on": 1},
            "positions": {"rope": -1, "nope": 1},
            "mixer": {"softmax": -1, "kda_3_to_1": 1},
            "mtp": {"off": -1, "on": 1},
            "sparse": {
                "dense": [1, 0],
                "indexer_topk": [-1, 1],
                "indexer_topk_indexpool": [0, -1],
            },
        },
        "registered_terms": design.model_terms,
        "model_rank": design.model_rank,
        "model_columns": len(design.model_terms),
        "attempts": design.attempts,
        "alias_structure": {
            "registered_main_and_two_way_exact_aliases": [],
            "interpretation": (
                "No exact aliases inside the registered main+two-way model; "
                "higher-order effects are pooled into residual error."
            ),
        },
    }
    (QUEUE_DIR / "alias_structure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    slice_4c, design = _slice_4c()
    runnable_cells = [*_slice_4a(), *_slice_4b(), *slice_4c]
    runnable_cells.sort(
        key=lambda cell: (
            cell["slice_order"],
            0 if cell["seq_len"] == SCIENCE_SEQ_LEN else 1,
            ARM_ORDER.get(str(cell["arm"]), 1000),
            cell["cell_id"],
        )
    )
    deferred_4d = _deferred_4d_templates()
    deferred_c1 = _deferred_c1_templates()

    _write_jsonl(QUEUE_DIR / "cells.jsonl", runnable_cells)
    _write_jsonl(QUEUE_DIR / "deferred_4d_templates.jsonl", deferred_4d)
    _write_jsonl(QUEUE_DIR / "deferred_c1_templates.jsonl", deferred_c1)
    _write_design_doc(
        runnable_cells=runnable_cells,
        deferred_4d=deferred_4d,
        deferred_c1=deferred_c1,
        design=design,
    )
    _write_alias_json(design)

    summary = {
        "runnable_cells": len(runnable_cells),
        "deferred_4d_templates": len(deferred_4d),
        "deferred_c1_templates": len(deferred_c1),
        "slice_counts": {
            slice_name: sum(1 for cell in runnable_cells if cell["slice"] == slice_name)
            for slice_name in ("4a", "4b", "4c")
        },
        "four_c_model_rank": design.model_rank,
        "four_c_model_columns": len(design.model_terms),
        "four_c_generator_attempts": design.attempts,
    }
    (QUEUE_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
