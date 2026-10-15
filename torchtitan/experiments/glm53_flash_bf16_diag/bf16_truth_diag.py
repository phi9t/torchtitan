# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare GLM-5.3-Flash BF16 slice paths against an FP32 slice truth."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.experiments.glm53_flash import hf_parity


DEFAULT_OUTPUT = Path(
    ".scratch/glm53-flash-sft-ladder/evidence/task-08c/bf16-truth-compare.json"
)
DEFAULT_TRUTH_TENSORS = Path(
    ".scratch/glm53-flash-sft-ladder/evidence/task-08c/fp32-truth-tensors.pt"
)


def _current_rss_bytes() -> int:
    with Path("/proc/self/status").open() as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _memory_snapshot(label: str) -> dict[str, Any]:
    hwm_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "label": label,
        "time_unix": time.time(),
        "rss_gib": _current_rss_bytes() / (1024**3),
        "maxrss_gib": (hwm_kb * 1024) / (1024**3),
    }


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _configure_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def _load_slice_state(
    *,
    checkpoint: Path,
    index_json: Path | None,
    layer_start: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    index = hf_parity._read_index(checkpoint, index_json)
    layers = range(layer_start, layer_start + num_layers)
    keys = hf_parity._slice_hf_keys(index["weight_map"], layers)
    return hf_parity._load_safetensors_subset(
        checkpoint,
        index["weight_map"],
        keys,
        device=str(device),
        dtype=dtype,
    )


def _build_tt_slice(
    *,
    checkpoint: Path,
    index_json: Path | None,
    layer_start: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
):
    hf_state = _load_slice_state(
        checkpoint=checkpoint,
        index_json=index_json,
        layer_start=layer_start,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )
    model = hf_parity._build_tt_model(
        hf_state,
        layer_start=layer_start,
        num_layers=num_layers,
        dtype=dtype,
    ).to(device)
    del hf_state
    _cleanup_cuda()
    return model


def _build_hf_slice(
    *,
    checkpoint: Path,
    index_json: Path | None,
    layer_start: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
):
    hf_state = _load_slice_state(
        checkpoint=checkpoint,
        index_json=index_json,
        layer_start=layer_start,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )
    model, lm_head = hf_parity._build_hf_text_model(
        checkpoint,
        hf_state,
        layer_start=layer_start,
        num_layers=num_layers,
        dtype=dtype,
    )
    model = model.to(device)
    lm_head = lm_head.to(device)
    del hf_state
    _cleanup_cuda()
    return model, lm_head


def _clone_sample_to_device(
    sample: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return sample["input_ids"].to(device), sample["attention_mask"].to(device)


@torch.no_grad()
def _run_tt_sample(
    model, sample: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    input_ids, attention_mask = _clone_sample_to_device(sample, device)
    hidden_BLD = model.tok_embeddings(input_ids)
    streams_BLND = (
        hidden_BLD.unsqueeze(2)
        .expand(-1, -1, model.glm_config.hc_mult, -1)
        .contiguous()
    )
    streams_BLND, traces = hf_parity._tt_step(model, streams_BLND, attention_mask)
    final_hidden_BLD = model.norm(streams_BLND.mean(dim=2))
    valid_flat_T = attention_mask.reshape(-1)
    logits_TV = model.lm_head(final_hidden_BLD.reshape(-1, final_hidden_BLD.shape[-1]))[
        valid_flat_T
    ]
    return {
        "attention_mask": attention_mask.detach().cpu(),
        "hidden": [trace["hidden"].detach().cpu().float() for trace in traces],
        "layer_kinds": [trace["kind"] for trace in traces],
        "logits": logits_TV.detach().cpu().float(),
    }


@torch.no_grad()
def _run_hf_sample(
    model,
    lm_head,
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    input_ids, attention_mask = _clone_sample_to_device(sample, device)
    hidden_BLD = model.embed_tokens(input_ids)
    streams_BLND = (
        hidden_BLD.unsqueeze(2).expand(-1, -1, model.config.hc_mult, -1).contiguous()
    )
    streams_BLND, traces = hf_parity._hf_step(
        model, streams_BLND, input_ids, attention_mask
    )
    final_hidden_BLD = model.norm(model.hc_head(streams_BLND))
    valid_flat_T = attention_mask.reshape(-1)
    logits_TV = lm_head(final_hidden_BLD.reshape(-1, final_hidden_BLD.shape[-1]))[
        valid_flat_T
    ]
    return {
        "attention_mask": attention_mask.detach().cpu(),
        "hidden": [trace["hidden"].detach().cpu().float() for trace in traces],
        "layer_kinds": [trace["kind"] for trace in traces],
        "logits": logits_TV.detach().cpu().float(),
    }


@torch.no_grad()
def _run_hf_one_layer_sample(
    model,
    sample: dict[str, Any],
    streams_BLND: torch.Tensor | None,
    prev_topk: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None]:
    input_ids, attention_mask = _clone_sample_to_device(sample, device)
    if streams_BLND is None:
        hidden_BLD = model.embed_tokens(input_ids)
        streams_BLND = (
            hidden_BLD.unsqueeze(2)
            .expand(-1, -1, model.config.hc_mult, -1)
            .contiguous()
        )
    else:
        streams_BLND = streams_BLND.to(device)
    causal_masks = {
        "indexed_attention": attention_mask,
        "linear_attention": attention_mask,
    }
    position_ids = torch.arange(streams_BLND.shape[1], device=device).unsqueeze(0)
    layer = model.layers[0]
    capture: dict[str, torch.Tensor] = {}
    hook = None
    if getattr(layer.self_attn, "indexer", None) is not None:
        hook = layer.self_attn.indexer.register_forward_hook(
            lambda _module, _args, output, capture=capture: capture.setdefault(
                "topk", output.detach().cpu()
            )
        )
    streams_BLND, prev_topk = layer(
        streams_BLND,
        attention_mask=causal_masks[layer.block_type],
        position_ids=position_ids,
        position_embeddings=None,
        input_ids=input_ids,
        past_key_values=None,
        prev_topk_indices=prev_topk.to(device) if prev_topk is not None else None,
    )
    if hook is not None:
        hook.remove()
    trace = {
        "kind": layer.block_type,
        "hidden": streams_BLND.detach().cpu(),
        "topk": capture.get("topk"),
    }
    return (
        streams_BLND.detach().cpu(),
        trace,
        (prev_topk.detach().cpu() if prev_topk is not None else None),
    )


@torch.no_grad()
def _run_tt_one_layer_sample(
    model,
    sample: dict[str, Any],
    streams_BLND: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    input_ids, attention_mask = _clone_sample_to_device(sample, device)
    if streams_BLND is None:
        hidden_BLD = model.tok_embeddings(input_ids)
        streams_BLND = (
            hidden_BLD.unsqueeze(2)
            .expand(-1, -1, model.glm_config.hc_mult, -1)
            .contiguous()
        )
    else:
        streams_BLND = streams_BLND.to(device)
    streams_BLND, traces = hf_parity._tt_step(model, streams_BLND, attention_mask)
    trace = traces[0]
    return streams_BLND.detach().cpu(), {
        "kind": trace["kind"],
        "hidden": trace["hidden"].detach().cpu(),
        "topk": trace["topk"],
    }


def _kl_truth_to_candidate(
    candidate_logits_TV: torch.Tensor,
    truth_logits_TV: torch.Tensor,
) -> float:
    candidate_log_probs_TV = F.log_softmax(candidate_logits_TV.float(), dim=-1)
    truth_probs_TV = F.softmax(truth_logits_TV.float(), dim=-1)
    return float(
        F.kl_div(
            candidate_log_probs_TV,
            truth_probs_TV,
            reduction="batchmean",
            log_target=False,
        ).item()
    )


def _compare_tensor(
    candidate: torch.Tensor,
    truth: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    return hf_parity._max_diff_stats(candidate.float(), truth.float(), mask=mask)


def _compare_outputs(
    *,
    candidate: dict[str, Any],
    truth: dict[str, Any],
    layer_start: int,
) -> dict[str, Any]:
    mask = truth["attention_mask"]
    layer_stats = []
    for local_layer, (candidate_hidden, truth_hidden) in enumerate(
        zip(candidate["hidden"], truth["hidden"])
    ):
        stats = _compare_tensor(candidate_hidden, truth_hidden, mask=mask)
        stats["layer"] = layer_start + local_layer
        stats["candidate_kind"] = candidate["layer_kinds"][local_layer]
        stats["truth_kind"] = truth["layer_kinds"][local_layer]
        layer_stats.append(stats)

    candidate_logits = candidate["logits"].float()
    truth_logits = truth["logits"].float()
    top1 = (candidate_logits.argmax(dim=-1) == truth_logits.argmax(dim=-1)).float()
    final_stats = {
        **_compare_tensor(candidate_logits, truth_logits),
        "top1_agreement": float(top1.mean().item()),
        "kl_truth_to_candidate": _kl_truth_to_candidate(candidate_logits, truth_logits),
    }
    return {
        "layer_stats": layer_stats,
        "final_logits": final_stats,
    }


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    layer_max_abs = 0.0
    layer_mean_abs = 0.0
    final_max_abs = 0.0
    min_top1 = 1.0
    max_kl = 0.0
    for sample in samples:
        if sample["layer_stats"]:
            layer_max_abs = max(
                layer_max_abs,
                max(stat["max_abs"] for stat in sample["layer_stats"]),
            )
            layer_mean_abs = max(
                layer_mean_abs,
                max(stat["mean_abs"] for stat in sample["layer_stats"]),
            )
        final = sample["final_logits"]
        final_max_abs = max(final_max_abs, final["max_abs"])
        min_top1 = min(min_top1, final["top1_agreement"])
        max_kl = max(max_kl, final["kl_truth_to_candidate"])
    return {
        "max_layer_abs": layer_max_abs,
        "max_layer_mean_abs": layer_mean_abs,
        "max_final_abs": final_max_abs,
        "min_top1_agreement": min_top1,
        "max_kl_truth_to_candidate": max_kl,
    }


def _compare_stream_hidden(
    *,
    candidate_hidden: torch.Tensor,
    truth_hidden: torch.Tensor,
    mask: torch.Tensor,
    layer: int,
    candidate_kind: str,
    truth_kind: str,
) -> dict[str, Any]:
    stats = _compare_tensor(candidate_hidden, truth_hidden, mask=mask)
    stats["layer"] = layer
    stats["candidate_kind"] = candidate_kind
    stats["truth_kind"] = truth_kind
    return stats


def _finalize_hf_logits(
    model,
    lm_head,
    streams_by_sample: dict[str, torch.Tensor],
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    logits_by_sample = {}
    for sample in samples:
        streams_BLND = streams_by_sample[sample["name"]].to(device)
        _, attention_mask = _clone_sample_to_device(sample, device)
        final_hidden_BLD = model.norm(model.hc_head(streams_BLND))
        valid_flat_T = attention_mask.reshape(-1)
        logits_TV = lm_head(final_hidden_BLD.reshape(-1, final_hidden_BLD.shape[-1]))[
            valid_flat_T
        ]
        logits_by_sample[sample["name"]] = logits_TV.detach().cpu().float()
    return logits_by_sample


def _finalize_tt_logits(
    model,
    streams_by_sample: dict[str, torch.Tensor],
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    logits_by_sample = {}
    for sample in samples:
        streams_BLND = streams_by_sample[sample["name"]].to(device)
        _, attention_mask = _clone_sample_to_device(sample, device)
        final_hidden_BLD = model.norm(streams_BLND.mean(dim=2))
        valid_flat_T = attention_mask.reshape(-1)
        logits_TV = model.lm_head(
            final_hidden_BLD.reshape(-1, final_hidden_BLD.shape[-1])
        )[valid_flat_T]
        logits_by_sample[sample["name"]] = logits_TV.detach().cpu().float()
    return logits_by_sample


@torch.no_grad()
def _stream_hf_truth(
    *,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    device: torch.device,
    memory: list[dict[str, Any]],
) -> dict[str, Any]:
    streams_by_sample: dict[str, torch.Tensor] = {
        sample["name"]: None for sample in samples
    }
    prev_topk_by_sample: dict[str, torch.Tensor | None] = {
        sample["name"]: None for sample in samples
    }
    layer_hidden: dict[int, dict[str, torch.Tensor]] = {}
    layer_kinds: dict[int, str] = {}
    last_model = None
    last_lm_head = None
    last_layer = args.layer_start + args.num_layers - 1
    for real_layer in range(args.layer_start, args.layer_start + args.num_layers):
        memory.append(_memory_snapshot(f"before_hf_fp32_layer_{real_layer}_build"))
        model, lm_head = _build_hf_slice(
            checkpoint=Path(args.checkpoint),
            index_json=args.index_json,
            layer_start=real_layer,
            num_layers=1,
            device=device,
            dtype=torch.float32,
        )
        memory.append(_memory_snapshot(f"after_hf_fp32_layer_{real_layer}_build"))
        sample_hidden = {}
        for sample in samples:
            streams_BLND, trace, prev_topk = _run_hf_one_layer_sample(
                model,
                sample,
                streams_by_sample[sample["name"]],
                prev_topk_by_sample[sample["name"]],
                device,
            )
            streams_by_sample[sample["name"]] = streams_BLND
            prev_topk_by_sample[sample["name"]] = prev_topk
            sample_hidden[sample["name"]] = streams_BLND.float()
            layer_kinds[real_layer] = trace["kind"]
        layer_hidden[real_layer] = sample_hidden
        memory.append(_memory_snapshot(f"after_hf_fp32_layer_{real_layer}_forward"))
        if real_layer == last_layer:
            last_model = model
            last_lm_head = lm_head
        else:
            del model, lm_head
            _cleanup_cuda()

    assert last_model is not None and last_lm_head is not None
    logits = _finalize_hf_logits(
        last_model,
        last_lm_head,
        streams_by_sample,
        samples,
        device,
    )
    memory.append(_memory_snapshot("after_hf_fp32_stream_final_logits"))
    del last_model, last_lm_head
    _cleanup_cuda()
    memory.append(_memory_snapshot("after_hf_fp32_stream_cleanup"))
    return {
        "layer_hidden": layer_hidden,
        "layer_kinds": layer_kinds,
        "logits": logits,
    }


@torch.no_grad()
def _stream_hf_candidate(
    *,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    truth: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    label: str,
    memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    streams_by_sample: dict[str, torch.Tensor | None] = {
        sample["name"]: None for sample in samples
    }
    prev_topk_by_sample: dict[str, torch.Tensor | None] = {
        sample["name"]: None for sample in samples
    }
    results_by_sample = {
        sample["name"]: {
            "name": sample["name"],
            "seq_len": int(sample["input_ids"].shape[1]),
            "layer_stats": [],
        }
        for sample in samples
    }
    last_model = None
    last_lm_head = None
    last_layer = args.layer_start + args.num_layers - 1
    for real_layer in range(args.layer_start, args.layer_start + args.num_layers):
        memory.append(_memory_snapshot(f"before_{label}_layer_{real_layer}_build"))
        model, lm_head = _build_hf_slice(
            checkpoint=Path(args.checkpoint),
            index_json=args.index_json,
            layer_start=real_layer,
            num_layers=1,
            device=device,
            dtype=dtype,
        )
        memory.append(_memory_snapshot(f"after_{label}_layer_{real_layer}_build"))
        for sample in samples:
            streams_BLND, trace, prev_topk = _run_hf_one_layer_sample(
                model,
                sample,
                streams_by_sample[sample["name"]],
                prev_topk_by_sample[sample["name"]],
                device,
            )
            streams_by_sample[sample["name"]] = streams_BLND
            prev_topk_by_sample[sample["name"]] = prev_topk
            results_by_sample[sample["name"]]["layer_stats"].append(
                _compare_stream_hidden(
                    candidate_hidden=streams_BLND,
                    truth_hidden=truth["layer_hidden"][real_layer][sample["name"]],
                    mask=sample["attention_mask"],
                    layer=real_layer,
                    candidate_kind=trace["kind"],
                    truth_kind=truth["layer_kinds"][real_layer],
                )
            )
        memory.append(_memory_snapshot(f"after_{label}_layer_{real_layer}_forward"))
        if real_layer == last_layer:
            last_model = model
            last_lm_head = lm_head
        else:
            del model, lm_head
            _cleanup_cuda()

    assert last_model is not None and last_lm_head is not None
    candidate_logits = _finalize_hf_logits(
        last_model,
        last_lm_head,
        streams_by_sample,
        samples,
        device,
    )
    for sample in samples:
        compared = _compare_tensor(
            candidate_logits[sample["name"]],
            truth["logits"][sample["name"]],
        )
        candidate_final = candidate_logits[sample["name"]].float()
        truth_final = truth["logits"][sample["name"]].float()
        top1 = (candidate_final.argmax(dim=-1) == truth_final.argmax(dim=-1)).float()
        compared["top1_agreement"] = float(top1.mean().item())
        compared["kl_truth_to_candidate"] = _kl_truth_to_candidate(
            candidate_final,
            truth_final,
        )
        results_by_sample[sample["name"]]["final_logits"] = compared
    memory.append(_memory_snapshot(f"after_{label}_final_logits"))
    del last_model, last_lm_head
    _cleanup_cuda()
    memory.append(_memory_snapshot(f"after_{label}_cleanup"))
    return [results_by_sample[sample["name"]] for sample in samples]


@torch.no_grad()
def _stream_tt_candidate(
    *,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    truth: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    label: str,
    memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    streams_by_sample: dict[str, torch.Tensor | None] = {
        sample["name"]: None for sample in samples
    }
    results_by_sample = {
        sample["name"]: {
            "name": sample["name"],
            "seq_len": int(sample["input_ids"].shape[1]),
            "layer_stats": [],
        }
        for sample in samples
    }
    last_model = None
    last_layer = args.layer_start + args.num_layers - 1
    for real_layer in range(args.layer_start, args.layer_start + args.num_layers):
        memory.append(_memory_snapshot(f"before_{label}_layer_{real_layer}_build"))
        model = _build_tt_slice(
            checkpoint=Path(args.checkpoint),
            index_json=args.index_json,
            layer_start=real_layer,
            num_layers=1,
            device=device,
            dtype=dtype,
        )
        memory.append(_memory_snapshot(f"after_{label}_layer_{real_layer}_build"))
        for sample in samples:
            streams_BLND, trace = _run_tt_one_layer_sample(
                model,
                sample,
                streams_by_sample[sample["name"]],
                device,
            )
            streams_by_sample[sample["name"]] = streams_BLND
            results_by_sample[sample["name"]]["layer_stats"].append(
                _compare_stream_hidden(
                    candidate_hidden=streams_BLND,
                    truth_hidden=truth["layer_hidden"][real_layer][sample["name"]],
                    mask=sample["attention_mask"],
                    layer=real_layer,
                    candidate_kind=trace["kind"],
                    truth_kind=truth["layer_kinds"][real_layer],
                )
            )
        memory.append(_memory_snapshot(f"after_{label}_layer_{real_layer}_forward"))
        if real_layer == last_layer:
            last_model = model
        else:
            del model
            _cleanup_cuda()

    assert last_model is not None
    candidate_logits = _finalize_tt_logits(
        last_model,
        streams_by_sample,
        samples,
        device,
    )
    for sample in samples:
        compared = _compare_tensor(
            candidate_logits[sample["name"]],
            truth["logits"][sample["name"]],
        )
        candidate_final = candidate_logits[sample["name"]].float()
        truth_final = truth["logits"][sample["name"]].float()
        top1 = (candidate_final.argmax(dim=-1) == truth_final.argmax(dim=-1)).float()
        compared["top1_agreement"] = float(top1.mean().item())
        compared["kl_truth_to_candidate"] = _kl_truth_to_candidate(
            candidate_final,
            truth_final,
        )
        results_by_sample[sample["name"]]["final_logits"] = compared
    memory.append(_memory_snapshot(f"after_{label}_final_logits"))
    del last_model
    _cleanup_cuda()
    memory.append(_memory_snapshot(f"after_{label}_cleanup"))
    return [results_by_sample[sample["name"]] for sample in samples]


@torch.no_grad()
def _capture_truth(
    *,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    device: torch.device,
    memory: list[dict[str, Any]],
) -> dict[str, Any]:
    memory.append(_memory_snapshot("before_hf_fp32_truth_build"))
    model, lm_head = _build_hf_slice(
        checkpoint=Path(args.checkpoint),
        index_json=args.index_json,
        layer_start=args.layer_start,
        num_layers=args.num_layers,
        device=device,
        dtype=torch.float32,
    )
    memory.append(_memory_snapshot("after_hf_fp32_truth_build"))
    truth = {
        sample["name"]: _run_hf_sample(model, lm_head, sample, device)
        for sample in samples
    }
    memory.append(_memory_snapshot("after_hf_fp32_truth_forward"))
    del model, lm_head
    _cleanup_cuda()
    memory.append(_memory_snapshot("after_hf_fp32_truth_cleanup"))
    return truth


@torch.no_grad()
def _compare_tt_fp32(
    *,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    truth: dict[str, Any],
    device: torch.device,
    memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    memory.append(_memory_snapshot("before_tt_fp32_build"))
    model = _build_tt_slice(
        checkpoint=Path(args.checkpoint),
        index_json=args.index_json,
        layer_start=args.layer_start,
        num_layers=args.num_layers,
        device=device,
        dtype=torch.float32,
    )
    memory.append(_memory_snapshot("after_tt_fp32_build"))
    results = []
    for sample in samples:
        output = _run_tt_sample(model, sample, device)
        compared = _compare_outputs(
            candidate=output,
            truth=truth[sample["name"]],
            layer_start=args.layer_start,
        )
        compared["name"] = sample["name"]
        compared["seq_len"] = int(sample["input_ids"].shape[1])
        results.append(compared)
    memory.append(_memory_snapshot("after_tt_fp32_forward"))
    del model
    _cleanup_cuda()
    memory.append(_memory_snapshot("after_tt_fp32_cleanup"))
    return results


@torch.no_grad()
def _compare_candidate(
    *,
    args: argparse.Namespace,
    candidate_name: str,
    samples: list[dict[str, Any]],
    truth: dict[str, Any],
    device: torch.device,
    memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dtype = torch.bfloat16
    if candidate_name == "tt_bf16":
        memory.append(_memory_snapshot("before_tt_bf16_build"))
        model = _build_tt_slice(
            checkpoint=Path(args.checkpoint),
            index_json=args.index_json,
            layer_start=args.layer_start,
            num_layers=args.num_layers,
            device=device,
            dtype=dtype,
        )
        memory.append(_memory_snapshot("after_tt_bf16_build"))
        outputs = [_run_tt_sample(model, sample, device) for sample in samples]
        del model
    elif candidate_name == "hf_bf16":
        memory.append(_memory_snapshot("before_hf_bf16_build"))
        model, lm_head = _build_hf_slice(
            checkpoint=Path(args.checkpoint),
            index_json=args.index_json,
            layer_start=args.layer_start,
            num_layers=args.num_layers,
            device=device,
            dtype=dtype,
        )
        memory.append(_memory_snapshot("after_hf_bf16_build"))
        outputs = [_run_hf_sample(model, lm_head, sample, device) for sample in samples]
        del model, lm_head
    else:
        raise ValueError(f"unknown candidate {candidate_name!r}")

    results = []
    for sample, output in zip(samples, outputs):
        compared = _compare_outputs(
            candidate=output,
            truth=truth[sample["name"]],
            layer_start=args.layer_start,
        )
        compared["name"] = sample["name"]
        compared["seq_len"] = int(sample["input_ids"].shape[1])
        results.append(compared)
    memory.append(_memory_snapshot(f"after_{candidate_name}_forward"))
    _cleanup_cuda()
    memory.append(_memory_snapshot(f"after_{candidate_name}_cleanup"))
    return results


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    _configure_device(device)
    memory = [_memory_snapshot("start")]
    input_samples, rendered_prompts = hf_parity._build_parity_inputs(
        checkpoint,
        Path(args.data_file),
        long_seq_len=None if args.no_long_sequence else args.long_seq_len,
        device=torch.device("cpu"),
    )

    truth = _capture_truth(
        args=args,
        samples=input_samples,
        device=device,
        memory=memory,
    )
    if args.truth_tensors is not None:
        args.truth_tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save(truth, args.truth_tensors)

    tt_fp32_samples = _compare_tt_fp32(
        args=args,
        samples=input_samples,
        truth=truth,
        device=device,
        memory=memory,
    )
    candidate_results = {
        "tt_fp32_vs_hf_fp32_truth": {
            "samples": tt_fp32_samples,
            "summary": _summarize_samples(tt_fp32_samples),
        }
    }

    for candidate_name in ("tt_bf16", "hf_bf16"):
        samples = _compare_candidate(
            args=args,
            candidate_name=candidate_name,
            samples=input_samples,
            truth=truth,
            device=device,
            memory=memory,
        )
        candidate_results[f"{candidate_name}_vs_hf_fp32_truth"] = {
            "samples": samples,
            "summary": _summarize_samples(samples),
        }

    memory.append(_memory_snapshot("end"))
    return {
        "claim_label": "deterministic_comparison",
        "dtype_truth": "float32",
        "truth_implementation": "hf_fp32",
        "device": args.device,
        "checkpoint": str(checkpoint),
        "data_file": str(args.data_file),
        "layer_start": args.layer_start,
        "num_layers": args.num_layers,
        "long_seq_len": None if args.no_long_sequence else args.long_seq_len,
        "rendered_prompts": rendered_prompts,
        "comparisons": candidate_results,
        "memory": memory,
    }


def run_stream_compare(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    _configure_device(device)
    memory = [_memory_snapshot("start")]
    input_samples, rendered_prompts = hf_parity._build_parity_inputs(
        checkpoint,
        Path(args.data_file),
        long_seq_len=None if args.no_long_sequence else args.long_seq_len,
        device=torch.device("cpu"),
    )
    truth = _stream_hf_truth(
        args=args,
        samples=input_samples,
        device=device,
        memory=memory,
    )
    if args.truth_tensors is not None:
        args.truth_tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save(truth, args.truth_tensors)

    comparisons = {}
    tt_fp32_samples = _stream_tt_candidate(
        args=args,
        samples=input_samples,
        truth=truth,
        device=device,
        dtype=torch.float32,
        label="tt_fp32",
        memory=memory,
    )
    comparisons["tt_fp32_vs_hf_fp32_truth"] = {
        "samples": tt_fp32_samples,
        "summary": _summarize_samples(tt_fp32_samples),
    }
    tt_bf16_samples = _stream_tt_candidate(
        args=args,
        samples=input_samples,
        truth=truth,
        device=device,
        dtype=torch.bfloat16,
        label="tt_bf16",
        memory=memory,
    )
    comparisons["tt_bf16_vs_hf_fp32_truth"] = {
        "samples": tt_bf16_samples,
        "summary": _summarize_samples(tt_bf16_samples),
    }
    hf_bf16_samples = _stream_hf_candidate(
        args=args,
        samples=input_samples,
        truth=truth,
        device=device,
        dtype=torch.bfloat16,
        label="hf_bf16",
        memory=memory,
    )
    comparisons["hf_bf16_vs_hf_fp32_truth"] = {
        "samples": hf_bf16_samples,
        "summary": _summarize_samples(hf_bf16_samples),
    }
    memory.append(_memory_snapshot("end"))
    return {
        "claim_label": "deterministic_comparison",
        "mode": "stream_one_layer_at_a_time",
        "dtype_truth": "float32",
        "truth_implementation": "hf_fp32",
        "device": args.device,
        "checkpoint": str(checkpoint),
        "data_file": str(args.data_file),
        "layer_start": args.layer_start,
        "num_layers": args.num_layers,
        "long_seq_len": None if args.no_long_sequence else args.long_seq_len,
        "rendered_prompts": rendered_prompts,
        "comparisons": comparisons,
        "memory": memory,
    }


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=hf_parity.DEFAULT_CHECKPOINT)
    parser.add_argument("--data-file", type=Path, default=hf_parity.DEFAULT_DATA_FILE)
    parser.add_argument("--index-json", type=Path, default=None)
    parser.add_argument("--layer-start", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--long-seq-len", type=int, default=4096)
    parser.add_argument("--no-long-sequence", action="store_true")
    parser.add_argument("--device", default="cpu")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare")
    _add_common_args(compare)
    compare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    compare.add_argument("--truth-tensors", type=Path, default=DEFAULT_TRUTH_TENSORS)
    compare.set_defaults(func=run_compare)

    stream = sub.add_parser("stream-compare")
    _add_common_args(stream)
    stream.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    stream.add_argument("--truth-tensors", type=Path, default=DEFAULT_TRUTH_TENSORS)
    stream.set_defaults(func=run_stream_compare)

    args = parser.parse_args()
    result = args.func(args)
    if getattr(args, "output", None) is not None:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
