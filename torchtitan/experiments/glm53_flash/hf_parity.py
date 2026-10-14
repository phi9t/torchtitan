# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real-weight parity utilities for the GLM-5.3-Flash experiment."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.glm53_flash.config_registry import (
    _countdown_sft_process_sample,
)
from torchtitan.experiments.glm53_flash.flavors import glm53_flash
from torchtitan.experiments.glm53_flash.model import (
    Glm53FlashForCausalLM,
    GlmSparseMLAAttention,
)
from torchtitan.experiments.glm53_flash.state_dict_adapter import (
    _EXPERT_RE,
    Glm53FlashStateDictAdapter,
)


DEFAULT_CHECKPOINT = Path(
    "/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16"
)
DEFAULT_DATA_FILE = Path(
    "/project/formal-cache/datasets/countdown-sft/v1/train/formatting.jsonl"
)
TRANSFORMERS_COMMIT = "e0eb2810ac1b2caf6d082f247d8d37162ecf67a0"


def _read_index(checkpoint: Path, index_json: Path | None = None) -> dict[str, Any]:
    path = index_json or checkpoint / "model.safetensors.index.json"
    with path.open() as f:
        return json.load(f)


def _root_keys() -> tuple[str, ...]:
    return (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    )


def _slice_hf_keys(weight_map: dict[str, str], layers: range) -> set[str]:
    selected = set(_root_keys())
    layer_prefixes = tuple(f"model.language_model.layers.{idx}." for idx in layers)
    for key in weight_map:
        if key.startswith(layer_prefixes):
            selected.add(key)
    return selected


def _load_safetensors_subset(
    checkpoint: Path,
    weight_map: dict[str, str],
    keys: set[str],
    *,
    device: str,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in sorted(keys):
        shard = weight_map.get(key)
        if shard is not None:
            by_shard[shard].append(key)

    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in sorted(by_shard.items()):
        path = checkpoint / shard
        if not path.exists():
            raise FileNotFoundError(f"missing shard {path}")
        with safe_open(path, framework="pt", device=device) as sf:
            for key in shard_keys:
                tensor = sf.get_tensor(key)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                out[key] = tensor
    return out


def _remap_hf_layers_to_local(
    hf_state: dict[str, torch.Tensor],
    *,
    layer_start: int,
    num_layers: int,
) -> dict[str, torch.Tensor]:
    layer_end = layer_start + num_layers
    remapped: dict[str, torch.Tensor] = {}
    for key, value in hf_state.items():
        prefix = "model.language_model.layers."
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            layer_text, suffix = rest.split(".", 1)
            real_layer = int(layer_text)
            if layer_start <= real_layer < layer_end:
                local_layer = real_layer - layer_start
                key = f"{prefix}{local_layer}.{suffix}"
        remapped[key] = value
    return remapped


def _build_tt_model_config(
    layer_start: int, num_layers: int
) -> Glm53FlashForCausalLM.Config:
    cfg = replace(
        glm53_flash(),
        num_hidden_layers=num_layers,
        layer_offset=layer_start,
        attn_impl="reference",
    )
    cfg.validate()
    return Glm53FlashForCausalLM.Config(
        glm_config=cfg,
        experiment_flavor=f"glm53_flash_layers_{layer_start}_{layer_start + num_layers}",
    )


def _build_tt_model(
    hf_state: dict[str, torch.Tensor],
    *,
    layer_start: int,
    num_layers: int,
    dtype: torch.dtype,
) -> Glm53FlashForCausalLM:
    model_config = _build_tt_model_config(layer_start, num_layers)
    model = model_config.build().to(dtype=dtype)
    local_hf = _remap_hf_layers_to_local(
        hf_state,
        layer_start=layer_start,
        num_layers=num_layers,
    )
    adapter = Glm53FlashStateDictAdapter(model_config, None)
    native = adapter.from_hf(local_hf)
    missing, unexpected = model.load_state_dict(native, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"TorchTitan load mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    model.eval()
    return model


def _slice_text_config_dict(
    checkpoint: Path,
    *,
    layer_start: int,
    num_layers: int,
) -> dict[str, Any]:
    with (checkpoint / "config.json").open() as f:
        raw = json.load(f)["text_config"]
    sliced = dict(raw)
    layer_end = layer_start + num_layers
    sliced["num_hidden_layers"] = num_layers
    sliced["layer_types"] = [
        "indexed_attention" if item == "deepseek_sparse_attention" else item
        for item in raw["layer_types"][layer_start:layer_end]
    ]
    sliced["mlp_layer_types"] = raw["mlp_layer_types"][layer_start:layer_end]
    sliced["indexer_types"] = raw.get("indexer_types", ["full"] * 45)[
        layer_start:layer_end
    ]
    if "linear_attn_config" in raw:
        linear = raw["linear_attn_config"]
        local_kda = [
            idx - layer_start
            for idx in linear.get("kda_layers", [])
            if layer_start <= idx < layer_end
        ]
        local_sparse = [
            idx - layer_start
            for idx in linear.get("full_attn_layers", [])
            if layer_start <= idx < layer_end
        ]
        sliced["linear_attn_config"] = {
            **linear,
            "kda_layers": local_kda,
            "full_attn_layers": local_sparse,
        }
    return sliced


def _tt_native_to_hf_text_state(
    native: dict[str, torch.Tensor],
    *,
    num_layers: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    lm_head = native["lm_head.weight"]
    for key, value in native.items():
        if key == "tok_embeddings.weight":
            state["embed_tokens.weight"] = value
            continue
        if key == "norm.weight":
            state["norm.weight"] = value
            continue
        if key == "lm_head.weight":
            continue
        if not key.startswith("layers."):
            continue
        parts = key.split(".")
        local_layer = int(parts[1])
        suffix = ".".join(parts[2:])
        if local_layer >= num_layers:
            continue
        prefix = f"layers.{local_layer}."
        if suffix == "self_attn.dt_bias":
            state[prefix + "self_attn.forget_gate.dt_bias"] = value.reshape(-1)
        elif suffix == "self_attn.A_log":
            state[prefix + "self_attn.forget_gate.A_log"] = value
        elif suffix in {
            "self_attn.f_a_proj.weight",
            "self_attn.f_b_proj.weight",
        }:
            state[
                prefix + suffix.replace("self_attn.", "self_attn.forget_gate.")
            ] = value
        elif suffix == "self_attn.q_conv1d.weight":
            q = value
            k = native[f"layers.{local_layer}.self_attn.k_conv1d.weight"]
            v = native[f"layers.{local_layer}.self_attn.v_conv1d.weight"]
            state[prefix + "self_attn.conv1d.weight"] = torch.cat([q, k, v], dim=0)
        elif suffix in {"self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"}:
            continue
        elif suffix == "self_attn.indexer.index_pool.index_kpool_compress_ape":
            state[prefix + "self_attn.indexer.index_kpool_compress_ape"] = value
        elif suffix == "self_attn.indexer.index_pool.index_kpool_compress_gate":
            state[prefix + "self_attn.indexer.index_kpool_compress_gate"] = value
        else:
            state[prefix + suffix] = value
    return state, lm_head


def _patch_hf_sparse_attention_mask_device() -> None:
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

    original = Glm5NextTextAttention.build_attention_mask_from_topk
    if getattr(original, "_torchtitan_device_patch", False):
        return

    def build_attention_mask_from_topk(
        self,
        topk_indices: torch.Tensor,
        query_states: torch.Tensor,
        kv_length: int,
    ) -> torch.Tensor | None:
        topk_indices = topk_indices.to(device=query_states.device)
        topk_valid = topk_indices.ge(0) & topk_indices.lt(kv_length)
        safe_indices = topk_indices.clamp(0, kv_length - 1).long()
        selected_counts = torch.zeros(
            topk_indices.shape[0],
            topk_indices.shape[1],
            kv_length,
            dtype=torch.int32,
            device=query_states.device,
        )
        selected_counts.scatter_add_(-1, safe_indices, topk_valid.to(torch.int32))
        mask = selected_counts.ne(0).unsqueeze(1)
        if self.config._attn_implementation == "sdpa":
            return mask

        zero = torch.full((), 0.0, device=query_states.device, dtype=query_states.dtype)
        min_value = torch.full(
            (),
            torch.finfo(query_states.dtype).min,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        return torch.where(mask, zero, min_value)

    build_attention_mask_from_topk._torchtitan_device_patch = True  # type: ignore[attr-defined]
    Glm5NextTextAttention.build_attention_mask_from_topk = (
        build_attention_mask_from_topk
    )


def _patch_hf_kda_reference() -> None:
    from transformers.models.glm5_next import modeling_glm5_next

    chunk_kda = modeling_glm5_next.chunk_kimi_delta_attention
    fallback = getattr(chunk_kda, "__wrapped__", None)
    if fallback is not None:
        modeling_glm5_next.chunk_kimi_delta_attention = fallback


def _build_hf_text_model(
    checkpoint: Path,
    hf_state: dict[str, torch.Tensor],
    *,
    layer_start: int,
    num_layers: int,
    dtype: torch.dtype,
):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    _patch_hf_kda_reference()
    _patch_hf_sparse_attention_mask_device()
    text_config = Glm5NextTextConfig(
        **_slice_text_config_dict(
            checkpoint,
            layer_start=layer_start,
            num_layers=num_layers,
        )
    )
    text_config._attn_implementation = "eager"
    model = Glm5NextTextModel(text_config).to(dtype=dtype)

    tt_model_config = _build_tt_model_config(layer_start, num_layers)
    local_hf = _remap_hf_layers_to_local(
        hf_state,
        layer_start=layer_start,
        num_layers=num_layers,
    )
    native = Glm53FlashStateDictAdapter(tt_model_config, None).from_hf(local_hf)
    state, lm_head_weight = _tt_native_to_hf_text_state(native, num_layers=num_layers)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"HF text load mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    lm_head = torch.nn.Linear(
        text_config.hidden_size, text_config.vocab_size, bias=False
    )
    lm_head = lm_head.to(dtype=dtype)
    lm_head.weight.data.copy_(lm_head_weight.to(dtype=dtype))
    model.eval()
    lm_head.eval()
    return model, lm_head


def _read_countdown_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= count:
                break
    if len(rows) < count:
        raise ValueError(f"wanted {count} countdown rows, found {len(rows)} in {path}")
    return rows


def _load_hf_tokenizer(checkpoint: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)


def _build_parity_inputs(
    checkpoint: Path,
    data_file: Path,
    *,
    long_seq_len: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[str]]:
    tokenizer = _load_hf_tokenizer(checkpoint)
    rows = _read_countdown_rows(data_file, 4)
    rendered = []
    samples = []
    for row in rows:
        messages = [{"role": "user", "content": row["question"]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            clear_thinking=True,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        rendered.append(text)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
        samples.append(
            {
                "name": row.get("problem_id", f"countdown_{len(samples)}"),
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            }
        )

    vocab_size = int(getattr(tokenizer, "vocab_size", 154_880))
    generator = torch.Generator(device="cpu").manual_seed(20261014)
    long_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(long_seq_len,),
        generator=generator,
        dtype=torch.long,
    ).view(1, -1)
    long_ids = long_ids.to(device)
    samples.append(
        {
            "name": f"random_{long_seq_len}",
            "input_ids": long_ids,
            "attention_mask": torch.ones_like(long_ids, dtype=torch.bool),
        }
    )
    return samples, rendered


def _max_diff_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    if mask is not None:
        while mask.ndim < actual_f.ndim:
            mask = mask.unsqueeze(-1)
        actual_f = actual_f[mask.expand_as(actual_f)].reshape(-1)
        expected_f = expected_f[mask.expand_as(expected_f)].reshape(-1)
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1.0e-12)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
    }


def _top1_kl_stats(
    tt_logits: torch.Tensor, hf_logits: torch.Tensor
) -> dict[str, float]:
    tt = tt_logits.float()
    hf = hf_logits.float()
    top1 = (tt.argmax(dim=-1) == hf.argmax(dim=-1)).float().mean()
    log_p = F.log_softmax(tt, dim=-1)
    log_q = F.log_softmax(hf, dim=-1)
    kl = F.kl_div(log_p, log_q.exp(), reduction="batchmean", log_target=False)
    return {
        "top1_agreement": float(top1.item()),
        "kl_hf_to_tt": float(kl.item()),
    }


def _selection_mask(topk_BLS: torch.Tensor, seq_len: int) -> torch.Tensor:
    valid_BLS = topk_BLS.ge(0) & topk_BLS.lt(seq_len)
    safe_BLS = topk_BLS.clamp(0, max(seq_len - 1, 0)).long()
    counts_BLL = torch.zeros(
        topk_BLS.shape[0],
        topk_BLS.shape[1],
        seq_len,
        dtype=torch.int32,
        device=topk_BLS.device,
    )
    counts_BLL.scatter_add_(-1, safe_BLS, valid_BLS.to(torch.int32))
    return counts_BLL.ne(0)


def _tt_step(
    model: Glm53FlashForCausalLM,
    h_streams: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    traces = []
    for local_layer, layer in enumerate(model.layers.values()):
        capture: dict[str, torch.Tensor] = {}
        hook = None
        if isinstance(layer.self_attn, GlmSparseMLAAttention):
            hook = layer.self_attn.indexer.register_forward_hook(
                lambda _module, _args, output, capture=capture: capture.setdefault(
                    "topk", output.detach().cpu()
                )
            )
        h_streams = layer(h_streams, attention_mask_BL=attention_mask)
        if hook is not None:
            hook.remove()
        traces.append(
            {
                "layer": local_layer,
                "kind": layer.block_type,
                "hidden": h_streams.detach(),
                "topk": capture.get("topk"),
            }
        )
    return h_streams, traces


def _hf_step(model, h_streams, input_ids, attention_mask):
    traces = []
    causal_masks = {
        "indexed_attention": attention_mask,
        "linear_attention": attention_mask,
    }
    position_ids = torch.arange(h_streams.shape[1], device=h_streams.device).unsqueeze(
        0
    )
    prev_topk = None
    for local_layer, layer in enumerate(model.layers):
        capture: dict[str, torch.Tensor] = {}
        hook = None
        if getattr(layer.self_attn, "indexer", None) is not None:
            hook = layer.self_attn.indexer.register_forward_hook(
                lambda _module, _args, output, capture=capture: capture.setdefault(
                    "topk", output.detach().cpu()
                )
            )
        h_streams, prev_topk = layer(
            h_streams,
            attention_mask=causal_masks[layer.block_type],
            position_ids=position_ids,
            position_embeddings=None,
            input_ids=input_ids,
            past_key_values=None,
            prev_topk_indices=prev_topk,
        )
        if hook is not None:
            hook.remove()
        traces.append(
            {
                "layer": local_layer,
                "kind": layer.block_type,
                "hidden": h_streams.detach(),
                "topk": capture.get("topk"),
            }
        )
    return h_streams, traces


@torch.no_grad()
def run_slice_parity(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    index = _read_index(checkpoint, args.index_json)
    weight_map = index["weight_map"]
    layers = range(args.layer_start, args.layer_start + args.num_layers)
    keys = _slice_hf_keys(weight_map, layers)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    hf_state = _load_safetensors_subset(
        checkpoint,
        weight_map,
        keys,
        device=str(device),
        dtype=dtype,
    )
    tt_model = _build_tt_model(
        hf_state,
        layer_start=args.layer_start,
        num_layers=args.num_layers,
        dtype=dtype,
    ).to(device)
    hf_model, hf_lm_head = _build_hf_text_model(
        checkpoint,
        hf_state,
        layer_start=args.layer_start,
        num_layers=args.num_layers,
        dtype=dtype,
    )
    hf_model = hf_model.to(device)
    hf_lm_head = hf_lm_head.to(device)

    input_samples, rendered = _build_parity_inputs(
        checkpoint,
        Path(args.data_file),
        long_seq_len=args.long_seq_len,
        device=device,
    )
    sample_results = []
    for sample in input_samples:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        tt_h = tt_model.tok_embeddings(input_ids)
        tt_streams = (
            tt_h.unsqueeze(2)
            .expand(-1, -1, tt_model.glm_config.hc_mult, -1)
            .contiguous()
        )
        hf_h = hf_model.embed_tokens(input_ids)
        hf_streams = (
            hf_h.unsqueeze(2).expand(-1, -1, hf_model.config.hc_mult, -1).contiguous()
        )

        tt_streams, tt_traces = _tt_step(tt_model, tt_streams, attention_mask)
        hf_streams, hf_traces = _hf_step(
            hf_model, hf_streams, input_ids, attention_mask
        )
        layer_stats = []
        index_stats = []
        for tt_trace, hf_trace in zip(tt_traces, hf_traces):
            stats = _max_diff_stats(
                tt_trace["hidden"],
                hf_trace["hidden"],
                mask=attention_mask,
            )
            stats["layer"] = args.layer_start + tt_trace["layer"]
            stats["tt_kind"] = tt_trace["kind"]
            stats["hf_kind"] = hf_trace["kind"]
            layer_stats.append(stats)
            tt_topk = tt_trace["topk"]
            hf_topk = hf_trace["topk"]
            if tt_topk is not None or hf_topk is not None:
                order_equal = (
                    tt_topk is not None
                    and hf_topk is not None
                    and torch.equal(tt_topk, hf_topk)
                )
                mask_equal = False
                mask_mismatch_count = None
                if tt_topk is not None and hf_topk is not None:
                    tt_mask = _selection_mask(tt_topk, input_ids.shape[1])
                    hf_mask = _selection_mask(hf_topk, input_ids.shape[1])
                    mask_equal = torch.equal(tt_mask, hf_mask)
                    mask_mismatch_count = int((tt_mask != hf_mask).sum().item())
                index_stats.append(
                    {
                        "layer": args.layer_start + tt_trace["layer"],
                        "match": bool(order_equal),
                        "tt_shape": list(tt_topk.shape)
                        if tt_topk is not None
                        else None,
                        "hf_shape": list(hf_topk.shape)
                        if hf_topk is not None
                        else None,
                        "mismatch_count": int((tt_topk != hf_topk).sum().item())
                        if tt_topk is not None and hf_topk is not None
                        else None,
                        "selection_mask_match": bool(mask_equal),
                        "selection_mask_mismatch_count": mask_mismatch_count,
                    }
                )

        tt_hidden = tt_model.norm(tt_streams.mean(dim=2))
        hf_hidden = hf_model.norm(hf_model.hc_head(hf_streams))
        valid_flat = attention_mask.reshape(-1)
        tt_logits = tt_model.lm_head(tt_hidden.reshape(-1, tt_hidden.shape[-1]))[
            valid_flat
        ]
        hf_logits = hf_lm_head(hf_hidden.reshape(-1, hf_hidden.shape[-1]))[valid_flat]
        final_stats = {
            **_max_diff_stats(tt_logits, hf_logits),
            **_top1_kl_stats(tt_logits, hf_logits),
        }
        sample_results.append(
            {
                "name": sample["name"],
                "seq_len": int(input_ids.shape[1]),
                "layer_stats": layer_stats,
                "index_stats": index_stats,
                "final_logits": final_stats,
            }
        )
    return {
        "claim_label": "deterministic_comparison",
        "transformers_commit": TRANSFORMERS_COMMIT,
        "dtype": args.dtype,
        "device": args.device,
        "layer_start": args.layer_start,
        "num_layers": args.num_layers,
        "long_seq_len": args.long_seq_len,
        "rendered_prompts": rendered,
        "samples": sample_results,
    }


def _expected_hf_shape_from_native(
    key: str,
    native_shapes: dict[str, tuple[int, ...]],
    adapter: Glm53FlashStateDictAdapter,
) -> tuple[int, ...] | None:
    expert_match = _EXPERT_RE.match(key)
    if expert_match is not None:
        _, _, projection = expert_match.groups()
        cfg = adapter.glm_config
        if projection in {"gate", "up"}:
            return (cfg.moe_intermediate_size, cfg.hidden_size)
        return (cfg.hidden_size, cfg.moe_intermediate_size)
    native_key = adapter._map_hf_key(key)
    if native_key is None:
        return None
    shape = native_shapes.get(native_key)
    if shape is None:
        return None
    if native_key.endswith(".dt_bias"):
        return (math.prod(shape),)
    return shape


def _make_fake_hf_state_from_index(
    weight_map: dict[str, str],
    adapter: Glm53FlashStateDictAdapter,
    native_shapes: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    state = {}
    for key in weight_map:
        skip = adapter._skip_reason(key)
        if skip is not None:
            state[key] = torch.empty(1, device="meta")
            continue
        shape = _expected_hf_shape_from_native(key, native_shapes, adapter)
        if shape is None:
            state[key] = torch.empty(1, device="meta")
        else:
            state[key] = torch.empty(shape, device="meta")
    return state


def run_key_coverage(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    index = _read_index(checkpoint, args.index_json)
    weight_map = index["weight_map"]
    model_config = Glm53FlashForCausalLM.Config(glm_config=glm53_flash())
    with torch.device("meta"):
        model = model_config.build()
    native_shapes = {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    }
    adapter = Glm53FlashStateDictAdapter(model_config, None)

    skip_counts: dict[str, int] = defaultdict(int)
    unknown = []
    shape_mismatches = []
    mapped = 0
    for key in weight_map:
        skip = adapter._skip_reason(key)
        if skip is not None:
            skip_counts[skip] += 1
            continue
        expected_shape = _expected_hf_shape_from_native(key, native_shapes, adapter)
        if expected_shape is None:
            unknown.append(key)
            continue
        mapped += 1

    if args.validate_shard_shapes:
        keys_by_shard: dict[str, list[str]] = defaultdict(list)
        for key, shard in weight_map.items():
            keys_by_shard[shard].append(key)
        for shard, shard_keys in sorted(keys_by_shard.items()):
            path = checkpoint / shard
            if not path.exists():
                for key in shard_keys:
                    if key in unknown or adapter._skip_reason(key) is not None:
                        continue
                    expected_shape = _expected_hf_shape_from_native(
                        key, native_shapes, adapter
                    )
                    if expected_shape is not None:
                        shape_mismatches.append(
                            {
                                "key": key,
                                "expected": list(expected_shape),
                                "actual": "missing_shard",
                            }
                        )
                continue
            with safe_open(path, framework="pt", device="cpu") as sf:
                for key in shard_keys:
                    if key in unknown or adapter._skip_reason(key) is not None:
                        continue
                    expected_shape = _expected_hf_shape_from_native(
                        key, native_shapes, adapter
                    )
                    if expected_shape is None:
                        continue
                    actual_shape = tuple(sf.get_slice(key).get_shape())
                    if tuple(actual_shape) != tuple(expected_shape):
                        shape_mismatches.append(
                            {
                                "key": key,
                                "expected": list(expected_shape),
                                "actual": list(actual_shape),
                            }
                        )

    fake_state = _make_fake_hf_state_from_index(weight_map, adapter, native_shapes)
    adapter_error = None
    restored_shapes: dict[str, tuple[int, ...]] = {}
    try:
        restored = adapter.from_hf(fake_state)
        restored_shapes = {key: tuple(value.shape) for key, value in restored.items()}
    except Exception as exc:
        adapter_error = repr(exc)

    native_key_set = set(native_shapes)
    restored_key_set = set(restored_shapes)
    missing_native = sorted(native_key_set - restored_key_set)
    unexpected_native = sorted(restored_key_set - native_key_set)
    restored_shape_mismatches = [
        {
            "key": key,
            "expected": list(native_shapes[key]),
            "actual": list(restored_shapes[key]),
        }
        for key in sorted(native_key_set & restored_key_set)
        if native_shapes[key] != restored_shapes[key]
    ]
    return {
        "claim_label": "deterministic_comparison",
        "total_hf_keys": len(weight_map),
        "mapped_hf_keys": mapped,
        "skip_counts": dict(sorted(skip_counts.items())),
        "unknown_hf_keys": unknown,
        "shape_mismatches": shape_mismatches,
        "adapter_error": adapter_error,
        "native_keys": len(native_shapes),
        "restored_native_keys": len(restored_shapes),
        "missing_native_keys": missing_native,
        "unexpected_native_keys": unexpected_native,
        "restored_shape_mismatches": restored_shape_mismatches,
    }


def inspect_chat_mask(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = HuggingFaceTokenizer(
        HuggingFaceTokenizer.Config(), tokenizer_path=str(args.checkpoint)
    )
    rows = _read_countdown_rows(Path(args.data_file), args.count)
    samples = []
    for row in rows:
        messages = _countdown_sft_process_sample(row)
        full_text = tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != tokenizer.eos_id:
            full_tokens.append(tokenizer.eos_id)
        prompt_text = tokenizer.apply_chat_template(
            messages[:1],
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        labels = full_tokens[1:].copy()
        for idx in range(max(len(prompt_tokens) - 1, 0)):
            labels[idx] = IGNORE_INDEX
        alignment = []
        for pos, (token_id, label_id) in enumerate(zip(full_tokens[:-1], labels)):
            alignment.append(
                {
                    "pos": pos,
                    "token_id": int(token_id),
                    "token": tokenizer.decode([token_id]),
                    "label_id": None if label_id == IGNORE_INDEX else int(label_id),
                    "label": None
                    if label_id == IGNORE_INDEX
                    else tokenizer.decode([label_id]),
                }
            )
        samples.append(
            {
                "problem_id": row.get("problem_id"),
                "full_text": full_text,
                "prompt_text": prompt_text,
                "num_tokens": len(full_tokens),
                "num_supervised_labels": sum(label != IGNORE_INDEX for label in labels),
                "alignment": alignment,
            }
        )
    return {
        "claim_label": "deterministic_comparison",
        "samples": samples,
    }


def wait_for_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        index_path = checkpoint / "model.safetensors.index.json"
        staged_path = checkpoint / ".staged.json"
        if index_path.exists():
            index = _read_index(checkpoint)
            shards = sorted(set(index["weight_map"].values()))
            missing = [shard for shard in shards if not (checkpoint / shard).exists()]
            if not missing and (staged_path.exists() or not args.require_staged_json):
                return {
                    "ready": True,
                    "checkpoint": str(checkpoint),
                    "num_shards": len(shards),
                    "staged_json": staged_path.exists(),
                }
        else:
            missing = ["model.safetensors.index.json"]
        if time.monotonic() >= deadline:
            return {
                "ready": False,
                "checkpoint": str(checkpoint),
                "missing": missing[:20],
                "staged_json": staged_path.exists(),
            }
        time.sleep(args.poll_seconds)


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--index-json", type=Path, default=None)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    wait = sub.add_parser("wait-for-checkpoint")
    wait.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    wait.add_argument("--timeout-seconds", type=int, default=0)
    wait.add_argument("--poll-seconds", type=int, default=120)
    wait.add_argument("--require-staged-json", action="store_true")
    wait.set_defaults(func=wait_for_checkpoint)

    coverage = sub.add_parser("coverage")
    _add_common_paths(coverage)
    coverage.add_argument("--validate-shard-shapes", action="store_true")
    coverage.set_defaults(func=run_key_coverage)

    chat = sub.add_parser("chat-mask")
    _add_common_paths(chat)
    chat.add_argument("--count", type=int, default=2)
    chat.set_defaults(func=inspect_chat_mask)

    parity = sub.add_parser("slice-parity")
    _add_common_paths(parity)
    parity.add_argument("--layer-start", type=int, required=True)
    parity.add_argument("--num-layers", type=int, required=True)
    parity.add_argument("--long-seq-len", type=int, default=4096)
    parity.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parity.add_argument("--device", default="cpu")
    parity.set_defaults(func=run_slice_parity)

    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
