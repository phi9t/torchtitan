# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import ChatProcessor
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.glm53_flash.config_registry import (
    glm53_flash_mini_countdown_full,
    glm53_flash_mini_countdown_lora,
)
from torchtitan.experiments.glm53_flash.flavors import debugmodel
from torchtitan.experiments.glm53_flash.hf_parity import run_key_coverage
from torchtitan.experiments.glm53_flash.model import (
    _initialize_glm53_flash_weights,
    _l2norm,
    Glm53FlashForCausalLM,
    glm_hc_post_mix,
    GlmIndexPool,
    GlmKDAAttention,
    GlmSparseIndexer,
    GlmSparseMLAAttention,
)
from torchtitan.experiments.glm53_flash.state_dict_adapter import (
    Glm53FlashStateDictAdapter,
)


def _init(module: torch.nn.Module) -> torch.nn.Module:
    cfg = debugmodel()
    _initialize_glm53_flash_weights(module, cfg)
    return module


def _oracle_index_pool(
    pool: GlmIndexPool,
    k_TK: torch.Tensor,
    gate_TK: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pooled = []
    indices = []
    valid = []
    ratio = pool.index_kpool
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        doc_k = k_TK[start:end]
        doc_gate = gate_TK[start:end]
        for local_start in range(0, doc_k.shape[0], ratio):
            local_end = min(local_start + ratio, doc_k.shape[0])
            idx = torch.full((ratio,), -1, dtype=torch.long)
            idx[: local_end - local_start] = torch.arange(
                start + local_start,
                start + local_end,
            )
            indices.append(idx.to(k_TK.device))
            valid.append(
                torch.tensor(local_end - local_start == ratio, device=k_TK.device)
            )
            keys = doc_k[local_start:local_end]
            gates = doc_gate[local_start:local_end]
            if keys.shape[0] != ratio:
                pad = ratio - keys.shape[0]
                keys = F.pad(keys, (0, 0, 0, pad))
                gates = F.pad(gates, (0, 0, 0, pad), value=float("-inf"))
            logits = gates.float() + pool.index_kpool_compress_ape.float()
            probs = torch.nan_to_num(logits.softmax(dim=0)).to(k_TK.dtype)
            pooled.append((probs * keys).sum(dim=0))
    return torch.stack(pooled), torch.stack(indices), torch.stack(valid)


def _oracle_topk(indexer: GlmSparseIndexer, x_BLD, q_resid_BLR, mask_BL):
    batch, seq_len = x_BLD.shape[:2]
    q_BLHK = indexer.wq_b(q_resid_BLR).view(
        batch, seq_len, indexer.n_heads, indexer.head_dim
    )
    k_BLK = indexer.k_norm(indexer.wk(x_BLD))
    gate_BLK = F.linear(x_BLD, indexer.index_kpool_compress_gate)
    output_width = indexer.index_topk + indexer.index_kpool - 1
    out = torch.full(
        (batch, seq_len, output_width),
        -1,
        dtype=torch.int32,
        device=x_BLD.device,
    )
    for b in range(batch):
        first_key = (
            int(mask_BL[b].long().argmax().item()) if mask_BL[b].any() else seq_len
        )
        pools = []
        pool_indices = []
        for start in range(first_key, seq_len, indexer.index_kpool):
            end = min(start + indexer.index_kpool, seq_len)
            if end - start != indexer.index_kpool or not mask_BL[b, start:end].all():
                continue
            logits = (
                gate_BLK[b, start:end].float()
                + indexer.index_kpool_compress_ape.float()
            )
            probs = logits.softmax(dim=0).to(k_BLK.dtype)
            pools.append((probs * k_BLK[b, start:end]).sum(dim=0))
            pool_indices.append(torch.arange(start, end, device=x_BLD.device))
        if pools:
            pool_keys_PK = torch.stack(pools)
            pool_indices_PR = torch.stack(pool_indices)
        else:
            pool_keys_PK = x_BLD.new_zeros(0, indexer.head_dim)
            pool_indices_PR = torch.empty(
                0, indexer.index_kpool, dtype=torch.long, device=x_BLD.device
            )
        for t in range(seq_len):
            if not mask_BL[b, t]:
                continue
            selected_tokens = []
            if pool_keys_PK.numel():
                scores_HP = torch.einsum(
                    "hk,pk->hp", q_BLHK[b, t].float(), pool_keys_PK.float()
                )
                scores_P = (
                    F.relu(scores_HP * indexer.softmax_scale)
                    * indexer.weights_proj(x_BLD[b : b + 1, t : t + 1])
                    .view(-1)
                    .float()[:, None]
                    * (indexer.n_heads**-0.5)
                ).sum(dim=0)
                visible = pool_indices_PR[:, -1] <= t
                scores_P = scores_P.masked_fill(
                    ~visible, torch.finfo(scores_P.dtype).min
                )
                select_k = min(
                    indexer.index_topk // indexer.index_kpool, scores_P.numel()
                )
                top = scores_P.topk(select_k).indices
                for pool_idx in top:
                    if visible[pool_idx]:
                        selected_tokens.extend(pool_indices_PR[pool_idx].tolist())
            visible_count = int((mask_BL[b, : t + 1]).sum().item())
            tail_count = visible_count % indexer.index_kpool
            tail_start = first_key + visible_count - tail_count
            chosen = selected_tokens[: indexer.index_topk]
            chosen.extend([-1] * (indexer.index_topk - len(chosen)))
            for idx in range(tail_start, tail_start + tail_count):
                if 0 <= idx <= t and bool(mask_BL[b, idx]):
                    chosen.append(idx)
                else:
                    chosen.append(-1)
            chosen = chosen[:output_width]
            if chosen:
                out[b, t, : len(chosen)] = torch.tensor(
                    chosen, dtype=torch.int32, device=x_BLD.device
                )
    return out


def _oracle_kda_attention(attn: GlmKDAAttention, x_BLD: torch.Tensor) -> torch.Tensor:
    batch, seq_len = x_BLD.shape[:2]
    q_BLC = attn._causal_conv(attn.q_proj(x_BLD), attn.q_conv1d.weight)
    k_BLC = attn._causal_conv(attn.k_proj(x_BLD), attn.k_conv1d.weight)
    v_BLC = attn._causal_conv(attn.v_proj(x_BLD), attn.v_conv1d.weight)
    q_BLHK = q_BLC.view(batch, seq_len, attn.num_heads, attn.head_dim)
    k_BLHK = k_BLC.view_as(q_BLHK)
    v_BLHK = v_BLC.view_as(q_BLHK)
    raw_gate_BLHK = attn.f_b_proj(attn.f_a_proj(x_BLD)).view_as(q_BLHK)
    beta_BLH = torch.sigmoid(attn.b_proj(x_BLD))
    core_BLHK = _oracle_kda_core(attn, q_BLHK, k_BLHK, v_BLHK, raw_gate_BLHK, beta_BLH)
    gate_BLHK = attn.g_b_proj(attn.g_a_proj(x_BLD)).view_as(core_BLHK)
    normed_BLHK = _oracle_gated_norm(
        core_BLHK, gate_BLHK, attn.o_norm.weight, attn.o_norm.eps
    )
    return attn.o_proj(normed_BLHK.reshape(batch, seq_len, -1))


def _oracle_kda_core(attn, q_BLHK, k_BLHK, v_BLHK, raw_gate_BLHK, beta_BLH):
    q_BLHK = _l2norm(q_BLHK).float()
    k_BLHK = _l2norm(k_BLHK).float()
    v_BLHK = v_BLHK.float()
    gate_BLHK = attn._prepare_gate(raw_gate_BLHK)
    beta_BLH = beta_BLH.float()
    state_BHKK = torch.zeros(
        q_BLHK.shape[0],
        attn.num_heads,
        attn.head_dim,
        attn.head_dim,
        device=q_BLHK.device,
    )
    outs = []
    scale = attn.head_dim**-0.5
    for idx in range(q_BLHK.shape[1]):
        state_BHKK = torch.exp(gate_BLHK[:, idx]).unsqueeze(-1) * state_BHKK
        k_BHK = k_BLHK[:, idx]
        old_BHV = torch.einsum("bhkv,bhk->bhv", state_BHKK, k_BHK)
        delta_BHV = beta_BLH[:, idx].unsqueeze(-1) * (v_BLHK[:, idx] - old_BHV)
        state_BHKK = state_BHKK + torch.einsum("bhk,bhv->bhkv", k_BHK, delta_BHV)
        outs.append(torch.einsum("bhkv,bhk->bhv", state_BHKK, q_BLHK[:, idx] * scale))
    return torch.stack(outs, dim=1).to(v_BLHK.dtype)


def _oracle_gated_norm(x_BLHK, gate_BLHK, weight_K, eps):
    x_float = x_BLHK.float()
    x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    x_float = x_float * weight_K.float()
    return (x_float * torch.sigmoid(gate_BLHK.float())).to(x_BLHK.dtype)


def _oracle_sparse_attention(sparse: GlmSparseMLAAttention, x_BLD: torch.Tensor):
    mask_BL = torch.ones(x_BLD.shape[:2], dtype=torch.bool, device=x_BLD.device)
    batch, seq_len = x_BLD.shape[:2]
    q_resid_BLR = sparse.q_a_layernorm(sparse.q_a_proj(x_BLD))
    q_BLHK = sparse.q_b_proj(q_resid_BLR).view(
        batch, seq_len, sparse.num_heads, sparse.qk_nope_head_dim
    )
    kv_BLR = sparse.kv_a_layernorm(sparse.kv_a_proj_with_mqa(x_BLD))
    topk_BLS = sparse.indexer(x_BLD.detach(), q_resid_BLR.detach(), mask_BL)
    kv_weight_HXR = sparse.kv_b_proj.weight.view(
        sparse.num_heads,
        sparse.qk_nope_head_dim + sparse.v_head_dim,
        sparse.kv_lora_rank,
    )
    k_weight_HKR = kv_weight_HXR[:, : sparse.qk_nope_head_dim]
    v_weight_HVR = kv_weight_HXR[:, sparse.qk_nope_head_dim :]
    q_latent_BLHR = torch.einsum("blhk,hkr->blhr", q_BLHK, k_weight_HKR)
    out_latent_BLHR = _oracle_sparse_latent_attention(
        q_latent_BLHR, kv_BLR, topk_BLS, sparse.scaling
    )
    out_BLHV = torch.einsum("blhr,hvr->blhv", out_latent_BLHR, v_weight_HVR)
    return sparse.o_proj(out_BLHV.reshape(batch, seq_len, -1))


def _oracle_sparse_latent_attention(q_BLHR, kv_BLR, topk_BLS, scale):
    batch, seq_len = q_BLHR.shape[:2]
    valid_BLS = topk_BLS.ge(0) & topk_BLS.lt(seq_len)
    safe_BLS = topk_BLS.clamp(0, max(seq_len - 1, 0)).long()
    counts_BLL = torch.zeros(
        batch, seq_len, seq_len, dtype=torch.int32, device=q_BLHR.device
    )
    counts_BLL.scatter_add_(-1, safe_BLS, valid_BLS.to(torch.int32))
    scores_BHLL = torch.einsum("blhr,bsr->bhls", q_BLHR.float(), kv_BLR.float())
    scores_BHLL = scores_BHLL * scale
    scores_BHLL = scores_BHLL.masked_fill(~counts_BLL.ne(0)[:, None], float("-inf"))
    probs_BHLL = torch.nan_to_num(
        torch.softmax(scores_BHLL, dim=-1, dtype=torch.float32)
    )
    return torch.einsum("bhls,bsr->blhr", probs_BHLL.to(kv_BLR.dtype), kv_BLR)


def _manual_debug_forward(model: Glm53FlashForCausalLM, tokens: torch.Tensor):
    mask = torch.ones_like(tokens, dtype=torch.bool)
    h_BLD = model.tok_embeddings(tokens)
    h_BLND = (
        h_BLD.unsqueeze(2).expand(-1, -1, model.glm_config.hc_mult, -1).contiguous()
    )
    for layer in model.layers.values():
        residual_BLND = h_BLND
        post_BLN, comb_BLNN, h_BLD = layer.attn_hc(h_BLND)
        h_BLD = layer.input_layernorm(h_BLD)
        if isinstance(layer.self_attn, GlmKDAAttention):
            h_BLD = _oracle_kda_attention(layer.self_attn, h_BLD)
        else:
            h_BLD = _oracle_sparse_attention(layer.self_attn, h_BLD)
        h_BLND = glm_hc_post_mix(h_BLD, residual_BLND, post_BLN, comb_BLNN)
        residual_BLND = h_BLND
        post_BLN, comb_BLNN, h_BLD = layer.ffn_hc(h_BLND)
        h_BLD = layer.post_attention_layernorm(h_BLD)
        h_BLD = layer.mlp(h_BLD)
        h_BLND = glm_hc_post_mix(h_BLD, residual_BLND, post_BLN, comb_BLNN)
    h_BLD = model.norm(h_BLND.mean(dim=2))
    return model.lm_head(h_BLD.reshape(-1, h_BLD.shape[-1]))


def test_mhc_post_mix_uses_comb_transpose_oracle():
    torch.manual_seed(1)
    x_BLD = torch.randn(2, 3, 5)
    residual_BLND = torch.randn(2, 3, 4, 5)
    post_BLN = torch.rand(2, 3, 4)
    comb_BLNN = torch.randn(2, 3, 4, 4).softmax(dim=-1)

    out = glm_hc_post_mix(x_BLD, residual_BLND, post_BLN, comb_BLNN)
    expected = post_BLN.unsqueeze(-1) * x_BLD.unsqueeze(-2) + torch.einsum(
        "blij,blid->bljd", comb_BLNN, residual_BLND
    )
    wrong_axis = post_BLN.unsqueeze(-1) * x_BLD.unsqueeze(-2) + torch.einsum(
        "blij,bljd->blid", comb_BLNN, residual_BLND
    )

    torch.testing.assert_close(out, expected)
    assert not torch.allclose(out, wrong_axis)


def test_index_pool_matches_oracle_with_document_boundaries():
    torch.manual_seed(2)
    cfg = debugmodel()
    pool = _init(GlmIndexPool(cfg))
    k_TK = torch.randn(9, cfg.index_head_dim)
    gate_TK = torch.randn(9, cfg.index_head_dim)
    cu = torch.tensor([0, 5, 9])

    actual = pool(k_TK, gate_TK, cu_seqlens=cu)
    expected = _oracle_index_pool(pool, k_TK, gate_TK, cu)

    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
    assert actual[2].tolist() == [True, False, True]


def test_indexer_topk_and_causal_tail_match_oracle():
    torch.manual_seed(3)
    cfg = debugmodel()
    cfg.index_topk = 8
    indexer = _init(GlmSparseIndexer(cfg))
    x_BLD = torch.randn(2, 9, cfg.hidden_size)
    q_BLR = torch.randn(2, 9, cfg.q_lora_rank)
    mask_BL = torch.ones(2, 9, dtype=torch.bool)
    mask_BL[1, :2] = False

    actual = indexer(x_BLD, q_BLR, mask_BL)
    expected = _oracle_topk(indexer, x_BLD, q_BLR, mask_BL)

    torch.testing.assert_close(actual, expected)
    assert (actual[:, :, 0] <= torch.arange(9).view(1, 9)).all()
    assert (actual[1, :2] == -1).all()


def test_sparse_reference_matches_dense_causal_when_topk_covers_all_tokens():
    torch.manual_seed(4)
    cfg = debugmodel()
    sparse = _init(GlmSparseMLAAttention(cfg))
    q_BLHR = torch.randn(1, 7, cfg.num_attention_heads, cfg.kv_lora_rank)
    kv_BLR = torch.randn(1, 7, cfg.kv_lora_rank)
    causal = torch.arange(7).view(1, 7, 1) >= torch.arange(7).view(1, 1, 7)
    topk_BLS = torch.where(
        causal,
        torch.arange(7).view(1, 1, 7).expand(1, 7, 7),
        torch.full((1, 7, 7), -1),
    ).to(torch.int32)

    actual = sparse._reference_attention(q_BLHR, kv_BLR, topk_BLS)
    scores_BHLL = torch.einsum("blhr,bsr->bhls", q_BLHR.float(), kv_BLR.float())
    scores_BHLL = scores_BHLL * sparse.scaling
    scores_BHLL = scores_BHLL.masked_fill(~causal[:, None], float("-inf"))
    probs_BHLL = torch.softmax(scores_BHLL, dim=-1)
    expected = torch.einsum("bhls,bsr->blhr", probs_BHLL.to(kv_BLR.dtype), kv_BLR)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_sparse_dense_causal_ablation_path_is_finite():
    torch.manual_seed(4)
    cfg = debugmodel()
    cfg.sparse_attention_mode = "dense_causal"
    sparse = _init(GlmSparseMLAAttention(cfg)).float()
    x_BLD = torch.randn(2, 7, cfg.hidden_size, requires_grad=True)
    mask_BL = torch.ones(2, 7, dtype=torch.bool)
    mask_BL[1, :2] = False

    actual = sparse(x_BLD, attention_mask_BL=mask_BL)
    loss = actual.square().mean()
    loss.backward()

    assert sparse.last_attention_path == "dense_causal"
    assert torch.isfinite(actual).all()
    assert torch.isfinite(x_BLD.grad).all()
    assert actual[1, :2].abs().max().item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_fast_path_matches_reference_on_gpu_bf16():
    torch.manual_seed(5)
    cfg_ref = debugmodel()
    cfg_fast = debugmodel()
    cfg_ref.attn_impl = "reference"
    cfg_fast.attn_impl = "fast"
    ref = _init(GlmSparseMLAAttention(cfg_ref)).cuda().bfloat16()
    fast = GlmSparseMLAAttention(cfg_fast).cuda().bfloat16()
    fast.load_state_dict(ref.state_dict())
    x_BLD = torch.randn(1, 64, cfg_ref.hidden_size, device="cuda", dtype=torch.bfloat16)
    mask_BL = torch.ones(1, 64, device="cuda", dtype=torch.bool)

    out_ref = ref(x_BLD, attention_mask_BL=mask_BL)
    out_fast = fast(x_BLD, attention_mask_BL=mask_BL)

    torch.testing.assert_close(out_fast, out_ref, atol=2e-2, rtol=2e-2)
    assert fast.last_attention_path.startswith("attn_gym_gather_attn")


def test_kda_layer_matches_oracle_with_low_rank_output_gate():
    torch.manual_seed(6)
    cfg = debugmodel()
    attn = _init(GlmKDAAttention(cfg)).float()
    x_BLD = torch.randn(1, 6, cfg.hidden_size)

    actual = attn(x_BLD)
    expected = _oracle_kda_attention(attn, x_BLD)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert attn.g_a_proj.weight.shape == (cfg.linear_head_dim, cfg.hidden_size)
    assert attn.g_b_proj.weight.shape == (
        cfg.linear_num_heads * cfg.linear_head_dim,
        cfg.linear_head_dim,
    )


def test_whole_debug_model_forward_matches_manual_reference_oracle():
    torch.manual_seed(7)
    cfg = debugmodel()
    cfg.attn_impl = "reference"
    model = Glm53FlashForCausalLM.Config(glm_config=cfg).build()
    model.to_empty(device="cpu")
    model.init_weights()
    tokens = torch.arange(10).view(1, 10) % cfg.vocab_size

    actual = model(tokens)
    expected = _manual_debug_forward(model, tokens)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_model_forward_accepts_flat_packed_text_tokens():
    torch.manual_seed(7)
    cfg = debugmodel()
    model = Glm53FlashForCausalLM.Config(glm_config=cfg).build()
    model.to_empty(device="cpu")
    model.init_weights()
    tokens_L = torch.arange(10) % cfg.vocab_size
    mask_L = torch.ones_like(tokens_L, dtype=torch.bool)

    actual = model(tokens_L, attention_mask=mask_L)
    expected = model(tokens_L.unsqueeze(0), attention_mask=mask_L.unsqueeze(0))

    assert actual.shape == (10, cfg.vocab_size)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [512, 4096])
def test_debug_model_forward_backward_is_finite_on_gpu(seq_len):
    torch.manual_seed(8)
    cfg = debugmodel()
    cfg.attn_impl = "fast"
    model = Glm53FlashForCausalLM.Config(glm_config=cfg).build().cuda().bfloat16()
    model.init_weights()
    tokens = torch.arange(seq_len, device="cuda").view(1, seq_len) % cfg.vocab_size

    logits = model(tokens)
    loss = logits.float().square().mean()
    loss.backward()

    assert torch.isfinite(logits).all()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_meta_to_empty_init_has_nonzero_norms_and_distinct_layers():
    torch.manual_seed(9)
    cfg = debugmodel()
    with torch.device("meta"):
        model = Glm53FlashForCausalLM.Config(glm_config=cfg).build()
    model.to_empty(device="cpu")
    model.init_weights()
    tokens = torch.arange(12).view(1, 12) % cfg.vocab_size
    layer_outputs = []
    h_BLD = model.tok_embeddings(tokens)
    h_BLND = h_BLD.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    for layer in model.layers.values():
        h_BLND = layer(
            h_BLND, attention_mask_BL=torch.ones_like(tokens, dtype=torch.bool)
        )
        layer_outputs.append(h_BLND.detach())

    for name, param in model.named_parameters():
        if "norm.weight" in name or name.endswith(".scale"):
            assert param.norm().item() > 0.0, name
    assert not torch.allclose(layer_outputs[0], layer_outputs[-1])


def test_mini_countdown_configs_use_chat_data_qwen_vocab_and_fast_path(
    monkeypatch, tmp_path
):
    data_file = tmp_path / "formatting.jsonl"
    data_file.write_text(
        '{"question":"Numbers: 1, 2\\nTarget: 3","answer":"1 + 2 = 3\\nFINAL: 3"}\n'
    )
    monkeypatch.setenv("GLM53_FLASH_COUNTDOWN_DATA_FILE", str(data_file))
    monkeypatch.setenv(
        "GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH", str(tmp_path / "tokenizer")
    )

    lora_config = glm53_flash_mini_countdown_lora()
    full_config = glm53_flash_mini_countdown_full()

    assert lora_config.hf_assets_path == str(tmp_path / "tokenizer")
    assert isinstance(lora_config.tokenizer, HuggingFaceTokenizer.Config)
    assert isinstance(lora_config.dataloader, GrainDataLoader.Config)
    assert isinstance(lora_config.dataloader.dataset, FirstFitPackingConfig)
    dataset = lora_config.dataloader.dataset.dataset
    assert isinstance(dataset, SingleDatasetConfig)
    assert isinstance(dataset.source, HuggingFaceRandomAccessSource.Config)
    assert dataset.source.load_dataset_kwargs["data_files"] == str(data_file)
    assert dataset.source.split == "train[:64]"
    assert isinstance(dataset.processor, ChatProcessor.Config)
    assert dataset.processor.messages_fn({"question": "q", "answer": "a"}) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert lora_config.training.max_context_length == 512
    assert lora_config.training.num_tokens_per_train_step == 64 * 512
    assert lora_config.model.glm_config.vocab_size == 151_936
    assert lora_config.model.glm_config.attn_impl == "fast"
    assert lora_config.model.glm_config.sparse_attention_mode == "indexer"
    assert lora_config.model.glm_config.lora_rank == 16
    assert full_config.model.glm_config.lora_rank is None
    assert isinstance(full_config.loss, ChunkedLossWrapper.Config)


def test_state_dict_adapter_round_trip_uses_real_bf16_index_schema():
    torch.manual_seed(10)
    cfg = debugmodel()
    model_cfg = Glm53FlashForCausalLM.Config(glm_config=cfg)
    model = model_cfg.build()
    model.to_empty(device="cpu")
    model.init_weights()
    native = model.state_dict()
    adapter = Glm53FlashStateDictAdapter(model_cfg, None)

    hf = adapter.to_hf(dict(native))
    repo_root = Path(__file__).resolve().parents[4]
    index_path = (
        repo_root
        / ".scratch/glm53-flash-sft-ladder/evidence/bf16-model.safetensors.index.json"
    )
    weight_map = json.loads(index_path.read_text())["weight_map"]
    assert "model.language_model.layers.3.self_attn.indexer.wk.weight" in weight_map
    assert "model.language_model.layers.3.mlp.experts.0.gate_proj.weight" in weight_map
    assert "model.language_model.layers.0.self_attn.dt_bias" in hf

    hf["model.language_model.layers.45.self_attn.q_a_proj.weight"] = torch.zeros(1)
    hf["vision_tower.patch_embed.proj.weight"] = torch.zeros(1)
    hf["model.visual.patch_embed.proj.weight"] = torch.zeros(1)
    hf["model.language_model.layers.0.self_attn.q_proj.weight_scale_inv"] = torch.zeros(
        1
    )
    restored = adapter.from_hf(hf)

    assert adapter.last_skip_counts == {
        "mtp": 1,
        "vision_tower": 2,
        "weight_scale_inv": 1,
    }
    assert restored.keys() == native.keys()
    for key in native:
        torch.testing.assert_close(restored[key], native[key])


def test_real_glm53_flash_checkpoint_key_coverage_when_present():
    checkpoint = Path("/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16")
    if not (checkpoint / "model.safetensors.index.json").exists():
        pytest.skip("GLM-5.3-Flash BF16 checkpoint is not staged in the rootfs")
    result = run_key_coverage(
        type(
            "Args",
            (),
            {
                "checkpoint": checkpoint,
                "index_json": None,
                "validate_shard_shapes": False,
            },
        )()
    )

    assert result["unknown_hf_keys"] == []
    assert result["adapter_error"] is None
    assert result["missing_native_keys"] == []
    assert result["unexpected_native_keys"] == []
    assert result["restored_shape_mismatches"] == []
