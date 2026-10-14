# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from torchtitan.experiments.glm53_arch_speedrun.elements import (
    apply_rope,
    arm_config,
    b0_config,
    build_attention,
    causal_token_indices,
    collapse_streams,
    collect_element_aux_losses,
    compose_arms,
    dense_causal_attention,
    expand_streams,
    indexer_auxiliary_loss,
    IndexPool,
    kda_attention_gym,
    kda_chunked_reference,
    kda_recurrent,
    LightningIndexer,
    maybe_apply_rope,
    MHCResidualBlock,
    MTPHead,
    sparse_causal_attention,
    sparse_causal_attention_reference,
    SpeedrunElementsConfig,
)


LONG_SEQS = (1024, 4096)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for speedrun-width long-sequence gates.")
    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    cap = min(0.95, (24 * 1024**3) / total_memory)
    torch.cuda.set_per_process_memory_fraction(cap, device=device)
    torch.cuda.empty_cache()
    return device


def _assert_finite_grads(tensors: list[torch.Tensor | nn.Parameter]) -> None:
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def _dense_topk_recall(
    indexer: LightningIndexer,
    hidden_BLD: torch.Tensor,
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
) -> float:
    seq_len = hidden_BLD.shape[1]
    scale = q_BLHK.shape[-1] ** -0.5
    positions_L = torch.arange(seq_len, device=hidden_BLD.device)
    causal_LL = positions_L[:, None] >= positions_L[None, :]
    scores_BHLL = torch.einsum("blhk,bshk->bhls", q_BLHK, k_BLHK) * scale
    scores_BHLL = scores_BHLL.masked_fill(~causal_LL[None, None], float("-inf"))
    probs_BLL = torch.softmax(scores_BHLL, dim=-1, dtype=torch.float32).sum(dim=1)
    selected_BLS = indexer(hidden_BLD)
    recalls = []
    for query_idx in range(1, seq_len):
        width = min(indexer.top_k, query_idx + 1)
        target_S = probs_BLL[:, query_idx].topk(width, dim=-1).indices
        chosen_BS = selected_BLS[:, query_idx, : indexer.top_k]
        hit_BS = chosen_BS[..., None].eq(target_S[:, None]).any(dim=-1)
        recalls.append(hit_BS.float().sum(dim=-1) / width)
    return float(torch.cat(recalls).mean().item())


def test_indexer_auxiliary_loss_improves_dense_topk_recall_on_toy_task() -> None:
    torch.manual_seed(11)
    batch_size, seq_len, num_heads, head_dim = 4, 12, 2, 8
    hidden_size = num_heads * head_dim
    hidden_BLD = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        requires_grad=True,
    )
    q_BLHK = hidden_BLD.view(batch_size, seq_len, num_heads, head_dim)
    k_BLHK = q_BLHK
    indexer = LightningIndexer(
        hidden_size,
        top_k=2,
        index_num_heads=num_heads,
        index_head_dim=head_dim,
    )
    optimizer = torch.optim.AdamW(indexer.parameters(), lr=0.05)

    initial_recall = _dense_topk_recall(indexer, hidden_BLD.detach(), q_BLHK, k_BLHK)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = indexer_auxiliary_loss(
            indexer,
            hidden_BLD,
            q_BLHK,
            k_BLHK,
            target="all",
            query_chunk_size=4,
        )
        loss.backward()
        optimizer.step()
    final_recall = _dense_topk_recall(indexer, hidden_BLD.detach(), q_BLHK, k_BLHK)

    assert hidden_BLD.grad is None
    assert final_recall > initial_recall + 0.10
    assert final_recall > 0.55


@pytest.mark.parametrize("seq_len", LONG_SEQS)
def test_mhc_identity_init_equals_plain_residual_fwd_bwd(seq_len: int) -> None:
    torch.manual_seed(1)
    device = _device()
    cfg = arm_config("A1")
    branch = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, device=device)
    block = MHCResidualBlock(
        cfg.hidden_size,
        branch,
        num_streams=cfg.mhc_streams,
        sinkhorn_iters=cfg.mhc_sinkhorn_iters,
        eps=cfg.mhc_eps,
        identity_at_init=True,
    ).to(device)
    x_BLD = torch.randn(1, seq_len, cfg.hidden_size, device=device, requires_grad=True)
    streams_BLND = expand_streams(x_BLD, num_streams=cfg.mhc_streams).contiguous()

    out_BLND = block(streams_BLND)
    expected_BLD = x_BLD + branch(x_BLD)

    torch.testing.assert_close(collapse_streams(out_BLND), expected_BLD)
    out_BLND.float().square().mean().backward()
    _assert_finite_grads([x_BLD, branch.weight, block.identity_mix])
    finite_mhc_grads = [
        param.grad
        for name, param in block.named_parameters()
        if name.startswith("pre.") and param.grad is not None
    ]
    assert finite_mhc_grads
    assert all(torch.isfinite(grad).all() for grad in finite_mhc_grads)


@pytest.mark.parametrize("seq_len", LONG_SEQS)
def test_indexer_topk_covering_context_matches_dense_causal_fwd_bwd(
    seq_len: int,
) -> None:
    torch.manual_seed(2)
    device = _device()
    cfg = b0_config()
    q_BLHK = torch.randn(
        1,
        seq_len,
        cfg.num_attention_heads,
        cfg.head_dim,
        device=device,
        requires_grad=True,
    )
    k_BLHK = torch.randn_like(q_BLHK, requires_grad=True)
    v_BLHK = torch.randn_like(q_BLHK, requires_grad=True)
    selected_BLS = causal_token_indices(1, seq_len, device=device)
    indexer = LightningIndexer(
        cfg.hidden_size,
        top_k=seq_len,
        index_num_heads=cfg.indexer_num_heads,
        index_head_dim=cfg.indexer_head_dim,
    ).to(device)
    hidden_BLD = torch.randn(1, seq_len, cfg.hidden_size, device=device)

    torch.testing.assert_close(indexer(hidden_BLD), selected_BLS)

    with torch.no_grad():
        dense_out_BLHK = dense_causal_attention(q_BLHK, k_BLHK, v_BLHK)
        reference_out_BLHK = sparse_causal_attention_reference(
            q_BLHK,
            k_BLHK,
            v_BLHK,
            selected_BLS,
        )
    sparse_out_BLHK = sparse_causal_attention_reference(
        q_BLHK,
        k_BLHK,
        v_BLHK,
        selected_BLS,
    )

    torch.testing.assert_close(
        sparse_out_BLHK, dense_out_BLHK, atol=1.0e-5, rtol=1.0e-5
    )
    torch.testing.assert_close(
        sparse_out_BLHK, reference_out_BLHK, atol=1.0e-5, rtol=1.0e-5
    )
    sparse_out_BLHK.float().square().mean().backward()
    _assert_finite_grads([q_BLHK, k_BLHK, v_BLHK])


@pytest.mark.parametrize("seq_len", LONG_SEQS)
def test_sparse_gather_matches_reference_for_topk_fwd_bwd(seq_len: int) -> None:
    torch.manual_seed(22)
    device = _device()
    cfg = b0_config()
    top_k = 64
    q_BLHK = torch.randn(
        1,
        seq_len,
        cfg.num_attention_heads,
        cfg.head_dim,
        device=device,
        requires_grad=True,
    )
    k_BLHK = torch.randn_like(q_BLHK, requires_grad=True)
    v_BLHK = torch.randn_like(q_BLHK, requires_grad=True)
    selected_BLS = causal_token_indices(1, seq_len, device=device)[..., :top_k]

    with torch.no_grad():
        reference_out_BLHK = sparse_causal_attention_reference(
            q_BLHK,
            k_BLHK,
            v_BLHK,
            selected_BLS,
        )
    gather_out_BLHK = sparse_causal_attention(
        q_BLHK,
        k_BLHK,
        v_BLHK,
        selected_BLS,
        query_chunk_size=16,
    )

    torch.testing.assert_close(
        gather_out_BLHK,
        reference_out_BLHK,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    gather_out_BLHK.float().square().mean().backward()
    _assert_finite_grads([q_BLHK, k_BLHK, v_BLHK])


@pytest.mark.parametrize("seq_len", LONG_SEQS)
def test_index_pool_size_one_reduces_to_a4_fwd_bwd(seq_len: int) -> None:
    torch.manual_seed(3)
    device = _device()
    cfg = arm_config("A4")
    pool = IndexPool(cfg.indexer_head_dim, pool_size=1).to(device)
    keys_BLK = torch.randn(
        1,
        seq_len,
        cfg.indexer_head_dim,
        device=device,
        requires_grad=True,
    )
    gate_BLK = torch.randn_like(keys_BLK, requires_grad=True)
    mask_BL = torch.ones(1, seq_len, dtype=torch.bool, device=device)

    pooled_BPK, indices_BPR, valid_BP = pool(keys_BLK, gate_BLK, mask_BL)

    expected_indices_BPR = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    torch.testing.assert_close(pooled_BPK, keys_BLK)
    torch.testing.assert_close(indices_BPR, expected_indices_BPR)
    assert valid_BP.all()
    pooled_BPK.float().square().mean().backward()
    _assert_finite_grads([keys_BLK])
    assert gate_BLK.grad is None


@pytest.mark.parametrize("seq_len", LONG_SEQS)
def test_kda_chunked_matches_recurrent_oracle_fwd_bwd(seq_len: int) -> None:
    torch.manual_seed(4)
    device = _device()
    cfg = arm_config("A3")
    shape = (1, seq_len, cfg.num_attention_heads, cfg.head_dim)
    q_BLHK = (torch.randn(shape, device=device) * 0.02).requires_grad_()
    k_BLHK = (torch.randn(shape, device=device) * 0.02).requires_grad_()
    v_BLHK = (torch.randn(shape, device=device) * 0.02).requires_grad_()
    gate_logits_BLHK = torch.randn(shape, device=device, requires_grad=True)
    gate_BLHK = -0.1 * torch.sigmoid(gate_logits_BLHK)
    beta_logits_BLH = torch.randn(
        1,
        seq_len,
        cfg.num_attention_heads,
        device=device,
        requires_grad=True,
    )
    beta_BLH = torch.sigmoid(beta_logits_BLH)

    with torch.no_grad():
        recurrent_BLHK, _ = kda_recurrent(
            q_BLHK.detach(),
            k_BLHK.detach(),
            v_BLHK.detach(),
            gate_BLHK.detach(),
            beta_BLH.detach(),
        )
    chunked_BLHK = kda_chunked_reference(q_BLHK, k_BLHK, v_BLHK, gate_BLHK, beta_BLH)

    torch.testing.assert_close(chunked_BLHK, recurrent_BLHK, atol=2.0e-4, rtol=2.0e-4)
    chunked_BLHK.float().square().mean().backward()
    _assert_finite_grads([q_BLHK, k_BLHK, v_BLHK, gate_logits_BLHK, beta_logits_BLH])


@pytest.mark.parametrize("seq_len", LONG_SEQS)
def test_kda_attention_gym_matches_recurrent_oracle_fwd_bwd(seq_len: int) -> None:
    torch.manual_seed(44)
    device = _device()
    cfg = arm_config("A3")
    shape = (1, seq_len, cfg.num_attention_heads, cfg.head_dim)
    q_BLHK = (torch.randn(shape, device=device) * 0.02).requires_grad_()
    k_BLHK = (torch.randn(shape, device=device) * 0.02).requires_grad_()
    v_BLHK = (torch.randn(shape, device=device) * 0.02).requires_grad_()
    gate_logits_BLHK = torch.randn(shape, device=device, requires_grad=True)
    gate_BLHK = -0.1 * torch.sigmoid(gate_logits_BLHK)
    beta_logits_BLH = torch.randn(
        1,
        seq_len,
        cfg.num_attention_heads,
        device=device,
        requires_grad=True,
    )
    beta_BLH = torch.sigmoid(beta_logits_BLH)
    A_log_H = torch.zeros(cfg.num_attention_heads, device=device, requires_grad=True)
    dt_bias_HK = torch.zeros(
        cfg.num_attention_heads,
        cfg.head_dim,
        device=device,
        requires_grad=True,
    )

    with torch.no_grad():
        recurrent_BLHK, _ = kda_recurrent(
            q_BLHK.detach(),
            k_BLHK.detach(),
            v_BLHK.detach(),
            gate_BLHK.detach(),
            beta_BLH.detach(),
        )
    kernel_BLHK = kda_attention_gym(
        q_BLHK,
        k_BLHK,
        v_BLHK,
        gate_logits_BLHK,
        beta_logits_BLH,
        A_log_H,
        dt_bias_HK,
        lower_bound=-0.1,
    )

    torch.testing.assert_close(kernel_BLHK, recurrent_BLHK, atol=2.0e-4, rtol=2.0e-4)
    kernel_BLHK.float().square().mean().backward()
    _assert_finite_grads(
        [
            q_BLHK,
            k_BLHK,
            v_BLHK,
            gate_logits_BLHK,
            beta_logits_BLH,
            A_log_H,
            dt_bias_HK,
        ]
    )


@pytest.mark.parametrize("seq_len", LONG_SEQS)
def test_nope_flag_removes_rope_only_fwd_bwd(seq_len: int) -> None:
    torch.manual_seed(5)
    device = _device()
    cfg = b0_config()
    q_BLHK = torch.randn(
        1,
        seq_len,
        cfg.num_attention_heads,
        cfg.head_dim,
        device=device,
        requires_grad=True,
    )
    k_BLHK = torch.randn_like(q_BLHK, requires_grad=True)
    q_nope_BLHK, k_nope_BLHK = maybe_apply_rope(
        q_BLHK,
        k_BLHK,
        rope_enabled=False,
    )
    q_rope_BLHK, k_rope_BLHK = apply_rope(q_BLHK, k_BLHK)

    torch.testing.assert_close(q_nope_BLHK, q_BLHK)
    torch.testing.assert_close(k_nope_BLHK, k_BLHK)
    assert not torch.allclose(q_rope_BLHK[:, 1:], q_BLHK[:, 1:])
    assert not torch.allclose(k_rope_BLHK[:, 1:], k_BLHK[:, 1:])
    dense_causal_attention(
        q_nope_BLHK, k_nope_BLHK, q_BLHK
    ).float().square().mean().backward()
    _assert_finite_grads([q_BLHK, k_BLHK])


def test_hybrid_layout_and_optional_mtp_hooks_are_declared() -> None:
    cfg = arm_config("A3")
    assert [cfg.attention_kind(i) for i in range(8)] == [
        "kda",
        "kda",
        "kda",
        "softmax",
        "kda",
        "kda",
        "kda",
        "softmax",
    ]
    assert build_attention(0, cfg).__class__.__name__ == "KDALayer"
    assert build_attention(3, cfg).__class__.__name__ == "TopKSparseCausalAttention"

    mtp_cfg = arm_config("A6")
    mtp = MTPHead(
        mtp_cfg.hidden_size,
        mtp_cfg.vocab_size,
        aux_loss_weight=mtp_cfg.mtp_aux_loss_weight,
    )
    hidden_BLD = torch.randn(2, 5, mtp_cfg.hidden_size, requires_grad=True)
    labels_BL = torch.randint(0, mtp_cfg.vocab_size, (2, 5))
    loss = mtp.aux_loss(hidden_BLD, labels_BL)
    loss.backward()
    assert torch.isfinite(loss)
    _assert_finite_grads([hidden_BLD, mtp.proj.weight, mtp.lm_head.weight])


def test_sparse_attention_exposes_indexer_aux_loss_and_dense_warmup() -> None:
    torch.manual_seed(12)
    cfg = SpeedrunElementsConfig(
        hidden_size=16,
        num_layers=1,
        num_attention_heads=2,
        head_dim=8,
        mlp_hidden_size=64,
        vocab_size=32,
        max_seq_len=8,
        sparse_attention_enabled=True,
        sparse_top_k=2,
        indexer_num_heads=2,
        indexer_head_dim=8,
        indexer_aux_loss_weight=0.25,
        indexer_aux_target="all",
        indexer_dense_warmup_steps=2,
        sparse_attention_query_chunk_size=4,
    )
    attention = build_attention(0, cfg)
    x_BLD = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)

    warmup_out_BLD = attention(x_BLD, train_step=0)
    assert attention.last_attention_path == "dense_warmup"
    warmup_loss = collect_element_aux_losses(attention)
    assert warmup_loss is not None
    assert torch.isfinite(warmup_loss)

    sparse_out_BLD = attention(x_BLD, train_step=2)
    assert attention.last_attention_path == "gather"
    loss = sparse_out_BLD.float().square().mean()
    aux_loss = collect_element_aux_losses(attention)
    assert aux_loss is not None
    (loss + aux_loss).backward()

    indexer_grads = [
        param.grad
        for name, param in attention.named_parameters()
        if name.startswith("indexer.") and param.grad is not None
    ]
    assert indexer_grads
    assert all(torch.isfinite(grad).all() for grad in indexer_grads)
    assert warmup_out_BLD.shape == sparse_out_BLD.shape


def test_arm_configs_are_one_knob_switches() -> None:
    b0 = b0_config()
    a1 = arm_config("A1")
    assert a1.mhc_enabled and a1.mhc_init == "glm" and not a1.mhc_identity_at_init
    assert not b0.mhc_enabled
    assert not arm_config("A2").rope_enabled and b0.rope_enabled
    assert arm_config("A3").kda_enabled and not b0.kda_enabled
    assert arm_config("A4").sparse_attention_enabled and b0.index_pool_size == 1
    assert arm_config("A5").index_pool_size == 4
    assert arm_config("A6").mtp_enabled and b0.mtp_aux_loss_weight == 0.0
    with pytest.raises(ValueError, match="hidden_size"):
        replace(b0, hidden_size=512).validate()


def test_composed_arms_and_dose_knobs_are_exposed() -> None:
    cfg = compose_arms(
        "A1",
        "A2",
        "A3",
        "A5",
        "A6",
        sparse_top_k=256,
        index_pool_size=2,
        mtp_aux_loss_weight=0.05,
        mhc_sinkhorn_iters=3,
        mhc_init="gated_identity",
        indexer_aux_loss_weight=0.2,
        indexer_aux_target="all",
        indexer_dense_warmup_steps=7,
        kda_conv_kernel_size=2,
    )

    assert cfg.mhc_enabled
    assert not cfg.rope_enabled
    assert cfg.kda_enabled
    assert cfg.sparse_attention_enabled
    assert cfg.index_pool_size == 2
    assert cfg.sparse_top_k == 256
    assert cfg.mtp_enabled
    assert cfg.mtp_aux_loss_weight == 0.05
    assert cfg.mhc_sinkhorn_iters == 3
    assert cfg.mhc_init == "gated_identity"
    assert cfg.mhc_identity_at_init
    assert cfg.indexer_aux_loss_weight == 0.2
    assert cfg.indexer_aux_target == "all"
    assert cfg.indexer_dense_warmup_steps == 7
    assert cfg.kda_conv_kernel_size == 2
