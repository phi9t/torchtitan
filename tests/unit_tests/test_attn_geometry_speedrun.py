# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from torchtitan.experiments.attn_geometry_speedrun.attention_geometry import (
    apply_upstream_rotary,
    explicit_gaussian_attention,
    geometry_sdpa,
    upstream_rotary_factors,
)
from torchtitan.experiments.attn_geometry_speedrun.recall_probe import (
    evaluate_recall_distances,
    make_recall_example,
)


def test_gauss_augmented_sdpa_matches_explicit_science_seq_len_and_backward():
    torch.manual_seed(11)
    seq_len = 1024
    num_heads = 2
    head_dim = 16
    q = torch.randn(1, seq_len, num_heads, head_dim, requires_grad=True)
    k = torch.randn(1, seq_len, num_heads, head_dim, requires_grad=True)
    v = torch.randn(1, seq_len, num_heads, head_dim, requires_grad=True)
    log_sigma = torch.tensor([math.log(3.0), math.log(5.0)], requires_grad=True)

    actual = geometry_sdpa(
        q,
        k,
        v,
        log_sigma,
        geometry="gauss",
        anchor_softmax_scale=0.1,
        gaussian_pad_to_multiple=1,
    )
    expected = explicit_gaussian_attention(q, k, v, log_sigma)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    loss = actual.square().mean()
    loss.backward()
    for grad in (q.grad, k.grad, v.grad, log_sigma.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()


def test_cosine_matches_anchor_rms_norm_path_at_matched_temperature():
    torch.manual_seed(12)
    seq_len = 64
    num_heads = 3
    head_dim = 32
    softmax_scale = 0.1
    q = torch.randn(1, seq_len, num_heads, head_dim)
    k = torch.randn(1, seq_len, num_heads, head_dim)
    v = torch.randn(1, seq_len, num_heads, head_dim)
    anchor_log = torch.zeros(num_heads)
    cosine_log = torch.full((num_heads,), math.log(head_dim * softmax_scale))

    anchor = geometry_sdpa(
        q,
        k,
        v,
        anchor_log,
        geometry="anchor",
        anchor_softmax_scale=softmax_scale,
        eps=0.0,
    )
    cosine = geometry_sdpa(
        q,
        k,
        v,
        cosine_log,
        geometry="cosine",
        anchor_softmax_scale=softmax_scale,
        eps=0.0,
    )

    torch.testing.assert_close(cosine, anchor, rtol=3e-5, atol=3e-5)


def test_dot_matches_plain_sdpa_at_inverse_sqrt_head_dim_scale():
    torch.manual_seed(13)
    seq_len = 48
    num_heads = 4
    head_dim = 24
    q = torch.randn(1, seq_len, num_heads, head_dim)
    k = torch.randn(1, seq_len, num_heads, head_dim)
    v = torch.randn(1, seq_len, num_heads, head_dim)
    log_scale = torch.full((num_heads,), math.log(head_dim**-0.5))

    actual = geometry_sdpa(
        q,
        k,
        v,
        log_scale,
        geometry="dot",
        anchor_softmax_scale=0.1,
    )
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=True,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def test_rope_preserves_key_norm_for_gauss_path():
    torch.manual_seed(14)
    seq_len = 128
    head_dim = 64
    k = torch.randn(1, seq_len, 6, head_dim)
    factor1, factor2 = upstream_rotary_factors(
        head_dim=head_dim,
        seq_len=seq_len,
        device=k.device,
    )
    rotated = apply_upstream_rotary(k, factor1, factor2)
    torch.testing.assert_close(
        rotated.norm(dim=-1),
        k.norm(dim=-1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_recall_probe_builds_fresh_kernel_aligned_examples():
    generator = torch.Generator().manual_seed(15)
    example = make_recall_example(
        distance=61,
        vocab_size=50304,
        generator=generator,
    )
    assert example.input_ids.numel() % 16 == 0
    assert example.retrieval_distance == 61


def test_recall_probe_reports_accuracy_by_distance():
    key_to_value = {}

    def logits_fn(input_ids: torch.Tensor) -> torch.Tensor:
        query_key = int(input_ids[-1])
        logits = torch.full((input_ids.numel(), 50304), -100.0)
        logits[-1, key_to_value[query_key]] = 100.0
        return logits

    original = make_recall_example

    def patched_example(**kwargs):
        example = original(**kwargs)
        prefix_len = (-(kwargs["distance"] + 3)) % 16
        key = int(example.input_ids[prefix_len + 1])
        value = int(example.target_id)
        key_to_value[key] = value
        return example

    import torchtitan.experiments.attn_geometry_speedrun.recall_probe as probe

    previous = probe.make_recall_example
    probe.make_recall_example = patched_example
    try:
        results = evaluate_recall_distances(
            logits_fn=logits_fn,
            distances=[16, 32],
            samples_per_distance=3,
            vocab_size=50304,
            seed=16,
        )
    finally:
        probe.make_recall_example = previous

    assert [result.accuracy for result in results] == [1.0, 1.0]
