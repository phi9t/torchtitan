# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU recurrent Falcon-1 / Falcon-1A mixers.

Legend: B batch, L sequence, N heads, K write/query feature dim, V value dim.
State S is stored as ``state_BNKV``.
"""

from __future__ import annotations

from typing import Literal

import torch

FalconVariant = Literal["falcon1", "falcon1a"]
FalconAlignment = Literal["delayed", "same_step"]
# phi is the query/key feature map applied before the fast-weight write:
# - "rms": QK-RMSNorm (paper default), - "l2": QK-L2 (paper 1A.1), - "none":
# identity (kernel identity tests). Legacy callers pass normalize_qk; see
# _resolve_phi for how the two are reconciled.
FalconPhi = Literal["rms", "l2", "none"]


def _rms_normalize(x_BLND: torch.Tensor, eps: float) -> torch.Tensor:
    return x_BLND * torch.rsqrt(x_BLND.pow(2).mean(dim=-1, keepdim=True) + eps)


def _l2_normalize(x_BLND: torch.Tensor, eps: float) -> torch.Tensor:
    return x_BLND * torch.rsqrt(x_BLND.pow(2).sum(dim=-1, keepdim=True) + eps)


def _resolve_phi(phi: FalconPhi | None, normalize_qk: bool) -> FalconPhi:
    """Reconcile the ``phi`` knob with the legacy ``normalize_qk`` flag.

    ``phi`` is authoritative when supplied. Otherwise ``normalize_qk=True``
    keeps the historical QK-RMSNorm default and ``normalize_qk=False`` selects
    the identity map, so existing callsites and tests are unchanged.
    """
    if phi is not None:
        if phi not in ("rms", "l2", "none"):
            raise ValueError(f"unsupported Falcon phi: {phi}")
        return phi
    return "rms" if normalize_qk else "none"


def _apply_phi(x_BLND: torch.Tensor, phi: FalconPhi, eps: float) -> torch.Tensor:
    if phi == "rms":
        return _rms_normalize(x_BLND, eps)
    if phi == "l2":
        return _l2_normalize(x_BLND, eps)
    return x_BLND


def falcon_recurrent_forward(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    beta_BLN: torch.Tensor,
    lambda_BLN: torch.Tensor,
    *,
    variant: FalconVariant = "falcon1a",
    alignment: FalconAlignment = "delayed",
    normalize_qk: bool = True,
    phi: FalconPhi | None = None,
    eps: float = 1.0e-6,
    eps_gamma: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read-after-write Falcon recurrence.

    ``alignment`` selects the write pairing:

    - ``delayed`` (default, paper): ``(x_t, y_t) = (phi(k_{t-1}), v_t)`` with
      ``x_0 = 0`` and ``eta_0 = 0``. Output at step ``t`` reads the updated
      state, so the first real write pairs ``phi(k_0)`` with ``v_1``.
    - ``same_step`` (ablation only): ``(x_t, y_t) = (phi(k_t), v_t)`` at every
      ``t`` including ``t = 0``, still read-after-write, with ``eta_0``
      computed normally. The first write pairs ``phi(k_0)`` with ``v_0``.

    ``phi`` selects the QK feature map: ``rms`` (QK-RMSNorm, paper default),
    ``l2`` (QK-L2, arm A5), or ``none`` (identity). When ``phi`` is unset the
    legacy ``normalize_qk`` flag is used (``True`` -> ``rms``, ``False`` ->
    ``none``).

    Same-step is an ablation switch, never the default; see HARNESS.md.
    """
    if variant not in ("falcon1", "falcon1a"):
        raise ValueError(f"unsupported Falcon variant: {variant}")
    if alignment not in ("delayed", "same_step"):
        raise ValueError(f"unsupported Falcon alignment: {alignment}")
    if q_BLNK.shape != k_BLNK.shape:
        raise ValueError("q and k must share shape BLNK")
    if q_BLNK.shape[:3] != v_BLNV.shape[:3]:
        raise ValueError("q and v must share B, L, and N")
    if beta_BLN.shape != q_BLNK.shape[:3] or lambda_BLN.shape != q_BLNK.shape[:3]:
        raise ValueError("beta and lambda must have shape BLN")

    phi_kind = _resolve_phi(phi, normalize_qk)
    q_work = q_BLNK.float()
    k_work = k_BLNK.float()
    v_work = v_BLNV.float()
    beta_work = beta_BLN.float()
    lambda_work = lambda_BLN.float()
    q_work = _apply_phi(q_work, phi_kind, eps)
    k_work = _apply_phi(k_work, phi_kind, eps)

    batch, seq_len, heads, key_dim = q_work.shape
    value_dim = v_work.shape[-1]
    out_BLNV = q_work.new_zeros(batch, seq_len, heads, value_dim)
    state_BNKV = q_work.new_zeros(batch, heads, key_dim, value_dim)

    for step in range(seq_len):
        # Delayed pairing writes phi(k_{step-1}) -> v_step, so t=0 is a no-op.
        # Same-step pairing writes phi(k_step) -> v_step at every t, including
        # t=0, with eta computed normally.
        delayed_noop = alignment == "delayed" and step == 0
        if delayed_noop:
            x_BNK = q_work.new_zeros(batch, heads, key_dim)
            write_BNV = q_work.new_zeros(batch, heads, value_dim)
            gamma_BN11 = q_work.new_ones(batch, heads, 1, 1)
            eta_BN1 = q_work.new_zeros(batch, heads, 1)
        else:
            key_step = step - 1 if alignment == "delayed" else step
            x_BNK = k_work[:, key_step]
            y_BNV = v_work[:, step]
            energy_BN = x_BNK.pow(2).sum(dim=-1)
            lambda_BN = lambda_work[:, step]
            beta_BN = beta_work[:, step]
            denom_BN = energy_BN + lambda_BN + eps
            eta_BN = torch.where(
                denom_BN == 0,
                torch.zeros_like(denom_BN),
                beta_BN / denom_BN,
            )
            alpha_BN = torch.minimum(
                eta_BN * lambda_BN,
                torch.full_like(eta_BN, 1.0 - eps_gamma),
            )
            gamma_BN11 = (1.0 - alpha_BN).unsqueeze(-1).unsqueeze(-1)
            eta_BN1 = eta_BN.unsqueeze(-1)
            if variant == "falcon1":
                pred_BNV = torch.einsum("bnkv,bnk->bnv", state_BNKV, x_BNK)
                write_BNV = y_BNV - pred_BNV
            else:
                write_BNV = y_BNV

        state_BNKV = state_BNKV * gamma_BN11 + torch.einsum(
            "bnk,bnv->bnkv",
            x_BNK,
            eta_BN1 * write_BNV,
        )
        out_BLNV[:, step] = torch.einsum(
            "bnkv,bnk->bnv",
            state_BNKV,
            q_work[:, step],
        )

    return out_BLNV.to(q_BLNK.dtype), state_BNKV


def _falcon_write_terms(
    q_work: torch.Tensor,
    k_work: torch.Tensor,
    v_work: torch.Tensor,
    beta_work: torch.Tensor,
    lambda_work: torch.Tensor,
    *,
    variant: FalconVariant,
    alignment: FalconAlignment,
    eps: float,
    eps_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute per-step write feature, gate, carry, and Hebbian target.

    Returns four float tensors, each aligned to the recurrence step ``t``:

    - ``x_BLNK``: write feature ``phi(k_{t-1})`` (delayed) or ``phi(k_t)``
      (same_step), with the ``delayed`` ``t = 0`` no-op left as zeros.
    - ``eta_BLN1``: NLMS-normalized step size ``eta_t`` (zero at the
      ``delayed`` ``t = 0`` no-op).
    - ``carry_BLN``: scalar carry ``gamma_t = 1 - min(eta_t lambda_t,
      1 - eps_gamma)`` (one at the ``delayed`` ``t = 0`` no-op).
    - ``y_BLNV``: the Hebbian target ``v_t`` (used directly by Falcon-1A and as
      the base of the Falcon-1 residual).

    Legend: B batch, L sequence, N heads, K key/write dim, V value dim.
    """
    batch, seq_len, heads, _ = q_work.shape

    if alignment == "delayed":
        # x_t = phi(k_{t-1}); x_0 = 0 so t=0 is a no-op write.
        x_BLNK = torch.zeros_like(k_work)
        x_BLNK[:, 1:] = k_work[:, :-1]
    else:
        x_BLNK = k_work

    energy_BLN = x_BLNK.pow(2).sum(dim=-1)
    denom_BLN = energy_BLN + lambda_work + eps
    eta_BLN = torch.where(
        denom_BLN == 0,
        torch.zeros_like(denom_BLN),
        beta_work / denom_BLN,
    )
    alpha_BLN = torch.minimum(
        eta_BLN * lambda_work,
        torch.full_like(eta_BLN, 1.0 - eps_gamma),
    )
    carry_BLN = 1.0 - alpha_BLN

    if alignment == "delayed":
        # The t=0 no-op keeps eta_0 = 0 and gamma_0 = 1 regardless of inputs.
        eta_BLN = eta_BLN.clone()
        carry_BLN = carry_BLN.clone()
        eta_BLN[:, 0] = 0.0
        carry_BLN[:, 0] = 1.0

    eta_BLN1 = eta_BLN.unsqueeze(-1)
    del variant, batch, seq_len, heads
    return x_BLNK, eta_BLN1, carry_BLN, v_work


def falcon_masked_parallel_forward(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    beta_BLN: torch.Tensor,
    lambda_BLN: torch.Tensor,
    *,
    variant: FalconVariant = "falcon1a",
    alignment: FalconAlignment = "delayed",
    normalize_qk: bool = True,
    phi: FalconPhi | None = None,
    eps: float = 1.0e-6,
    eps_gamma: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked-parallel view of :func:`falcon_recurrent_forward`.

    This computes the identical read-after-write Falcon output as the recurrent
    oracle, but with the sequential scan replaced by masked matmuls over the
    whole sequence. The recurrence

        S_t = gamma_t S_{t-1} + eta_t x_t w_t^T,   o_t = S_t^T q_t

    unrolls to

        S_t = sum_{s<=t} (prod_{r=s+1..t} gamma_r) eta_s x_s w_s^T

    so, writing the cumulative log-carry ``c_t = sum_{r<=t} log gamma_r``, the
    decay from write step ``s`` to read step ``t`` is ``exp(c_t - c_s)``. Both
    ``S_t`` and the read-after-write output ``o_t`` are then masked, decay-
    weighted sums of rank-one writes.

    For Falcon-1A the write target ``w_s = v_s`` is known up front. For
    Falcon-1 the write is the residual ``w_s = v_s - S_{s-1}^T x_s``, which
    depends on the state just before step ``s``. That still admits a parallel
    form: collecting ``r_s = w_s`` gives a lower-triangular linear system

        r_s + eta over prior writes acting through x_s = v_s,

    which we solve with a forward substitution over ``L`` (still O(L) but
    vectorized across B, N, K, V). ``L`` is tiny in these identity tests, so
    this is a faithful parallel-view oracle, not a production kernel.

    Same signature and tolerance contract as the recurrent oracle. Documented
    fp32 tolerance: rtol/atol 1e-5 on random inputs (see the identity tests).
    """
    if variant not in ("falcon1", "falcon1a"):
        raise ValueError(f"unsupported Falcon variant: {variant}")
    if alignment not in ("delayed", "same_step"):
        raise ValueError(f"unsupported Falcon alignment: {alignment}")
    if q_BLNK.shape != k_BLNK.shape:
        raise ValueError("q and k must share shape BLNK")
    if q_BLNK.shape[:3] != v_BLNV.shape[:3]:
        raise ValueError("q and v must share B, L, and N")
    if beta_BLN.shape != q_BLNK.shape[:3] or lambda_BLN.shape != q_BLNK.shape[:3]:
        raise ValueError("beta and lambda must have shape BLN")

    q_work = q_BLNK.float()
    k_work = k_BLNK.float()
    v_work = v_BLNV.float()
    beta_work = beta_BLN.float()
    lambda_work = lambda_BLN.float()
    phi_kind = _resolve_phi(phi, normalize_qk)
    q_work = _apply_phi(q_work, phi_kind, eps)
    k_work = _apply_phi(k_work, phi_kind, eps)

    batch, seq_len, heads, key_dim = q_work.shape
    value_dim = v_work.shape[-1]

    x_BLNK, eta_BLN1, carry_BLN, y_BLNV = _falcon_write_terms(
        q_work,
        k_work,
        v_work,
        beta_work,
        lambda_work,
        variant=variant,
        alignment=alignment,
        eps=eps,
        eps_gamma=eps_gamma,
    )

    # Cumulative log-carry per step; decay from write s to read t is
    # exp(cumlog_t - cumlog_s). Carry is strictly positive by the eps_gamma
    # clamp, so the log is well defined.
    log_carry_BLN = torch.log(carry_BLN)
    cumlog_BLN = torch.cumsum(log_carry_BLN, dim=1)

    # write_BLNV holds w_s, the value actually written at step s.
    if variant == "falcon1a":
        write_BLNV = y_BLNV
    else:
        # Falcon-1 residual: w_s = v_s - S_{s-1}^T x_s. Rebuild S_{s-1} (the
        # state before step s's write) incrementally, matching the recurrent
        # oracle order: predict against the pre-carry state, then apply carry
        # and fold in the new write.
        write_BLNV = torch.zeros_like(y_BLNV)
        state_BNKV = q_work.new_zeros(batch, heads, key_dim, value_dim)
        for step in range(seq_len):
            x_BNK = x_BLNK[:, step]
            pred_BNV = torch.einsum("bnkv,bnk->bnv", state_BNKV, x_BNK)
            w_BNV = y_BLNV[:, step] - pred_BNV
            write_BLNV[:, step] = w_BNV
            gamma_BN = carry_BLN[:, step]
            state_BNKV = state_BNKV * gamma_BN.unsqueeze(-1).unsqueeze(-1)
            state_BNKV = state_BNKV + torch.einsum(
                "bnk,bnv->bnkv",
                x_BNK,
                eta_BLN1[:, step] * w_BNV,
            )

    # Decay from write s to read t is exp(cumlog_t - cumlog_s). For the causal
    # entries (t >= s) the exponent is <= 0, since cumlog is non-increasing
    # (carry <= 1 after the eps_gamma clamp), so exp is in (0, 1] and never
    # overflows. For the masked upper triangle (t < s) the raw exponent can be
    # large and positive, so exp would overflow to inf there; a later
    # masked_fill zeros the forward, but inf * 0 poisons the backward with NaN.
    # We therefore clamp the exponent at 0 before exp so every entry stays in
    # (0, 1] and both forward and backward are finite; masked entries are
    # dropped by the causal mask below regardless of their value.
    qx_LL = torch.einsum("btnk,bsnk->bnts", q_work, x_BLNK)
    cumlog_BNL = cumlog_BLN.transpose(1, 2)
    exponent_BNLL = (cumlog_BNL.unsqueeze(-1) - cumlog_BNL.unsqueeze(-2)).clamp(max=0.0)
    decay_BNLL = torch.exp(exponent_BNLL)
    attn_BNLL = qx_LL * decay_BNLL * eta_BLN1.squeeze(-1).transpose(1, 2).unsqueeze(-2)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q_work.device)
    )
    attn_BNLL = attn_BNLL.masked_fill(~causal_mask, 0.0)
    out_BNLV = torch.einsum("bnts,bsnv->bntv", attn_BNLL, write_BLNV)
    out_BLNV = out_BNLV.transpose(1, 2)

    # Final state S_L: same decay-weighted accumulation with cumlog_{L-1} as the
    # read frame (all writes decayed to the last step). exp(cumlog_{L-1} -
    # cumlog_s) has exponent <= 0 since L-1 >= s, so no overflow.
    last_cumlog_BN = cumlog_BLN[:, -1]
    state_coeff_BLN1 = eta_BLN1 * torch.exp(
        last_cumlog_BN.unsqueeze(1) - cumlog_BLN
    ).unsqueeze(-1)
    state_BNKV = torch.einsum(
        "blnk,blnv->bnkv",
        x_BLNK * state_coeff_BLN1,
        write_BLNV,
    )

    return out_BLNV.to(q_BLNK.dtype), state_BNKV
