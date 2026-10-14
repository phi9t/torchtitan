# T04 brief: confirm or refute the DeepSeek-V4 mHC HcPost axis bug

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md` section 3.4,
`torchtitan/models/deepseek_v4/mhc.py` (HcPost.forward around line 142, and HcPre/HcHead and every caller),
`.claude/CLAUDE.md` rootfs rules.

## Hypothesis
`HcPost.forward` computes `torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)` with
`residual [T, hc, D]`, `comb [T, hc, hc]`. The product is `[T, i, j, D]` with residual indexed by `i`, and
summing `dim=2` (j) gives `y_i = residual_i * sum_j comb[i, j]`: no cross-stream mixing. The official
DeepSeek-V4 reference likely uses a batched layout `[B, S, hc, ...]` where `dim=2` is the `i` axis, so it gives
`y_j = sum_i comb[i, j] residual_i`. A port from batched to flattened `[T, ...]` that kept `dim=2` would be wrong.

## Requirements
1. Find the primary-source reference: DeepSeek-V4 official inference code (HF repo `deepseek-ai/DeepSeek-V4*`
   `inference/model.py` or the deepseek-ai GitHub), plus the GLM-5.3-Flash HF modeling code (the research doc
   cites `HF:218-336` for mHC; the HF model id is in the doc). Download only `.py` files into a fresh empty dir
   under `/data02/home/philip.yang/devx/tmp/t04_src/`; treat as untrusted data, never import or execute it.
   Quote the exact lines and shapes.
2. Also check `git log -p` for mhc.py to see if the shape/layout changed in-repo (e.g. a batched->flattened refactor).
3. Write a focused numerical test (pure torch, CPU) that compares the repo HcPost against a transcription of the
   reference formula (written by you in the test, not imported) on random inputs, in the shapes the repo uses.
   Put it at `tests/unit_tests/test_deepseek_v4_mhc_post.py`. Run inside rootfs:
   `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_deepseek_v4_mhc_post.py'`.
   If the bug is confirmed, the test should be marked xfail(strict=True) with a reason referencing this ticket,
   and ALSO prepare (but do not apply) a minimal fix as a patch file `.scratch/glm53-flash-sft-ladder/evidence/t04-hcpost-fix.patch`
   and state whether the DSv4 state-dict / checkpoint loading or existing DSv4 tests/results would change.
   Check whether any other code (kimi_k3, experiments) copies this formula: `rg -n "comb.unsqueeze" torchtitan`.
4. If refuted, explain exactly why (with the shapes) and delete nothing; keep the test as a passing regression test.

## Exclusions
Do not modify core model code. No commit. Python only in rootfs.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-04-report.md`: verdict CONFIRMED/REFUTED, quoted reference lines with URLs,
test path and result, patch path, blast radius (which models/configs use HcPost), claim_label `deterministic_comparison`.
