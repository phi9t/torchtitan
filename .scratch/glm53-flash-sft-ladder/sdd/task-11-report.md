# T11 report: GLM-5.3-Flash expert-parallelism deep dive

Status: COMPLETE for phase 1. I produced the requested source-backed design note
and did not edit GLM experiment code or common MoE code. No commit was made.

Deliverable:

- `docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`

## Surface

This ticket was classified as repo-local research/design for the GLM SFT ladder,
not core training implementation and not online RL. The work stayed in
read/analyze/design mode. No GPU training was launched. I skipped the optional
prototype because the existing sources were sufficient to answer the phase-1
question, and adding a prototype would have introduced code churn without
changing the conclusion.

## Main findings

TorchTitan's current MoE EP stack has the right ingredients for GLM:

- `ep` owns routed expert placement;
- `edp_shard = dp_shard * cp * tp // ep` is the leftover sparse-FSDP shard axis;
- common MoE dispatches routed tokens with all-to-all and combines them back;
- routed expert weights use `Shard(0)` on the expert dimension;
- FSDP already switches routed expert parameters to the sparse mesh when
  `ep_degree > 1`;
- Qwen3, DeepSeek V3/V4, and Kimi K3 show the expected wiring pattern.

GLM is structurally close to that path because its routed expert tensors are
already grouped:

- `gate_up_proj`: `(num_experts, 2 * moe_intermediate_size, hidden)`;
- `down_proj`: `(num_experts, hidden, moe_intermediate_size)`.

The current GLM experiment still lacks token-dispatch EP execution. Its
validation explicitly says the seam does not implement token-dispatch EP, TP, or
CP. It also rejects TP/CP/PP, so on 4 GPUs the valid current EP choices are
only EP=1, EP=2, and EP=4. EP=8 on 4 GPUs would require adding TP or CP support,
not just changing `expert_parallel_degree`.

Memory conclusion:

- EP does not reduce the steady BF16 frozen-base shard at the same world size.
- On 8 GPUs, static remains about 73.046 GiB/GPU.
- On 4 GPUs, static remains about 146.092 GiB/GPU.
- EP does shrink the largest routed MoE unshard/all-gather component:
  - 4 GPUs EP=1: static + largest unshard about 159.902 GiB/GPU;
  - 4 GPUs EP=2: about 152.842 GiB/GPU;
  - 4 GPUs EP=4: about 149.467 GiB/GPU.
- Therefore EP helps the 4-GPU picture but does not replace the T09 memory
  directions. It should be paired with another memory mechanism for a full
  4-GPU step loop.

Communication conclusion:

- All-to-all traffic grows with EP.
- At 32,768 tokens/step over 42 MoE layers, estimated forward aggregate traffic
  is 84 GiB at EP=2, 126 GiB at EP=4, and 147 GiB at EP=8.
- The rough training aggregate doubles those values.
- For 4 GPUs EP=4, that is about 31.5 GiB/GPU forward and 63 GiB/GPU training
  payload per optimizer step before count traffic, padding, skew, or runtime
  overhead.

LoRA correction:

- The brief said LoRA targets include experts. The checked-in GLM target list
  includes attention projections and shared experts, not routed experts.
- The note calls this out explicitly and treats routed-expert LoRA as a possible
  phase-2 extension, not the current configuration.
- Common grouped LoRA already has the right placement concept for future routed
  expert LoRA: grouped adapter weights inherit the base grouped projection's
  expert-axis placement.

Loader/checkpoint conclusion:

- The existing GLM rank-local safetensors loader is already close to EP-ready.
  It computes each live DTensor local shape/global offset and loops over local
  grouped experts when reading HF keys.
- If routed experts become true EP `Shard(0)` tensors, the loader should be able
  to read only each EP rank's local expert IDs, subject to native key mapping if
  the implementation replaces `GlmExperts` with common `RoutedExperts`.

## Phase-2 plan captured in the note

The deliverable proposes seven follow-up tickets:

1. isolated `glm53_flash_ep` prototype with debug/mini fake-backend checks;
2. router and dispatch parity on tiny/truncated models;
3. rank-local loader and checkpoint compatibility under EP;
4. truncated-depth full-width numerical parity;
5. full-depth 8-GPU EP ladder;
6. full-depth 4-GPU feasibility paired with the selected T09 memory direction;
7. optional Dist-MoE evaluation after plain all-to-all EP is correct.

Only the first three are CPU/fake-backend friendly. Tickets 4-7 need GPUs.

## Files touched

Added:

- `docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`
- `.scratch/glm53-flash-sft-ladder/sdd/task-11-report.md`

Not touched:

- `torchtitan/experiments/glm53_flash/`
- `torchtitan/models/common/`

## Verification

Passed:

- Markdown file readback with `wc -l` and `sed`.
- `git diff --check -- docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md .scratch/glm53-flash-sft-ladder/sdd/task-11-report.md`.
- ASCII check:
  `LC_ALL=C grep -nP '[^\x00-\x7F]' docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md .scratch/glm53-flash-sft-ladder/sdd/task-11-report.md`
  returned no matches for the final docs.
- Focused pre-commit hooks in the `glm53_train` rootfs env:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && for hook in \
    trailing-whitespace end-of-file-fixer check-merge-conflict \
    check-added-large-files codespell; do \
      pre-commit run "$hook" --files \
        docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md \
        .scratch/glm53-flash-sft-ladder/sdd/task-11-report.md || exit $?; \
    done'
```

- Focused status inspection confirmed this turn added only the two T11 docs.
  The status view also shows unrelated pre-existing untracked files under
  `torchtitan/experiments/glm53_flash/`, which I did not touch.

No training, fake-backend launch, or GPU work was run for T11.
