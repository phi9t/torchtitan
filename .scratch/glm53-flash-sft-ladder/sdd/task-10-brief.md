# T10 brief: campaign documentation (living doc; first full draft now)

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit. No GPU use. No code edits.
Python (if any) only inside the rootfs per `env-facts.md`; plain reading/writing Markdown needs no Python.

Write `docs/research/2026-10-14-glm-5.3-flash-sft-ladder.md`: the engineering record of the GLM-5.3-Flash SFT ladder
campaign on 8x B200, written for a TorchTitan engineer who was not here. Sources (cite file paths for every claim; never
invent numbers; quote numbers exactly as the reports give them):
- `.scratch/glm53-flash-sft-ladder/{spec.md,map.md}`, `sdd/progress.md`, every `sdd/task-*-brief.md` and `task-*-report.md`;
- `docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md` (architecture; link to it, do not duplicate);
- `git log --oneline 96760e290^..HEAD` and the files those commits touch;
- `experiments/glm53_flash_sft_ladder/README.md` and runner scripts; `torchtitan/experiments/glm53_flash*/`.

Required sections:
1. Goal and done-definition (from spec.md); claim-label vocabulary used.
2. Storage contract: waystone HDFS layout, URIs of every published artifact (datasets, weights, checkpoints, runs), MD5
   verification, auth keepalive (kinit + `waystone auth keepalive`, cron), measured HDFS throughput.
3. Environment: rootfs envs (default/vLLM, `glm53_train` torch nightly, `glm53_ref` Transformers oracle), `unset CC CXX`,
   networked mode, host RAM guard.
4. Ladder table: rung, model, configs, gates (parity, LoRA vs full, 2/4/8 GPU, resume, eval), results, claim labels,
   HDFS URIs, commit. Rungs 0, 1, 3, 3b(T07c parity), 4-probe(T08b). Leave clearly marked `TODO(T06)`, `TODO(T08d)`,
   `TODO(T09a/b/c)`, `TODO(T11)` placeholders for in-flight tickets -- do not guess their results.
5. GLM integration seam: model/flavors/state-dict adapter/sharding (FSDP only; TP/CP/PP rejected; no EP dispatch yet),
   LoRA trainable policy, rank-local sharded HF loader, trainable-only checkpoint, chat template + assistant mask.
6. Numerics: DSv4 HcPost dim=-3 fix (96760e290) and why; T07c fp32 parity; T08c bf16 drift diagnosis (both bf16 paths
   equally far from fp32 truth).
7. Memory: T07b analytical floors (4 vs 8 GPUs), T08b measured 72.99 GiB static + 131.56 GiB peak, and the three
   4-GPU directions (T09a streaming, T09b FP8 quantize-at-load, T09c overhead cuts) with their rationale.
8. Incidents and lessons: T08 host-RAM incident (CPU offload + full HF load, ~248-263 GiB RSS/rank) and the guard;
   stale doas token hangs; rootfs dep drift; pyrefly hook auto-edits; worker reverting a fix; pkill self-match;
   Triton autotune OOM + prewarm; vLLM AOT cache; shared-host GPU etiquette.
9. Open items / next steps: EP deep dive (T11), TP/CP, bf16 sensitivity, 4-GPU path, auth side project.
Keep it dense and factual, ASCII only, tables where they help. Also add a short "how to reproduce" section with the exact
runner commands from the reports. When done, write `sdd/task-10-report.md` listing every TODO placeholder left.
