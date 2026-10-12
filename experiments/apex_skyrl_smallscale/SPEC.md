# Smaller-scale APEX-Agents SkyRL post-training run — SPEC

Source of intent: deep dive of the Mercor × SkyRL guide
*"Training frontier knowledge work agents: A 397B RL training guide with SkyRL"*
(https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/),
recipe repo `github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe`.

**Goal:** a faithful, *locally-adapted* port of the guide's methodology at small
scale on **one 8× B200 node**. Per the 2026-09-05 conversation the adaptations
are: (1) use a **dense Qwen3** model to avoid the gated GDN complexity, (2) do
an **SFT baseline first**, before any RL, (3) **single local node** only.

See `notes/00-recon-2026-09-05.md` for the reconnaissance behind every choice
below.

---

## Part A — Deep dive: what the guide actually does

### A.1 The two runs in the paper

| | Small run | Hero run |
|---|---|---|
| Base model | `Qwen/Qwen3.6-35B-A3B` (MoE, GDN hybrid, ~3B active) | `Qwen/Qwen3.5-397B-A17B` (MoE, ~17B active) |
| APEX-Agents Pass@1 | 13.96% base → **22.71%** trained (MeanReward 28.69→38.69) | 16.11% → **27.29%** (MeanReward 31.29→43.18) |
| Inference : train split | 10 engines ×TP8 : 4 train nodes (~14 H100 nodes) | 12 engines ×TP8 : 8 train nodes (20 H200 nodes) |
| Train parallelism | Megatron TP=8, EP=8, CP=1, PP=1 | Megatron TP=4, EP=16, CP=2, PP=4 (64-way, DP=1) |
| `MAX_CONCURRENCY` | 550 | 300 |
| `gpu_memory_utilization` | 0.85 (GDN hybrid: KV-offload connectors break EngineCore) | 0.93 (relies on KV-offload-during-sync) |

Central claim: *"the only real difference between a 35B run and a 397B run is
the RL systems work."* The **algorithm + recipe knobs are identical** across
both.

### A.2 Shared algorithm + training config (the recipe)
From `scripts/run_qwen3{5,6}_*_fully_async.sh`:

- **Framework**: SkyRL fully-async loop, vLLM inference, **Megatron** trainer,
  in-flight NCCL weight sync. Tinker-compatible.
- **Harness**: Harbor (rollout lifecycle) + custom `ArchipelagoAgent`
  (Harbor `BaseAgent`) doing MCP tool-calling, exchanging **raw token IDs**.
- **Advantage**: `grpo`. **Policy loss**: `dppo` (binary-TV masking of
  train/inference-divergent tokens), `dppo.delta_low=delta_high=0.15`.
- **Loss reduction**: `prompt_mean` (DAPO/ScaleRL). Biggest ablation win **+3.9
  pts** over `token_mean`.
- **Clips (GLM-5 style)**: `eps_clip_high=4`, `eps_clip_low=0.5`. **No KL**
  (`use_kl_loss=false`), no norm-by-std, zero-variance filter on (tol `1e-6`).
- **Optimizer**: Adam `lr=1.0e-6`, betas `[0.9,0.98]`, wd `0.01`, grad-norm
  `1.0`, CPU optimizer offload (`fraction=1.0`). Temperature 1.0.
- **Batching**: `train_batch_size=16`, `policy_mini_batch_size=16`,
  `n_samples_per_prompt=16` → 256 trajectories/step. `use_sample_packing=true`,
  `max_tokens_per_microbatch=160000` (dynamic micro-batching).
- **Fully-async**: `max_staleness_steps=3`,
  `num_parallel_generation_workers=64` (=`mini_batch×(staleness+1)`),
  `sample_full_batch=true`. Algorithmic concurrency ceiling
  `(3+1)×16×16=1024`; systems (KV-cache) ceiling binds → 550 / 300.
- **Context**: train `MAX_MODEL_LEN=160000`, eval `EVAL_MAX_MODEL_LEN=262144`.
- **Epochs=3**, eval every 20 steps, ckpt every 5.
- **Harness**: per-turn `max_tokens` 40k/50k, `agent_timeout=3600s`,
  `llm_call_timeout=1800s`, `tool_result_max_chars=42000`, **context nudge**
  (`tito_budget_warning_ratio=0.2`, worth **+3.0 pts**). Parsers
  `qwen3_xml`/`qwen3`. **TITO** via `/completions`-style exact token bookkeeping.

### A.3 The 6 steps (the methodology we port)
1. **Environment, harness, token accounting** — timeouts on everything, MCP
   client per Ray task, error classification, non-model error → 0 at target
   concurrency, TITO. *Harness fixes alone: 22.74% → 28.69% mean reward, zero
   training.*
2. **RL systems tuning** — Megatron knob sweep; split cluster so trainer never
   waits (`wait_for_generation_buffer=0`); concurrency = min(KV systems ceiling,
   algorithmic ceiling); train-inference logprob diff < 0.03.
3. **Overfitting run** — 32 nonzero-variance tasks, `batch=32`, `n_samples=8`,
   **synchronous**. Can't overfit ⇒ Step 1/2 is broken.
4. **Algorithm ablations** (small model) — `prompt_mean` +3.9, nudge +3.0, DPPO
   vs GLM-5 ~tie. OLF/ALP didn't help. Chose **DPPO+prompt_mean+nudge**.
5. **Hero run** — same knobs, bigger model, only systems differ.
6. **Evaluation & generalization** — 3-pass mean over held-out 480; cross-harness
   (Archipelago→OpenCode); Terminal-Bench 2.1; HLE+GPQA no-regression.

### A.4 Hard dependencies of the original recipe
Modal + AWS ECR (world `image.tar`s), Harbor + Archipelago, SkyRL+Megatron on
Ray, an LLM judge for reward, and APEX datasets in Harbor task-dir format
(`apex-agents-dev-1928` proprietary train + public `apex-agents-eval-480`).

---

## Part B — This host & the local constraints

- **8× NVIDIA B200, 183 GB, single node.** `bwrap`, `uv` present.
- **No `modal`/`aws`/`s5cmd`** → the guide's Modal+ECR sandbox substrate and
  S3 checkpointing are absent.
- `/data02` **99% full (~30 GB free)** → hard cap on model & checkpoint size.
- **Mercor released eval-trace datasets are license-forbidden for training**
  (CC-BY-4.0, "evaluation only; fine-tuning forbidden"). Usable for *eval
  reference* only, not as SFT targets. (recon note §Mercor)

---

## Part C — Local adaptations (the three the user asked for)

1. **Dense Qwen3, no GDN.** Base = **`Qwen3-1.7B`** (`model_type=qwen3`,
   `Qwen3ForCausalLM`, pure attention, no `layer_types`, 3.8 GB, already at
   `torchtitan/assets/hf/Qwen3-1.7B`). Sidesteps the gated GDN hybrid entirely
   and fits the 30 GB disk. (The paper's 35B/397B are MoE+GDN; we deliberately
   trade model class for a runnable, GDN-free dense baseline.)
2. **SFT baseline first.** Establish an SFT checkpoint *before* touching RL, as
   the reference point RL must beat. This is a deviation from the paper (which
   is explicitly *RL-without-SFT-warmup*), taken because the user wants an SFT
   baseline; we keep the RL recipe intact for a later phase so the SFT→RL delta
   is measurable.
3. **Single local node.** `colocate_all`, no multi-node NCCL/EFA — deletes the
   paper's heaviest Step-2 systems work while preserving the algorithm.

---

## Part D — Phase 1: the SFT baseline (do this first)

### D.1 Stack & model
- **Trainer**: **verl `sft_trainer.py`** (`SFTTrainer`, FSDP `TrainingWorker`;
  `verl/examples/sft`). Chosen because verl is already in the workspace, is
  single-node/FSDP-friendly, and keeps SFT and the later RL phase in one
  framework family.
- **Model**: `Qwen3-1.7B` dense (above). Full fine-tune fits on 8× B200;
  LoRA optional to shrink checkpoints given the disk cap.
- **Data schema**: verl `MultiTurnSFTDataset` wants **Parquet with a `messages`
  column** (multi-turn chat), optional `tools` column, chat-template applied,
  loss on assistant turns only.

### D.2 SFT data — self-generated APEX traces (rejection-sampling SFT)
Chosen approach (no license issue — our own trajectories, tasks used only as
prompts):
1. Take the **public APEX-Agents-480 tasks** as the prompt pool (tasks, not the
   forbidden Mercor trajectories).
2. Run an agent (Qwen3-1.7B, and/or a stronger local model as teacher) through
   a **simplified local agent loop** against each task, scoring with the task
   verifier / a local judge.
3. **Keep only high-reward trajectories** (rejection sampling), reformat the
   kept `(system+user prompt, assistant turns, tool calls)` into the verl
   `messages`-Parquet schema.
4. SFT Qwen3-1.7B on that Parquet.

**Dependency this creates:** to generate traces we need the APEX tasks
*runnable locally* — i.e. the sandbox/MCP world + a reward signal. This is the
same Step-0 infra the RL phase needs (Harbor + a local sandbox provider +
judge), pulled forward. If standing up the full APEX world locally proves too
heavy for the disk/infra, the documented fallback for the *baseline only* is to
generate traces on a lighter local verifiable task and treat APEX as the
eval/RL target — flagged as a fidelity deviation, decided when we hit it.

### D.3 SFT knobs (baseline)
- Full FT (or LoRA), context **8k–16k** to start (well under the 30 GB disk and
  1.7B memory; APEX rollouts can be long — truncate/pack).
- Standard SFT LR (~1e-5 full-FT / ~1e-4 LoRA), cosine, a few epochs over the
  kept traces, `max_samples` capped for iteration speed.
- Eval the SFT checkpoint on a held-out APEX split (MeanReward + Pass@1),
  matching the guide's 3-pass-mean discipline where feasible.

### D.4 SFT deliverables
- `scripts/phase1_sft/` — verl SFT launch wrapper + the trace-gen + parquet-pack
  scripts.
- SFT checkpoint + a short `.org` results note (base vs SFT MeanReward/Pass@1).

---

## Part E — Phase 2: the RL run (after SFT lands)

Keep the paper's **recipe knobs unchanged**; shrink only scale knobs; run on the
one node.

- **Model**: SFT checkpoint from Phase 1 (RL warm-started from SFT — again a
  deviation from the paper's cold RL, deliberate given the SFT-first goal).
- **Placement**: `colocate_all=true`, `policy_num_nodes=1`,
  `policy_num_gpus_per_node=8`. Dense Qwen3-1.7B → Megatron TP small (2–4),
  PP=CP=1, no EP (dense, not MoE). vLLM 1–2 colocated engines,
  `gpu_memory_utilization≈0.5–0.6`.
- **Recipe knobs (unchanged)**: `advantage_estimator=grpo`,
  `policy_loss_type=dppo`, `dppo.delta=0.15`, `loss_reduction=prompt_mean`,
  `eps_clip_high=4`, `eps_clip_low=0.5`, `use_kl_loss=false`,
  `grpo_norm_by_std=false`, `zero_variance_filter=true`, `temperature=1.0`,
  `lr=1.0e-6`, betas `[0.9,0.98]`, wd `0.01`, grad-norm `1.0`, TITO on, context
  nudge on (`tito_budget_warning_ratio=0.2`), `qwen3_xml`/`qwen3` parsers.
- **Scale knobs (shrunk)**: context 32k train/eval; `train/mini/n_samples`
  start 8/8/8 → toward 16/16/16 if throughput allows;
  `max_tokens_per_microbatch≈32k`; `max_staleness_steps=3` (or synchronous for
  the overfit step); `MAX_CONCURRENCY` from KV systems ceiling, raise until
  `wait_for_generation_buffer→0`; epochs 1–3.
- **Steps ported**: Step 1 smoke (`Qwen3-1.7B` on 1 GPU, adapt
  `run_1gpu_colocated_smoke.sh`) → eval pass at concurrency, drive error→0 →
  Step 2 colocated systems tuning + logprob-diff<0.03 → Step 3 overfit 32 tasks
  synchronous → Step 4 (optional) confirm prompt_mean/nudge → Step 5 small RL
  run → Step 6 eval (3-pass) vs the SFT baseline.

---

## Part F — Execution order (top-level)
- **Step 0 — Local infra**: resolve the **30 GB disk** blocker (scratch mount or
  redirect ckpt/export); stand up a local sandbox provider + reward/judge so
  APEX tasks are runnable locally (needed for both trace-gen and RL). Fail
  loudly if missing.
- **Phase 1 — SFT baseline** (Part D): trace-gen → parquet → verl SFT →
  eval. **This is the first thing we build.**
- **Phase 2 — RL** (Part E): the ported 6-step recipe, warm-started from SFT.

---

## Part G — Open risks / honest blockers
1. **Disk 30 GB free** — cannot hold big checkpoints; Qwen3-1.7B + LoRA is the
   mitigation, but trace dumps and RL ckpts still need space. Hard blocker for
   Step 0.
2. **No Modal/ECR/local sandbox yet** — required to make APEX tasks runnable for
   trace-gen *and* RL reward. Largest infra task.
3. **Mercor traces license-forbidden for training** — hence self-generated
   traces (Part D.2); resolved but shapes the whole SFT-data plan.
4. **Model class deviation** — dense 1.7B vs the paper's MoE+GDN 35B/397B;
   numbers won't be comparable to the paper, by design.
5. **SFT-first + RL-from-SFT** are deviations from the paper's cold-RL protocol,
   taken on the user's instruction; documented so the SFT→RL delta stays
   interpretable.
6. **LLM judge / reward** endpoint for APEX rewards at generation concurrency.

These are surfaced per the guide's own "fail loudly" Step-1 philosophy, not
assumed away.
