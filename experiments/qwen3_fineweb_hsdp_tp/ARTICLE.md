# HSDP + TP on Qwen3, from config line to collective: a torchtitan deep dive

This article traces exactly what happens when you launch the
[`run.sh`](run.sh) experiment in this folder:

```bash
NGPU=8 MODULE=qwen3 CONFIG=qwen3_debugmodel_fineweb ./run_train.sh \
    --parallelism.data_parallel_replicate_degree=2 \
    --parallelism.data_parallel_shard_degree=2 \
    --parallelism.tensor_parallel_degree=2
```

`dp_replicate(2) x dp_shard(2) x tp(2) = 8` GPUs. `dp_replicate x dp_shard` is
HSDP (hybrid sharded data parallel); `tp` is tensor parallel. Every claim below
is tied to a concrete `file:line` in the current tree and to an observed result
from the companion [`probe_parallelism.py`](probe_parallelism.py) and
[`sweep.sh`](sweep.sh) (outputs land in `results/`, which is gitignored). Line
numbers are against the tree at the time of writing; if the source has moved,
search for the quoted snippet.

All file references are repo-root-relative (e.g.
`torchtitan/distributed/parallel_dims.py`).

## Contents

1. [The experiment and the mesh math](#1-the-experiment-and-the-mesh-math)
2. [From config to DeviceMesh](#2-from-config-to-devicemesh)
3. [Tensor Parallel on Qwen3](#3-tensor-parallel-on-qwen3)
4. [HSDP via FSDP2 `fully_shard`](#4-hsdp-via-fsdp2-fully_shard)
5. [The training step and its collectives](#5-the-training-step-and-its-collectives)
6. [Gradient norm across a two-axis sharded parameter](#6-gradient-norm-across-a-two-axis-sharded-parameter)
7. [Reading the sweep results](#7-reading-the-sweep-results)
8. [Reproduce](#8-reproduce)

---

## 1. The experiment and the mesh math

The config is `qwen3_debugmodel_fineweb`
(`torchtitan/models/qwen3/config_registry.py:65`). It reuses
`qwen3_debugmodel()` (dim 256, 8 layers, 16 heads / 8 KV heads, hidden 3072,
vocab 2048, `seq_len=2048`, `local_batch_size=8`, `steps=10`, `lr=8e-4`,
weight-tied embeddings, fused QKV) and only swaps the dataset to the locally
prefetched `fineweb_test` and enables checkpointing
(`config_registry.py:69-73`). Parallelism is left at defaults so the launcher
picks the mesh.

The mesh is `dp_replicate=2`, `dp_shard=2`, `tp=2`. That single line of config
turns into a stack of `DeviceMesh` submeshes and per-parameter DTensor
placements. The probe confirms the enabled one-dimensional meshes on 8 GPUs:

```
batch          size=4     # dp_replicate * dp_shard
loss           size=4     # dp_replicate * dp_shard * cp
dp_replicate   size=2
fsdp           size=2     # dp_shard * cp
tp             size=2
```

To reproduce this experiment: prefetch once online
(`python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py`), then run
`run.sh` offline. To reproduce the *analysis* in this article, run
`probe_parallelism.py` (Section 2, 3, 6) and `sweep.sh` (Section 7).

---

## 2. From config to DeviceMesh

`ParallelDims` is the bridge between the four integer degrees and the actual
process groups.

### Validation

`ParallelDims.from_config` (`torchtitan/distributed/parallel_dims.py:148-160`)
copies the parallelism config into a dataclass, and `__post_init__` calls
`_validate` (`parallel_dims.py:165`). The load-bearing check is:

```python
assert dp_replicate * dp_shard * cp * tp * pp == self.world_size  # parallel_dims.py:181
```

For our run: `2 * 2 * 1 * 2 * 1 == 8`. If you mistype a degree, the job dies
here at startup -- a clean startup is itself proof the mesh multiplies to world
size. (`dp_shard=-1` means "infer to fill world size"; `parallel_dims.py:176`.)

### Building the mesh

`build_mesh` (`parallel_dims.py:203`) constructs a flat 1-D world mesh and
*unflattens* it into named axes:

```python
self._world_mesh = init_device_mesh(device_type, (self.world_size,),
                                     mesh_dim_names=("world",))            # :276
dataloading_mesh = unflatten_mesh(self._world_mesh,
                                  ("pp", "batch", "cp", "tp"),
                                  (self.pp, batch, self.cp, self.tp))       # :279
loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")          # :284
```

Two derived degrees matter (`parallel_dims.py:272-273`):

```python
batch = self.dp_replicate * self.dp_shard   # 4: data-loading + global batch
fsdp  = self.dp_shard * self.cp             # 2: the FSDP shard axis
```

`batch` folds replicate and shard together because both parallelize *data*;
`fsdp` folds `dp_shard` with `cp` because both drive the FSDP weight all-gather
and gradient reduce-scatter. `loss` is a *flatten* of `[batch, cp]`, used to
all-reduce the loss (Section 5).

The default (legacy) backend then unflattens the dense parameter mesh as
`["dp_replicate", "fsdp", "tp"]` (`parallel_dims.py:324-329`):

```python
candidate_spmd_dense_axes = ["dp_replicate", "fsdp", "tp"]                  # :324
full_dense_mesh_for_fsdp = unflatten_mesh(
    self._world_mesh, ("pp", "dp_replicate", "fsdp", "tp"),
    (self.pp, self.dp_replicate, fsdp, self.tp))                           # :325
```

This `[dp_replicate, fsdp, tp]` mesh is the one every parameter lives on -- the
probe prints exactly these axis names for every sampled parameter.

### The "fake" backend for degree-1 axes

`unflatten_mesh` (`parallel_dims.py:244`) builds a `backend_override` dict: any
axis whose degree is 1 (and which is not force-kept) gets the `"fake"` backend,
so no real process group is created for it (`parallel_dims.py:254-263`). The
exception is `fsdp`, which `_mesh_exist` keeps real even at size 1
(`parallel_dims.py:186-190`) so `fully_shard` can still install its
`MixedPrecisionPolicy`. In our run `pp=cp=ep=1`, so those axes are fake; only
`dp_replicate`, `fsdp`, and `tp` carry real NCCL groups.

`get_all_one_dimensional_meshes` (`parallel_dims.py:612`) returns exactly the
axes with `ndim == 1 and size > 1` -- the list the probe prints and the set of
process groups whose collective timeout gets synchronized
(`torchtitan/distributed/utils.py:552-555`).

Underneath, `init_device_mesh` and `DeviceMesh._unflatten` are PyTorch upstream:
`_unflatten` slices the flat world rank space into a strided N-D layout and
creates (or fakes) one process group per named axis.

---

## 3. Tensor Parallel on Qwen3

TP is applied by `parallelize_qwen3` (`torchtitan/models/qwen3/parallelize.py:30`).

### The declarative path

Modern torchtitan does not walk the model applying `ParallelStyle` objects.
Instead the model *config* carries a `sharding_config` on each submodule, and
`Module.parallelize` (`torchtitan/protocols/module.py:248`) walks children,
distributes each module's parameters/buffers, and wraps its `forward` with an
input/output redistribution:

```python
def forward_with_redistribution(*args, **kwargs):
    args, kwargs = self._redistribute_inputs(parallel_dims, args, kwargs)   # module.py:290
    outputs = fn(*args, **kwargs)
    return self._redistribute_outputs(parallel_dims, outputs)               # module.py:292
```

For the default backend, `parallelize_qwen3` calls `model.parallelize` only when
TP or EP is enabled (`parallelize.py:57-58`). The sharding annotations are set
up front by `set_qwen3_sharding_config` (`torchtitan/models/qwen3/sharding.py:37`),
which delegates the root modules (`tok_embeddings`, `norm`, `lm_head`) to
`set_decoder_sharding_config` (`torchtitan/models/common/decoder_sharding.py:253`)
and wires each layer via `_set_qwen3_layer_sharding` (`sharding.py:59`).

`enable_sp` flows in from `parallelism.enable_sequence_parallel`
(`torchtitan/models/qwen3/model.py:103`), which defaults to `True`
(`torchtitan/config/configs.py:140`). Sequence parallel is thus a *separate*
toggle that is on by default; it only has an effect when TP > 1. The
`hsdp_tp_no_seq_parallel` sweep config flips it off with
`--parallelism.no-enable-sequence-parallel`.

### The layouts

The reusable layouts live in `decoder_sharding.py`:

- **Colwise** (`colwise_config`, `decoder_sharding.py:65`): weight `Shard(0)`,
  output `Shard(-1)`. Applied to Q/K/V projections and FFN `w1`/`w3`
  (`decoder_sharding.py:186, 248-249`). The output feature dim is split across
  TP ranks.
- **Rowwise** (`rowwise_config`, `decoder_sharding.py:76`): weight `Shard(1)`.
  With SP the output reduce-scatters to `Shard(1)`; without SP it all-reduces
  to replicated. Applied to attention `wo` and FFN `w2`
  (`decoder_sharding.py:187, 250`).
- **Sequence-parallel activation** (`dense_sequence_parallel_placement`,
  `decoder_sharding.py:53`): activations `(batch, seq, hidden)` sharded on the
  sequence dim over `(CP, TP)` via `partition_spec=(DP, (CP, TP), None)`.

The inner attention kernel gets a `LocalMapConfig`
(`set_gqa_inner_attention_local_map`, `decoder_sharding.py:190`): q/k/v arrive as
`(bs, seq, heads, head_dim)` DTensors with heads TP-sharded (`Shard(2)`), and
`local_map` converts them to local tensors before the flash/SDPA kernel and
wraps the outputs back (`decoder_sharding.py:214-229`). This is how GQA works
under TP: heads are split across TP ranks, the kernel runs on local heads, and
the output projection (`wo`, rowwise) reduces across them.

Placements are DTensor `Shard(dim)` / `Replicate()` / `Partial()` composed per
mesh axis. The old `ParallelStyle` machinery is gone; only `NoParallel` survives
as a marker for modules that stay replicated.

### Observed placements

The probe prints, for the landed HSDP+TP mesh:

```
attention.qkv_linear.wqkv.weight  global(4096, 256) local(1024, 256)
    mesh_axes=[dp_replicate, fsdp, tp] placements=[dp_replicate=R, fsdp=_S(0, 2), tp=S(0)]
attention.wo.weight               global(256, 2048) local(128, 1024)
    mesh_axes=[dp_replicate, fsdp, tp] placements=[dp_replicate=R, fsdp=S(0), tp=S(1)]
feed_forward.w1.weight            global(3072, 256) local(768, 256)
    mesh_axes=[dp_replicate, fsdp, tp] placements=[dp_replicate=R, fsdp=_S(0, 2), tp=S(0)]
feed_forward.w2.weight            global(256, 3072) local(128, 1536)
    mesh_axes=[dp_replicate, fsdp, tp] placements=[dp_replicate=R, fsdp=S(0), tp=S(1)]
```

Read `wqkv` (colwise): `tp=S(0)` splits the 4096 output features into 2048 per
TP rank; then FSDP shards that again on dim 0 with `fsdp=_S(0, 2)` (a *strided*
`Shard(0)` -- `_StridedShard` is how FSDP composes its dim-0 shard *on top of*
an existing TP dim-0 shard so the flat parameter still reconstructs correctly).
Net local shape `(1024, 256)` = 4096 / (tp=2 * fsdp=2). `wo` (rowwise) has
`tp=S(1)` on the *input* dim and plain `fsdp=S(0)`, local `(128, 1024)` =
`(256/2, 2048/2)`. This is the concrete composition of TP `Shard` with FSDP
`Shard` over the `[dp_replicate, fsdp, tp]` mesh; `dp_replicate=R` means the
parameter is replicated across the two HSDP replica groups.

`enable_async_tensor_parallel` (`parallelize.py:60-61`, def at
`torchtitan/distributed/tensor_parallel.py:102`) is a separate optimization: it
sets `torch._inductor.config._micro_pipeline_tp = True` to overlap TP
all-gathers/reduce-scatters with the matmuls, and *requires* `torch.compile`
with `model` in the compiled components (`tensor_parallel.py:108-111`) -- which
is why the `hsdp_tp_async_tp` sweep config also passes `--compile.enable`.

---

## 4. HSDP via FSDP2 `fully_shard`

After TP (and AC/compile), `parallelize_qwen3` selects the data-parallel mesh
and applies FSDP2 (`parallelize.py:79-108`):

```python
dp_mesh_names = (["dp_replicate", "fsdp"]
                 if parallel_dims.dp_replicate_enabled else ["fsdp"])       # parallelize.py:80
dp_mesh = parallel_dims.get_mesh(dp_mesh_names)                             # :83
...
apply_fsdp_to_decoder(model, dp_mesh, param_dtype=..., reduce_dtype=...,
                      reshard_after_forward_policy=..., ...)                # :95
```

Because `dp_replicate=2 > 1`, `dp_mesh` is the 2-D `[dp_replicate, fsdp]`
submesh -- this is what makes it HSDP rather than pure FSDP.

`apply_fsdp_to_decoder` (`torchtitan/distributed/fsdp.py:110`) builds a
`MixedPrecisionPolicy` (`fsdp.py:165-169`, bf16 params by default, fp32 reduce
if configured, `cast_forward_inputs=False`), resolves the reshard policy
(`get_fsdp_reshard_after_forward_policy`, `fsdp.py:54`), and calls
`fully_shard` per unit:

- Weight-tied embedding + norm + head as one FSDP unit
  (`fsdp.py:180-192`) so the shared parameter is all-gathered once. Our debug
  model is weight-tied, so there is no separate `lm_head.weight` (the probe
  reports it absent).
- Each transformer block as its own FSDP unit (`fsdp.py:209`, dense fall-through
  `fsdp.py:304-308`).
- The whole model as a root unit (`fsdp.py:310`).

The reshard policy (`fsdp.py:66-78`): `"always"` reshards after every forward
(minimum memory, extra all-gather in backward); `"never"` keeps parameters
gathered (more memory, fewer collectives); `"default"` reshards unless PP is on.
The `hsdp_tp_reshard_always` / `hsdp_tp_reshard_never` sweep configs exercise
both extremes.

Two torchtitan-specific tweaks follow: `disable_fsdp_gradient_division`
(`fsdp.py:27`, called at `fsdp.py:316`) sets FSDP's gradient divide factor to
1.0 because torchtitan scales gradients itself by the global token count
(Section 5); and the HSDP-vs-FSDP log branch (`fsdp.py:319-322`):

```python
if "dp_replicate" in (dp_mesh.mesh_dim_names or ()):
    logger.info("Applied HSDP to the model")   # our run
else:
    logger.info("Applied FSDP to the model")
```

The sweep logs confirm: `fsdp8` prints "Applied FSDP", every `hsdp_*` config
prints "Applied HSDP".

**FSDP2 semantics.** Each parameter is stored sharded on the `fsdp` axis (dim 0,
possibly strided on top of TP as shown above). In forward, the block's
parameters are **all-gathered** across the `fsdp` axis just before use and (if
`reshard_after_forward`) freed after. In backward, gradients are
**reduce-scattered** across the `fsdp` axis. Under HSDP the second axis
`dp_replicate` turns the reduce-scatter into a reduce-scatter *within* each
shard group plus an **all-reduce across** replica groups -- i.e. the replica
groups keep identical sharded parameters and average gradients across
themselves. The memory win is visible in the sweep: HSDP+TP peaks at ~2.4 GiB
vs ~4.4 GiB for pure FSDP on this tiny model, because TP further splits each
parameter.

---

## 5. The training step and its collectives

`Trainer.train_step` (`torchtitan/trainer.py:805`) drives one optimizer step.
It first fetches the microbatches and computes a **global valid-token count** by
all-reducing over the `batch` mesh (`trainer.py:830-834`):

```python
batch_mesh = parallel_dims.get_mesh("batch")
global_valid_tokens = dist_utils.dist_sum(local_valid_tokens.to(self.device), batch_mesh)
```

That token count is threaded into `forward_backward_step` (`trainer.py:731`),
which runs the model, computes the loss (already divided by
`global_valid_tokens` inside the loss fn), and calls `loss.backward()`
(`trainer.py:756-762`). This is why FSDP's own gradient division is disabled
(Section 4): torchtitan divides by the true global token count, not by the DP
degree.

### Data sharding

Each rank reads a disjoint slice of data indexed by its flat `batch` rank
(`trainer.py:279-283`):

```python
batch_mesh = parallel_dims.get_mesh("batch")
batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
```

passed to the dataloader as `dp_world_size=batch_degree, dp_rank=batch_rank`
(`trainer.py:530-531`), which forwards them to
`split_dataset_by_node(ds, dp_rank, dp_world_size)`
(`torchtitan/hf_datasets/text_datasets.py:100`). With `batch=4`, the 8 GPUs form
4 data-parallel groups of 2 (the TP pair shares a data slice), so each of the 4
`batch` ranks reads a distinct shard of `fineweb_test`.

### Loss all-reduce

After the accumulation loop, the loss is reduced over the `loss` mesh
(`trainer.py:888-911`):

```python
loss_mesh = parallel_dims.get_optional_mesh("loss")
global_avg_loss = dist_utils.dist_sum(loss, loss_mesh)
global_max_loss = dist_utils.dist_max(local_avg_loss, loss_mesh)
```

`dist_sum` / `dist_max` are thin wrappers over `_dist_reduce`
(`torchtitan/distributed/utils.py:78`, called via `:133`/`:123`), which uses
functional collectives (`funcol.all_reduce`, `utils.py:115/119`). A subtle but
important detail (`utils.py:95-111`): if the loss is a DTensor, `_dist_reduce`
calls `to_local()` rather than `full_tensor()`, because `full_tensor()` on a
loss that is Partial over both `dp_shard` and `cp` would issue *two* all-reduces
and could change the reduction order; `to_local()` plus one explicit all-reduce
over the `loss` mesh keeps loss parity deterministic. The averaging math is
documented inline (`trainer.py:892-900`): `global_avg_loss` is a SUM of
already-normalized local losses, so it equals `sum(local_loss_sum) /
global_valid_tokens`.

---

## 6. Gradient norm across a two-axis sharded parameter

Back in `train_step`, gradient clipping calls `clip_grad_norm_`
(`trainer.py:866-872` -> `torchtitan/distributed/utils.py:561`):

```python
total_norm = torch.nn.utils.get_total_norm(grads, norm_type, ...)   # utils.py:614
...
if isinstance(total_norm, DTensor):
    total_norm = total_norm.full_tensor()                           # utils.py:627
```

`get_total_norm` computes a p-norm over sharded gradients. Because each
parameter's gradient is sharded on both `fsdp` and `tp` (Section 3), the per-rank
partial norms are a DTensor whose placement on each sharded axis is
`_NormPartial` (an upstream PyTorch placement that knows how to combine partial
p-norms: raise to the p-th power, sum across the axis, take the p-th root).
`full_tensor()` then reduces across every `_NormPartial` axis to produce the
true scalar norm.

The probe reproduces this in miniature over the `[dp_replicate, fsdp, tp]` mesh:

```
vector_norm returned type=DTensor
placements (partial, pre-reduction): [_NormPartial(2.0), _NormPartial(2.0), _NormPartial(2.0)]
mesh axes: [dp_replicate, fsdp, tp]
full_tensor() reduces over 3 axes -> scalar norm = 102.058807
```

and PyTorch emits the tell-tale warning during `full_tensor()`:

```
While redistributing from (_NormPartial(2.0), _NormPartial(2.0), _NormPartial(2.0))
to (Replicate(), Replicate(), Replicate()), 3 sequential all_reduce operations
will be performed. ... To optimize, flatten mesh dimensions
["dp_replicate", "fsdp", "tp"] so DTensor can use a single operation instead.
```

That is the concrete cost of a norm over a multi-axis sharded parameter: one
all-reduce per sharded axis, issued sequentially. It is correct (the p-norm
composition is associative across axes) but suboptimal in latency -- hence the
"flatten mesh dimensions" hint. If PP were enabled, `clip_grad_norm_` would then
all-reduce once more across the `pp` mesh (`utils.py:628-634`); the EP case is
handled separately by `_clip_grad_norm_with_ep` (`utils.py:642`).

---

## 7. Reading the sweep results

`sweep.sh` runs eight 8-GPU configurations against `qwen3_debugmodel_fineweb`
with `--debug.seed=42 --debug.deterministic --training.steps=10`, tees each to
`results/<name>.log`, and `summarize_sweep.py` extracts the comparison table to
`results/summary.md`. See that file for the exact captured numbers; the
qualitative findings:

- **Mesh shape is confirmed per config.** Every log's
  `Building device mesh with parallelism:` line and its `Applied HSDP`/`Applied
  FSDP` line match the requested degrees: `fsdp8` -> "Applied FSDP",
  `hsdp_2x4` / `hsdp_tp_2x2x2` / etc. -> "Applied HSDP".
- **Memory drops as you add TP.** On this tiny model, pure FSDP (`fsdp8`) and
  HSDP (`hsdp_2x4`) peak around ~4.4 GiB, while HSDP+TP (`hsdp_tp_2x2x2`) peaks
  around ~2.4 GiB -- TP splits each parameter and its activations further.
- **Throughput drops as you add TP** *for a model this small*: HSDP+TP runs at a
  fraction of the tps/tflops of pure FSDP, because the extra TP all-gather /
  reduce-scatter per layer is pure overhead when the matmuls are tiny. On a
  real-size model the tradeoff flips; the debug model is for correctness and
  mechanism, not for a throughput verdict.
- **Reshard ablation.** `hsdp_tp_reshard_never` keeps parameters gathered after
  forward (fewer backward all-gathers, higher peak memory) versus
  `hsdp_tp_reshard_always` (minimum memory, extra all-gather); the memory delta
  is the readable signal at this scale.
- **Async-TP** (`hsdp_tp_async_tp`, with `--compile.enable`) enables
  micro-pipelined TP collectives; on the debug model the effect is small but the
  config demonstrates the required compile coupling.

### Numeric equivalence: why every mesh reshape drifts

`scripts/loss_compare.py` compares loss and grad_norm across two runs with
`--debug.deterministic` always on. This experiment ran two pairs (full data in
`results/loss_compare_findings.md`), and the empirical result is stronger than
"close but not identical": **no mesh pair in this sweep is bitwise identical,
and each differs for a concrete, mechanistic reason.**

**Pair A -- `fsdp8` vs `hsdp_tp_2x2x2`** (`--assert-equal` FAILS at step 1:
7.88904 vs 8.06982). Two independent causes stack:

1. The `batch` axis is `dp_replicate * dp_shard` (Section 2), and TP does *not*
   widen it. So `tp=2` shrinks `batch` from 8 to 4, taking global batch from 64
   to 32. A different global batch is simply a different optimization step.
2. Even at equal batch, TP adds the `tp` axis to the grad-norm `_NormPartial`
   reduction (Section 6) and adds TP activation collectives, changing float
   reduction order.

**Pair B -- `fsdp8` vs `hsdp_2x4`** (`--assert-equal` FAILS at step 1: 7.88904
vs 8.35181). This pair is the interesting control: both have `tp=1` and the
*same* global batch 64, so the reduction structure is identical. It still
diverges, because **data assignment changes.** Data is sharded by the flat
`batch`-mesh rank (Section 5, `text_datasets.py:100`). Refactoring the DP degrees
from `[dp_replicate=1, dp_shard=8]` to `[dp_replicate=2, dp_shard=4]` re-maps
each GPU's flat `batch` coordinate, so each rank consumes a different slice of
the FineWeb stream -> a different minibatch -> a different loss from step 1.

So the honest statement is: reshaping the DP factorization already breaks bitwise
equality via data re-assignment, and adding TP breaks it further via a smaller
global batch and a different reduction order. `docs/converging.md` frames the
general policy -- exact reproducibility across different distributed settings is
not a goal; convergence to the same neighbourhood is. All eight configs here do
converge together (loss[10] between 3.79 and 3.83). Per AGENTS.md we never use
`--debug.deterministic_warn_only`; `loss_compare.py` keeps determinism on and
reports the measured deltas rather than silencing them.

---

## 8. Reproduce

```bash
# 0. One-time online prefetch of the FineWeb-edu slice (only network step).
python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py

# 1. The landed HSDP+TP training run (offline).
experiments/qwen3_fineweb_hsdp_tp/run.sh

# 2. Read-only mesh / DTensor / _NormPartial introspection (8 GPUs).
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HF_HOME="$PWD/.hf_cache" PYTHONPATH="$PWD" \
  torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    experiments/qwen3_fineweb_hsdp_tp/probe_parallelism.py \
    --module qwen3 --config qwen3_debugmodel_fineweb \
    --parallelism.data_parallel_replicate_degree=2 \
    --parallelism.data_parallel_shard_degree=2 \
    --parallelism.tensor_parallel_degree=2

# 3. Mesh sweep + ablations + summary table (writes results/, gitignored).
experiments/qwen3_fineweb_hsdp_tp/sweep.sh

# 4. Numeric equivalence check for one config pair (writes results/loss_compare_*).
# Pass '.' for both commits so no git checkout happens (keeps the untracked
# experiment data/venv in place). --checkpoint.no-enable avoids reloading a
# stale checkpoint written under a different mesh; --no-seed-checkpoint skips
# seed-checkpoint creation. --assert-equal reports the delta (never use
# --debug.deterministic_warn_only).
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HF_HOME="$PWD/.hf_cache" CC=/usr/bin/gcc CXX=/usr/bin/g++
python scripts/loss_compare.py . . \
  --baseline-module=qwen3 --baseline-config=qwen3_debugmodel_fineweb \
  --test-module=qwen3 --test-config=qwen3_debugmodel_fineweb \
  --baseline-options='--parallelism.data_parallel_shard_degree=8 --checkpoint.no-enable' \
  --test-options='--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --checkpoint.no-enable' \
  --steps 10 --no-seed-checkpoint --assert-equal \
  --output-folder=experiments/qwen3_fineweb_hsdp_tp/results/loss_compare_fsdp8_vs_hsdp_tp
```

Everything in this article is grounded in the source at the cited lines and in
the captured `results/`. When the code moves, re-run the probe and the sweep --
they regenerate the observations verbatim.
