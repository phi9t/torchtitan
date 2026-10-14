# T04 report: DeepSeek-V4 mHC HcPost axis

Verdict: **CONFIRMED**

claim_label: `deterministic_comparison`

## Finding

TorchTitan `HcPost.forward` is wrong for its flattened `[T, hc, D]` layout.
The official DeepSeek-V4-Flash inference source uses the same text expression,
but in a batched `[b, s, hc, d]` layout where summing `dim=2` is the source HC
stream axis. After TorchTitan folds batch and sequence into `T`, `dim=2` is the
target HC stream axis, so the current code computes:

```text
y_i = post_i * x + residual_i * sum_j comb[i, j]
```

instead of the reference formula:

```text
y_j = post_j * x + sum_i comb[i, j] * residual_i
```

## Primary sources

DeepSeek-V4-Flash official HF inference source:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py

```text
673:     def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
674:         # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
...
680:         y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
681:         return y.to(dtype), post, comb

683:     def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
684:         # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
685:         y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
686:         return y.type_as(x)
```

GLM-5.3-Flash Transformers source:
https://github.com/huggingface/transformers/blob/e0eb2810ac1b2caf6d082f247d8d37162ecf67a0/src/transformers/models/glm5_next/modeling_glm5_next.py

```text
223:     When using mHC, each token is projected onto `hc_mult` streams, so the shape of decoder layer inputs changes from
224:     [batch_size, sequence_length, hidden_size] to [batch_size, sequence_length, hc_mult, hidden_size].
...
317:         # The comb weight is a bit different: it dictates how the input streams (In) are added to the output streams
318:         # (Out) in this way: Mixed = In @ Comb + Out.
...
328:         collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
329:         return post, comb, collapsed

1335:         hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
1336:             comb.to(dtype).transpose(-1, -2), residual
1337:         )
```

The GLM source is an independent cross-check of the source-to-target stream
orientation: `comb.transpose(-1, -2) @ residual`.

Source downloads were kept as inert `.py` files under
`/data02/home/philip.yang/devx/tmp/t04_src/`; no reference source was imported
or executed.

## Repo history and callsites

`git log -p -- torchtitan/models/deepseek_v4/mhc.py` shows:

- `0ff2464637d0947d9f30b7da0dee3973a08b8f1a` introduced `mhc.py` with
  `HcPre`/`HcPost` documented as `[T, hc_mult, D]` and the current
  `torch.sum(..., dim=2)` expression.
- `6f22255756ca247f343f2f517671c61f0a407e9b` only changed the `Module`
  import path during repo reorganization. No batched-to-flattened refactor
  appears after introduction.

`rg -n "comb.unsqueeze" torchtitan` finds only
`torchtitan/models/deepseek_v4/mhc.py`; I found no Kimi K3 or experiment copy of
this formula.

Blast radius: all `DeepSeekV4TransformerBlock` instances use one shared
`HcPost` instance twice per layer, after attention and after FFN. Affected model
flavors/configs are:

- `debugmodel`
- `deepseek_v4_flash`
- `deepseek_v4_pro`
- MTP debug/config paths whose `MTPBlock` inherits `DeepSeekV4TransformerBlock`

Checkpoint and state-dict loading should not change: the minimal fix changes
only tensor algebra in `HcPost.forward`; no parameters, keys, shapes, or config
fields change. Existing DeepSeek-V4 numerical results and any tests that compare
losses/logits would change and should be regenerated after applying the fix.

## Test and verification

Added test:

```text
tests/unit_tests/test_deepseek_v4_mhc_post.py
```

It compares the real repo `HcPost` against a local transcription of the
reference formula on CPU random tensors in TorchTitan's `[T, hc, D]` shape. The
test is marked `xfail(strict=True)` because the bug is confirmed and core model
code was excluded from this ticket.

Required command:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_deepseek_v4_mhc_post.py'
```

Result:

```text
x                                                                        [100%]
1 xfailed, 14 warnings in 4.05s
```

Lint note: direct `ruff` was not installed in the rootfs (`bash: ruff: command
not found`). `uv run ruff --version` started creating a `.venv` but did not
finish promptly, so I stopped it and removed the transient `.venv`.

## Prepared fix

Prepared but not applied:

```text
.scratch/glm53-flash-sft-ladder/evidence/t04-hcpost-fix.patch
```

Patch check:

```bash
git apply --check .scratch/glm53-flash-sft-ladder/evidence/t04-hcpost-fix.patch
```

passed. The minimal fix is:

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(
    comb.transpose(-1, -2), residual
)
```
