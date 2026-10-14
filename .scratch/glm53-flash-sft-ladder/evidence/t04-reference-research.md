# T04 primary-source mHC reference notes

Scope: primary-source source-code check only. Files were downloaded as text under
`/data02/home/philip.yang/devx/tmp/t04_src/`; no reference code was imported or
executed.

## DeepSeek-V4 / DeepSeek-V4-Flash official inference

Primary source found: Hugging Face repo `deepseek-ai/DeepSeek-V4-Flash`,
`inference/model.py` and `inference/kernel.py`, resolved with `X-Repo-Commit:
60d8d70770c6776ff598c94bb586a859a38244f1`.

URLs:

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py

`deepseek-ai/DeepSeek-V4` itself returned HTTP 401 for
`/resolve/main/inference/model.py` from this unauthenticated environment, so the
available official source here is the V4-Flash repo.

Exact model.py lines:

```text
78:     hc_mult: int = 4
79:     hc_sinkhorn_iters: int = 20
80:     hc_eps: float = 1e-6
```

```text
648:     """Transformer block with Hyper-Connections (HC) mixing.
649:     Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
650:     hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
651:     hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
```

```text
673:     def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
674:         # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
675:         shape, dtype = x.size(), x.dtype
676:         x = x.flatten(2).float()
677:         rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
678:         mixes = F.linear(x, hc_fn) * rsqrt
679:         pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
680:         y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
681:         return y.to(dtype), post, comb
```

```text
683:     def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
684:         # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
685:         y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
686:         return y.type_as(x)
```

```text
688:     def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
689:         residual = x
690:         x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
691:         x = self.attn_norm(x)
692:         x = self.attn(x, start_pos)
693:         x = self.hc_post(x, residual, post, comb)
...
695:         residual = x
696:         x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
697:         x = self.ffn_norm(x)
698:         x = self.ffn(x, input_ids)
699:         x = self.hc_post(x, residual, post, comb)
```

```text
803:         h = self.embed(input_ids)
804:         # Expand to hc_mult copies for Hyper-Connections
805:         h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
```

Exact kernel.py lines:

```text
379:         mixes: T.Tensor[(n, mix_hc), FP32],
380:         hc_scale: T.Tensor[(3,), FP32],
381:         hc_base: T.Tensor[(mix_hc,), FP32],
382:         pre: T.Tensor[(n, hc), FP32],
383:         post: T.Tensor[(n, hc), FP32],
384:         comb: T.Tensor[(n, hc, hc), FP32],
```

```text
391:             for j in T.Parallel(hc):
392:                 pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
393:             for j in T.Parallel(hc):
394:                 post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
395:             for j, k in T.Parallel(hc, hc):
396:                 comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]
```

```text
401:             # comb = comb.softmax(-1) + eps
402:             row_max = T.alloc_fragment(hc, FP32)
403:             T.reduce_max(comb_frag, row_max, dim=1)
404:             for j, k in T.Parallel(hc, hc):
405:                 comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
406:             T.reduce_sum(comb_frag, row_sum, dim=1)
407:             for j, k in T.Parallel(hc, hc):
408:                 comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps
409:
410:             # comb = comb / (comb.sum(-2) + eps)
411:             T.reduce_sum(comb_frag, col_sum, dim=0)
412:             for j, k in T.Parallel(hc, hc):
413:                 comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)
414:
415:             for _ in T.serial(sinkhorn_iters - 1):
416:                 # comb = comb / (comb.sum(-1) + eps)
417:                 T.reduce_sum(comb_frag, row_sum, dim=1)
418:                 for j, k in T.Parallel(hc, hc):
419:                     comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
420:                 # comb = comb / (comb.sum(-2) + eps)
421:                 T.reduce_sum(comb_frag, col_sum, dim=0)
422:                 for j, k in T.Parallel(hc, hc):
423:                     comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)
```

```text
430: def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
431:     b, s, _ = mixes.size()
432:     pre = mixes.new_empty(b, s, hc_mult)
433:     post = mixes.new_empty(b, s, hc_mult)
434:     comb = mixes.new_empty(b, s, hc_mult, hc_mult)
435:     kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
436:     kernel(mixes.view(-1, (2 + hc_mult) * hc_mult), hc_scale, hc_base,
437:            pre.view(-1, hc_mult), post.view(-1, hc_mult), comb.view(-1, hc_mult, hc_mult))
438:     return pre, post, comb
```

Formula reading: with `residual: [b,s,hc,d]` and `comb: [b,s,hc,hc]`,
`comb.unsqueeze(-1) * residual.unsqueeze(-2)` has shape `[b,s,i,j,d]` if
`residual` indexes source stream `i` and `comb[i,j]` maps source `i` to target
`j`. Summing `dim=2` therefore computes `sum_i comb[i,j] * residual_i`, i.e.
cross-stream mixing into target stream `j`.

## GLM-5.3-Flash Transformers `glm5_next`

Primary source: Hugging Face Transformers,
`src/transformers/models/glm5_next/modeling_glm5_next.py`, commit
`e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`.

URL:

- https://github.com/huggingface/transformers/blob/e0eb2810ac1b2caf6d082f247d8d37162ecf67a0/src/transformers/models/glm5_next/modeling_glm5_next.py

Model repo metadata checked: `zai-org/GLM-5.3-Flash`, SHA
`91b69215804f2d2036a8867f6ce44848e278dcda`, declares
`model_type: glm5_next`, but does not ship custom `.py` siblings; the source
implementation is in Transformers.

Exact lines for mHC shape and formulas:

```text
223:     When using mHC, each token is projected onto `hc_mult` streams, so the shape of decoder layer inputs changes from
224:     [batch_size, sequence_length, hidden_size] to [batch_size, sequence_length, hc_mult, hidden_size].
```

Line 231 defines the diagram notation as
`B = batch_size, S = seq_length, N = hc_mult, D = hidden_size`; the diagram
then labels the mHC stream tensor as `[B, S, N, D]` and the generated weights
as `[B, S, N]`, `[B, S, N]`, and `[B, S, N, N]`.

```text
299:         # Flatten and norm the hidden streams
300:         flattened = hidden_streams.view(batch_size, seq_len, -1).float()
301:         flattened = self.input_norm(flattened)
302:         # Mix the streams together to infer the weight coefficients
303:         flattened = F.linear(flattened, self.fn.float())
304:         # Split the weight coefficients
305:         pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
306:         pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
307:         pre_scale, post_scale, comb_scale = self.scale.unbind(0)
308:
309:         comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)  # these are matrix weights, unlike pre or post
310:         comb_b = comb_b.view(hc, hc)
```

```text
312:         # All weights are computed with a one layer perceptron. For pre and post, this is it.
313:         pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
314:         post = 2 * torch.sigmoid(post_w * post_scale + post_b)
315:         comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps
```

```text
317:         # The comb weight is a bit different: it dictates how the input streams (In) are added to the output streams
318:         # (Out) in this way: Mixed = In @ Comb + Out. To make sure the norm of "Mixed" does not blow up, we constrain
319:         # the comb weight to be doubly-stochastic (ie. its rows and columns must sum to 1) with a few iterations of the
320:         # Sinkhorn-Knopp algorithm, which iteratively normalizes the rows and columns to sum to 1.
321:         comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
322:         for _ in range(self.hc_sinkhorn_iters - 1):
323:             comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
324:             comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
```

```text
326:         # Since "pre" is meant to be used with the input streams (available here as `hidden_streams`), we collapse the
327:         # streams here and return `collapsed` tensor, which will be the input for the next attention or MLP block.
328:         collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
329:         return post, comb, collapsed
```

Exact decoder application lines:

```text
1335:         hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
1336:             comb.to(dtype).transpose(-1, -2), residual
1337:         )
```

The same decoder formula is repeated for the FFN path at lines 1343-1345.

Exact initial/final stream handling:

```text
1493:         hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
```

```text
332: class Glm5NextTextHyperHead(nn.Module):
333:     """Final GLM-5.3-Flash HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""
334:
335:     def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
336:         return hidden_streams.mean(dim=2)
```

```text
1509:         hidden_states = self.norm(self.hc_head(hidden_states))
```

Formula reading: GLM computes `post_j * block_output + sum_i comb[i,j] *
residual_i`, because `torch.matmul(comb.transpose(-1, -2), residual)` treats
`residual` as `[B,S,N,D]` and multiplies by `comb^T`.

## Uncertainty

- I found and quoted the official `deepseek-ai/DeepSeek-V4-Flash` inference
  source. The non-Flash `deepseek-ai/DeepSeek-V4` HF repo was not accessible
  unauthenticated from this environment, and the public GitHub repos checked
  (`DeepSeek-V3`, `DeepSeek-V3.2`) do not contain mHC/HcPost.
- The DeepSeek and GLM post formulas agree on the source-to-target stream
  interpretation when the tensor layout is `[B,S,hc,d]`: `sum_i comb[i,j] *
  residual_i`.
