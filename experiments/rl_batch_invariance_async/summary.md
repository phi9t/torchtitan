# RL Batch-Invariance Summary

| Metric | Arm | First | Last | Min | Max | Mean | Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bit_wise/logprob_diff/max` | `bi` | 0 @ 1 | 0 @ 2 | 0 | 0 | 0 | 2 |
| `bit_wise/logprob_diff/max` | `no_bi` | 0.114651 @ 1 | 0.0996521 @ 2 | 0.0996521 | 0.114651 | 0.107151 | 2 |
| `bit_wise/logprob_diff/mean` | `bi` | 0 @ 1 | 0 @ 2 | 0 | 0 | 0 | 2 |
| `bit_wise/logprob_diff/mean` | `no_bi` | 0.0124135 @ 1 | 0.000402601 @ 2 | 0.000402601 | 0.0124135 | 0.00640805 | 2 |
| `bit_wise/ratio_tokens_different/mean` | `bi` | 0 @ 1 | 0 @ 2 | 0 | 0 | 0 | 2 |
| `bit_wise/ratio_tokens_different/mean` | `no_bi` | 1 @ 1 | 1 @ 2 | 1 | 1 | 1 | 2 |
| `rollout_reward/_mean` | `bi` | 0 @ 1 | 0 @ 2 | 0 | 0 | 0 | 2 |
| `rollout_reward/_mean` | `no_bi` | 0 @ 1 | 0 @ 2 | 0 | 0 | 0 | 2 |
| `validation_reward/_mean` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation_reward/_mean` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation_reward/_max` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation_reward/_max` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation/reward/_mean` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation/reward/_mean` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation/reward/_max` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation/reward/_max` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `perf/trainer/tokens_per_second_full_step` | `bi` | 0.389898 @ 1 | 3.92152 @ 2 | 0.389898 | 3.92152 | 2.15571 | 2 |
| `perf/trainer/tokens_per_second_full_step` | `no_bi` | 0.284379 @ 1 | 0.84602 @ 2 | 0.284379 | 0.84602 | 0.5652 | 2 |
| `perf/trainer/tokens_per_second_fwd_bwd` | `bi` | 0.716713 @ 1 | 26.9058 @ 2 | 0.716713 | 26.9058 | 13.8112 | 2 |
| `perf/trainer/tokens_per_second_fwd_bwd` | `no_bi` | 0.542733 @ 1 | 3.14866 @ 2 | 0.542733 | 3.14866 | 1.84569 | 2 |
| `perf/trainer/full_step_throughput` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `perf/trainer/full_step_throughput` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `perf/trainer/tokens_per_second` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `perf/trainer/tokens_per_second` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `generator/inter_token_latency_ms/mean` | `bi` | 192.288 @ 1 | 99.0855 @ 2 | 99.0855 | 192.288 | 145.687 | 2 |
| `generator/inter_token_latency_ms/mean` | `no_bi` | 111.511 @ 1 | 67.3516 @ 2 | 67.3516 | 111.511 | 89.4312 | 2 |
| `generator/decode_time_ms/mean` | `bi` | 2307.45 @ 1 | 1189.03 @ 2 | 1189.03 | 2307.45 | 1748.24 | 2 |
| `generator/decode_time_ms/mean` | `no_bi` | 1338.13 @ 1 | 808.219 @ 2 | 808.219 | 1338.13 | 1073.17 | 2 |
| `generator/queue_time_ms/mean` | `bi` | 0.290988 @ 1 | 0.250842 @ 2 | 0.250842 | 0.290988 | 0.270915 | 2 |
| `generator/queue_time_ms/mean` | `no_bi` | 0.349745 @ 1 | 0.264285 @ 2 | 0.264285 | 0.349745 | 0.307015 | 2 |
| `timing/step/total` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `timing/step/total` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
