# RL Batch-Invariance Summary

| Metric | Arm | First | Last | Min | Max | Mean | Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bit_wise/logprob_diff/max` | `bi` | 0 @ 1 | 0.0755811 @ 2 | 0 | 0.11875 | 0.0647769 | 3 |
| `bit_wise/logprob_diff/max` | `no_bi` | 0.323327 @ 1 | 0.465602 @ 150 | 0.0676317 | 2.72901 | 0.698053 | 152 |
| `bit_wise/logprob_diff/mean` | `bi` | 0 @ 1 | -0.0009266 @ 2 | -0.0166257 | 0 | -0.00585077 | 3 |
| `bit_wise/logprob_diff/mean` | `no_bi` | -9.81366e-05 @ 1 | -0.000143948 @ 150 | -0.00583036 | 0.00288853 | -0.000125723 | 152 |
| `bit_wise/ratio_tokens_different/mean` | `bi` | 0 @ 1 | 1 @ 2 | 0 | 1 | 0.666667 | 3 |
| `bit_wise/ratio_tokens_different/mean` | `no_bi` | 0.69722 @ 1 | 0.485288 @ 150 | 0.40295 | 1 | 0.526288 | 152 |
| `rollout_reward/_mean` | `bi` | 0.0520833 @ 1 | 0 @ 2 | 0 | 0.0520833 | 0.0173611 | 3 |
| `rollout_reward/_mean` | `no_bi` | 0.0803571 @ 1 | 0.4625 @ 150 | 0 | 0.694444 | 0.332471 | 152 |
| `validation_reward/_mean` | `bi` | 0 @ 0 | 0 @ 0 | 0 | 0 | 0 | 1 |
| `validation_reward/_mean` | `no_bi` | 0.0333333 @ 0 | 0.266667 @ 150 | 0.0333333 | 0.266667 | 0.15 | 2 |
| `validation_reward/_max` | `bi` | 0 @ 0 | 0 @ 0 | 0 | 0 | 0 | 1 |
| `validation_reward/_max` | `no_bi` | 1 @ 0 | 1 @ 150 | 1 | 1 | 1 | 2 |
| `validation/reward/_mean` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation/reward/_mean` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation/reward/_max` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `validation/reward/_max` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `perf/trainer/tokens_per_second_full_step` | `bi` | 25.8702 @ 1 | 3.30862 @ 2 | 0.503274 | 25.8702 | 9.89403 | 3 |
| `perf/trainer/tokens_per_second_full_step` | `no_bi` | 169.689 @ 1 | 6690.09 @ 150 | 0.900662 | 11594.9 | 2704.8 | 152 |
| `perf/trainer/tokens_per_second_fwd_bwd` | `bi` | 2923.6 @ 1 | 26.6806 @ 2 | 0.530205 | 2923.6 | 983.602 | 3 |
| `perf/trainer/tokens_per_second_fwd_bwd` | `no_bi` | 5887.13 @ 1 | 12621.7 @ 150 | 2.11804 | 13901.4 | 11959.5 | 152 |
| `perf/trainer/full_step_throughput` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `perf/trainer/full_step_throughput` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `perf/trainer/tokens_per_second` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `perf/trainer/tokens_per_second` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `generator/inter_token_latency_ms/mean` | `bi` | 124.743 @ 1 | 102.711 @ 2 | 102.711 | 124.743 | 114.99 | 3 |
| `generator/inter_token_latency_ms/mean` | `no_bi` | 70.916 @ 1 | 77.8237 @ 150 | 66.7186 | 84.3936 | 73.9502 | 152 |
| `generator/decode_time_ms/mean` | `bi` | 110128 @ 1 | 1232.53 @ 2 | 1232.53 | 110128 | 37590.3 | 3 |
| `generator/decode_time_ms/mean` | `no_bi` | 53109 @ 1 | 237704 @ 150 | 817.757 | 326167 | 208886 | 152 |
| `generator/queue_time_ms/mean` | `bi` | 1.805 @ 1 | 0.290572 @ 2 | 0.290572 | 1.805 | 0.800281 | 3 |
| `generator/queue_time_ms/mean` | `no_bi` | 72.4056 @ 1 | 2116.37 @ 150 | 0.310376 | 2523.73 | 939.427 | 152 |
| `timing/step/total` | `bi` | n/a | n/a | n/a | n/a | n/a | 0 |
| `timing/step/total` | `no_bi` | n/a | n/a | n/a | n/a | n/a | 0 |
