# Countdown Reduced Pilot Before Full Run

The Countdown search-distillation experiment will use a reduced pilot before
the full-size run: Qwen3-1.7B remains the evidence model, calibration must hit
the target pass@1/pass@32 band, and the first non-smoke comparison trains raw,
hindsight, and curriculum arms under the same sequential 8-GPU envelope. This
trades slower time-to-full-result for a cheaper interpretability gate: the full
pilot is justified only after the reduced pilot shows a meaningful dev pass@1
lift without materially degrading pass@32.

Adapter evaluation should use vLLM LoRA runtime support rather than merged model
copies so the adapter pass@k path matches base rollout collection. The reduced
pilot aborts before adapter training if calibration misses the target band or
matched coverage is too low; the result closes with both a current `RESULTS.md`
and an immutable timestamped report, and the full-pilot gate is measured on the
base-elicitable dev bucket while still reporting all-dev metrics.
