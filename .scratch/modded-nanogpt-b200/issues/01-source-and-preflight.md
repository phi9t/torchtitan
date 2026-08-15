# Source checkout and B200 preflight

Type: task
Status: ready-for-agent
Blocked by: -

## Requirement

Create the first runnable slice of the Modded NanoGPT B200 benchmark harness:
pin or verify the upstream source and fail loudly before any long run if the
host/runtime cannot support the benchmark.

Use `experiments/modded_nanogpt_b200/benchmark_design.md` as the design source.

## Scope

Allowed:

- add gitignored experiment-local `sources/`, `data/`, and `results/` paths;
- add a source-fetch or source-verify script;
- add a preflight script;
- add minimal README usage for the preflight;
- run cheap host-side or rootfs preflight checks.

Excluded:

- no full training run;
- no FineWeb bulk download;
- no upstream source edits;
- no TorchTitan core code edits;
- no optimization ablations;
- no commit, push, or pull request unless explicitly authorized.

## Behavior

The preflight must verify or record:

- upstream source path and commit, expected
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`;
- 8 visible CUDA devices;
- every visible device matches the expected B200 device class unless overridden;
- PyTorch, CUDA, Triton, and NCCL version evidence where available;
- BF16 allocation smoke;
- FP8 tensor allocation and `torch._scaled_mm` smoke;
- Flash Attention 3 import and tiny varlen smoke, or a clear unsupported status
  for an explicit fallback arm;
- 8-rank NCCL all-reduce smoke, unless the script is in a documented dry-run
  mode;
- repo-local cache and compiler settings.

## Verification Evidence

- Run the preflight on the current host or document the exact blocker.
- Show that source verification fails on a wrong commit, or cover it with a
  focused unit/helper test.
- Inspect the diff to confirm no generated source, data, logs, or caches are
  tracked.
