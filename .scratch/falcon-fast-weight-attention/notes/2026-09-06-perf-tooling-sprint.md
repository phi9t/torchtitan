# Perf / MLSYS tooling sprint (2026-09-06)

Not ticket 08. Not a hero. Claim label: `smoke`.

One science-shape base run is the attach target. Tool variations are
the experiment, not mixer knobs. Ticket 08 stays frozen on A5.

## Base profile (V1)

Frozen A5 knobs on the ticket-05/06 science shell:

- mixer=`falcon`, variant=`falcon1a`, alignment=`delayed`, phi=`l2`
- 4 layers, hidden 256, 8 heads, head_dim 32, SwiGLU 1024
- seq_len 512, local_batch_size 32, **16,384 tok/step**, fp32, 1x B200
- FineWeb-10B GPT-2 `.bin`, no checkpoints
- 20,000 steps (~328M tokens). Prior A2 wall time was ~57 ms/step
  including val, so this should stay up ~15-20 min for attach windows.
- Eager, no `torch.compile`, no TensorBoard on V1 (ablation-faithful)
- `log_freq=10` so step/tps/tflops join to attach windows

Driver: `experiments/falcon/run.sh perf --arm A5 --steps 20000`

## Variations

| ID | GPU | What | Why |
| --- | --- | --- | --- |
| V1 | 0 | A5 science 20k, eager | Live-sample NVIDIA + eBPF + Trainer tps |
| V2 | 1 | Same shape, ~200 steps under `nsys` | Launch-wrap timeline after DCGM pause on that GPU |
| V3 | 2 | `falcon_tiny_overfit` stretched to many steps | live-inject (py-spy, memray) without risking V1 |

Do not ncu, Fil-wrap, or memray-attach V1.

## Tool waves on V1 (after first logged step)

1. USE + `nvidia-smi dmon` / `pmon` / `gpm` / topo
2. `dcgmi dmon` 150/155/203/204/252, then profiling 1002/1003
3. `attach_live.sh` (own-process `perf cpu-clock:u`)
4. docker-priv tracepoints
5. Host `/proc` I/O vs `iostat` (FineWeb page cache)

V3 only: py-spy `top`, tracemalloc if we add a wrap, memray attach last.

## Success

Written: `experiments/falcon/results/tooling/REPORT.md`.

- V1 stayed `R` after NVML, DCGM A.0, `perf`, docker-priv tracepoints,
  and host py-spy.
- DCGM 1003 occupancy is real on this B200 (0.33 on A5 fp32).
- Host `nsys` around `enter_rootfs.sh` does not see CUDA.
- In-rootfs `nsys` works after binding host `/sys`. A5 20-step report
  has `cutlass3x_sm100_simt_sgemm_*` kernels.
- In-rootfs `ncu` wrote a one-kernel report.
- py-spy Python-only works host→bwrap; `--native` is `UNW_EBADREG`.
- memray / Scalene / Fil installed in rootfs `/usr/bin/python`.
- BPF loads from inside `--profile` via `run_privileged.sh`
  (bpftrace, BCC cachestat/runqlat/profile). Proof:
  `experiments/falcon/results/tooling/rootfs_all_tools/REPORT.md`.
- Superseded: `enter_rootfs.sh --privileged` loads BPF natively, no
  Docker. `bpf()` checks caps in the initial user namespace, so
  `--unshare-all` can never work (`CapEff=0`); privileged mode starts
  bwrap from root without `--unshare-user` and keeps `CAP_BPF`.
  Separate bug: the image has zero-byte placeholders for host-only libs,
  so bound host `perf` said `file too short`; now filled from `ldd`.
  Adds `offcputime`/`biolatency` and fixes `perf` in the default
  sandbox. Proof:
  `experiments/falcon/results/tooling/rootfs_privileged/REPORT.md`.
