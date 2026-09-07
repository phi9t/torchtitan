# Perf tooling sprint (2026-09-06)

Claim label: `smoke`. Not ticket 08. Artifacts next to this file.

## Profiles

| ID | GPU | Shape | Purpose |
| --- | --- | --- | --- |
| V1 | 0 `GPU-c3f9bd88-...` | A5 science, 20k steps, fp32, 16,384 tok/step | Live-sample attach target |
| V2 | 1 | A5 science, 200 steps under host `nsys` | Launch-wrap |
| V3 | 2 | `falcon_tiny_overfit` stretched to 30k steps | py-spy inject guinea pig |

V1 driver: `experiments/falcon/run.sh perf --arm A5 --steps 20000`.
Host pid `2176064` (bwrap inner pid 2). Finished all **20,000** steps
in **1140.13 s** (~57 ms/step, ~287k tok/s including init). Live
attach saw FB 20.9 GiB, 97–98% SM busy, 741–753 W, 1965 MHz, no
throttle. GPU 0 is free.

V2 finished 200 steps in **14.56 s** (~73 ms/step including init;
~225k tok/s including init). Init-heavy; V1 is the better A5 rate.

V3 first launch died: overlaying `--training.steps 30000` onto
`falcon_tiny_overfit` left `decay_ratio=0` / `decay_steps=0`. Relaunch
needed `--lr_scheduler.total_steps 30000 --lr_scheduler.decay_ratio 0.8`.
Then ~14.4k tok/s, loss already 0.011 (memorized).

## MLSYS numbers (V1, DCGM group A.0)

First `dcgmi dmon -e 1001..` sample is `N/A` (counter warmup). After that,
on GPU 0, stable across 7 samples:

| Field | Value | Meaning |
| --- | --- | --- |
| GRACT 1001 | 0.973 | Graphics engine active (matches `nvidia-smi` 97%) |
| SMACT 1002 | 0.889 | SM active |
| SMOCC 1003 | **0.333** | Occupancy. `%gpu` is not this. |
| TENSO 1004 | 0.000 | No tensor-core (fp32 science run) |
| DRAMA 1005 | 0.226 | DRAM active |
| FP32A 1007 | 0.434 | FP32 pipe |
| FP16A 1008 | 0.006 | |
| INTAC 1016 | 0.176 | Integer |

33% occupancy on a 4×256 fp32 mixer is the expected small-wave shape,
not a hang. Hero-scale or bf16 is what would move TENSO / SMOCC.

Host RSS ~2.8 GiB. HBM 20.9 / 183 GiB. Disk `%util` ~0–3%.
`read_bytes` 49 MiB vs `rchar` 290 MiB: FineWeb is mostly page cache.

## Tool outcomes

| Tool | Class | Result |
| --- | --- | --- |
| `nvidia-smi query/dmon/pmon/topo` | live-sample | Works, no sudo. Job stayed `R`. |
| `nvidia-smi gpm` | live-sample | CLI is only `-g/-s` stream state. Stream **DISABLED**. No counter dump. |
| `dcgmi dmon` 150/155/203/204/252 | live-sample | Works. Matches NVML. |
| `dcgmi` 1001–1008 | live-sample | **Works on B200** after one N/A row. Occupancy is real. |
| `dcgmi health` | live-sample | "Health watches not enabled." |
| `dcgmi profile --pause` | control | `Feature not supported` on this host-engine. |
| `dcgmi nvlink` | live-sample | All 18 links `U` per GPU. |
| `perf cpu-clock:u` | live-sample | 1K samples / 12 s. Dominated by `cuStreamSynchronize` / vdso / libcuda. |
| docker-priv tracepoints | live-sample | `ioctl=0`, `sched_switch=203/8s`, `futex=176`. Same spin signature as 2026-09-05. |
| BCC / bpftrace | — | Still not installed. |
| host `nsys` wrapping `enter_rootfs.sh` | launch-wrap | Wrote 3.6M `.nsys-rep` but **no CUDA kernels**. Injection libs
  (`/data00/home/zfc/cuda_13_2/nsight-systems-...`) fail `LD_PRELOAD`
  inside bwrap. Do not do this. |
| py-spy `--native` | live-sample | `UNW_EBADREG` (libunwind vs this CPU). |
| py-spy Python-only | live-sample | Host → bwrap pid works. V3: 998 samples, 0 errors, process lived.
  Top: `_engine_run_backward`. V1: 249 samples; Python sits in
  `labels.to(device)` and backward (GPU wait, not a Python scan). |
| memray / Scalene / Fil / tracemalloc | — | Later: installed in rootfs `/usr/bin/python`. See below. |
| `ncu` | stop-job | Later: in-rootfs fill-kernel report. See below. |

## How to read the split (V1)

High GPU util + DCGM SMOCC 0.33 + perf `cuStreamSynchronize` + py-spy
asleep in `.to(device)` / `backward` = **GPU-bound eager fp32 Falcon
on a tiny width**. Not FineWeb disk, not an O(L) Python mixer.

Next perf levers (not this sprint): bf16 / tensor cores, compile,
wider model (hero), or a real CUDA kernel. Occupancy will stay low
until the wave is larger.

## Launch notes

- Redirected `console.log` is block-buffered. `perf_base.py` now sets
  `PYTHONUNBUFFERED=1` for the next attempt.
- `run.sh train` defaults `COMM_MODE=fake_backend` and forces 1 step.
  Long tiny runs must go through `run_train.sh` with scheduler overlays.
- Host `nsys` needs `TMPDIR` on a writable disk (`/tmp/nvidia` denied).

## Artifact map

- V1 attach: `attach_v1/` (`nvidia_smi.txt`, `dcgm_live.txt`,
  `dcgm_prof.txt`, `use_io.txt`, `own_process/`, `pyspy_python.raw`)
- V2: `nsys_v2/a5_200.nsys-rep` (CPU/host only; superseded by in-rootfs)
- V3: `attach_v3/pyspy_python.raw`

## In-rootfs tools (same day, later)

`enter_rootfs.sh` now ro-binds host `/sys` and `/usr/local/cuda`. The
image `/sys` is an empty stub; without the bind, in-sandbox `nsys`
dies with `Connection to Agent lost` / `End of file` even on
`/bin/sleep`. Isolated PID + offline net is enough after `/sys`.

| Tool | In-rootfs result |
| --- | --- |
| `experiments/falcon/run.sh nsys` A5 20 steps | `nsys_in_rootfs/a5_20.nsys-rep`. Kernels: `cutlass3x_sm100_simt_sgemm_*`, softmax, elementwise. 20 steps in 4.59 s. |
| `ncu --launch-count 1` | `nsys_diag/ncu_fill.ncu-rep` on `vectorized_elementwise_kernel`. |
| `dcgmi` / `nvidia-smi` / `perf` / `py-spy` | On PATH inside the sandbox. |
| memray 1.20 / Scalene 2.3 / Fil 2024.11 | Installed into `/usr/bin/python`. Toy launch-wraps under `py_profilers/`. Fil warns that cgroup `memory.high` is missing. |

`--profile` first ran BPF from inside the rootfs via
`run_privileged.sh` (docker-priv). Proof: `rootfs_all_tools/REPORT.md`
(bpftrace, BCC cachestat/runqlat/profile, host `perf` tracepoints, dcgmi
health/stats, ncu sgemm, compute-sanitizer, memray attach, Scalene
`--gpu`). Default training stays isolated.

## Docker removed from the tracing path (same day, later)

`enter_rootfs.sh --privileged` now loads BPF natively in the sandbox, so
no probe needs Docker. The blocker was never a missing mount: `bpf()`
resolves capabilities in the **initial** user namespace, so anything
minted inside `--unshare-all` is ignored (`CapEff=0` there). Privileged
mode starts bwrap from root, skips `--unshare-user`, and keeps
`CAP_BPF`/`CAP_PERFMON`; `/proc/self/ns/user` then matches
`/proc/1/ns/user`.

A second, unrelated bug was hiding behind the same symptom: the image
ships **zero-byte** files where host-only libraries belong, so a bound
host `perf` died with `libunwind-x86_64.so.8: file too short` while
`[[ -e ]]` said it was present. `enter_rootfs.sh` now fills those from
`ldd` of each bound host binary, only where the rootfs copy is empty.
`perf` consequently works in the default sandbox too.

Proven in-rootfs with no Docker: bpftrace (profile + `ustack` +
tracepoints), BCC `cachestat`/`runqlat`/`profile`/`offcputime`/
`biolatency`, `perf` tracepoints and `perf record -g -p`, `bpftool`,
`dcgmi dmon`, nsys, ncu, py-spy. `attach_live.sh --privileged` writes 17
artifacts, all owned by the invoking user. Proof:
`rootfs_privileged/REPORT.md`. Default training is unchanged and still
runs with `--unshare-all` and no capabilities.
