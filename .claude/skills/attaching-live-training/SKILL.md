---
name: attaching-live-training
description: Use when a live TorchTitan or experiment training job looks slow, stuck, hung, CPU-bound, GPU-idle, or I/O-bound, and the user wants eBPF, perf, DCGM, nvidia-smi, or host attach without stopping the process.
---

# Attaching to live training

Non-intrusive **host attach** for a running trainer. Sample, do not stop.
This complements the repo observability tiers; it does not replace DCGM,
Flight Recorder, or Nsight.

**Core rule:** the job keeps running. If a tool pauses, traces every
syscall, or needs the job stopped, it is the wrong tool for this skill.

## When to use

- Live job is slow, hung, GPU-idle, CPU-spinning, or suspected I/O-bound.
- User asks for eBPF / BCC / bpftrace / perf / DCGM / nvidia-smi on a
  running trainer.
- Need to tell Python-scan vs CUDA-spin vs FineWeb disk vs scheduler
  queue without killing the run.

Use `numerics_debugging` for bitwise / activation drift. Use the paper
closure + Tier 3 for stopped-job EUD / Nsight Compute.

## Guardrails

Do these:

- Attach from the **host** against the **host PID**.
- Prefer tools that never send signals: `/proc`, `nvidia-smi`, `dcgmi dmon`,
  `perf record` sampling, short in-kernel histograms.
- Bound every probe to 5–30 s. Write artifacts next to the run, not into git.
- Join by host PID + GPU UUID + trainer step. Device index is not a key.

Use these only after the job is stopped, or the user explicitly accepts
a pause: Nsight Systems/Compute, DCGM EUD, `nccl-tests`, SuperBench,
`strace`, `gdb`, `py-spy dump`. `py-spy top` samples but still ptrace-pauses
briefly; skip it on a science run unless asked.

`kernel.unprivileged_bpf_disabled=1` and `perf_event_paranoid=2` here.
First path is still no-sudo own-process `perf` + `/proc` + NVML. `sudo` is
for BPF, not for the first look.

bwrap rootfs is the place for every attach tool, including BPF.
`enter_rootfs.sh` ro-binds host `/sys` (required by nsys),
`/usr/local/cuda`, DCGM, `perf` + the host libs it needs, `bpftool`, and
`py-spy`. Pick the mode:

| Mode | Use for |
| --- | --- |
| default | training, launch-wrap (`nsys`, memray, Scalene, Fil) |
| `--share-pid` | attach by host pid: `perf`, `py-spy`, `nsys`, DCGM |
| `--privileged` | bpftrace, BCC, `perf` tracepoints (needs `sudo -n`) |

`--privileged` starts bwrap from root *without* a new user namespace and
keeps `CAP_BPF`/`CAP_PERFMON`, because `bpf()` checks capabilities against
the initial user namespace and ignores any minted in a nested one. That is
why no bind or `--cap-add` can make BPF work under `--unshare-all`. It runs
as real host root: read-only diagnostics only, never training. Ownership of
`--out` and the runtime state is handed back on exit.

`--profile` (host PIDs + Docker) is the older docker-privileged path, kept
for probes that already depend on it. Prefer `--privileged`.

After a rootfs rebuild:

```
.claude/skills/attaching-live-training/scripts/install_ebpf_tools.sh
.claude/skills/attaching-live-training/scripts/install_python_profilers.sh
```

## Steps

1. **Find the host PID.** Completion: one live pid, cmdline, GPU UUID.
   ```bash
   pgrep -af 'python.*(train|ablation_runner)'
   nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
   ```
   If the process is under `bwrap`, use the host python pid, not the inner pid 4.

2. **60-second USE pass** (not eBPF). Completion: errors and saturation
   checked before utilization.
   `uptime`, `dmesg | tail`, `vmstat 1`, `mpstat -P ALL 1`,
   `pidstat -p $PID 1`, `iostat -xz 1`, `free -m`.
   Then GPU: `nvidia-smi dmon -s pucvmet -c 5` and
   `dcgmi dmon -e 150,155,203,204,252 -c 5`.
   `utilization.gpu` is busy time (a kernel ran), not SM occupancy.

3. **Own-process sample** (no sudo when you own the pid). Completion:
   10–15 s of `/proc` + `perf` written to a directory; process still `R`/`S`.
   ```bash
   .claude/skills/attaching-live-training/scripts/attach_live.sh --pid $PID
   .claude/skills/attaching-live-training/scripts/attach_live.sh --pid $PID --tracepoints
   ```
   Or the commands in [reference.md](reference.md).
   Add `--privileged` to also collect `offcputime`, `runqlat`, `cachestat`,
   and bpftrace user stacks. Do that once the USE pass says the split is
   off-CPU or queueing, not as the first move.

4. **Read the split.** Completion: one of the rows below is named.

   | Signal | Meaning |
   | --- | --- |
   | High GPU util + `cuStreamSynchronize` / vdso / `libcuda` on-CPU | GPU-bound. Driver **spin-wait**. `ioctl` may be 0. |
   | High GPU util + CPython eval / Python scan on-CPU | Leftover host loop (recurrent mixer). |
   | Low GPU util + `read_bytes` rising / `iostat` `%util` | Dataloader / cold `.bin`. |
   | `read_bytes=0` and large `rchar` | Page-cache hits. Exonerate disk. |
   | `vmstat r` >> nCPU, `runqlat` tail | Other tenants queued the core. |
   | Off-CPU stacks in `futex` / NCCL watchdog / `poll` only on helpers | Idle side threads. Not the step bottleneck. |
   | Xid / ECC / throttle | Stop ablating. Hardware path. |

   CUDA stream wait is **on-CPU spin** until the stacks say otherwise.
   `offcputime` alone misses it.

5. **Escalate only if the split is still ambiguous.** Completion: a 8–15 s
   bound attach, then detach.
   - Tracepoints and BCC/bpftrace via `enter_rootfs.sh --privileged`.
   - Commands: [reference.md](reference.md). Catalog: [tools.md](tools.md).
   Do not memray-attach, Fil-wrap, or ncu the live science pid.

Intel `iaprof` / AI Flame Graphs are Xe-only. This fleet is NVIDIA.

## Common mistakes

| Move | What to do instead |
| --- | --- |
| `strace -p` / `gdb` / `nsys profile` on the live job | Sample with `perf` / `/proc` |
| BCC in the default sandbox | `enter_rootfs.sh --privileged`; `CapEff=0` under `--unshare-all` |
| Binding more paths to fix BPF | It is a user-namespace problem, not a mount problem |
| `perf`/`bpftool` "not found for kernel" | Rootfs copies are distro wrappers; host binaries are bound in |
| `sudo` before checking own-pid `perf` | Own-process `cpu-clock:u` first |
| `offcputime` as the CUDA-wait tool | On-CPU `profile` / `perf` for `cuStreamSynchronize` |
| Treat `%gpu` as occupancy | Occupancy is DCGM 1003 (`SMOCC`); works on this B200 after one `N/A` warmup row |
| Host `nsys` around `enter_rootfs.sh` | `experiments/falcon/run.sh nsys` (nsys inside rootfs; needs host `/sys`) |
| py-spy `--native` | `UNW_EBADREG` here. Python-only `record` works host→bwrap |
| Leave `biosnoop` / high-Hz `profile` up for the whole table | 10–30 s windows |
| Hardware `cycles` failing => give up | `cpu-clock` software sampling still works |

## Sources

Campaign capture that proved the CUDA-spin split:
`experiments/falcon/results/ebpf_nosudo_20260906T050950Z/REPORT.md`.
Why BPF needs the initial user namespace, and the in-rootfs proof for every
tool: `experiments/falcon/results/tooling/rootfs_privileged/REPORT.md`.
Cited tool list and USE mapping:
`.scratch/falcon-fast-weight-attention/notes/2026-09-05-ebpf-training-observability.md`.
Repo tiers: `docs/research/2026-08-12-training-observability-paper-closure.md`.
Full catalog + TODOs (NVIDIA, eBPF, py-spy/memray/scalene/tracemalloc/Fil):
[tools.md](tools.md).
