# eBPF as a host-side complement for Falcon GPU training jobs

Status: research note, not an approved implementation spec. Not a competing
observability stack.

Operational playbook for non-intrusive attach:
`.claude/skills/attaching-live-training/SKILL.md`. This file keeps citations.

Campaign: Falcon Fast Weight Attention local jobs (single-GPU Python trainer,
FineWeb nanogpt `.bin` reader, possible later multi-GPU). Training Python runs
inside `scripts/rootfs/enter_rootfs.sh` (bwrap). eBPF is a **host / privileged**
tool; GPU counters are a **vendor** tool. Do not collapse those layers.

## Why this note exists

Gregg's overview page points at four methodologies this note follows, then
stops: it does not recommend installing 150 BPF tools.

- [Overview](https://www.brendangregg.com/overview.html): short list of
  useful material, including eBPF tracing tools, the USE method, Off-CPU
  analysis, and "Linux Performance Analysis in 60,000 Milliseconds".
- [eBPF tracing tools](https://www.brendangregg.com/ebpf.html): "The main and
  recommended front-ends for BPF tracing are BCC and bpftrace: BCC for complex
  tools and daemons, and bpftrace for one-liners and short scripts." The same
  page says the two repositories together provide over 100 tools, and Gregg's
  BPF book adds more than 100. That is a catalog, not a campaign checklist.
- [USE method](https://www.brendangregg.com/usemethod.html): "For every
  resource, check utilization, saturation, and errors." Intended early, to
  identify systemic bottlenecks.
- [Off-CPU analysis](https://www.brendangregg.com/offcpuanalysis.html):
  complementary to CPU profiling; measures blocked time (I/O, locks, timers,
  paging, involuntary switches) via the kernel scheduler, not by tracing every
  app function.
- [Linux Performance Analysis in 60,000 Milliseconds](https://www.brendangregg.com/Articles/Netflix_Linux_Perf_Analysis_60s.pdf)
  (Netflix tech blog, 2015): first 60 seconds on a box with ten *standard*
  Linux commands before reaching for tracing.

This campaign is NVIDIA GPU training. Gregg's [AI Flame Graphs](https://www.brendangregg.com/blog/2024-10-29/ai-flame-graphs.html)
are **not** the GPU tool here: the post says the first version is a preview on
Intel Tiber AI Cloud for Intel Data Center GPU Max Series, "based on Intel EU
stall profiling and eBPF". Intel's [iaprof README](https://github.com/intel/iaprof)
states "This version targets Xe-family consumer and data center GPUs" and lists
Arc B-series / Lunar Lake / other Xe2. Do not run `iaprof` against NVIDIA.

## How this maps onto the repo contract

TorchTitan already has a tiered capture policy. eBPF does not replace it.

[AGENTS.md](../../../AGENTS.md) (Observability Evidence Contract):

- Tier 0: structured metrics, bounded Flight Recorder, checkpoint lineage,
  low-rate DCGM. Must stay within 1% median steady-state throughput.
- Tier 1: scheduled PyTorch Profiler and memory capture.
- Tier 2: anomaly-triggered Flight Recorder, NCCL RAS, py-spy, evidence-tail.
- Tier 3: stopped-job EUD, `nccl-tests`, SuperBench, deterministic replay,
  Nsight.

[docs/research/2026-08-12-training-observability-paper-closure.md](../../../docs/research/2026-08-12-training-observability-paper-closure.md):

- ByteRobust always-on monitor uses "second-level, non-workload GPU/host/network
  queries" plus training series; implicit hangs use py-spy and Flight Recorder;
  stop-time diagnostics include GPU/network tests, then replay.
- Tool inventory already names DCGM, py-spy, Flight Recorder, Nsight Systems,
  Nsight Compute, NCCL RAS, EUD. Costs and questions are there.
- Design inference: "The repo should own training-semantic evidence ... a
  launcher or platform can join host health." eBPF is that host-health
  complement. Do not invent a second run-bundle schema.

Practical split for Falcon:

| Layer | Owner | What it answers |
| --- | --- | --- |
| Training semantics | TorchTitan JSONL / metrics / Flight Recorder | step, loss, MFU, collectives |
| GPU vendor counters | DCGM / nvidia-smi / CUPTI / Nsight | SM busy, occupancy, Xid, kernel timeline |
| Host OS | 60s checklist + a few BCC/bpftrace tools | disk, page cache, scheduler, off-CPU CUDA sync, Python/native CPU |

## What classic eBPF can and cannot see for CUDA

Classic BCC/bpftrace attach to **Linux kernel** events (kprobes, tracepoints,
perf events) and **host userspace** (uprobes / USDT). Gregg's eBPF page lists
the event sources as dynamic tracing, static tracing, and profiling events on
Linux. That is the host OS, not the GPU SM.

**Can see (host side of a CUDA job):**

- The Python trainer on-CPU (native stacks via `profile`; Python stacks via
  the already-prescribed py-spy, not via BCC).
- Off-CPU blocking, including threads sitting in `cudaStreamSynchronize` /
  `cudaDeviceSynchronize` / driver ioctl, disk reads of FineWeb `.bin` shards,
  page faults, and run-queue delay ([offcputime](https://github.com/iovisor/bcc/blob/master/man/man8/offcputime.8)
  man page: "This spans all types of blocking activity: disk I/O, network I/O,
  locks, page faults, involuntary context switches, etc.").
- VFS / block I/O of the nanogpt reader (`numpy.fromfile` of uint16 tokens).
- Page-cache hit/miss while those shards are scanned.
- Later: TCP connect / retransmit if NCCL uses the host network stack.

**Cannot see:**

- SM occupancy, warp stalls, tensor-pipe activity, kernel instruction mix.
  NVIDIA documents those as **DCGM profiling fields** and **Nsight Compute /
  CUPTI** metrics, not as Linux BPF programs. DCGM's profiling module says SM
  occupancy (`DCGM_FI_PROF_SM_OCCUPANCY`, ID 1003) is "the fraction of resident
  warps on a multiprocessor, relative to the maximum number of concurrent
  warps" and is an interval average, not a kernel trace
  ([DCGM Profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)).
- Which CUDA kernel is on the SM. CUPTI's Activity API "asynchronously collect[s]
  a trace of an application's CPU and GPU CUDA activity"
  ([CUPTI Usage](https://docs.nvidia.com/cupti/main/main.html)). BCC does not.
- NVML / nvidia-smi `gpu` utilization is **not** occupancy. NVML defines it as
  "Percent of time over the past sample period during which one or more kernels
  was executing on the GPU"
  ([nvmlUtilization_t](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html)).
  A kernel that occupies 5% of SMs still reports high `gpu` percent if it ran
  for the whole sample window.

Host-side uprobes on `libcudart.so` / `libcuda.so` can time **API calls**
(`cudaLaunchKernel`, `cudaMemcpy`, `cudaStreamSynchronize`). That is still
host time, not SM time. Nested recursive launches will confuse
[funclatency](https://github.com/iovisor/bcc/blob/master/man/man8/funclatency.8)
("Currently nested or recursive functions are not supported properly").

## NVIDIA GPU tracing that is actually public (and what to do with it)

These are **not** BCC. They sit in the existing Tier 0 / 1 / 3 slots.

| Tool | What NVIDIA / NVlabs actually say | Campaign use |
| --- | --- | --- |
| nvidia-smi / NVML | Busy-time utilization, memory util, clocks, power, Xid. Bound into the rootfs by `enter_rootfs.sh` (`--ro-bind /usr/bin/nvidia-smi`). | Always-on cheap GPU glance **inside or outside** the sandbox. |
| DCGM | Low-overhead device-level profiling fields at 1 Hz default (min 100 ms). Process/job stats need watches enabled **before** work. Profiling metrics "require administrator privileges starting with Linux drivers 418.43". Profiling counters can conflict with Nsight; pause with `dcgmi profile --pause` ([DCGM Profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html), [Process and Job Statistics](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/process-and-job-statistics.html)). | Already Tier 0. Use for GPU USE (util / thermal / ECC / Xid). Not source attribution. |
| CUPTI | Official CUDA Profiling Tools Interface; Activity API traces CPU and GPU CUDA activity with correlation IDs linking `cuLaunchKernel` / `cudaMemcpy` to GPU records ([CUPTI](https://docs.nvidia.com/cupti/main/main.html)). Nsight Systems and Nsight Compute are the supported clients. | Do not call CUPTI from Falcon code. Let Nsight (Tier 1/3) own it. |
| Nsight Systems | System timeline for CUDA, NVTX, OS runtime, scheduling. "Nsight Systems does not require any application changes"; NVTX/ranges make CPU/GPU join useful ([Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)). Paper closure already warns of nontrivial overhead. | On-demand / stop-job when host eBPF shows the process is in CUDA sync but you need kernel names and stream overlap. |
| Nsight Compute | Occupancy, pipelines, memory; "counter limits require replay"; never production throughput evidence (paper closure + [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)). | Last resort on a filtered kernel after Nsight Systems names it. |
| NVBit | NVlabs **research prototype**, "not part of the official CUDA toolkit", "provided as-is with no guarantee of support". Dynamic SASS instrumentation; "any injection of instrumentation function has an associated cost" ([NVBit README](https://github.com/NVlabs/NVBit)). | Not for Falcon training jobs. |
| eGPU / bpftime | Workshop paper: compile eBPF bytecode to PTX and inject into CUDA kernels ([eGPU, HCDS 2025](https://dl.acm.org/doi/10.1145/3723851.3726984)). Research prototype, not an NVIDIA product. | Not production-ready. Do not put on the Falcon path. |
| Third-party CUPTI+USDT+eBPF agents | Some vendors bridge CUPTI activity streams into eBPF via USDT. That is still CUPTI for GPU data plus eBPF for host transport. Not NVIDIA-supported for this repo. | Ignore unless a platform team owns it. |

There is **no** NVIDIA-supported equivalent of Intel `iaprof` that profiles SM
instruction stalls with eBPF on NVIDIA GPUs.

## Rootfs / bwrap boundary (do not skip)

`scripts/rootfs/enter_rootfs.sh` builds a bwrap command that includes
`--unshare-all` (new user, pid, net, ipc, uts, cgroup namespaces), `--proc /proc`,
`--dev /dev`, NVIDIA device binds, and an optional `--ro-bind` of host
`nvidia-smi`. It does **not** bind `/sys/kernel/debug/tracing`, `/sys/fs/bpf`,
or `/sys/kernel/btf`.

Consequences:

1. **eBPF programs load into the host kernel and need host privilege.** Linux
   [capabilities(7)](https://man7.org/linux/man-pages/man7/capabilities.7.html):
   `CAP_BPF` (since 5.8) "Employ privileged BPF operations"; `CAP_PERFMON`
   covers `perf_event_open` and BPF ops with performance implications; BCC man
   pages uniformly say "Since this uses BPF, only the root user can use this
   tool." A bwrap user namespace's "root" is not host `CAP_BPF`.
2. **This note does not claim BCC works inside the sandbox.** It was not
   verified. Default: run BCC/bpftrace on the **host**, targeting the **host
   PID** of the trainer (the process is visible on the host even though it has
   a different PID inside the pid namespace).
3. **nvidia-smi is the exception:** the rootfs explicitly bind-mounts it and
   the NVIDIA device nodes, so GPU counters can be sampled from inside the
   job. DCGM `nv-hostengine` is still a host daemon; do not assume `dcgmi`
   exists in the rootfs.
4. `--unshare-net` (default offline rootfs) means host-network eBPF
   (`tcpconnect`, `tcpretrans`) sees the host stack, not a namespaced veth,
   until `TORCHTITAN_ROOTFS_NETWORK=networked`. For current single-GPU Falcon
   jobs that is fine: there is no NCCL fabric yet.

Join key back to the run bundle: host hostname, host PID, GPU UUID, `run_id` /
`attempt_id` if present. CUDA device index is not enough (paper closure).

## This host (2026-09-05 snapshot)

Checked on `n116-077-207`, kernel `5.15.152.bsk.9`, `perf_event_paranoid=2`.

- Present: `perf`, `bpftool`, `nvidia-smi`, `dcgmi` 4.4.1.
- Missing: `bcc`, `bpftrace`, `py-spy`.
- Bounding set includes `cap_bpf` / `cap_perfmon`, but the login shell is
  unprivileged. Probes need `sudo` on the host.
- Live Falcon job at snapshot time: host pid `2211702`
  (`experiments.falcon.ablation_runner`, arms A0 A2 A4 A5, 8000 steps)
  on GPU 0 only: 98% util, 21 / 183 GiB, 66 C, 774 W, 1965 MHz, no throttle,
  71 threads, state `R`. GPUs 1-7 idle. DCGM profiling fields (`SMACT`,
  `TENSO`, occupancy) returned `N/A` without profiling mode.

That snapshot already says the current arm is GPU-busy and not HBM-bound.
Until host `bpfcc-tools` is installed, the runnable subset is the 60s
checklist, `nvidia-smi` / `dcgmi dmon`, and `perf record -F 99 -g -p $PID`.

## USE method for this campaign

Gregg: for every resource, check utilization, saturation, errors
([USE](https://www.brendangregg.com/usemethod.html)). Linux checklist:
[USE Linux](https://www.brendangregg.com/USEmethod/use-linux.html). GPU is a
resource the original Linux table does not name; treat it as a device with
vendor counters, the same way the paper closure already treats DCGM.

| Resource | Utilization | Saturation | Errors |
| --- | --- | --- | --- |
| CPU | 60s: `vmstat` us+sy, `mpstat -P ALL 1`, `pidstat 1`. On-demand: BCC `profile`. | 60s: `vmstat 1` `r` vs CPU count. On-demand: `runqlat`. | `dmesg \| tail` (MCE / hung task). |
| Memory | 60s: `free -m`. Trainer RSS vs GPU HBM (nvidia-smi / DCGM). | `vmstat` si/so; BCC `drsnoop` if `allocstall` rises; off-CPU stacks in `handle_mm_fault`. | OOM in `dmesg`; DCGM Xid / ECC. |
| Disk | 60s: `iostat -xz 1` `%util`. FineWeb `.bin` sequential reads. | `iostat` `avgqu-sz` / `await`; BCC `biolatency`. | `dmesg` I/O errors; smartctl (host). |
| Network | Idle on current single-GPU jobs. Later: `sar -n DEV 1`. | Later: `sar -n TCP,ETCP 1`; BCC `tcpretrans`. | Later: retransmits, Xid 79/etc. |
| GPU (vendor, not eBPF) | nvidia-smi / DCGM `GPU_UTIL` = busy time. DCGM `PROF_SM_ACTIVE` (1002) / `PROF_SM_OCCUPANCY` (1003) if profiling fields are enabled. | Host threads blocked in CUDA sync (`offcputime`); Nsight Systems launch gaps. DCGM does not expose a CUDA run-queue length. | DCGM `DCGM_FI_DEV_XID_ERROR` (230), ECC fields, clocks throttling. |

Low utilization over a long window can hide saturation: Gregg's USE page
warns that 5-minute averages can miss seconds of 100% CPU. Same trap for
`nvidia-smi` 1-second GPU busy-time.

The 60s article's ten commands, in order: `uptime`, `dmesg | tail`, `vmstat 1`,
`mpstat -P ALL 1`, `pidstat 1`, `iostat -xz 1`, `free -m`, `sar -n DEV 1`,
`sar -n TCP,ETCP 1`, `top`. Look at errors and saturation first, then
utilization. That is the first minute **before** BCC.

## Recommended small set

Three host eBPF tools that can sit quietly, four to attach when a job looks
wrong, and the existing vendor last-resort. Plus the 60s checklist, which is
not eBPF.

Find the host PID first:

```bash
# host shell, not enter_rootfs
pgrep -af 'python.*(falcon|train)'
# or: pstree -ap $(pgrep -n bwrap)
```

### Always-on / cheap (seconds, low overhead)

These are **seconds of looking**, not daemons in the Trainer. Tier 0 already
owns DCGM; do not add a second always-on GPU sampler that fights Nsight.

#### 1. 60-second Linux checklist (not eBPF)

- **Question:** Is this box CPU-saturated, memory-thrashing, or disk-bound
  while Falcon trains? Exonerate host resources before blaming the mixer.
- **Command:** the ten Netflix/Gregg commands above, 1-second samples, ~60s.
- **Expected signal:** For a healthy single-GPU FineWeb job: modest `us` on one
  CPU (Python + numpy scan), high GPU busy, `iostat` sequential reads of `.bin`
  shards, `r` not >> nCPU, no OOM in `dmesg`.
- **Overhead:** userspace polling of `/proc`. Negligible.
- **Privilege:** none. Can run on the host. `iostat`/`pidstat`/`sar` need
  sysstat.

#### 2. nvidia-smi / DCGM (GPU vendor, already prescribed)

- **Question:** Is the GPU busy, throttling, or throwing Xid while the host
  looks idle? (Busy ≠ occupancy.)
- **Command sketch:**
  ```bash
  # inside or outside rootfs; nvidia-smi is bind-mounted
  nvidia-smi dmon -s pucvmet -c 30
  nvidia-smi --query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv -l 1

  # host DCGM (Tier 0); pause before Nsight
  dcgmi dmon --entity-id gpu:0 --field-id 203,204,230 --delay 1000
  # occupancy / SM active need profiling fields + admin nv-hostengine:
  dcgmi dmon --entity-id gpu:0 --field-id 1002,1003,1005 --delay 1000
  ```
- **Expected signal:** High `utilization.gpu` + low host CPU => compute-bound
  (good for a kernel ablation). Low GPU util + host off-CPU in CUDA sync =>
  launch/sync problem. Low GPU util + `iostat` `%util` high => dataloader.
  Xid/ECC => stop, do not keep ablating.
- **Overhead:** NVML sampling is cheap. DCGM default 1 Hz profiling is the
  documented low-overhead path; min 100 ms. Paper: keep Tier 0 within 1%
  throughput. DCGM profiling conflicts with Nsight; `dcgmi profile --pause`.
- **Privilege:** nvidia-smi as the job user is enough for util. DCGM profiling
  and process accounting may need root / `nvidia-smi -am 1`.

#### 3. BCC `cachestat` (eBPF, host)

- **Question:** Are FineWeb `.bin` shards coming from page cache or hitting
  disk every window? The reader uses `numpy.fromfile` on a 256-int32 header
  plus a uint16 token stream (`torchtitan/experiments/falcon/bin_reader.py`).
  First pass over a shard larger than RAM will miss; later epochs should hit
  if the working set fits.
- **Command:** `sudo cachestat -T 1` on the host while the job runs.
- **Expected signal:** High `MISSES` + `iostat` activity => cold scan / too
  little RAM. High `HITS` + still-slow steps => not disk; look at GPU or CPU.
- **Overhead:** Man page: traces page-cache kernel functions, in-kernel
  counts; "the rate of operations can be very high (>1G/sec) we can have up
  to 34% overhead"; "still a relatively efficient way... Measure in a test
  environment." ([cachestat.8](https://github.com/iovisor/bcc/blob/master/man/man8/cachestat.8)).
  Treat as cheap **relative to fileslower**, not as free. Sample 1 Hz, do not
  leave it on a 10-hour run without a measurement.
- **Privilege:** root / `CAP_BPF`. Host only.

#### 4. BCC `biolatency` (eBPF, host) — optional always-on histogram

- **Question:** When disk is involved, is latency a few hundred microseconds
  (cache/SSD) or tens of milliseconds (queueing)?
- **Command:** `sudo biolatency -mT 1 10` (1s summaries, 10 times) or
  `sudo bpftrace /usr/share/bpftrace/tools/biolatency.bt`.
- **Expected signal:** Power-of-two histogram. A second mode in the tens of
  ms during `fromfile` windows is dataloader saturation (USE disk).
- **Overhead:** Man page: "overhead for most storage I/O rates (< 10k IOPS)
  should be negligible" ([biolatency.8](https://github.com/iovisor/bcc/blob/master/man/man8/biolatency.8)).
  In-kernel histogram, not per-I/O print.
- **Privilege:** root. Host only.

Skip always-on `fileslower` / `biosnoop` / `opensnoop`: `fileslower` warns
overhead can reach 2x on VFS-heavy workloads
([fileslower.8](https://github.com/iovisor/bcc/blob/master/man/man8/fileslower.8)).
`execsnoop` is cheap ([execsnoop.8](https://github.com/iovisor/bcc/blob/master/man/man8/execsnoop.8)
"< 1000/s") but a single-process Falcon trainer does not fork workers.

### On-demand when a job looks stuck or slow

Attach from the **host** after the 60s checklist plus nvidia-smi have named a
resource. Bound the duration (5–30s). This is closer to repo Tier 2 (py-spy,
evidence tail) than to Tier 0.

#### 5. BCC `offcputime` — GPU sync, I/O wait, page faults

- **Question:** Why is the trainer not on CPU? Waiting in CUDA synchronize,
  reading `.bin`, faulting, or sleeping on a lock?
- **Command:**
  ```bash
  sudo offcputime -df -p $HOST_PID 15
  # optional: only uninterruptible (disk/fault-ish)
  sudo offcputime -p $HOST_PID --state 2 15
  ```
  Folded `-f` output feeds an off-CPU flame graph
  ([Gregg off-CPU flame graphs](https://www.brendangregg.com/FlameGraphs/offcpuflamegraphs.html)).
- **Expected signal:** User stacks through `cudaStreamSynchronize` /
  `cuStreamSynchronize` / NVIDIA ioctl => GPU work or a missing overlap;
  confirm with nvidia-smi busy-time. Stacks through `vfs_read` /
  `blk_mq_start_request` => dataloader. `handle_mm_fault` => page faults
  (working set). Short involuntary switches => `runqlat` next.
- **Overhead:** Man page: in-kernel unique-stack summary, but "scheduler
  events ... can exceed 1 million events per second, and so caution should
  still be used. Test before production use." Raise `-m` (min block us) to
  drop short sleeps ([offcputime.8](https://github.com/iovisor/bcc/blob/master/man/man8/offcputime.8)).
- **Privilege:** root. Host PID. Frame pointers help user stacks (Gregg
  off-CPU prerequisites). For Python frames use py-spy (Tier 2), not this.

#### 6. BCC `profile` + existing py-spy — CPU-bound Python / native scan

- **Question:** Is the host CPU burning in CPython, numpy `fromfile`, or the
  CUDA user-mode driver while the GPU waits?
- **Command:**
  ```bash
  sudo profile -df -p $HOST_PID -F 49 10
  # already prescribed, from host if ptrace allows, or inside rootfs:
  py-spy dump --pid $PID --subprocesses
  py-spy record -o /tmp/falcon-cpu.svg --pid $PID --subprocesses -d 15
  ```
- **Expected signal:** `profile` native stacks: `fromfile` / memcpy / libcuda
  busy-wait. py-spy: Python dataloader vs `forward` vs optimizer. Paper
  closure: py-spy "cannot see GPU kernel progress."
- **Overhead:** `profile` man page: 49 Hz default, in-kernel unique-stack
  counts, "overhead ... should be negligible" at 49 Hz
  ([profile.8](https://github.com/iovisor/bcc/blob/master/man/man8/profile.8)).
  py-spy: out-of-process sampling; attach may need ptrace/root (paper
  inventory).
- **Privilege:** `profile` root; py-spy often ptrace. `[unknown]` frames =>
  missing frame pointers / JIT (same man page).

#### 7. BCC `fileslower` (and `biosnoop` if you need per-I/O)

- **Question:** Which `.bin` path is stalling the VFS read, and is it 10 ms or
  200 ms? `fileslower` is VFS-generic (ext4/xfs/nfs); `ext4slower` only if you
  have verified the FS.
- **Command:**
  ```bash
  sudo fileslower -p $HOST_PID 1    # sync reads/writes slower than 1 ms
  sudo biosnoop -d nvme0n1          # per-I/O device latency; higher volume
  ```
- **Expected signal:** `FILENAME` matching FineWeb shards, `D=R`, `LAT(ms)`
  spikes aligned with step-time bubbles in JSONL. Sequential large reads are
  expected; random small reads are not (the loader packs contiguous windows).
- **Overhead:** `fileslower`: "in the worst case slowing applications by 2x"
  because it traces VFS including cache hits. Threshold + `-p PID` is
  mandatory. Prefer `biolatency` histograms unless you need the filename.
  `biosnoop`: "block device I/O usually has a relatively low frequency
  (< 10,000/s), the overhead ... negligible" until high IOPS
  ([biosnoop.8](https://github.com/iovisor/bcc/blob/master/man/man8/biosnoop.8)).
- **Privilege:** root. Host only.

#### 8. BCC `runqlat` — scheduler

- **Question:** Is the trainer runnable but waiting for a CPU (noisy neighbor,
  too many Python threads)?
- **Command:** `sudo runqlat -mT -p $HOST_PID 1 10`
- **Expected signal:** Most mass in <100 us is healthy. Milliseconds of run-
  queue latency with `vmstat r` > nCPU is CPU saturation (USE). That can make
  CUDA launches look like "GPU idle" because the host thread never queued
  work.
- **Overhead:** "traces scheduler functions, which can become very frequent
  ... Measure in a lab"
  ([runqlat.8](https://github.com/iovisor/bcc/blob/master/man/man8/runqlat.8)).
  Bound to 10 seconds, filter `-p`.
- **Privilege:** root.

#### 9. BCC `funclatency` or a bpftrace uprobe — CUDA sync host time

- **Question:** How long does each `cudaStreamSynchronize` block? Complements
  `offcputime` with a histogram of one API instead of stacks.
- **Command:**
  ```bash
  # BCC: time the libc symbol as actually mapped by the process
  sudo funclatency -p $HOST_PID -u 'c:cudaStreamSynchronize'
  # if the symbol lives in libcudart inside the rootfs mapping:
  sudo funclatency -p $HOST_PID -u 'libcudart:cudaStreamSynchronize'

  sudo bpftrace -e '
    uprobe:/proc/'$HOST_PID'/root/usr/lib/x86_64-linux-gnu/libcudart.so.12:cudaStreamSynchronize { @start[tid] = nsecs; }
    uretprobe:/proc/'$HOST_PID'/root/usr/lib/x86_64-linux-gnu/libcudart.so.12:cudaStreamSynchronize /@start[tid]/ {
      @ns = hist(nsecs - @start[tid]); delete(@start[tid]);
    }'
  ```
  Adjust the `.so` path to whatever `/proc/$HOST_PID/maps` shows (rootfs CUDA
  lives at `/opt/cuda-synth/lib64` inside the sandbox).
- **Expected signal:** Tight histogram of kernel-length times => GPU compute.
  Long tail while nvidia-smi util is low => CPU not launching / memcpy /
  sync-on-every-op.
- **Overhead:** funclatency: efficient in-kernel histogram, but "rate of
  kernel functions can also be very high (>1M/sec)". CUDA sync should be far
  below that. Nested launches: dubious timestamps (man page warning).
- **Privilege:** root. Uprobes on the host see the process's mapped files,
  including bwrap binds.

#### 10. BCC `drsnoop` — only if memory looks tight

- **Question:** Is the FineWeb scan (or host pinned memory) forcing direct
  reclaim?
- **Command:** `sudo drsnoop -p $HOST_PID -d 20`
- **Expected signal:** Any events during steady state are a smell (`allocstall`
  rising). Overhead "generally expected to be low (< 1000/s)"
  ([drsnoop.8](https://github.com/iovisor/bcc/blob/master/man/man8/drsnoop.8)).
- **Privilege:** root. Skip unless `vmstat` / `free` already look bad.

### Stop-the-job / last resort

Do not stack these on a live ablation that you still want as throughput
evidence. Matches repo Tier 3.

| Action | Why | Source |
| --- | --- | --- |
| Pause DCGM profiling, short Nsight Systems capture | Kernel names, stream overlap, CPU/GPU idle gaps on one timeline | Paper inventory; [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html); DCGM `profile --pause` |
| Filtered Nsight Compute | Occupancy, pipelines, memory for **one** kernel | [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html); "behavior-changing" |
| DCGM EUD / `nccl-tests` (later multi-GPU) | Hardware vs fabric vs code | Paper Tier 3 |
| Deterministic replay / loss compare | Correctness, not speed | AGENTS.md numerical proof |
| NVBit / eGPU / iaprof | — | NVBit is a research SASS injector; eGPU is a workshop prototype; iaprof is Intel Xe only |

Gregg: observability tools drop events under load and are not security or
audit tools ([eBPF Observability Tools Are Not Security Tools](https://www.brendangregg.com/blog/2023-04-28/ebpf-security-issues.html)).
Do not treat BCC histograms as complete CUDA accounting.

### Later: multi-GPU / NCCL

Keep using the repo tools first: Flight Recorder, NCCL RAS, `NCCL_DEBUG`.
Host eBPF only answers "is the host TCP path dropping/retransmitting?"

- `sudo tcpretrans` — "To keep overhead very low, only the TCP retransmit
  functions are traced. This does not trace every packet"
  ([tcpretrans.8](https://github.com/iovisor/bcc/blob/master/man/man8/tcpretrans.8)).
- `sudo tcpconnect` / bpftrace `tcpconnect.bt` — new connections at init, not
  per-byte.
- Off-CPU stacks in `ibv_*` / `nccl*` on the host thread => waiting on
  fabric; still need Flight Recorder to know **which** collective.

RDMA/NCCL device memory is not visible to BCC. Do not pretend `sar -n DEV`
explains NVLink.

## bpftrace vs BCC for this campaign

Gregg: try BCC if you want tools; bpftrace if you want one-liners
([eBPF page](https://www.brendangregg.com/ebpf.html)). Official bpftrace:
"general purpose tracing tool and language for Linux" with canonical tools
under `tools/` ([bpftrace README](https://github.com/bpftrace/bpftrace/blob/master/README.md),
[tools/README.md](https://github.com/bpftrace/bpftrace/blob/master/tools/README.md)).

Use BCC for `offcputime`, `profile`, `fileslower`, `cachestat` (richer flags,
PID filters, folded stacks). Use packaged bpftrace tools as drop-in twins when
BCC Python is painful: `biolatency.bt`, `biosnoop.bt`, `runqlat.bt`,
`execsnoop.bt`, `opensnoop.bt`, `tcpretrans.bt`. Use a bpftrace one-liner only
for the CUDA uprobe whose `.so` path is job-specific.

## Practical 5-minute playbook for a slow Falcon ablation

1. Host: 60s checklist. Note `r`, `%util` disk, `free`, `dmesg`.
2. `nvidia-smi dmon` (in rootfs is fine). Busy GPU + idle host CPU => stop
   host-tracing; occupancy/kernel work is DCGM profiling or Nsight, not BCC.
3. If GPU util is low: `offcputime -p HOST_PID` for 15s. CUDA sync vs `.bin`
   read vs run-queue.
4. If stacks show VFS/block: `cachestat` + `biolatency`; only then
   `fileslower -p HOST_PID 1`.
5. If stacks show Python/native on-CPU: py-spy + `profile`.
6. If still unexplained GPU-side: pause DCGM profiling, one short Nsight
   Systems window, then resume. Occupancy last, Nsight Compute, stopped job.

Join artifacts into the existing run-attempt bundle (host PID, GPU UUID, step,
clock). Do not start a parallel "ebpf traces/" convention that never gets a
`run_id`.

## What not to do

- Do not install the full BCC zoo. Gregg's book and the iovisor README list
  80+ tracing tools; this campaign needs the dozen above.
- Do not run `strace` on the trainer (Gregg: [strace Wow Much Syscall](https://www.brendangregg.com/blog/2014-05-11/strace-wow-much-syscall.html)
  — high overhead; listed on the overview as a warning).
- Do not treat nvidia-smi `%` as SM occupancy.
- Do not recommend Intel AI Flame Graphs / `iaprof` on NVIDIA.
- Do not run BCC inside bwrap unless a later note **measures** that
  `bpf()` succeeds, BTF is visible, and PIDs match. Default is host.
- Do not leave `fileslower` or unfiltered `runqlat` on for an overnight
  FineWeb run.
- Do not replace DCGM, Flight Recorder, py-spy, or Nsight with eBPF.

## Primary sources

- Brendan Gregg: [overview](https://www.brendangregg.com/overview.html),
  [eBPF tools](https://www.brendangregg.com/ebpf.html),
  [USE](https://www.brendangregg.com/usemethod.html),
  [USE Linux](https://www.brendangregg.com/USEmethod/use-linux.html),
  [Off-CPU](https://www.brendangregg.com/offcpuanalysis.html),
  [60,000 Milliseconds PDF](https://www.brendangregg.com/Articles/Netflix_Linux_Perf_Analysis_60s.pdf),
  [AI Flame Graphs](https://www.brendangregg.com/blog/2024-10-29/ai-flame-graphs.html),
  [eBPF tools are not security tools](https://www.brendangregg.com/blog/2023-04-28/ebpf-security-issues.html).
- iovisor BCC: [README tool list](https://github.com/iovisor/bcc), man pages
  linked per tool (`offcputime`, `profile`, `cachestat`, `biolatency`,
  `fileslower`, `biosnoop`, `runqlat`, `funclatency`, `execsnoop`, `drsnoop`,
  `tcpretrans`).
- bpftrace: [README](https://github.com/bpftrace/bpftrace/blob/master/README.md),
  [canonical tools](https://github.com/bpftrace/bpftrace/blob/master/tools/README.md),
  [docs](https://bpftrace.org/docs).
- Linux: [capabilities(7)](https://man7.org/linux/man-pages/man7/capabilities.7.html)
  `CAP_BPF` / `CAP_PERFMON`.
- NVIDIA: [nvmlUtilization_t](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html),
  [DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html),
  [DCGM process/job stats](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/process-and-job-statistics.html),
  [DCGM field IDs](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)
  (`DCGM_FI_DEV_GPU_UTIL_RATIO` 203, `DCGM_FI_DEV_XID_ERROR` 230),
  [CUPTI](https://docs.nvidia.com/cupti/main/main.html),
  [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
  [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
  [NVBit](https://github.com/NVlabs/NVBit).
- Intel (explicit non-goal): [iaprof](https://github.com/intel/iaprof).
- Research-only GPU eBPF: [eGPU](https://dl.acm.org/doi/10.1145/3723851.3726984).
- Repo: [AGENTS.md](../../../AGENTS.md),
  [observability paper closure](../../../docs/research/2026-08-12-training-observability-paper-closure.md),
  [enter_rootfs.sh](../../../scripts/rootfs/enter_rootfs.sh),
  [falcon bin_reader.py](../../../torchtitan/experiments/falcon/bin_reader.py).
