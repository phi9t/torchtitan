# Tool survey (NVIDIA, eBPF, Python)

Inventory for live and near-live training attach. Privilege ladder stays:

1. Own-process `/proc` + `perf` + NVML
2. `enter_rootfs.sh --privileged` (root bwrap, `CAP_BPF`) for BPF
3. Docker `--privileged --pid=host` only where a probe already depends on it

`nsys` / `ncu` live on this host at `/usr/local/cuda/bin/` and are on
`PATH` **inside the rootfs**. `dcgmi` and `nvidia-smi` are bound in.
bpftrace and BCC are installed in the rootfs and load natively under
`enter_rootfs.sh --privileged`. Host `perf` and `bpftool` are bound along
with the host libraries they need, since the rootfs copies are distro
wrappers for a kernel package that does not exist here. py-spy is bound
from `~/.local/bin`. memray / Scalene / Fil / gdb are in the rootfs
`/usr/bin/python` (reinstall after rebuild). `tracemalloc` is stdlib.

Attach class:

| Class | Meaning |
| --- | --- |
| live-sample | Attach to a running pid. Bound 5–30 s. Default for this skill. |
| live-inject | Attach, but injects into the process (ptrace/gdb/`import`). Can pause or crash. Prove safe before a science run. |
| launch-wrap | Must start the process under the tool. New attempt, not the current job. |
| stop-job | Exclusive GPU/counters, replay, or active diagnostics. Job must be down. |

## NVIDIA (this box)

Sources: [NVML / nvidia-smi](https://docs.nvidia.com/deploy/nvml-api/),
[DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/),
[DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html),
[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/),
[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/),
[CUPTI](https://docs.nvidia.com/cupti/main/main.html),
[NCCL RAS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html),
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/).

| Tool | Class | Privilege | On this host | Answers | Do not use it for |
| --- | --- | --- | --- | --- | --- |
| `nvidia-smi query` / `dmon` / `pmon` | live-sample | job user | yes | Busy time, FB, power, temp, throttle, which pid owns the GPU | Occupancy, kernel names |
| `nvidia-smi nvlink` / `topo` / `c2c` / `pci` | live-sample | job user | yes | Fabric topology, link state | Step-time CPU vs GPU split |
| `nvidia-smi gpm` | live-sample? | often admin | yes (verb exists) | Hopper+ GPM counters | Unproven on B200 here |
| `nvidia-smi clocks` / `power-*` | live-sample / mutate | admin to set | yes | Query clocks; **setting** clocks is intrusive | Changing clocks on a comparative run |
| `dcgmi dmon` (150/155/203/204/252) | live-sample | job user | yes | Same as NVML + host-engine join | SM occupancy |
| `dcgmi dmon` profiling fields 1002/1003/... | live-sample | admin; conflicts with Nsight | fields were `N/A` | SM active / occupancy (interval average, not a kernel trace) | Comparative hero throughput while Nsight is reserved |
| `dcgmi health` / `nvlink` / `stats` | live-sample | host-engine | yes | Xid/ECC/health watches, job stats if started **before** work | Attribution |
| `dcgmi profile --pause/--resume` | control | admin | yes | Release counters for Nsight | Leaving profiling paused |
| `dcgmi diag` / `mndiag` / EUD | stop-job | admin, job down | yes | Hardware gray failure | Live science table |
| `nsys` 2025.6.3 | launch-wrap or short capture-range | user; pause DCGM profiling | `/usr/local/cuda/bin/nsys` | CPU+CUDA+NVTX+NCCL timeline | Overnight attach; production MFU |
| `ncu` 2026.1.0 | stop-job / replay | admin counters | `/usr/local/cuda/bin/ncu` | Occupancy, pipes, roofline of **one** kernel | Throughput evidence |
| CUPTI | library under nsys/ncu | — | via toolkit | Correlation IDs | Calling from Falcon code |
| Compute Sanitizer | launch-wrap | user | `/usr/local/cuda/bin/compute-sanitizer` | memcheck / sync / race | Live job |
| NCCL RAS / `NCCL_DEBUG` | live query / launch log | job user | NCCL in rootfs | Hang membership, collective counts | Single-GPU Falcon default |
| `nccl-tests` | stop-job | exclusive GPUs | images exist | Fabric bandwidth/correctness | Live trainer |
| NVBit / eGPU | research | — | no | Experimental SASS / BPF-to-PTX | This campaign |

`utilization.gpu` is "fraction of time a kernel was executing"
([nvmlUtilization_t](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html)).
Not occupancy.

## eBPF / perf (host)

Sources: [Gregg eBPF](https://www.brendangregg.com/ebpf.html),
[BCC](https://github.com/iovisor/bcc), [Off-CPU](https://www.brendangregg.com/offcpuanalysis.html).
Install into the rootfs (`install_ebpf_tools.sh`). Load BPF with
`enter_rootfs.sh --privileged`. `bpf()` resolves capabilities in the
initial user namespace, so an unprivileged bwrap userns can never load a
program regardless of binds or `--cap-add`.

| Tool | Class | Privilege | On this host | Answers |
| --- | --- | --- | --- | --- |
| `perf record -e cpu-clock:u` | live-sample | own pid | yes | On-CPU userspace (`cuStreamSynchronize` spin) |
| `perf stat` hardware `cycles` | live-sample | often root; still `<not supported>` here | yes, PMU missing | IPC if PMU appears |
| `perf` tracepoints (`sched:sched_switch`, `sys_enter_*`) | live-sample | `--privileged` | **proven in rootfs** | Syscall mix, off-CPU helpers |
| `bpftool` | inspect | `--privileged` | **proven in rootfs** | What is already loaded |
| BCC `profile` | live-sample | `--privileged` | **proven** (`profile-bpfcc`) | Same as `perf` + kernel stacks |
| BCC `offcputime` | live-sample | `--privileged` | **proven** | Blocked stacks. **Misses CUDA spin.** |
| BCC `runqlat` | live-sample | `--privileged` | **proven** | Scheduler wait vs other tenants |
| BCC `cachestat` | live-sample | `--privileged` | **proven** | Page-cache hit/miss (FineWeb) |
| BCC `biolatency` | live-sample | `--privileged` | **proven** | Disk latency histogram |
| BCC `biosnoop` | live-sample | `--privileged` | in rootfs | Per-I/O; keep the window short |
| BCC `opensnoop` / `fileslower` / `ext4slower` | live-sample | `--privileged` | in rootfs | Which shard stalled. `fileslower` can be 2× |
| BCC `execsnoop` | live-sample | `--privileged` | in rootfs | Forked children; useless for a long-lived trainer |
| BCC `tcpconnect` / `tcpretrans` / `tcplife` | live-sample | `--privileged` | in rootfs | Host TCP only. Not NVLink/NCCL device |
| BCC `funccount` / `funclatency` / `uprobe cudaStreamSynchronize` | live-inject | `--privileged` | in rootfs | API wait histogram |
| `bpftrace` one-liners | live-sample | `--privileged` | **proven** 0.20.2 | Same questions, shorter scripts |
| Intel `iaprof` | n/a | — | no | Xe EU stalls only |

## Python CPU / memory

Official docs, not the Medium roundups.

| Tool | Class | Privilege | On this host | What the project actually says | GPU / native | Live-job risk |
| --- | --- | --- | --- | --- | --- | --- |
| [py-spy](https://github.com/benfred/py-spy) | live-sample (`top`/`record`); `dump` pauses | ptrace; `ptrace_scope=0` here | **missing** | Out-of-process sampling. Low overhead. `--native` costs more. | No CUDA kernels. `--native` sees C frames | `dump` stops threads. `top` still ptrace-samples |
| [memray](https://bloomberg.github.io/memray/overview.html) | launch-wrap **or** [attach](https://bloomberg.github.io/memray/attach.html) | ptrace/gdb inject; package must exist **inside** the target env | missing | Traces allocations in Python, native extensions, interpreter. Flame graphs. Attach does **not** see pre-attach heap. Failed inject can **crash** the process ([#846](https://github.com/bloomberg/memray/issues/846)) | Native yes. CUDA device allocs generally **no** | live-inject. Not default on a science pid |
| [Scalene](https://github.com/plasma-umass/scalene) | launch-wrap | job user | missing | Line-level CPU + memory + GPU (NVML sampling). Not an attach profiler | GPU via NVML, not kernel names | Restart under `scalene` |
| [tracemalloc](https://docs.python.org/3/library/tracemalloc.html) | launch (`PYTHONTRACEMALLOC=1`) or mid-run `tracemalloc.start()` | in-process | stdlib 3.11 | Python allocator only. Snapshots + diffs. Start early for most blocks | No C++, no CUDA | Safe if we add a gated hook; useless on an already-started job without inject |
| [Fil](https://pythonspeed.com/fil/docs/index.html) | launch-wrap | job user | missing | Peak / high-water mark for data-processing. Authors: **offline**, enough overhead to skip production | Native-ish Python objects; not CUDA | Never attach to a live trainer |
| PyTorch CUDA memory snapshot | in-process, gated | job user | in Trainer | [Official](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html): allocator history, not all NCCL/external | **The** GPU heap tool | Enable on a later attempt; do not inject into a running ablation |
| `cProfile` / `pyinstrument` | launch-wrap | job user | stdlib / no | Function CPU, not memory, not GPU | no | Restart only |

Memray/Scalene/Fil do **not** replace DCGM or PyTorch CUDA snapshots for
HBM. They answer "why is RSS 3 GiB" and "which Python line churns
allocations", not "why is FB 21 GiB".

## TODOs (how to use them here)

Check these off by writing a short note under
`experiments/<prog>/results/tooling/` after a **non-science** pid test
(tiny overfit or a throwaway `sleep` python). Do not use the hero pid
for first injects.

### NVIDIA

Evidence: `experiments/falcon/results/tooling/REPORT.md` (2026-09-06).

- [x] **GPM on B200.** `nvidia-smi gpm` is only `-g/-s` stream state.
  Stream DISABLED. No counter dump. Use DCGM 1002/1003 instead.
- [x] **DCGM profiling fields.** Group A.0 works on B200 after one `N/A`
  warmup row. V1 A5 fp32: GRACT 0.97, SMACT 0.89, SMOCC 0.33, TENSO 0.
  `dcgmi profile --pause` is `Feature not supported` here.
- [x] **`dcgmi stats` job window.** `--profile` + `dcgmi stats -g 0 -e`
  / `-s JOB` / `-x JOB` / `-j JOB -v` works. Short gemm window was
  Healthy, 0 Xid; process attribution needs a longer job started
  after enable.
- [x] **`dcgmi health`.** `--profile` (shared net to host-engine).
  `-g 0 -s pm` then `-c` = **Healthy**. First query of some watches
  still wants 60 s.
- [x] **nsys inside rootfs.** `enter_rootfs.sh` ro-binds host `/sys`
  and the `/usr/local/cuda` tree. Empty image `/sys` made nsys die
  with "Connection to Agent lost". Offline + isolated PID is enough
  once `/sys` is bound. `experiments/falcon/run.sh nsys`. Never wrap
  host `nsys` around `enter_rootfs.sh`. `TMPDIR=/project/tmp`.
- [x] **Prove CUDA kernels in an in-rootfs nsys report.** A5 20-step
  wrap wrote `nsys_in_rootfs/a5_20.nsys-rep`. Top kernels are
  `cutlass3x_sm100_simt_sgemm_*` (fp32 SIMT, not tensor-core).
- [x] **ncu filtered kernel.** In-rootfs `ncu --launch-count 1` on
  `torch.zeros().fill_()` wrote `ncu_fill.ncu-rep` (elementwise, 9
  passes). Ready for a named cutlass kernel when needed.
- [x] **PATH.** Rootfs `PATH` already prepends `/usr/local/cuda/bin`
  and sets `TMPDIR=/project/tmp`. `attach_live.sh` re-enters rootfs.
- [ ] **Step-time tax of DCGM A.0.** Occupancy fields work; still need
  on vs off median step time (Tier 0 1% budget).

### eBPF / perf

- [x] **`bpfcc-tools` in the rootfs.** `sudo chroot` apt, no fleet
  ticket. `cachestat-bpfcc`, `runqlat-bpfcc`, `profile-bpfcc`,
  `offcputime-bpfcc`, and `biolatency-bpfcc` compile in-sandbox once
  `--privileged` ro-binds host `/lib/modules` + `/usr/src`.
- [x] **BPF natively in bwrap, no Docker.** `enter_rootfs.sh
  --privileged` starts bwrap from root without a new user namespace and
  keeps `CAP_BPF`/`CAP_PERFMON`. `/proc/self/ns/user` then equals
  `/proc/1/ns/user`, which is what `bpf()` requires. Docker is no longer
  in any tracing path. Proof:
  `experiments/falcon/results/tooling/rootfs_privileged/REPORT.md`.
- [x] **Host `perf`/`bpftool` in the rootfs.** The rootfs copies are
  distro wrappers that look for a `5.15.152.bsk.9` package. The bound
  host binaries then hit `file too short` because the image ships
  zero-byte placeholders for host-only libs (libunwind, libbfd, ...).
  `enter_rootfs.sh` fills those from `ldd`, only where the rootfs copy
  is empty. `perf` now also works in the default sandbox.
- [x] **`bpftrace` one-liner pack.** `BEGIN`, `profile:hz:99` with
  `ustack`, and `tracepoint:sched:sched_switch` all work under
  `--privileged`. Uprobe on `cuStreamSynchronize` not yet tried.
- [x] **Ambient caps as uid 1018.** `setpriv --ambient-caps` inside a root
  bwrap does keep `CapEff` (`000000c400280000`), but `bpftrace` refuses
  to run as non-root and BCC fails kprobe detach. Rejected; run as root
  in the sandbox and hand ownership back on exit.
- [x] **`cachestat` vs `/proc/io`.** Idle box: HITRATIO 98–100%,
  CACHED_MB ~2.7e6. Matches FineWeb `read_bytes=0` / large `rchar`.
- [x] **`runqlat` under multi-tenant load.** Most mass 0–7 us; a
  tail to 64 ms exists (Ray / other tenants). No science threshold yet.
- [x] **PMU.** `cycles` stays `<not supported>` even as
  container-root. Stop retrying; use `cpu-clock`.

### Python profilers

- [x] **py-spy from host.** `pip install --user --break-system-packages
  py-spy` (0.4.2). Python-only `record` against a bwrap trainer works
  (V3: 998 samples, 0 errors; V1 lived). `--native` dies with
  `UNW_EBADREG`. GPU wait shows up as `_engine_run_backward` /
  `labels.to(device)`, not a mixer scan. Rootfs `b200-runtime` venv
  path is missing; do not rely on `uv pip` there until the venv exists.
- [x] **ptrace across bwrap.** Host py-spy → namespaced python works
  for Python stacks. `--native` unwind does not.
- [x] **Python profilers in the rootfs interpreter.** Training uses
  `/usr/bin/python` (the `b200-runtime` venv path is still missing).
  Installed into that interpreter: memray 1.20.0, Scalene 2.3.0, Fil
  2024.11.2. Reinstall after a rootfs rebuild with
  `install_python_profilers.sh`. Toy launch-wraps:
  `memray run` wrote a bin; `scalene run` starts (toy too short);
  `fil-profile run` wrote a flamegraph (cgroup `memory.high` missing,
  expected).
- [x] **memray attach on a throwaway.** Needs rootfs `gdb` (do **not**
  bind host gdb; it links libpython3.11). gdb script `SUCCESS`,
  `memray_attach.bin` written. Do not attach the hero pid.
- [ ] **memray vs PyTorch CUDA snapshot.** Same OOM-ish toy: which
  allocations each tool sees. Expect memray=host RSS, snapshot=caching
  allocator.
- [x] **Scalene launch wrap.** `scalene run --gpu` on a CUDA toy
  wrote `scalene_gpu.json`. Full overfit overhead still open.
- [x] **tracemalloc.** Stdlib snapshot in-rootfs: 1.6 MiB list.
  Keep it off science defaults.
- [ ] **Fil on a real CPU job.** Toy wrap works. Next: addition
  dataset build. Do not wrap Trainer.
- [ ] **Default recipe.** After the above, write a 10-line "if RSS
  vs if HBM vs if CPU" chooser back into `SKILL.md` step 4.

## Default chooser (until TODOs land)

| Symptom | First tool | Next |
| --- | --- | --- |
| Slow step, GPU busy | `nvidia-smi dmon` + `perf cpu-clock:u` | nsys 1-step if still ambiguous |
| Slow step, GPU idle | `/proc/io` + `iostat` | `cachestat` / `biolatency` |
| RSS growth | `perf` + `/proc/smaps_rollup` | PyTorch CUDA snapshot (new attempt) or py-spy |
| HBM / OOM | PyTorch memory snapshot | nsys mem; not memray/Fil |
| Hang / NCCL | Flight Recorder + RAS | py-spy `dump` after timeout |
| Kernel occupancy | DCGM 1003 **or** ncu after pause | never both at once |
