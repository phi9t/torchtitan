# Live Falcon attach (no sudo, job not stopped)

Captured 2026-09-06T05:09Z against host pid `2211702`
(`python -m experiments.falcon.ablation_runner`, CUDA_VISIBLE_DEVICES=0).
Table at capture: seed-0 A0/A2/A4/A5 done; seed-1 A0/A2 done; addition empty.
Process still `Rl` / ~99.8% CPU after every probe.

## Privilege reality

| Path | Result |
| --- | --- |
| Unprivileged eBPF (`kernel.unprivileged_bpf_disabled=1`) | Blocked |
| `bpftool` as login user | `Operation not permitted` |
| Own-process `perf` (`perf_event_paranoid=2`) | Software `cpu-clock:u` only. Hardware `cycles:u` not supported |
| Docker group + `--privileged --pid=host` chroot to host | No sudo. `bpftool` works; tracepoints work; debugfs/tracefs visible |
| Hardware `cycles` even as container-root | Still `<not supported>` (PMU not exposed) |

Classic BCC (`profile`/`offcputime`/`cachestat`) is not installed. The runnable
stand-ins were `perf` sampling + `sched:sched_switch` stacks + `/proc` + NVML.

## What the job is doing

GPU 0 (B200 `GPU-c3f9bd88-...`): 97-98% SM busy, 21.0 / 183.4 GiB, 64-66 C,
765-775 W, 1965 MHz, no power/thermal violation. GPUs 1-7 idle.
26 fds to `/dev/nvidia0`, plus nvidiactl/uvm. Host disk `%util` ~0-3%;
process `read_bytes=0` while `rchar` ~3.9 GiB (page-cache FineWeb reads).
No `.bin` fds held open (open/close per window).

Host: 216 CPUs, ~98% idle, load 5.3. One Python thread eats a whole core.

3 s thread delta: only tid `2211702` (`python`, state `R`) gained +300
jiffies. Helpers (`cuda-EvtHandlr`, `pt_nccl_watchdg`, `pt_nccl_heartbt`,
`pt_autograd_0`, `pt_tcpstore_uv`) stayed `S` at +0.

## On-CPU (12 s, 99 Hz `cpu-clock:u`, 1K samples, no job pause)

Dominant symbol: `cuStreamSynchronize` in `libcuda.so.1` / `[vdso]`.
Top leaves: vdso 10.8%, then several `libcuda` PCs at 9.1%, 5.3%, 3.8%, ...
Almost every stack ends `cuStreamSynchronize` -> ATen/C10 -> hex (no
frame pointers). This is a **userspace spin-wait on the GPU stream**,
not a Python mixer loop and not a disk wait.

Unprivileged `perf stat` 10 s: 0.80 CPUs (`:u` clock), 0 reported
context-switches:u, 1264 minor faults, 0 major faults.

## Off-CPU / syscalls (8-10 s as container-root, not sudo)

`perf stat` tracepoints on the pid:

- `sched:sched_switch` 196
- `sched:sched_wakeup` 12
- `sys_enter_read` 8
- `sys_enter_ioctl` 0
- `sys_enter_futex` 176
- context-switches 197, migrations 3, page-faults 0
- task-clock 1.000 CPUs (includes kernel)

`sched:sched_switch` stacks (250 samples) are helper threads going idle:

- 33% `pt_nccl_watchdg` `futex_wait` / `pthread_cond_timedwait` /
  `ProcessGroupNCCL::Watchdog::runLoop`
- 40% `cuda-EvtHandlr` `poll` / `schedule_hrtimeout`
- 8% `pt_nccl_heartbt` `recv`/`futex` in `HeartbeatMonitor::runLoop`

The main trainer thread almost never leaves the CPU. CUDA wait is on-CPU
spin (`ioctl` count 0), so BCC `offcputime` would have **missed** the
bottleneck that `profile`/`perf` caught.

## Conclusion for this arm

Compute-bound on GPU 0, host cache-hot, not I/O, not scheduler-queued,
not a leftover Python scan. Step time is GPU kernels; the CPU core is
busy only because the driver spins in `cuStreamSynchronize`.
