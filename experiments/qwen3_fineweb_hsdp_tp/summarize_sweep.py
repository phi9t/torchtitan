# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parse sweep logs into results/summary.md.

Reads each ``results/<name>.log`` produced by ``sweep.sh``, extracts the mesh
line, the HSDP/FSDP application line, and the per-step metrics that the
torchtitan metrics logger prints (metrics.py:523-532), and writes a markdown
comparison table. Read-only over the logs; only writes ``summary.md``.
"""

import argparse
import os
import re


# The metrics logger wraps fields in ANSI color codes; strip them first.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Fields from metrics.py:523-532. tps uses thousands separators.
STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[-\d.]+)\s+"
    r"grad_norm:\s*(?P<grad_norm>[-\d.]+)\s+"
    r"memory:\s*(?P<memory>[\d.]+)GiB\((?P<mem_pct>[\d.]+)%\)\s+"
    r"tps:\s*(?P<tps>[\d,]+)\s+"
    r"tflops:\s*(?P<tflops>[\d,.]+)\s+"
    r"mfu:\s*(?P<mfu>[\d.]+%|N/A)"
)

MESH_RE = re.compile(r"Building device mesh with parallelism:\s*(?P<mesh>.+)$")
APPLIED_RE = re.compile(r"Applied (HSDP|FSDP) to the model")


def parse_log(path: str) -> dict:
    """Extract mesh line, HSDP/FSDP flavor, and per-step metrics from one log."""
    mesh = ""
    applied = ""
    steps: list[dict] = []
    if not os.path.exists(path):
        return {"missing": True}
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw).rstrip("\n")
            m = MESH_RE.search(line)
            if m:
                mesh = m.group("mesh").strip()
            a = APPLIED_RE.search(line)
            if a:
                applied = a.group(0)
            s = STEP_RE.search(line)
            if s:
                steps.append(
                    {
                        "step": int(s.group("step")),
                        "loss": float(s.group("loss")),
                        "grad_norm": float(s.group("grad_norm")),
                        "memory": float(s.group("memory")),
                        "tps": int(s.group("tps").replace(",", "")),
                        "tflops": float(s.group("tflops").replace(",", "")),
                        "mfu": s.group("mfu"),
                    }
                )
    return {"mesh": mesh, "applied": applied, "steps": steps, "missing": False}


def summarize(name: str, info: dict) -> dict:
    """Reduce a parsed log to the row we tabulate."""
    if info.get("missing"):
        return {"name": name, "status": "LOG MISSING"}
    steps = info["steps"]
    if not steps:
        return {
            "name": name,
            "status": "no steps parsed",
            "mesh": info.get("mesh", ""),
            "applied": info.get("applied", ""),
        }
    first, last = steps[0], steps[-1]
    # Peak memory and median-ish throughput over the logged steps (skip step 1
    # for throughput since startup skews it).
    peak_mem = max(s["memory"] for s in steps)
    warm = steps[1:] or steps
    avg_tps = sum(s["tps"] for s in warm) / len(warm)
    avg_tflops = sum(s["tflops"] for s in warm) / len(warm)
    return {
        "name": name,
        "status": "ok",
        "mesh": info.get("mesh", ""),
        "applied": info.get("applied", ""),
        "n_steps": len(steps),
        "loss_first": first["loss"],
        "loss_last": last["loss"],
        "grad_norm_last": last["grad_norm"],
        "peak_mem": peak_mem,
        "avg_tps": avg_tps,
        "avg_tflops": avg_tflops,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument(
        "--names",
        required=True,
        help="space-separated config names, in table order",
    )
    args = ap.parse_args()

    names = [n for n in args.names.split() if n]
    rows = [
        summarize(n, parse_log(os.path.join(args.results_dir, f"{n}.log")))
        for n in names
    ]

    out = os.path.join(args.results_dir, "summary.md")
    lines: list[str] = []
    lines.append("# HSDP+TP sweep summary\n")
    lines.append(
        "Per-config metrics from `sweep.sh` "
        "(qwen3_debugmodel_fineweb, 8 GPUs, --debug.seed=42 "
        "--debug.deterministic, 10 steps). Throughput/tflops are averaged over "
        "steps 2-10 to drop warmup; peak memory is the max reserved over all "
        "steps.\n"
    )
    header = (
        "| config | applied | loss[1] | loss[10] | grad_norm[10] | "
        "peak mem (GiB) | avg tps | avg tflops |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        if r["status"] != "ok":
            lines.append(
                f"| {r['name']} | {r.get('applied', '')} | "
                f"({r['status']}) |  |  |  |  |  |"
            )
            continue
        lines.append(
            f"| {r['name']} | {r['applied']} | {r['loss_first']:.5f} | "
            f"{r['loss_last']:.5f} | {r['grad_norm_last']:.4f} | "
            f"{r['peak_mem']:.2f} | {r['avg_tps']:,.0f} | {r['avg_tflops']:.2f} |"
        )
    lines.append("\n## Mesh lines\n")
    for r in rows:
        lines.append(f"- **{r['name']}**: `{r.get('mesh', '')}`")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
