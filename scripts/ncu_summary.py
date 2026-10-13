#!/usr/bin/env python3
"""Condense Nsight Compute's raw CSV export into the per-kernel numbers this project discusses.

usage: scripts/ncu_summary.py results/ncu/raw_N4096.csv 4096
writes results/ncu/summary.csv and results/ncu/summary.md
"""
import csv
import math
import re
import sys
from pathlib import Path

# Substring of the demangled kernel name -> kernel id used everywhere else in the project.
# Anything that does not match is a cuBLAS kernel.
KERNEL_IDS = [
    ("sgemm_v1_naive", "1_naive"),
    ("sgemm_v2_coalesced", "2_coalesced"),
    ("sgemm_v3_smem_tiled<16", "3a_smem_tile16"),
    ("sgemm_v3_smem_tiled<32", "3b_smem_tile32"),
    ("sgemm_v4_regblock_1d", "4_regblock_1d"),
    ("sgemm_v5_regblock_2d", "5_regblock_2d"),
    ("sgemm_v6_vectorized", "6_vectorized"),
]
STALL_RE = re.compile(r"smsp__average_warps?_issue_stalled_(.+?)_per_issue_active\.ratio$")


def kernel_id(name):
    for key, kid in KERNEL_IDS:
        if key in name:
            return kid
    return "cublas"


def num(row, key):
    value = row.get(key, "").replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return math.nan


def ratio(a, b):
    return a / b if b and not math.isnan(b) and b != 0 else math.nan


def main():
    raw_path, n = Path(sys.argv[1]), int(sys.argv[2])
    with raw_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{raw_path}: no data")
    units, data = rows[0], rows[1:]  # first row after the header holds the units

    out = []
    for r in data:
        name = r.get("Kernel Name", "")
        t_s = num(r, "gpu__time_duration.sum") * 1e-9  # --print-units base: nanoseconds
        flops = (2 * num(r, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
                 + num(r, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
                 + num(r, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"))
        dram = num(r, "dram__bytes_read.sum") + num(r, "dram__bytes_write.sum")
        l2 = num(r, "lts__t_bytes.sum")
        stalls = []
        for key, value in r.items():
            m = STALL_RE.match(key or "")
            if m:
                v = num(r, key)
                if not math.isnan(v):
                    stalls.append((v, m.group(1)))
        stalls.sort(reverse=True)
        out.append({
            "kernel": kernel_id(name),
            "N": n,
            "ncu_kernel_name": name.split("(")[0].strip(),
            "duration_ms": t_s * 1e3,
            "gflops_under_ncu": ratio(flops, t_s) / 1e9,
            "flops": flops,
            "dram_bytes": dram,
            "dram_gbs": ratio(dram, t_s) / 1e9,
            "dram_pct_of_peak": num(r, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            "l2_bytes": l2,
            "ai_dram": ratio(flops, dram),
            "ai_l2": ratio(flops, l2),
            "sm_throughput_pct": num(r, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            "fma_pipe_pct": num(r, "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
            "issue_active_pct": num(r, "smsp__issue_active.avg.pct_of_peak_sustained_active"),
            "achieved_occupancy_pct": num(r, "sm__warps_active.avg.pct_of_peak_sustained_active"),
            "theoretical_occupancy_pct": num(r, "sm__maximum_warps_per_active_cycle_pct"),
            "registers_per_thread": num(r, "launch__registers_per_thread"),
            "global_ld_sectors_per_request": ratio(num(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
                                                   num(r, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")),
            "smem_ld_bank_conflicts": num(r, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
            "smem_st_bank_conflicts": num(r, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
            "smem_ld_wavefronts": num(r, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
            "top_stalls": "; ".join(f"{s} {v:.1f}" for v, s in stalls[:3]),
        })

    # If cuBLAS launched helper kernels, keep only its main (longest) kernel.
    cublas = [o for o in out if o["kernel"] == "cublas"]
    if len(cublas) > 1:
        keep = max(cublas, key=lambda o: o["duration_ms"])
        out = [o for o in out if o["kernel"] != "cublas" or o is keep]

    order = [kid for _, kid in KERNEL_IDS] + ["cublas"]
    out.sort(key=lambda o: order.index(o["kernel"]))

    out_dir = raw_path.parent
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for o in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in o.items()})

    def f(v, fmt):
        return "n/a" if isinstance(v, float) and math.isnan(v) else format(v, fmt)

    lines = [
        f"Nsight Compute, N = {n}, `--clock-control none` (GPU boosts as in the benchmark).",
        "",
        "| Kernel | DRAM traffic | DRAM throughput | AI vs DRAM (FLOP/B) | AI vs L2 (FLOP/B) | FMA pipe busy | "
        "Achieved / theoretical occupancy | Global load sectors per request | Shared-mem bank conflicts (ld / st) "
        "| Top warp stall reasons (cycles per issued instruction) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for o in out:
        lines.append(
            f"| {o['kernel']} | {f(o['dram_bytes'] / 1e6, ',.0f')} MB | {f(o['dram_gbs'], '.1f')} GB/s "
            f"({f(o['dram_pct_of_peak'], '.0f')}%) | {f(o['ai_dram'], '.1f')} | {f(o['ai_l2'], '.1f')} | "
            f"{f(o['fma_pipe_pct'], '.0f')}% | {f(o['achieved_occupancy_pct'], '.0f')}% / "
            f"{f(o['theoretical_occupancy_pct'], '.0f')}% | {f(o['global_ld_sectors_per_request'], '.1f')} | "
            f"{f(o['smem_ld_bank_conflicts'], ',.0f')} / {f(o['smem_st_bank_conflicts'], ',.0f')} | "
            f"{o['top_stalls']} |")
    cub = [o for o in out if o["kernel"] == "cublas"]
    if cub:
        lines += ["", f"cuBLAS kernel chosen at N = {n}: `{cub[0]['ncu_kernel_name']}`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
