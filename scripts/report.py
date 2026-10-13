#!/usr/bin/env python3
"""STEP 4: turn the CSV files in results/ into charts and markdown tables.

  results/gflops.png     GFLOPS vs N for every kernel and cuBLAS
  results/roofline.png   roofline at the profiled size (Nsight Compute data if available)
  results/results.md     all tables; the README's results block is refreshed from it

Runs as the last step of `make bench`, or alone with `make report`.
"""
import json
import math
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    import pandas as pd
except ImportError as e:  # pragma: no cover
    sys.exit(f"report.py needs matplotlib, numpy and pandas ({e}). "
             "Install them (pip install -r requirements.txt) or run make with PYTHON=/path/to/python.")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# id, legend label, analytical arithmetic intensity = FLOPs per byte the kernel loads from global
# memory, assuming no cache hits (derivation in docs/LEARN.md).
KERNELS = [
    ("1_naive", "1 naive", 2 / 8),                        # 1 FMA (2 FLOP) per 2 floats (8 B) loaded
    ("2_coalesced", "2 coalesced", 2 / 8),
    ("3a_smem_tile16", "3a smem tiling, 16x16", 16 / 4),  # T/4 FLOP per byte for a TxT tile
    ("3b_smem_tile32", "3b smem tiling, 32x32", 32 / 4),
    ("4_regblock_1d", "4 1D register blocking", 2 * 64 * 64 / (4 * (64 + 64))),    # BM*BN/(2*(BM+BN))
    ("5_regblock_2d", "5 2D register blocking", 2 * 128 * 128 / (4 * (128 + 128))),
    ("6_vectorized", "6 2D + float4 loads", 2 * 128 * 128 / (4 * (128 + 128))),
]
CUBLAS = ("cublas", "cuBLAS", math.nan)
LABEL = {k: label for k, label, _ in KERNELS + [CUBLAS]}
COLORS = dict(zip([k for k, _, _ in KERNELS],
                  ["#d62728", "#ff7f0e", "#bcbd22", "#8c564b", "#2ca02c", "#17becf", "#1f77b4"]))
COLORS["cublas"] = "black"


def gfmt(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def pfmt(v):
    return f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"


def load():
    df = pd.read_csv(RES / "results.csv")
    dev = json.loads((RES / "device.json").read_text())
    kinfo = pd.read_csv(RES / "kernels.csv") if (RES / "kernels.csv").exists() else None
    ncu_path = RES / "ncu" / "summary.csv"
    ncu = pd.read_csv(ncu_path) if ncu_path.exists() else None
    return df, dev, kinfo, ncu


def plot_gflops(df, dev):
    sizes = sorted(df.N.unique())
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for kid, label, _ in KERNELS + [CUBLAS]:
        d = df[df.kernel == kid].set_index("N").reindex(sizes)
        y = d.gflops.where(d.status == "ok")  # failed/skipped sizes become gaps in the line
        if y.notna().sum() == 0:
            continue
        is_ref = kid == "cublas"
        ax.plot(sizes, y, marker="s" if is_ref else "o", ms=6 if is_ref else 5, lw=2.8 if is_ref else 1.8,
                color=COLORS[kid], label=label, zorder=3 if is_ref else 2)
    peak = dev["peak_gflops"]
    ax.axhline(peak, ls="--", color="gray", lw=1)
    ax.text(sizes[0], peak * 1.1, f"theoretical FP32 peak: {peak:,.0f} GFLOPS", color="gray", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(top=peak * 1.6)
    ax.set_xlabel("matrix size N   (C = A·B with N×N FP32 matrices)")
    ax.set_ylabel("GFLOPS = 2N³ / time   (log scale)")
    ax.set_title(f"SGEMM performance on {dev['name']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.savefig(RES / "gflops.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def roofline_inputs(df, ncu):
    """Arithmetic intensity per kernel: measured DRAM-level (ncu) if available, else analytical."""
    if ncu is not None and not ncu.empty:
        n = int(ncu.N.iloc[0])
        ai = dict(zip(ncu.kernel, ncu.ai_dram))
        return n, ai, True
    n = int(df[df.status == "ok"].N.max())
    return n, {k: a for k, _, a in KERNELS}, False


def plot_roofline(df, dev, ncu):
    n, ai, measured = roofline_inputs(df, ncu)
    ok = df[(df.status == "ok") & (df.N == n)]
    peak, bw = dev["peak_gflops"], dev["peak_bandwidth_gbs"]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.logspace(-1.5, 3.5, 500)
    ax.loglog(x, np.minimum(bw * x, peak), color="black", lw=2.2,
              label=f"theoretical roof ({bw:.0f} GB/s, {peak:,.0f} GFLOPS)")
    if dev.get("measured_copy_gbs", 0) > 0 and dev.get("measured_fma_gflops", 0) > 0:
        mb, mf = dev["measured_copy_gbs"], dev["measured_fma_gflops"]
        ax.loglog(x, np.minimum(mb * x, mf), color="gray", ls="--", lw=1.5,
                  label=f"measured roof (copy {mb:.0f} GB/s, FMA loop {mf:,.0f} GFLOPS)")
    ridge = peak / bw
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.text(ridge * 1.08, peak * 0.012, f"ridge point\n{ridge:.0f} FLOP/byte", fontsize=8, color="gray")
    ax.text(x[0] * 1.3, bw * x[0] * 3.2, "memory-bound", rotation=33, fontsize=9, color="gray")
    ax.text(ridge * 4, peak * 1.12, "compute-bound", fontsize=9, color="gray")

    for kid, label, _ in KERNELS + [CUBLAS]:
        a = ai.get(kid, math.nan)
        row = ok[ok.kernel == kid]
        if row.empty or a is None or math.isnan(a):
            continue
        g = float(row.gflops.iloc[0])
        ax.plot(a, g, marker="s" if kid == "cublas" else "o", ms=9, color=COLORS[kid], ls="none",
                markeredgecolor="white", label=f"{label}: AI {a:.1f}, {gfmt(g)} GFLOPS")

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(min(10, peak / 1000), peak * 2.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("arithmetic intensity (FLOP per byte of " + ("DRAM traffic, measured by Nsight Compute)"
                                                               if measured else "global-memory loads, analytical)"))
    ax.set_ylabel("achieved GFLOPS (benchmark median)")
    src = ("x: DRAM bytes measured with Nsight Compute" if measured else
           "x: analytical, every load counted as DRAM traffic (Nsight Compute unavailable);\n"
           "caches serve many loads, so points can sit above the roof")
    ax.set_title(f"Roofline at N = {n}, {dev['name']}\n{src}; y: benchmark", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    fig.savefig(RES / "roofline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return n, measured


def table_by_size(df):
    sizes = sorted(df.N.unique())
    lines = ["| Kernel | " + " | ".join(f"N = {n}" for n in sizes) + " |",
             "|---|" + "---:|" * len(sizes)]
    for kid, label, _ in [CUBLAS] + KERNELS:
        cells = []
        for n in sizes:
            r = df[(df.kernel == kid) & (df.N == n)]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            if r.status != "ok":
                cells.append(str(r.status))
            elif kid == "cublas":
                cells.append(f"**{gfmt(r.gflops)}**")
            else:
                cells.append(f"{gfmt(r.gflops)} ({pfmt(r.pct_of_cublas)})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_detail(df, n):
    d = df[df.N == n]
    lines = ["| Kernel | Time (ms) | GFLOPS | % of cuBLAS | % of peak | Max rel. error vs cuBLAS | SM clock (MHz) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for kid, label, _ in [CUBLAS] + KERNELS:
        r = d[d.kernel == kid]
        if r.empty:
            continue
        r = r.iloc[0]
        if r.status != "ok":
            lines.append(f"| {label} | {r.status} | | | | | |")
            continue
        err = "reference" if kid == "cublas" else f"{r.max_rel_err_vs_cublas:.1e}"
        clock = "" if pd.isna(r.sm_clock_mhz) else f"{int(r.sm_clock_mhz)}"
        lines.append(f"| {label} | {r.ms_median:.2f} | {gfmt(r.gflops)} | {pfmt(r.pct_of_cublas)} | "
                     f"{pfmt(r.pct_of_peak)} | {err} | {clock} |")
    return "\n".join(lines)


def table_resources(kinfo):
    lines = ["| Kernel | Threads / block | Registers / thread | Shared memory / block | Spills (local mem) | "
             "Blocks / SM | Theoretical occupancy | Limited by |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in kinfo.iterrows():
        lines.append(f"| {LABEL.get(r.kernel, r.kernel)} | {r.threads_per_block} | {r.registers_per_thread} | "
                     f"{r.static_smem_bytes / 1024:.1f} KiB | {r.local_mem_bytes} B | {r.blocks_per_sm} | "
                     f"{r.theoretical_occupancy_pct:.0f}% | {r.occupancy_limited_by} |")
    return "\n".join(lines)


def table_ai():
    lines = ["| Kernel | Analytical AI (FLOP per byte loaded from global memory) |", "|---|---:|"]
    for kid, label, a in KERNELS:
        lines.append(f"| {label} | {a:g} |")
    return "\n".join(lines)


def update_readme(block):
    readme = ROOT / "README.md"
    start, end = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
    if not readme.exists():
        return
    text = readme.read_text()
    if start in text and end in text:
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        readme.write_text(pre + start + "\n" + block + "\n" + end + post)


def main():
    df, dev, kinfo, ncu = load()
    plot_gflops(df, dev)
    n_roof, measured = plot_roofline(df, dev, ncu)
    n_max = int(df[df.status == "ok"].N.max())

    failed = df[df.status == "FAILED"]
    sections = [
        "### GFLOPS by matrix size (percent of cuBLAS in parentheses)\n\n" + table_by_size(df),
        f"### Details at N = {n_max}\n\n" + table_detail(df, n_max),
    ]
    if not failed.empty:
        sections.append("**Correctness failures (not timed):** " +
                        ", ".join(f"{r.kernel} at N={r.N}" for _, r in failed.iterrows()))
    block = "\n\n".join(sections)

    extra = []
    if kinfo is not None:
        extra.append("### Kernel resources (from the compiled binary)\n\n" + table_resources(kinfo))
    extra.append("### Analytical arithmetic intensity\n\n" + table_ai())
    ncu_md = RES / "ncu" / "summary.md"
    if ncu_md.exists():
        extra.append("### Nsight Compute summary\n\n" + ncu_md.read_text().strip())
    else:
        extra.append("### Nsight Compute summary\n\nNot available (profiling was not permitted or ncu is "
                     "missing); the roofline uses the analytical intensities above.")

    (RES / "results.md").write_text(f"# Results: {dev['name']}\n\n" + block + "\n\n" + "\n\n".join(extra) + "\n")
    update_readme(block)
    print(block)
    print(f"\nWrote results/gflops.png, results/roofline.png (N = {n_roof}, "
          f"{'measured' if measured else 'analytical'} arithmetic intensity), results/results.md")


if __name__ == "__main__":
    main()
