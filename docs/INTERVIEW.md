# 20 likely NVIDIA DevTech interview questions, answered with this project's numbers

Every number below was measured on the RTX 3050 6GB Laptop GPU (20 SMs, compute capability 8.6, peak
7,639 FP32 GFLOPS, 168 GB/s) by this repository. Sources: [`results/results.csv`](../results/results.csv),
[`results/ncu/summary.csv`](../results/ncu/summary.csv), [`results/kernels.csv`](../results/kernels.csv),
[`results/experiments.md`](../results/experiments.md). Unless stated otherwise, N = 4096. Concepts are
explained in [LEARN.md](LEARN.md).

A good interview answer has three parts: the concept in one sentence, the evidence from your own
measurement, and the trade-off or limitation. The answers below follow that shape.

---

### 1. What is a warp, and why does it matter for performance?

A warp is 32 consecutive threads of a block that the SM schedules as one unit: they execute the same
instruction at the same time on 32 lanes (SIMT). It matters because **memory accesses and branches are
handled per warp**. The 32 addresses of one load instruction are served together, so what matters is the
pattern across the warp. In my naive kernel, consecutive threads of a warp walked down a column of A,
and each load instruction needed 16.5 sectors on average; the coalesced version needed 2.5. That one
mapping change was a **6.8× speedup** (73.7 → 502 GFLOPS).

### 2. What is memory coalescing? How did you show it?

When the 32 threads of a warp access consecutive addresses, the hardware merges them into the minimum
number of 32-byte sectors: 4 sectors for 32 floats. Kernels 1 and 2 execute the same 4.30 billion load
instructions. Nsight Compute measured **70.9 billion sectors** for kernel 1 against **10.7 billion** for
kernel 2. Per request that is 16.5 vs 2.5 sectors, exactly (32 + 1)/2 and (4 + 1)/2, because the B load is
either scattered (32) or contiguous (4) while the A load is a broadcast (1). The main stall `lg_throttle`
(global-memory queue full) fell from 212 to 18 cycles per instruction.

### 3. If kernel 1 was memory-bound, why was DRAM only 3% busy?

Because the bottleneck was not DRAM bandwidth but the **number of transactions** the load/store path had to
handle. The L1 and L2 caches served most of the redundant requests: kernel 1's measured DRAM arithmetic
intensity is 14.5 FLOP/byte, not the 0.25 a cache-less model predicts. But each warp load still occupied
the L1 for 32 sectors, and the queue for memory instructions stayed full (`lg_throttle`). "Memory-bound"
can mean DRAM, L2, L1 or the load/store unit. The profiler tells you which.

### 4. What is shared memory and how did you use it?

Shared memory is fast on-chip SRAM (up to 100 KiB per SM here) that the threads of one block share and the
programmer manages explicitly. In kernel 3 each block loads a TILE×TILE tile of A and of B into shared
memory, one element per thread, and every thread then reuses those values TILE times. Global load requests
fell from 4.30 billion (kernel 2) to **269 M (TILE 16) and 135 M (TILE 32)**. Speed improved only 1.3×,
to 664 GFLOPS, because the bottleneck moved: each FMA still needed two *shared* loads, and `mio_throttle`
(shared-memory queue full) became the top stall at ~20 cycles per instruction.

### 5. Why does the tiled kernel need two `__syncthreads()`? What breaks if you remove one?

Warps of a block progress at different speeds.

- **The barrier after the loads** guarantees the whole tile is written before anyone reads it. Without
  it, a fast warp reads entries a slow warp hasn't stored yet.
- **The barrier after the compute loop** guarantees everyone has finished reading the tile before the next
  iteration overwrites it. Without it, a fast warp overwrites data others still need.

Both are race conditions that corrupt results only intermittently. That's why my harness compares every
output element against cuBLAS, and skips timing for any kernel that fails the check.

### 6. What are shared-memory bank conflicts? Did your kernels have any?

Shared memory has 32 banks, each 4 bytes wide. If threads of a warp access different words in the same
bank, the accesses are serialized. Reading the same word is a broadcast and costs nothing extra.

- **Kernel 3** is conflict-free by design (`As[ty][k]` broadcast, `Bs[k][tx]` consecutive). Nsight Compute
  measured 0 load conflicts for TILE 32.
- **Kernels 5 and 6** have **268 M load conflicts out of ~672 M wavefronts**. Each thread reads 8
  consecutive B values with two 128-bit loads, and thread columns are 8 words apart, so a warp's 16
  addresses use only half the banks.
- **cuBLAS's kernel has zero load conflicts** and needs 2.5× fewer shared-memory wavefronts than my best
  kernel. That is one concrete reason it's faster. The fix is a padded or swizzled shared-memory layout.

### 7. What is occupancy and how do you compute it?

Occupancy is resident warps per SM divided by the maximum, 48 on this GPU. It is capped by whichever
resource runs out first: threads (1536 per SM), registers (65,536), shared memory (100 KiB) or block slots
(16). My harness computes this with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`:

| Kernel | Block | Registers | Blocks / SM | Occupancy | Limit |
|---|---:|---:|---:|---:|---|
| 3a (tile 16) | 256 | 38 | 6 | 100% | threads |
| 3b (tile 32) | 1024 | 38 | 1 | 67% | threads |
| 6 | 256 | 102 | 2 | 33% | registers |

Nsight Compute's *achieved* occupancy matched the theoretical values within a percent for every kernel.

### 8. Is higher occupancy always better?

No. Kernel 3a has **100%** occupancy and runs at 664 GFLOPS. Kernel 6 has **33%** and runs at 4,173
GFLOPS, 6.3× faster. Occupancy is one way to hide latency (more warps to switch between). The other is
instruction-level parallelism: a kernel-6 thread has 64 independent FMAs to issue. What matters is having
enough independent work in flight. In kernel 6 the top stall reason is `not_selected`: warps are ready and
merely waiting for an issue slot.

### 9. What is register pressure? Show me its effect.

Registers are the fastest storage, but the 65,536 per SM are shared among all resident threads. More
registers per thread means fewer warps fit, and if the compiler runs out it **spills** to local memory,
which lives off-chip. Two experiments ([`results/experiments.md`](../results/experiments.md)):

- **Kernel 5 used 145 registers**, so only 1 block (8 warps) fit per SM, at ~3,000 GFLOPS.
  `__launch_bounds__(256, 2)` made the compiler fit into 128 registers with **zero spills**. Two blocks
  fit, and it ran at ~3,720 GFLOPS (**+24%**).
- **Kernel 6 forced to 80 registers** (to fit 3 blocks): 112 bytes per thread spilled, and it ran at
  746 GFLOPS instead of 4,234, **5.7× slower**.

### 10. Does warp divergence matter in GEMM?

Very little, and it didn't here. Divergence happens when threads of the same warp take different branches,
so the warp executes both paths with lanes masked. My kernels' only branch is the bounds check
`if (row < N && col < N)`, which is uniform within a warp except at the matrix edge. Kernels 4–6 have no
branches in the inner loop at all; they require N to be a multiple of the tile size. Divergence becomes a
real concern in things like irregular graph algorithms or per-element conditionals, not dense GEMM.

### 11. How does tile size affect performance?

Larger tiles mean more reuse per global load, but use more resources per block. TILE 32 halved DRAM
traffic compared with TILE 16 (9.2 GB vs 17.7 GB). But its 1024-thread blocks fit only one per SM (67%
occupancy), and each barrier waits for 32 warps. Net result at N = 4096: 663 vs 664 GFLOPS, a wash. At
N = 1024, TILE 16 wins (743 vs 646 GFLOPS). The same trade-off appears at block level. At N = 256 the
128×128-tile kernels launch only 4 blocks for 20 SMs and reach 618–762 GFLOPS, while the 64×64-tile
kernel 4 launches 16 blocks and reaches 1,260 GFLOPS.

### 12. What is register blocking (thread coarsening) and why did it help so much?

Each thread computes several outputs and keeps operands in registers, so one value loaded from shared
memory is reused many times.

- **Kernel 4** (8 outputs in a column, 1 B value reused 8 times) cut shared-load wavefronts from
  3.22 billion to 1.34 billion and ran **2.9×** faster than kernel 3 (1,894 GFLOPS).
- **Kernel 5** (an 8×8 outer product, 16 shared loads per 64 FMAs) cut them to 672 M and ran **1.9×**
  faster again (3,594 GFLOPS).

The FMA pipe went from 10% busy (kernel 3) to 23%, then 49%.

### 13. Explain the roofline model. Where do your kernels sit?

Attainable FLOP/s = min(peak compute, arithmetic intensity × peak bandwidth). The ridge point here is
7,639 / 168 = **45 FLOP/byte**. With DRAM bytes measured by Nsight Compute:

- **Left of the ridge:** naive 14.5, coalesced 14.1, tiling 7.8 and 14.9, kernel 4 29.3.
- **Right of the ridge:** kernels 5 and 6 at **56.6**, cuBLAS at 68.8.

But kernels 1–3 sit far *below* their memory roof. They are limited by on-chip bottlenecks (the
load/store queue, shared memory), not DRAM. The roofline tells you the ceiling; the profiler tells you
which pipe hits it first. See [`results/roofline.png`](../results/roofline.png).

### 14. How do you tell whether a kernel is compute-bound or memory-bound?

Compare its utilizations from Nsight Compute against their peaks: DRAM throughput %, SM/pipe throughput %,
and the stall reasons.

- **Kernel 2:** SM throughput **94.5%** (the load/store path saturated), DRAM **20%**, FMA pipe 18%, top
  stall `lg_throttle`. Bound by the memory pipeline inside the SM.
- **Kernel 6:** DRAM 48%, FMA pipe **58%**, issue slots 63%, top stall `not_selected`. Heading toward
  compute-bound (issue- and FMA-limited).

The analytical check is arithmetic intensity against the ridge point: 56.6 FLOP/byte against 45.

### 15. What is the arithmetic intensity of naive vs tiled GEMM?

Counting only the loads each kernel *issues*, with no caching:

| Kernel | Analytical AI | Reasoning |
|---|---:|---|
| Naive | 0.25 FLOP/byte | 2 FLOPs per 8 bytes |
| Tiled, tile T | T/4 | each loaded value used T times |
| Block tile BM×BN | BM·BN / (2(BM+BN)) | 16 for 64×64, 32 for 128×128 |

Measured at the DRAM level, the values are higher (14.5, 7.8/14.9, 29.3, 56.6) because the L1 and L2
caches reuse data across threads and blocks. The math itself (3N² values for 2N³ FLOPs) allows N/6 ≈ 680
at N = 4096. GEMM *can* be very compute-bound, if the implementation reuses data.

### 16. Why is cuBLAS still faster than your best kernel?

Kernel 6 reaches **95.1%** of cuBLAS at N = 4096 (4,173 vs 4,388 GFLOPS), but only 39% at N = 256.
Nsight Compute shows the difference: cuBLAS (`ampere_sgemm_128x64_nn`) uses **75% of issue slots vs 63%**,
keeps the **FMA pipe 64% busy vs 58%**, has **zero shared-memory bank conflicts vs 268 M**, needs **2.5×
fewer shared-memory wavefronts**, and moves 2.0 vs 2.4 GB from DRAM.

The techniques behind that are conflict-free (swizzled) shared layouts, warp-level tiling, double
buffering so loads overlap compute, SASS-level tuning, and choosing a different kernel per problem size.
That last one is why cuBLAS is much better at small N, where my 128×128 tiles leave 16 of 20 SMs idle.

### 17. How did you verify correctness? Why not compare for exact equality?

Every kernel runs once with non-trivial α = 0.75, β = 0.25 on a random initial C, which also tests the
`+ βC` part. Every output element is then compared with cuBLAS, and for N ≤ 512 with a double-precision
CPU loop (which also validates my cuBLAS call). Floating-point addition isn't associative, so different
summation orders give slightly different results; exact equality is the wrong test. I use a tolerance of
K·FLT_EPSILON relative to the largest reference value. Measured relative errors are 4.9·10⁻⁷ at N = 256
and 3.1·10⁻⁶ at N = 4096, against tolerances of 3·10⁻⁵ and 5·10⁻⁴. An indexing bug or a missing barrier
produces errors of order 1. A kernel that fails isn't timed.

### 18. How did you time the kernels, and what could go wrong?

- **CUDA events** around each launch measure GPU time, not CPU launch overhead.
- **3 warm-up launches** absorb one-time costs.
- **The median of 10** resists outliers.
- **A short busy-wait kernel is queued first**, so the CPU can enqueue all 10 launches before the GPU
  starts. Tiny kernels are then never timed with idle gaps between them.
- **A 1-second GPU spin-up** happens before the first measurement. I found I needed it when an early run
  showed the GPU still at 210 MHz.

On a laptop the biggest pitfall is **clocks**. This GPU is power-limited at about 60 W (`nvidia-smi`
throttle reason: software power cap). As a result, cuBLAS measured 5,533 GFLOPS at N = 1024 but 4,388 at
N = 4096, and runs vary by a few percent. I also found that NVML's reported clock doesn't track the real
boost clock. My FMA microbenchmark did more work per second than NVML's clock reading allows. Nsight
Compute's cycle counter is the trustworthy clock measurement.

### 19. How did performance change with matrix size, and why?

At small N there isn't enough parallel work. At N = 256 cuBLAS reaches only 1,973 GFLOPS (26% of peak)
and my 128×128-tile kernels launch just 4 blocks for 20 SMs. Performance rises as N grows until the GPU
is full: cuBLAS reaches 5,533 GFLOPS at N = 1024 and kernel 6 reaches 4,643 at N = 2048. At N = 4096 the
power cap pulls both down. The naive kernel is flat at ~73 GFLOPS from N = 512 onward, because it is
limited by memory transactions at every size.

### 20. What would you do next?

In order of expected payoff, each targeting a measured gap:

1. **Fix the shared-memory bank conflicts** (268 M) with padding or an XOR swizzle.
2. **Warp tiling**, so each warp owns a sub-tile and register fragments are reused more.
3. **Double buffering with `cp.async`** (Ampere's asynchronous global→shared copy), to overlap the next
   tile's load with the current compute and remove barrier stalls.
4. **Autotune tile sizes per N**, and use smaller tiles or split-K for small matrices to fill all 20 SMs.
5. **Tensor cores.** cuBLAS with TF32 or FP16 would leave this FP32 comparison behind entirely, at
   reduced input precision.
