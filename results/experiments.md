# Experiments: register pressure vs occupancy

Measured on the RTX 3050 6GB Laptop GPU on 2026-10-13 with `./bin/sgemm --sizes 2048,4096 --kernels ...`
(median of 10 timed runs each). These numbers are from separate runs of the harness, not from
`results/results.csv`, so they differ slightly from the main table (this laptop GPU is power-limited at
about 60 W, so clocks vary from run to run).

The knob is `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)`. With `minBlocksPerSM = b` the
compiler must fit `b` blocks of 256 threads into the SM's 65536 registers, so it limits itself to
65536 / (256·b) registers per thread, spilling to local memory if the code needs more.

## Kernel 5 (2D register blocking): capping registers helped

| `__launch_bounds__` | Registers / thread | Spills | Blocks / SM (occupancy) | GFLOPS at N=2048 | GFLOPS at N=4096 |
|---|---:|---:|---:|---|---|
| `(256)` | 145 | 0 B | 1 (16.7%) | 3062, 2710, 2999, 2982 | 2988, 3000, 3000, 2997 |
| `(256, 2)` (kept) | 128 | 0 B | 2 (33.3%) | 3539, 3791 | 3716, 3723 |

145 registers is just over the 128 that would let a second block fit, so each SM ran only 8 warps.
Capping at 128 cost no spills and doubled the resident warps, giving **+24% at N=4096**. The kernel
keeps `(256, 2)`.

## Kernel 6 (float4): forcing more occupancy backfired

| `__launch_bounds__` | Registers / thread | Spills | Blocks / SM (occupancy) | GFLOPS at N=2048 | GFLOPS at N=4096 |
|---|---:|---:|---:|---|---|
| `(256)` (kept) | 102 | 0 B | 2 (33.3%) | 4813, 4609, 4816 | 4272, 4238, 4234 |
| `(256, 3)` | 80 | 112 B per thread | 3 (50%) | 764, 764 | 746, 746 |

To fit a third block the compiler had to squeeze 102 registers into 80. The 64 accumulators no longer fit,
so some were **spilled** to local memory, which lives in off-chip DRAM (cached in L1/L2). Every inner-loop
FMA then waited on memory: **5.7× slower** despite 50% more occupancy. Occupancy is only useful if
the extra warps can do work without spilling.
