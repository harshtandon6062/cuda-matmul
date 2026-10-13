# Results: NVIDIA GeForce RTX 3050 6GB Laptop GPU

### GFLOPS by matrix size (percent of cuBLAS in parentheses)

| Kernel | N = 256 | N = 512 | N = 1024 | N = 2048 | N = 4096 |
|---|---:|---:|---:|---:|---:|
| cuBLAS | **1,973** | **4,228** | **5,533** | **5,124** | **4,388** |
| 1 naive | 52.3 (2.6%) | 72.4 (1.7%) | 72.9 (1.3%) | 74.0 (1.4%) | 73.7 (1.7%) |
| 2 coalesced | 428 (22%) | 588 (14%) | 533 (9.6%) | 513 (10%) | 502 (11%) |
| 3a smem tiling, 16x16 | 655 (33%) | 771 (18%) | 743 (13%) | 681 (13%) | 664 (15%) |
| 3b smem tiling, 32x32 | 541 (27%) | 728 (17%) | 646 (12%) | 676 (13%) | 663 (15%) |
| 4 1D register blocking | 1,260 (64%) | 1,827 (43%) | 2,081 (38%) | 1,859 (36%) | 1,894 (43%) |
| 5 2D register blocking | 618 (31%) | 2,545 (60%) | 3,452 (62%) | 3,908 (76%) | 3,594 (82%) |
| 6 2D + float4 loads | 762 (39%) | 2,621 (62%) | 4,178 (76%) | 4,643 (91%) | 4,173 (95%) |

### Details at N = 4096

| Kernel | Time (ms) | GFLOPS | % of cuBLAS | % of peak | Max rel. error vs cuBLAS | SM clock (MHz) |
|---|---:|---:|---:|---:|---:|---:|
| cuBLAS | 31.32 | 4,388 | 100% | 57% | reference | 1920 |
| 1 naive | 1865.96 | 73.7 | 1.7% | 1.0% | 3.1e-06 | 1920 |
| 2 coalesced | 273.60 | 502 | 11% | 6.6% | 3.1e-06 | 1680 |
| 3a smem tiling, 16x16 | 206.88 | 664 | 15% | 8.7% | 3.1e-06 | 1605 |
| 3b smem tiling, 32x32 | 207.32 | 663 | 15% | 8.7% | 3.1e-06 | 1830 |
| 4 1D register blocking | 72.57 | 1,894 | 43% | 25% | 3.1e-06 | 1612 |
| 5 2D register blocking | 38.25 | 3,594 | 82% | 47% | 3.1e-06 | 1815 |
| 6 2D + float4 loads | 32.93 | 4,173 | 95% | 55% | 3.1e-06 | 1642 |

### Kernel resources (from the compiled binary)

| Kernel | Threads / block | Registers / thread | Shared memory / block | Spills (local mem) | Blocks / SM | Theoretical occupancy | Limited by |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 naive | 1024 | 40 | 0.0 KiB | 0 B | 1 | 67% | threads + registers |
| 2 coalesced | 1024 | 40 | 0.0 KiB | 0 B | 1 | 67% | threads + registers |
| 3a smem tiling, 16x16 | 256 | 38 | 2.0 KiB | 0 B | 6 | 100% | threads + registers |
| 3b smem tiling, 32x32 | 1024 | 38 | 8.0 KiB | 0 B | 1 | 67% | threads + registers |
| 4 1D register blocking | 512 | 52 | 4.0 KiB | 0 B | 2 | 67% | registers |
| 5 2D register blocking | 256 | 128 | 8.0 KiB | 0 B | 2 | 33% | registers |
| 6 2D + float4 loads | 256 | 102 | 8.0 KiB | 0 B | 2 | 33% | registers |

### Analytical arithmetic intensity

| Kernel | Analytical AI (FLOP per byte loaded from global memory) |
|---|---:|
| 1 naive | 0.25 |
| 2 coalesced | 0.25 |
| 3a smem tiling, 16x16 | 4 |
| 3b smem tiling, 32x32 | 8 |
| 4 1D register blocking | 16 |
| 5 2D register blocking | 32 |
| 6 2D + float4 loads | 32 |

### Nsight Compute summary

Nsight Compute, N = 4096, `--clock-control none` (GPU boosts as in the benchmark).

| Kernel | SM clock | DRAM traffic | DRAM throughput | AI vs DRAM (FLOP/B) | AI vs L2 (FLOP/B) | FMA pipe busy | Achieved / theoretical occupancy | Global load sectors per request | Shared-mem bank conflicts (ld / st) | Top warp stall reasons (cycles per issued instruction) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1_naive | 1910 MHz | 9,488 MB | 5.1 GB/s (3%) | 14.5 | 6.4 | 2% | 66% / 67% | 16.5 | 0 / 0 | lg_throttle 211.6; long_scoreboard 8.7; not_selected 2.9 |
| 2_coalesced | 1574 MHz | 9,758 MB | 33.8 GB/s (20%) | 14.1 | 7.9 | 18% | 67% / 67% | 2.5 | 0 / 0 | lg_throttle 18.3; not_selected 3.3; long_scoreboard 2.6 |
| 3a_smem_tile16 | 1566 MHz | 17,719 MB | 84.1 GB/s (50%) | 7.8 | 4.0 | 10% | 100% / 100% | 3.9 | 115,625 / 12,573,476 | mio_throttle 19.1; long_scoreboard 7.3; barrier 6.5 |
| 3b_smem_tile32 | 1686 MHz | 9,213 MB | 41.6 GB/s (25%) | 14.9 | 7.9 | 8% | 67% / 67% | 4.0 | 0 / 31,561,616 | mio_throttle 21.1; long_scoreboard 5.5; not_selected 3.6 |
| 4_regblock_1d | 1501 MHz | 4,700 MB | 60.9 GB/s (36%) | 29.3 | 15.8 | 23% | 67% / 67% | 4.0 | 77,845 / 8,361,980 | mio_throttle 6.7; long_scoreboard 4.9; barrier 2.8 |
| 5_regblock_2d | 1598 MHz | 2,431 MB | 68.4 GB/s (41%) | 56.6 | 30.7 | 49% | 33% / 33% | 4.3 | 268,435,456 / 0 | not_selected 2.1; dispatch_stall 1.3; selected 1.0 |
| 6_vectorized | 1560 MHz | 2,429 MB | 80.1 GB/s (48%) | 56.6 | 31.0 | 58% | 33% / 33% | 15.6 | 268,435,456 / 16,777,216 | not_selected 1.6; selected 1.0; long_scoreboard 1.0 |
| cublas | 1471 MHz | 2,001 MB | 70.0 GB/s (42%) | 68.8 | 19.9 | 64% | 33% / 33% | 3.9 | 0 / 2,097,152 | not_selected 1.9; wait 1.1; selected 1.0 |

cuBLAS kernel chosen at N = 4096: `ampere_sgemm_128x64_nn`
