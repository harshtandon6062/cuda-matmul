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
