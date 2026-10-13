# Resume bullets

Both bullets use only numbers measured by this repository (see [`results/`](../results/)).

**SGEMM from scratch in CUDA (C++/CUDA, cuBLAS, Nsight Compute)** · github.com/harshtandon6062/cuda-matmul

- Wrote and optimized six FP32 matrix-multiply (SGEMM) CUDA kernels on an RTX 3050 Laptop GPU, adding
  global-memory coalescing, shared-memory tiling, 1D/2D register blocking and float4 vectorization; reached
  **4,173 GFLOPS at N=4096, 95% of cuBLAS** and **56.6× faster than the naive kernel**, with every output
  verified against cuBLAS.
- Built a benchmarking and profiling harness (CUDA events, cuBLAS and CPU correctness checks, Nsight
  Compute metrics, roofline charts) and used it to explain each bottleneck. For example: coalescing cut
  global-load sectors per request from 16.5 to 2.5 (6.8× speedup), and capping registers with
  `__launch_bounds__` doubled occupancy for a 24% gain, while over-capping caused spills and a 5.7×
  slowdown.

## If asked "what does 95% of cuBLAS mean?"

It is measured at N = 4096 in cuBLAS's default FP32 mode (no tensor cores), median of 10 runs on a
power-limited laptop GPU. At small sizes the gap is larger (39% of cuBLAS at N = 256), because 128×128
tiles leave most of the 20 SMs idle. Say this before the interviewer asks; it shows you understand your
own numbers.
