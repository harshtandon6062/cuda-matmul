// kernels.cuh — the interface every SGEMM kernel file implements.
//
// SGEMM = Single-precision GEneral Matrix Multiply:      C = alpha * (A x B) + beta * C
//
//   A is M x K, B is K x N, C is M x N, all stored ROW-MAJOR (the C/C++ convention):
//   element (row r, column c) of a matrix with `cols` columns lives at X[r * cols + c].
//
// Each file in src/kernels/ contains one __global__ kernel, a host "launcher" that chooses the
// grid and block shape, and a *_spec() function that describes it, so the benchmark harness
// (src/bench.cu) can test and time every kernel in exactly the same way.
#pragma once

#include <cuda_runtime.h>

// Enqueue one SGEMM on the default stream. All pointers are device pointers.
using LaunchFn = void (*)(int M, int N, int K, float alpha, const float* A, const float* B,
                          float beta, float* C);

struct KernelSpec {
    const char* name;         // short id used in CSV files and charts, e.g. "2_coalesced"
    const char* description;  // one line for console output
    int size_multiple;        // M, N and K must be multiples of this (1 = any size works)
    int threads_per_block;    // block size the launcher uses
    const void* kernel;       // the __global__ function itself, for register/occupancy queries
    LaunchFn launch;
};

KernelSpec naive_spec();        // 01_naive.cu
KernelSpec coalesced_spec();    // 02_coalesced.cu
KernelSpec smem_tile16_spec();  // 03_smem_tiled.cu, TILE = 16
KernelSpec smem_tile32_spec();  // 03_smem_tiled.cu, TILE = 32
KernelSpec regblock_1d_spec();  // 04_regblock_1d.cu
KernelSpec regblock_2d_spec();  // 05_regblock_2d.cu
KernelSpec vectorized_spec();   // 06_vectorized.cu
