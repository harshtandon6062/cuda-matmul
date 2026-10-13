// common.cuh — small helpers shared by every file in the project.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Every CUDA runtime call returns a cudaError_t. Ignoring it is the classic beginner mistake:
// a failed launch or an out-of-bounds access then surfaces much later as a confusing error
// somewhere else. CUDA_CHECK stops the program immediately and says where.
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n  in: %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(err__), cudaGetErrorName(err__),    \
                         #call);                                                          \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

// Integer division rounded up: how many blocks of size b are needed to cover a items.
__host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }
