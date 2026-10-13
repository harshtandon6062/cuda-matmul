// =============================================================================
// Kernel 2 — GLOBAL-MEMORY COALESCING: same work as kernel 1, better thread -> element mapping.
// =============================================================================
//
// Exactly the same arithmetic and the same number of load instructions as kernel 1. The only
// change is which thread computes which element:
//
//     column of C <-  blockIdx.x * blockDim.x + threadIdx.x     (x = the FAST-changing index)
//     row of C    <-  blockIdx.y * blockDim.y + threadIdx.y
//
// Now the 32 threads of a warp own 32 CONSECUTIVE COLUMNS of one row of C. In step k they load
//
//     A[row*K + k]        the same address for the whole warp -> 1 sector, broadcast to all
//     B[k*N + col + 0..31] 32 consecutive floats = 128 contiguous bytes = 4 sectors, all bytes used
//
// When the threads of a warp access consecutive addresses, the hardware merges ("coalesces")
// their 32 requests into the minimum number of memory transactions. Kernel 1 needed 32 sectors
// per warp per load of A; this kernel needs 4 (for B) + 1 (for A). The stores to C at the end
// are also contiguous now.
//
// Rule of thumb: make threadIdx.x walk along the dimension that is contiguous in memory
// (for row-major arrays: along a row, i.e. across columns).
// =============================================================================
#include "../common.cuh"
#include "../kernels.cuh"

__global__ void sgemm_v2_coalesced(int M, int N, int K, float alpha, const float* A, const float* B,
                                   float beta, float* C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x -> column: neighbours in a warp
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  //   touch neighbouring addresses

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];  // A: broadcast, B: coalesced
        C[row * N + col] = alpha * acc + beta * C[row * N + col];  // coalesced store
    }
}

namespace {

constexpr int kBlock = 32;

void launch(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const dim3 block(kBlock, kBlock);
    const dim3 grid(ceil_div(N, kBlock), ceil_div(M, kBlock));  // grid x now covers columns
    sgemm_v2_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace

KernelSpec coalesced_spec() {
    return {"2_coalesced", "one thread per C element, threadIdx.x -> column (coalesced)", 1,
            kBlock * kBlock, reinterpret_cast<const void*>(sgemm_v2_coalesced), launch};
}
