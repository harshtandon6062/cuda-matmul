// =============================================================================
// Kernel 1 — NAIVE: one thread computes one element of C.
// =============================================================================
//
// The most direct translation of the math  C[r][c] = sum over k of A[r][k] * B[k][c]:
// launch one thread per element of C; each thread walks one row of A and one column of B,
// doing K multiply-adds and 2*K loads from global memory (DRAM, through the caches).
//
// The thread -> element mapping below is the "obvious" one, copied from math notation (x, y):
//
//     row of C    <-  x  =  blockIdx.x * blockDim.x + threadIdx.x
//     column of C <-  y  =  blockIdx.y * blockDim.y + threadIdx.y
//
// It computes the right answer but is slow, because of how a WARP touches memory.
// A warp is 32 threads with consecutive threadIdx.x and the same threadIdx.y. They execute
// each instruction together, so each load instruction asks for 32 addresses at once. Here the
// 32 threads of a warp own 32 consecutive ROWS of C in the same column, so in step k they load
//
//     A[row*K + k]  for 32 different rows: 32 addresses that are K*4 bytes apart
//     B[k*N + col]  the same address for all 32 threads (served once, then broadcast)
//
// DRAM and the caches move data in 32-byte sectors. The 32 A addresses sit in 32 different
// sectors, so the load unit must handle 32 sectors to deliver 32 * 4 = 128 useful bytes.
// The final stores to C are spread out the same way. Kernel 2 changes only this mapping.
// =============================================================================
#include "../common.cuh"
#include "../kernels.cuh"

__global__ void sgemm_v1_naive(int M, int N, int K, float alpha, const float* A, const float* B,
                               float beta, float* C) {
    // Which element of C does this thread own? (x -> row: the "obvious" but slow choice.)
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    // The grid is rounded up to whole blocks, so threads past the edge of C must do nothing.
    if (row < M && col < N) {
        float acc = 0.0f;  // the running sum lives in a register
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];  // 2 global loads for every multiply-add
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

namespace {

constexpr int kBlock = 32;  // 32 x 32 = 1024 threads per block, the hardware maximum

void launch(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const dim3 block(kBlock, kBlock);
    const dim3 grid(ceil_div(M, kBlock), ceil_div(N, kBlock));  // grid x covers rows, grid y columns
    sgemm_v1_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace

KernelSpec naive_spec() {
    return {"1_naive", "one thread per C element, threadIdx.x -> row (uncoalesced)", 1, kBlock * kBlock,
            reinterpret_cast<const void*>(sgemm_v1_naive), launch};
}
