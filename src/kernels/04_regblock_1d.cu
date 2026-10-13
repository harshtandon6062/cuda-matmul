// =============================================================================
// Kernel 4 — 1D REGISTER BLOCKING: each thread computes TM outputs stacked in one column.
// =============================================================================
//
// Kernel 3 cut global-memory traffic, but every multiply-add still needs TWO shared-memory loads
// (As[ty][k] and Bs[k][tx]). Shared memory is fast, yet it has limited bandwidth and every load
// is an instruction the scheduler has to issue. With one FMA per two loads the SM spends most
// of its issue slots on loads, not math.
//
// Fix: give each thread more work. A thread now computes TM = 8 elements of C that sit in one
// column (rows r .. r+7, column c). For each k it:
//     loads ONE value  b = Bs[k][c]            from shared memory, keeps it in a register,
//     loads TM values  As[r+i][k], i = 0..7    from shared memory,
//     does  TM FMAs    acc[i] += As[r+i][k] * b
// That is TM + 1 = 9 shared loads for 8 FMAs instead of 16 loads, and the TM partial sums
// live in REGISTERS, the fastest storage on the GPU (no load instruction needed at all).
//
// Block tile: BM x BN = 64 x 64 elements of C per block, K consumed BK = 8 at a time:
//
//   As: BM x BK = 64 x 8 tile of A        Bs: BK x BN = 8 x 64 tile of B
//   threads per block = BM * BN / TM = 512; each loads exactly one element of As and of Bs.
//
// Global traffic per block-step: (BM + BN) * BK floats for BM * BN * BK FMAs, i.e. each
// loaded float feeds 32 FMAs on average (kernel 3 with TILE=32: 16).
//
// Requirement: M, N multiples of 64 and K a multiple of 8 (no edge handling, to keep the code
// readable). The harness only runs it on sizes that satisfy this.
// =============================================================================
#include "../common.cuh"
#include "../kernels.cuh"

template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
sgemm_v4_regblock_1d(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                     float* C) {
    constexpr int kThreads = (BM * BN) / TM;
    static_assert(BM * BK == kThreads, "each thread loads exactly one element of the A tile");
    static_assert(BK * BN == kThreads, "each thread loads exactly one element of the B tile");

    __shared__ float As[BM * BK];  // BM x BK, row-major
    __shared__ float Bs[BK * BN];  // BK x BN, row-major

    // Where this thread's TM x 1 strip of outputs sits inside the BM x BN block tile.
    // Consecutive threads take consecutive columns, so a warp (32 threads) covers 32 columns
    // of the same strip row.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;  // strip index; rows threadRow*TM .. threadRow*TM+TM-1

    // Which element of the A and B tiles this thread copies into shared memory.
    const int aRow = threadIdx.x / BK, aCol = threadIdx.x % BK;  // a warp reads 4 rows x 8 floats
    const int bRow = threadIdx.x / BN, bCol = threadIdx.x % BN;  // a warp reads 32 consecutive floats

    // Move the pointers to this block's tiles (block (x, y) owns C tile row y, column x).
    A += blockIdx.y * BM * K;                   // first row of the A tile
    B += blockIdx.x * BN;                       // first column of the B tile
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM] = {0.0f};  // TM running sums, kept in registers (loops below are fully unrolled)

    for (int t = 0; t < K; t += BK) {
        As[aRow * BK + aCol] = A[aRow * K + aCol];
        Bs[bRow * BN + bCol] = B[bRow * N + bCol];
        __syncthreads();  // tile complete before anyone reads it

        A += BK;      // next A tile: BK columns to the right
        B += BK * N;  // next B tile: BK rows down

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float b = Bs[k * BN + threadCol];  // loaded once, reused TM times from a register
#pragma unroll
            for (int i = 0; i < TM; ++i)
                acc[i] += As[(threadRow * TM + i) * BK + k] * b;  // same address across the warp: broadcast
        }
        __syncthreads();  // everyone done with this tile before it is overwritten
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        float* c = &C[(threadRow * TM + i) * N + threadCol];
        *c = alpha * acc[i] + beta * *c;
    }
}

namespace {

constexpr int BM = 64, BN = 64, BK = 8, TM = 8;

void launch(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const dim3 block((BM * BN) / TM);
    const dim3 grid(N / BN, M / BM);
    sgemm_v4_regblock_1d<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace

KernelSpec regblock_1d_spec() {
    return {"4_regblock_1d", "64x64 block tile, 8 outputs per thread in a column (1D register blocking)",
            BM, (BM * BN) / TM, reinterpret_cast<const void*>(sgemm_v4_regblock_1d<BM, BN, BK, TM>), launch};
}
