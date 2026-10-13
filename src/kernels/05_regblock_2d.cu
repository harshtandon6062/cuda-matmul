// =============================================================================
// Kernel 5 — 2D REGISTER BLOCKING: each thread computes a TM x TN = 8 x 8 tile of C.
// =============================================================================
//
// Kernel 4 reused one B value for 8 FMAs, but still loaded a fresh A value for every FMA.
// Going 2D removes that imbalance. For each k, a thread:
//
//     loads TM values of A's column k   (aReg[0..7])  from shared memory into registers
//     loads TN values of B's row k      (bReg[0..7])  from shared memory into registers
//     does the OUTER PRODUCT            acc[i][j] += aReg[i] * bReg[j]   for all 8 x 8 pairs
//
//                bReg (1 x TN)
//               [b0 b1 ... b7]
//     aReg  [a0]  .  .  ...  .
//    (TM x 1)[a1]  .  acc[i][j] += a_i * b_j      64 FMAs from only 16 shared-memory loads
//            [..]
//            [a7]  .  .  ...  .
//
// 16 loads feed 64 FMAs (kernel 4: 9 loads for 8 FMAs, kernel 3: 2 loads per FMA). The 64
// running sums plus aReg/bReg occupy roughly 100+ registers per thread. That is the price:
// REGISTER PRESSURE. Registers are split among all resident threads (65536 per SM on this GPU),
// so more registers per thread means fewer warps fit on an SM (lower occupancy). Here that trade
// pays off: each warp has a lot of independent FMAs to issue, which hides latency better than
// extra warps would.
//
// Block tile: BM x BN = 128 x 128 of C per block, BK = 8; 128*128 / 64 = 256 threads.
// The 256 threads copy the 128 x 8 A tile in 4 passes of 32 rows and the 8 x 128 B tile in
// 4 passes of 2 rows. Per block-step that is (128 + 128) * 8 floats loaded for 128*128*8 FMAs:
// each float loaded from global memory feeds 64 FMAs (kernel 4: 32).
//
// Requirement: M, N multiples of 128, K a multiple of 8.
// =============================================================================
#include "../common.cuh"
#include "../kernels.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_v5_regblock_2d(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                     float* C) {
    constexpr int kThreads = (BM * BN) / (TM * TN);
    constexpr int aStride = kThreads / BK;  // rows of the A tile copied per pass
    constexpr int bStride = kThreads / BN;  // rows of the B tile copied per pass
    static_assert(kThreads % BK == 0 && BM % aStride == 0, "A tile must split evenly over the threads");
    static_assert(kThreads % BN == 0 && BK % bStride == 0, "B tile must split evenly over the threads");

    __shared__ float As[BM * BK];  // BM x BK, row-major
    __shared__ float Bs[BK * BN];  // BK x BN, row-major

    // This thread's TM x TN output tile inside the block tile: a 16 x 16 grid of threads.
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // Copy positions for the cooperative tile loads.
    const int aRow = threadIdx.x / BK, aCol = threadIdx.x % BK;
    const int bRow = threadIdx.x / BN, bCol = threadIdx.x % BN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM][TN] = {};  // 64 running sums in registers
    float aReg[TM];          // one column slice of the A tile
    float bReg[TN];          // one row slice of the B tile

    for (int t = 0; t < K; t += BK) {
#pragma unroll
        for (int r = 0; r < BM; r += aStride)
            As[(aRow + r) * BK + aCol] = A[(aRow + r) * K + aCol];
#pragma unroll
        for (int r = 0; r < BK; r += bStride)
            Bs[(bRow + r) * BN + bCol] = B[(bRow + r) * N + bCol];
        __syncthreads();  // tiles complete

        A += BK;
        B += BK * N;

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Stage TM + TN values in registers...
#pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[(threadRow * TM + i) * BK + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[k * BN + threadCol * TN + j];
            // ...then TM * TN FMAs that touch nothing but registers.
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();  // everyone done before the tiles are overwritten
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            float* c = &C[(threadRow * TM + i) * N + threadCol * TN + j];
            *c = alpha * acc[i][j] + beta * *c;
        }
    }
}

namespace {

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

void launch(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(N / BN, M / BM);
    sgemm_v5_regblock_2d<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace

KernelSpec regblock_2d_spec() {
    return {"5_regblock_2d", "128x128 block tile, 8x8 outputs per thread (2D register blocking)", BM,
            (BM * BN) / (TM * TN),
            reinterpret_cast<const void*>(sgemm_v5_regblock_2d<BM, BN, BK, TM, TN>), launch};
}
