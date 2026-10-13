// =============================================================================
// Kernel 6 — VECTORIZED (float4) LOADS: kernel 5 with 128-bit memory instructions.
// =============================================================================
//
// Same 128 x 128 block tile and 8 x 8 outputs per thread as kernel 5. Two changes:
//
// 1. Global -> shared copies use float4 (16 bytes = 4 floats per load instruction, compiled to
//    LDG.128 instead of four LDG.32). Fewer instructions for the same bytes. With 256 threads,
//    the 128 x 8 A tile and the 8 x 128 B tile are each exactly one float4 per thread.
//
// 2. The A tile is stored TRANSPOSED in shared memory: As[k][m] instead of As[m][k].
//    In the inner loop a thread needs A values (rows m..m+7, column k). In kernel 5 those are
//    8 floats spaced BK apart; transposed they are 8 CONSECUTIVE floats, so they can be read
//    with two 128-bit shared loads (LDS.128) instead of eight 32-bit ones. B's 8 values were
//    already consecutive and get the same treatment.
//
// The epilogue reads and writes C four floats at a time as well.
//
// Requirements: everything kernel 5 needs, plus 16-byte aligned rows (N and K multiples of 4,
// cudaMalloc'd base pointers), which the 128-multiple sizes guarantee.
// =============================================================================
#include "../common.cuh"
#include "../kernels.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_v6_vectorized(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                    float* C) {
    constexpr int kThreads = (BM * BN) / (TM * TN);
    static_assert(BM * BK == 4 * kThreads, "each thread copies exactly one float4 of the A tile");
    static_assert(BK * BN == 4 * kThreads, "each thread copies exactly one float4 of the B tile");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "register tiles are read as float4");

    // 16-byte alignment is required for float4 access to shared memory.
    __shared__ __align__(16) float As[BK * BM];  // TRANSPOSED: As[k * BM + m]
    __shared__ __align__(16) float Bs[BK * BN];  // Bs[k * BN + n]

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // Copy positions: A tile row aRow, floats aCol4*4 .. +3; B tile row bRow, floats bCol4*4 .. +3.
    const int aRow = threadIdx.x / (BK / 4), aCol4 = threadIdx.x % (BK / 4);
    const int bRow = threadIdx.x / (BN / 4), bCol4 = threadIdx.x % (BN / 4);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM][TN] = {};
    float aReg[TM];
    float bReg[TN];

    for (int t = 0; t < K; t += BK) {
        // A: one 128-bit global load, then four 32-bit shared stores that transpose it.
        const float4 a4 = *reinterpret_cast<const float4*>(&A[aRow * K + aCol4 * 4]);
        As[(aCol4 * 4 + 0) * BM + aRow] = a4.x;
        As[(aCol4 * 4 + 1) * BM + aRow] = a4.y;
        As[(aCol4 * 4 + 2) * BM + aRow] = a4.z;
        As[(aCol4 * 4 + 3) * BM + aRow] = a4.w;
        // B: one 128-bit global load straight into one 128-bit shared store.
        *reinterpret_cast<float4*>(&Bs[bRow * BN + bCol4 * 4]) =
            *reinterpret_cast<const float4*>(&B[bRow * N + bCol4 * 4]);
        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Both register slices are now contiguous in shared memory: 128-bit loads.
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&As[k * BM + threadRow * TM + i]);
                aReg[i + 0] = v.x;
                aReg[i + 1] = v.y;
                aReg[i + 2] = v.z;
                aReg[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&Bs[k * BN + threadCol * TN + j]);
                bReg[j + 0] = v.x;
                bReg[j + 1] = v.y;
                bReg[j + 2] = v.z;
                bReg[j + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

    // Epilogue: read-modify-write C with 128-bit accesses.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4* c = reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j]);
            float4 v = *c;
            v.x = alpha * acc[i][j + 0] + beta * v.x;
            v.y = alpha * acc[i][j + 1] + beta * v.y;
            v.z = alpha * acc[i][j + 2] + beta * v.z;
            v.w = alpha * acc[i][j + 3] + beta * v.w;
            *c = v;
        }
    }
}

namespace {

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

void launch(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(N / BN, M / BM);
    sgemm_v6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace

KernelSpec vectorized_spec() {
    return {"6_vectorized", "kernel 5 + float4 global/shared loads, A tile transposed in shared memory", BM,
            (BM * BN) / (TM * TN),
            reinterpret_cast<const void*>(sgemm_v6_vectorized<BM, BN, BK, TM, TN>), launch};
}
