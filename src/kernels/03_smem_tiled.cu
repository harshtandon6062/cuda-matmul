// =============================================================================
// Kernel 3 — SHARED-MEMORY TILING (instantiated with TILE = 16 and TILE = 32).
// =============================================================================
//
// Problem with kernels 1-2: every thread loads its own full row of A and full column of B from
// global memory. Across the whole grid each element of A and B is loaded N times, so the GPU
// spends its time waiting for memory instead of computing.
//
// Idea: the threads of one block cooperate. The block computes a TILE x TILE tile of C. It walks
// along K in steps of TILE, and at each step:
//
//   1. every thread copies ONE element of A's current tile and ONE element of B's current tile
//      from global memory into SHARED MEMORY (fast on-chip memory visible to the whole block);
//   2. every thread then does TILE multiply-adds reading only shared memory.
//
//            K ->                          N ->
//         +--------+                    +--+--+--+
//      M  |  As -> |   x             K  |Bs|  |  |         C tile (TILE x TILE)
//         |        |                 |  |↓ |  |  |      = sum over tiles of  As x Bs
//         +--------+                    +--+--+--+
//
// Each value fetched from global memory is now reused TILE times (by the TILE threads in the
// same row or column of the tile), so global-memory traffic drops by a factor of TILE.
//
// Two __syncthreads() barriers per step are REQUIRED (explained at each one below). A barrier
// makes every thread of the block wait until all of them have reached it.
//
// TILE = 16 -> 256 threads per block, 2 KiB of shared memory, each value reused 16 times.
// TILE = 32 -> 1024 threads per block, 8 KiB, reused 32 times, but at most one block fits on an
//             SM (1024 of its 1536 thread slots), and a barrier must wait for 32 warps, not 8.
// =============================================================================
#include "../common.cuh"
#include "../kernels.cuh"

template <int TILE>
__global__ void sgemm_v3_smem_tiled(int M, int N, int K, float alpha, const float* A, const float* B,
                                    float beta, float* C) {
    // Shared memory is allocated per BLOCK and lives on the SM: ~20-30 cycles to access vs
    // hundreds for DRAM. Every thread of the block sees the same As/Bs.
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;              // column inside the tile (fast index -> coalescing)
    const int ty = threadIdx.y;              // row inside the tile
    const int row = blockIdx.y * TILE + ty;  // the element of C this thread owns
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {  // A's tile slides right, B's tile slides down
        // Step 1: cooperative load, one element per thread per matrix. Neighbouring tx read
        // neighbouring addresses, so both loads are coalesced. Out-of-range elements become 0,
        // which lets this kernel handle any M, N, K.
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + (t + tx)] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;

        // Barrier 1: wait until EVERY thread has written its element. Without it, a fast warp
        // would start the loop below and read As/Bs entries that a slower warp has not
        // written yet (they would still hold the previous tile, or garbage).
        __syncthreads();

        // Step 2: TILE multiply-adds from shared memory. Within a warp As[ty][k] is the same
        // address for every thread (broadcast) and Bs[k][tx] is 32 consecutive floats that fall
        // in 32 different banks, so neither read has bank conflicts.
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];

        // Barrier 2: wait until EVERY thread has finished reading this tile. Without it, a fast
        // warp would loop around and overwrite As/Bs with the NEXT tile while slower warps are
        // still multiplying with the current one.
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

namespace {

template <int TILE>
void launch(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const dim3 block(TILE, TILE);
    const dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    sgemm_v3_smem_tiled<TILE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace

KernelSpec smem_tile16_spec() {
    return {"3a_smem_tile16", "shared-memory tiling, 16x16 tile, 1 output per thread", 1, 16 * 16,
            reinterpret_cast<const void*>(sgemm_v3_smem_tiled<16>), launch<16>};
}

KernelSpec smem_tile32_spec() {
    return {"3b_smem_tile32", "shared-memory tiling, 32x32 tile, 1 output per thread", 1, 32 * 32,
            reinterpret_cast<const void*>(sgemm_v3_smem_tiled<32>), launch<32>};
}
