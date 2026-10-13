# Learning CUDA through SGEMM

This document teaches CUDA from zero using one problem, multiplying two matrices, on one GPU, the
**NVIDIA GeForce RTX 3050 6GB Laptop GPU** in this laptop. It assumes you know C/C++ and basic computer architecture (caches,
pipelines, memory latency) but have never written GPU code.

- **Part 1** explains the concepts: threads, blocks, grids, warps, SMs, the memory hierarchy, and how to
  measure performance.
- **Part 2** walks through the six kernels in [`src/kernels/`](../src/kernels/): what each one changes, why it
  should be faster, what the measurement actually showed, and the key lines of code.

Every performance number here was measured on this machine by this project's code. The raw data is in
[`results/`](../results/), and [`results/device.md`](../results/device.md) shows where each hardware number
comes from.

## Contents

- [Part 1: Concepts](#part-1-concepts)
  - [1. The problem: SGEMM](#1-the-problem-sgemm)
  - [2. CPU vs GPU](#2-cpu-vs-gpu)
  - [3. The CUDA programming model](#3-the-cuda-programming-model)
  - [4. How the hardware runs your code: SMs, warps, occupancy](#4-how-the-hardware-runs-your-code-sms-warps-occupancy)
  - [5. The memory hierarchy](#5-the-memory-hierarchy)
  - [6. Measuring performance](#6-measuring-performance)
- [Part 2: The kernels](#part-2-the-kernels)

---

# Part 1: Concepts

## 1. The problem: SGEMM

SGEMM ("**S**ingle-precision **GE**neral **M**atrix **M**ultiply") is the BLAS routine

```text
C = alpha * (A x B) + beta * C          A: M x K,  B: K x N,  C: M x N,  all 32-bit floats
```

This project uses square matrices (M = N = K, simply called N). Every element of C is a dot product of a
row of A and a column of B:

```text
C[i][j] = sum over k = 0 .. N-1 of  A[i][k] * B[k][j]
```

**Counting the work.** There are N² outputs, each needing N multiply-adds, so N³ multiply-adds. A
multiply-add counts as **2 floating-point operations (FLOPs)**, so one multiplication costs **2N³ FLOPs**:
137 billion at N = 4096. Speed is reported in **GFLOPS = 2N³ / time / 10⁹**.

**Counting the data.** The three matrices hold 3N² floats, 12N² bytes (192 MiB at N = 4096). Work grows
like N³ but data only like N², so **each input value is used N times**. That reuse is the whole story of
this project. If a kernel fetches each value from DRAM once and then reuses it from fast on-chip storage,
arithmetic speed can become the limit. If it fetches the value again every time it needs it, memory
bandwidth is the limit. Every optimization in Part 2 increases reuse.

**Why GEMM matters.** Fully-connected layers, attention and (after rearrangement) convolutions in neural
networks are GEMMs, as is most dense linear algebra in scientific computing. NVIDIA added tensor cores
specifically to speed it up. This project deliberately uses only the ordinary FP32 "CUDA cores", and so
does the cuBLAS baseline it compares against. Both do the same arithmetic, so the comparison is fair.

**Row-major storage.** Matrices are stored row by row, like C arrays: element (r, c) of a matrix with
`cols` columns is at `X[r * cols + c]`. **Neighbouring elements of a row are neighbours in memory.
Neighbouring elements of a column are `cols` floats (4·cols bytes) apart.** Remember this, because it
decides which access patterns are fast.

## 2. CPU vs GPU

A CPU core is built to finish one thread's instruction stream as quickly as possible. It uses deep
pipelines, branch prediction, out-of-order execution and large caches so that it rarely waits.

A GPU is built for **throughput**. It runs tens of thousands of threads, each individually slow. It does
not try to avoid waiting for memory; it **hides** the wait by always having other threads ready to run.

This GPU has **20 streaming multiprocessors (SMs)**, each with **128 FP32 lanes**, so
2560 lanes in total (marketing calls them "CUDA cores"). Every lane can complete one fused
multiply-add (FMA, `a*b + c` = 2 FLOPs) per clock cycle, which gives the theoretical peak:

```text
peak FP32 = SMs x lanes per SM x 2 FLOPs per FMA x clock
          = 20 x 128 x 2 x 1.492 GHz = 7,639 GFLOPS
```

Its DRAM delivers at most **168 GB/s**. Compare the two numbers. To keep every lane busy, a kernel
must perform 7,639 / 168 ≈ **45 FLOPs for every byte** it reads from DRAM, which is
about 180 FLOPs per float. A naive matrix multiply does 2 FLOPs per 2 floats it reads. That
gap of two orders of magnitude is what the optimizations in Part 2 have to close.

## 3. The CUDA programming model

### Host and device

The CPU and its RAM are the **host**. The GPU and its DRAM are the **device**. They have separate memories,
so data is allocated on the GPU and copied explicitly:

```cpp
float* dA;
cudaMalloc(&dA, bytes);                              // allocate GPU (global) memory
cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);   // copy CPU -> GPU
my_kernel<<<grid, block>>>(dA, ...);                 // launch on the GPU (asynchronous!)
cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);   // copy GPU -> CPU (waits for the kernel first)
```

### Kernels and threads

A **kernel** is a C++ function marked `__global__`. Launching it runs the *same function* in thousands
of **threads** at once. Each thread reads built-in variables to find out which part of the problem is its
own. That is the whole trick: you write the code for **one** thread.

### Threads, blocks, grids

Threads are grouped at two levels:

- A **block** is up to 1024 threads, arranged in 1, 2 or 3 dimensions. Threads of the same block run on
  the same SM, can share fast **shared memory**, and can wait for each other at a **barrier**
  (`__syncthreads()`).
- The **grid** is all the blocks of one launch. Blocks are **independent**: they may run in any order, on
  any SM, at the same time or one after another.

Inside a kernel each thread sees:

| Variable | Meaning |
|---|---|
| `threadIdx.{x,y,z}` | this thread's position inside its block |
| `blockIdx.{x,y,z}` | this block's position inside the grid |
| `blockDim.{x,y,z}` | the block's size (threads per block in each dimension) |
| `gridDim.{x,y,z}` | the grid's size (blocks in each dimension) |

The standard way to give every thread a unique global position:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;   // which column of C this thread owns
int row = blockIdx.y * blockDim.y + threadIdx.y;   // which row
```

and the matching launch, one thread per element of an N×N matrix, rounded up to whole blocks:

```cpp
dim3 block(32, 32);                                  // 1024 threads per block
dim3 grid(ceil_div(N, 32), ceil_div(N, 32));         // enough blocks to cover N x N
kernel<<<grid, block>>>(...);
// inside the kernel: if (row < N && col < N) { ... }  because the last blocks may stick out
```

**Why two levels?** A block always lives on one SM, so its threads can cooperate through that SM's fast
memory. Blocks never depend on each other, so the same program runs on a GPU with 10 SMs or 140 SMs. The
bigger GPU simply runs more blocks at the same time.

### Launches are asynchronous

`kernel<<<...>>>()` only *queues* the work and returns immediately; the CPU keeps going. A blocking call
such as `cudaMemcpy` or `cudaDeviceSynchronize()` waits for the GPU. Errors inside a kernel therefore
appear *later*. That is why every CUDA call in this project goes through `CUDA_CHECK`
([`src/common.cuh`](../src/common.cuh)), and every launch is followed by `cudaGetLastError()`.

## 4. How the hardware runs your code: SMs, warps, occupancy

### Streaming multiprocessors

An **SM** is roughly the GPU's equivalent of a CPU core. Each SM on this GPU (compute capability
8.6, the Ampere "GA10x" design) is split into **4 sub-partitions**. Each has its own warp scheduler, a
quarter of the register file, and 32 FP32 lanes. When a kernel is launched, a hardware block scheduler
hands blocks to SMs. An SM keeps as many blocks as its resources allow, and starts a new one whenever one
finishes.

### Warps (SIMT)

The SM does not schedule individual threads. It schedules **warps** of **32 consecutive threads** of a
block: threads 0–31, 32–63, and so on, numbered with `threadIdx.x` changing fastest, then `y`, then `z`.
All 32 threads of a warp execute **the same instruction at the same time**, each on its own data. NVIDIA
calls this SIMT (single instruction, multiple threads). It is like a 32-wide SIMD unit whose lanes each
look like a separate thread.

Two consequences matter for this project:

1. **Memory instructions are issued per warp.** When a warp loads, the hardware receives 32 addresses at
   once and decides how many memory transactions it needs (see coalescing below). Which thread touches
   which address therefore matters a great deal.
2. **Divergence.** If threads of one warp take different sides of an `if`, the warp runs both sides one
   after the other, with the inactive threads masked off. Divergence *between* warps costs nothing. The
   SGEMM kernels here have essentially none: their only branch is the bounds check `if (row < N && col < N)`,
   which is the same for every thread of a warp except at the matrix edge.

### Latency hiding

A load from DRAM takes hundreds of cycles. A CPU core would stall, or rely on out-of-order execution and
caches. An SM instead keeps **many warps resident** at the same time. Every cycle, each of its 4 schedulers
picks a warp whose next instruction has its operands ready, and issues it. While warp 5 waits for memory,
warps 6, 7 and 8 compute. The GPU therefore needs **plenty of independent work in flight**, from one of
two sources:

- **many warps** (high *occupancy*), or
- **many independent instructions within each warp** (*instruction-level parallelism*, ILP). Kernels 4–6
  rely on this: one thread computes 8 or 64 outputs, and their FMAs do not depend on each other.

### Occupancy

**Occupancy = resident warps per SM ÷ the maximum an SM supports.** On this GPU the maximum is
48 warps (1536 threads). What a kernel actually gets is capped by whichever
resource runs out first:

| Resource (per SM on this GPU) | Example of it being the limit |
|---|---|
| 1536 threads | a 1024-thread block: only one fits, so 1024/1536 = 67% occupancy |
| 65,536 32-bit registers | 128 registers/thread × 256 threads = 32768 per block, so only 2 blocks fit |
| 100 KiB shared memory | 48 KiB per block means at most 2 blocks |
| 16 blocks | very small blocks run out of block slots |

Higher occupancy gives the scheduler more warps to choose from, but **it is not a goal in itself**. Part 2
shows a kernel with lower occupancy beating one with higher occupancy by a wide margin, because each of its
warps has far more independent work. [`results/kernels.csv`](../results/kernels.csv) lists every kernel's
registers, shared memory and occupancy, and which resource limits it.

## 5. The memory hierarchy

| Level | Who can see it | Size on this GPU | Speed (typical, not measured here) | Managed by |
|---|---|---|---|---|
| **Registers** | one thread | 65,536 × 4 B = 256 KiB per SM, ≤ 255 per thread | operands of every instruction, no load needed | compiler |
| **Shared memory** | one block | up to 100 KiB per SM | ~20–30 cycles | **you** (`__shared__`) |
| **L1 cache** | one SM | shares one 128 KiB SRAM per SM with shared memory | ~30 cycles | hardware |
| **L2 cache** | whole GPU | 1.5 MiB | ~200 cycles | hardware |
| **Global memory (DRAM)** | whole GPU + host copies | 5.7 GiB, 168 GB/s | hundreds of cycles | you (`cudaMalloc`) |

Latencies are typical published figures for this GPU generation, quoted to give you a sense of scale.
This project measures bandwidth and throughput, not latency.

### Coalescing: how a warp talks to global memory

Global memory is moved in **32-byte sectors**, and four sectors make a 128-byte cache line. When a warp
executes a load, the hardware collects its 32 addresses and fetches the sectors that cover them:

| Warp access pattern | Sectors needed | Useful bytes / fetched bytes |
|---|---|---|
| 32 threads read 32 **consecutive** floats (a piece of a row) | 4 | 128 / 128 = **100%** |
| 32 threads read 32 floats from 32 different rows (a piece of a column) | 32 | 128 / 1024 = **12.5%** |
| 32 threads read the **same** float | 1 (broadcast) | all threads served by one sector |

Merging a warp's accesses into the minimum number of sectors is called **coalescing**. The rule of thumb:
**make `threadIdx.x`, the index that changes fastest within a warp, walk along the dimension that is
contiguous in memory.** Kernels 1 and 2 differ in exactly this.

### Shared memory and bank conflicts

Shared memory is split into **32 banks**, each 4 bytes wide. 4-byte word number `i` lives in bank
`i mod 32`. Each bank can deliver one word per cycle, so:

- 32 threads reading 32 words in 32 different banks: **1 pass**.
- Several threads reading **the same word**: 1 pass (broadcast).
- *k* threads reading **different words in the same bank**: *k* passes, a **k-way bank conflict**.

Example: `float tile[32][32]` and every thread of a warp reading `tile[threadIdx.x][0]`. The words are
32 apart, so they all sit in bank 0: a 32-way conflict, 32 times slower than it needs to be. Part 2 counts
the bank conflicts in each kernel with Nsight Compute.

## 6. Measuring performance

### Timing on the GPU

[`src/bench.cu`](../src/bench.cu) times kernels the way NVIDIA recommends:

- **CUDA events**: `cudaEventRecord` puts a marker in the GPU's work queue, and the GPU timestamps it
  when it gets there. The difference between two events is GPU execution time, not the CPU's launch
  overhead.
- **3 warm-up launches first.** The first launch pays one-time costs: loading the kernel's code, cold
  caches, and the GPU ramping its clock up from idle.
- **10 timed launches, median reported.** The median ignores the odd slow run, for example when another
  process briefly uses the GPU.
- **The clock is recorded.** A laptop GPU changes its clock with temperature and power. The harness reads
  the actual SM clock through NVML (the library behind `nvidia-smi`) during every measurement.
- **Correctness before speed.** Every kernel's output is compared element by element with cuBLAS (and with
  a double-precision CPU loop for N ≤ 512). A kernel that gives a wrong answer is not timed.

### Arithmetic intensity and the roofline model

**Arithmetic intensity (AI)** = FLOPs performed ÷ bytes moved to and from DRAM. The **roofline model**
says a kernel's speed is capped by

```text
attainable GFLOPS = min( peak GFLOPS,  AI x peak DRAM bandwidth )
```

Plotted on log-log axes, this is a sloped line (memory roof) that meets a flat line (compute roof) at the
**ridge point**, 45 FLOP/byte on this GPU. A kernel to the left of the ridge is **memory-bound**:
only more reuse (higher AI) or better bandwidth use can speed it up. A kernel to the right can be
**compute-bound**, and then only issuing more FMAs per cycle helps.

The *math* of a 4096×4096 GEMM needs only 3 × 4096² × 4 bytes of data for 2 × 4096³ FLOPs, an AI of about
680, far right of the ridge. Whether an *implementation* gets there depends entirely on how often it
re-reads the same data. Each kernel in Part 2 raises the effective AI. The roofline chart
([`results/roofline.png`](../results/roofline.png)) plots every kernel's measured AI against its measured
speed.

### Nsight Compute

`ncu` is NVIDIA's kernel profiler. It replays each kernel several times and reads hardware performance
counters: bytes moved at every memory level, achieved occupancy, how busy the FMA units were, bank
conflicts, and *why* warps were stalled. `make profile` collects these for every kernel; the raw output is
in [`results/ncu/`](../results/ncu/).

---

# Part 2: The kernels

All numbers in this part come from the final `make bench` run committed in `results/`. Unless a size is
named, they are for **N = 4096**: one multiplication = 137.4 GFLOP, cuBLAS = **4,388 GFLOPS**, theoretical
peak = 7,639 GFLOPS. The Nsight Compute counters are from `results/ncu/summary.csv`.

| # | Kernel | GFLOPS at N=4096 | vs previous | % of cuBLAS |
|---|---|---:|---:|---:|
| 1 | naive | 73.7 | | 1.7% |
| 2 | coalesced | 502 | 6.8× | 11.4% |
| 3a / 3b | shared-memory tiling, 16 / 32 | 664 / 663 | 1.3× | 15.1% |
| 4 | 1D register blocking | 1,894 | 2.9× | 43.2% |
| 5 | 2D register blocking | 3,594 | 1.9× | 81.9% |
| 6 | 2D + float4 loads | 4,173 | 1.16× | **95.1%** |

From kernel 1 to kernel 6: **56.6× faster**, with the same arithmetic. Every kernel sums over k in the
same order with one accumulator per output, and all six show exactly the same maximum error at every size:
a relative error of 3.1·10⁻⁶ against cuBLAS at N = 4096, and 1.1·10⁻⁶ against the double-precision CPU
reference at N = 512.

## How to read the profiler columns

- **Sectors per request**: for each warp-wide global load instruction, how many 32-byte sectors the
  hardware had to fetch. A perfectly coalesced 4-byte load is 4.
- **Warp stall reasons**: when a warp could not issue, *why*. The value is the average number of cycles
  spent in that state per issued instruction.
  - `lg_throttle`: the queue for global memory instructions is full.
  - `mio_throttle`: the queue for shared-memory instructions is full.
  - `long_scoreboard`: waiting for data from global memory.
  - `barrier`: waiting at `__syncthreads()`.
  - `not_selected`: the warp was ready, but the scheduler issued another warp's instruction. That is a
    *good* sign: there is more ready work than issue slots.
- **FMA pipe busy**: the fraction of cycles the FP32 FMA units were working. This is the closest thing to
  "% of peak" measured from the inside.

---

## Kernel 1: naive ([`01_naive.cu`](../src/kernels/01_naive.cu))

**What it does.** One thread per element of C. Thread (x, y) computes row x, column y, a direct copy of
the math notation C[x][y]:

```cpp
const int row = blockIdx.x * blockDim.x + threadIdx.x;   // x -> row
const int col = blockIdx.y * blockDim.y + threadIdx.y;   // y -> column
if (row < M && col < N) {                                // the grid is rounded up; skip the overhang
    float acc = 0.0f;                                    // running sum in a register
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];          // 2 global loads per multiply-add
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}
```

**Why it is slow.** A warp is 32 threads with consecutive `threadIdx.x`. Here that means 32 consecutive
**rows**. In every iteration the warp loads `A[row*K + k]` for 32 rows, which are 32 addresses 16 KiB
apart, so 32 different sectors. It also loads `B[k*N + col]`, the same address for all 32 threads, which
is one sector broadcast to all of them.

**Measured.** **73.7 GFLOPS, 1.0% of peak** (52–74 GFLOPS across all sizes). Nsight Compute:

- **16.5 sectors per load request**, exactly the average of the A load (32) and the B load (1).
- The dominant stall is `lg_throttle` at **212 cycles per instruction**. Warps spend nearly all their time
  waiting to get a memory instruction *into the queue*.
- The FMA units are busy **2%** of the time.
- DRAM throughput is only 5.1 GB/s (3% of peak). The L1 and L2 caches absorb most of the repeated
  traffic: the measured DRAM arithmetic intensity is 14.5 FLOP/byte, not the 0.25 a cache-less analysis
  predicts.

The bottleneck is the *number of memory transactions* the load/store unit must process, not DRAM.

## Kernel 2: global-memory coalescing ([`02_coalesced.cu`](../src/kernels/02_coalesced.cu))

**What changed.** Two lines: `threadIdx.x` now picks the **column**.

```cpp
const int col = blockIdx.x * blockDim.x + threadIdx.x;   // x -> column: a warp covers 32 neighbours
const int row = blockIdx.y * blockDim.y + threadIdx.y;
```

**Why it should be faster.** A warp is now 32 consecutive columns of one row:

- `B[k*N + col]` is 32 consecutive floats: 128 contiguous bytes, 4 sectors, every byte used.
- `A[row*K + k]` is the same address for the whole warp: 1 sector.
- The stores to C at the end are contiguous too.

Same instructions, same FLOPs; only the addresses each warp touches changed.

**Measured.** **502 GFLOPS: 6.8× faster than kernel 1** from a two-line change.

- Global loads now need **2.5 sectors per request** ((4 + 1)/2). The load instructions are the same
  4.30 billion as before, but the sectors moved fell from **70.9 billion to 10.7 billion** (6.6× fewer).
- `lg_throttle` fell from 212 to 18 cycles per instruction.
- The FMA units are busy 18% of the time.

**What limits it now.** Every multiply-add still needs two load instructions. The SM's load/store path
is saturated: Nsight Compute reports SM throughput at **94.5%** of its limit, while DRAM is only 20% busy.
To go faster we must issue *fewer loads per FMA*, which means reusing data on-chip.

## Kernel 3: shared-memory tiling ([`03_smem_tiled.cu`](../src/kernels/03_smem_tiled.cu))

**What changed.** The block cooperates. A TILE×TILE block computes a TILE×TILE tile of C, walking along K
one tile at a time. At each step every thread copies **one** element of A's tile and one of B's tile into
**shared memory**. Then all threads compute from shared memory, so each value fetched from global memory
is reused TILE times.

```cpp
__shared__ float As[TILE][TILE], Bs[TILE][TILE];
for (int t = 0; t < K; t += TILE) {
    As[ty][tx] = A[row * K + (t + tx)];       // coalesced: neighbouring tx, neighbouring addresses
    Bs[ty][tx] = B[(t + ty) * N + col];
    __syncthreads();                          // barrier 1: the tile is complete
    for (int k = 0; k < TILE; ++k)
        acc += As[ty][k] * Bs[k][tx];         // TILE FMAs from shared memory
    __syncthreads();                          // barrier 2: everyone is done reading it
}
```

**The two `__syncthreads()`.** A barrier makes every thread of the block wait until all have arrived.
Different warps of a block run at different speeds, so without the barriers:

- **Without barrier 1**, a fast warp would start multiplying while a slow warp has not yet written its
  part of `As`/`Bs`. It would read stale data from the previous tile, or garbage, and the result would be
  wrong.
- **Without barrier 2**, a fast warp would finish its multiplies, loop around and overwrite `As`/`Bs` with
  the *next* tile while slower warps are still reading the current one. That is the same bug from the
  other side.

Both are race conditions, and they produce wrong answers only *sometimes*. The harness's element-by-element
check exists to catch exactly that.

**Shared-memory banks.** In `As[ty][k]` every thread of a warp reads the same word, which is a broadcast.
In `Bs[k][tx]` 32 threads read 32 consecutive words, one per bank. Nsight Compute confirms no load bank
conflicts for TILE = 32, and only 115,625 for TILE = 16 (out of 3.2 billion shared-load wavefronts).

> The profiler also counts 12.6 M (tile 16) and 31.6 M (tile 32) shared *store* "bank conflicts". The store
> pattern `As[ty][tx]` is conflict-free by construction. Nsight Compute's conflict counter can also include
> stalls that the access pattern does not cause, so I report the counter as measured but did not
> investigate it further.

**TILE = 16 vs 32.**

| | TILE = 16 | TILE = 32 |
|---|---|---|
| Threads per block | 256 | 1024 |
| Blocks per SM, occupancy | 6, **100%** | 1, 67% |
| Reuse per global load (analytical AI) | 16× (4 FLOP/B) | 32× (8 FLOP/B) |
| Global load requests | 269 M | 135 M |
| Measured DRAM traffic | 17.7 GB | 9.2 GB |
| GFLOPS at N = 256 / 1024 / 4096 | 655 / 743 / 664 | 541 / 646 / 663 |

TILE = 32 halves global traffic; TILE = 16 keeps more warps resident and has cheaper barriers (8 warps to
wait for instead of 32). At N = 4096 the two balance out almost exactly. At small sizes TILE = 16 wins,
because 1024-thread blocks leave a third of each SM's thread slots empty.

**Measured.** **664 GFLOPS: only 1.3× faster than kernel 2**, even though global load requests dropped
16–32×. The stall profile explains why. `mio_throttle` (the shared-memory instruction queue is full)
now dominates at 19–21 cycles per instruction, and the FMA units are busy only 8–10%. Each FMA still
needs **two shared-memory loads**. We moved the bottleneck from global memory to shared memory.

## Kernel 4: 1D register blocking ([`04_regblock_1d.cu`](../src/kernels/04_regblock_1d.cu))

**What changed.** Each thread computes **8 outputs** stacked in one column, not 1. A block now covers a
64×64 tile of C with 512 threads and consumes K eight values at a time (BK = 8).

```cpp
float acc[TM] = {0.0f};                                   // 8 running sums, in registers
for (int k = 0; k < BK; ++k) {
    const float b = Bs[k * BN + threadCol];               // ONE shared load...
    for (int i = 0; i < TM; ++i)
        acc[i] += As[(threadRow * TM + i) * BK + k] * b;  // ...reused for 8 FMAs
}
```

**Why it should be faster.** Shared-memory loads per FMA drop from 2 to 9/8. Registers need no load
instruction at all. The 8 accumulators are independent, so each warp has 8 FMAs it can issue back to back
(instruction-level parallelism). Global traffic also halves again: (64 + 64)×8 floats are loaded for
64×64×8 FMAs, an analytical AI of 16 FLOP/byte.

**Measured.** **1,894 GFLOPS: 2.9× faster than kernel 3.**

- Shared-memory load wavefronts fell from 3.22 billion to **1.34 billion**.
- `mio_throttle` fell from ~20 to 6.7 cycles per instruction.
- The FMA units are busy 23% of the time.
- Measured DRAM traffic halved to 4.7 GB (AI 29.3 FLOP/byte).
- 52 registers per thread, so 2 blocks of 512 fit per SM (67% occupancy).

## Kernel 5: 2D register blocking ([`05_regblock_2d.cu`](../src/kernels/05_regblock_2d.cu))

**What changed.** Each thread computes an **8×8 tile** of outputs. For each k it loads 8 values of A's
column and 8 of B's row into registers and forms their **outer product**:

```cpp
for (int i = 0; i < TM; ++i) aReg[i] = As[(threadRow * TM + i) * BK + k];
for (int j = 0; j < TN; ++j) bReg[j] = Bs[k * BN + threadCol * TN + j];
for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j)
        acc[i][j] += aReg[i] * bReg[j];        // 64 FMAs from 16 shared loads
```

Block tile 128×128, 256 threads. Analytical AI 32 FLOP/byte, double kernel 4's.

**Why it should be faster.** Now 16 shared loads feed 64 FMAs, 0.25 loads per FMA. Each warp has 64
independent FMAs to issue.

**The register-pressure lesson.** The first version used **145 registers per thread**. 256 threads × 145
registers is more than half the SM's 65,536 registers, so only **one** block (8 warps, 16.7% occupancy)
fit per SM, and it ran at about 3,000 GFLOPS. Adding `__launch_bounds__(256, 2)` tells the compiler
"guarantee 2 blocks per SM". It then fits the code into **128 registers with zero spills**, and the kernel
runs at about **3,720 GFLOPS (+24%)**. The second experiment in
[`results/experiments.md`](../results/experiments.md) shows the opposite case. Forcing kernel 6 down to
80 registers made it spill 112 bytes per thread to local memory and run **5.7× slower** (4,234 → 746
GFLOPS). **Occupancy helps only if it does not cost spills.**

**Measured (final version).** **3,594 GFLOPS: 1.9× faster than kernel 4, 81.9% of cuBLAS.**

- Shared-load wavefronts halved again, to 672 M.
- The FMA units are busy **49%** of the time at only 33% occupancy.
- The top stall is now `not_selected` (2.1): warps are *ready* and waiting their turn, which is a sign of
  a healthy, compute-heavy kernel.
- Measured DRAM AI is **56.6 FLOP/byte, right of the ridge point (45)**: the kernel is now on the
  compute-bound side of the roofline.
- What remains: Nsight Compute counts **268 M shared-load bank conflicts** among those 672 M wavefronts.
  The compiler turned each thread's 8 consecutive B values into two 128-bit loads (`LDS.128`). Thread
  columns are 8 words apart, so the 16 different addresses a warp reads land on only half of the 32
  banks, and the hardware needs twice the minimum number of passes.

## Kernel 6: vectorized float4 loads ([`06_vectorized.cu`](../src/kernels/06_vectorized.cu))

**What changed.** Kernel 5's tiling, plus:

1. Global → shared copies use `float4` (16 bytes per instruction, `LDG.E.128`).
2. The A tile is stored **transposed** in shared memory so that a thread's 8 A values are contiguous.
3. The epilogue reads and writes C four floats at a time.

```cpp
const float4 a4 = *reinterpret_cast<const float4*>(&A[aRow * K + aCol4 * 4]);  // 1 instruction, 16 bytes
As[(aCol4 * 4 + 0) * BM + aRow] = a4.x;   // ...stored transposed: As[k][m]
```

**What the measurement showed, and what it did not.** **4,173 GFLOPS: 1.16× faster than kernel 5, 95.1%
of cuBLAS**. The *reason* is not quite the textbook one. Disassembling both kernels
(`cuobjdump -sass`) shows that the compiler had **already** turned kernel 5's shared reads into 128-bit
`LDS.128` instructions: both kernels contain exactly 32 `LDS.128` and 576 `FFMA`. Shared-load wavefronts
are identical (671 M vs 672 M), and so are the 268 M bank conflicts, so the transpose bought nothing
there. The real differences are:

- global loads: **18 `LDG.E.128` instead of 72 `LDG.E`** in the code, and 8.5 M instead of 34 M load
  requests at run time;
- fewer instructions per tile, so more issue slots for FMAs. Issue-active rose from 55% to 63%, and the
  FMA pipe from 49% to **58%**;
- 102 registers instead of 128, comfortably under the cap.

Lesson: check the SASS before claiming *why* an optimization worked.

## Why cuBLAS is still faster, and what would close the gap

At N = 4096 cuBLAS chose the kernel **`ampere_sgemm_128x64_nn`**. Side by side under Nsight Compute:

| | Kernel 6 | cuBLAS |
|---|---:|---:|
| GFLOPS (benchmark median) | 4,173 | 4,388 |
| FMA pipe busy | 58% | **64%** |
| Issue slots used | 63% | **75%** |
| Shared-memory load wavefronts | 671 M | **272 M** (2.5× fewer) |
| Shared-load bank conflicts | 268 M | **0** |
| DRAM traffic | 2.43 GB | **2.00 GB** |
| Registers / occupancy | 102 / 33% | 122 / 33% |

cuBLAS's kernels are hand-tuned close to the hardware:

- **conflict-free shared-memory layouts** (zero conflicts, and 2.5× fewer wavefronts per FMA);
- **warp-level tiling**, which organizes each warp's work so register fragments are reused across a
  larger tile;
- **software pipelining / double buffering**, which loads the next tile while computing the current one,
  so `__syncthreads()` stalls disappear;
- a different kernel chosen per problem size by heuristics.

Those are the natural next steps for this project.

The gap is largest at small sizes. At N = 256 our 128×128-tile kernels launch only (256/128)² = **4 blocks
on 20 SMs**, so 16 SMs sit idle. Kernel 4's smaller 64×64 tiles (16 blocks) are our best there at 1,260
GFLOPS, against cuBLAS's 1,973.

**A note on "% of peak".** cuBLAS itself reaches only 57% of the 7,639 GFLOPS theoretical peak at
N = 4096, yet 72% at N = 1024. This laptop GPU is **power-limited** at about 60 W: `nvidia-smi` reported the
throttle reason *software power cap* during long runs. Sustained heavy kernels therefore run at lower
clocks than short ones. The same effect is why single cold launches under Nsight Compute are faster than
the 10-run median: 4,533 GFLOPS for kernel 6 and 4,815 for cuBLAS.
