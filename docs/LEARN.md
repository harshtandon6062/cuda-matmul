# Learning CUDA through SGEMM

This document teaches CUDA from zero using one problem, multiplying two matrices, on one GPU, the
**{{GPU_NAME}}** in this laptop. It assumes you know C/C++ and basic computer architecture (caches,
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

This GPU has **{{SMS}} streaming multiprocessors (SMs)**, each with **{{LANES_PER_SM}} FP32 lanes**, so
{{LANES_TOTAL}} lanes in total (marketing calls them "CUDA cores"). Every lane can complete one fused
multiply-add (FMA, `a*b + c` = 2 FLOPs) per clock cycle, which gives the theoretical peak:

```text
peak FP32 = SMs x lanes per SM x 2 FLOPs per FMA x clock
          = {{SMS}} x {{LANES_PER_SM}} x 2 x {{CLOCK_GHZ}} GHz = {{PEAK_GFLOPS}} GFLOPS
```

Its DRAM delivers at most **{{BW_GBS}} GB/s**. Compare the two numbers. To keep every lane busy, a kernel
must perform {{PEAK_GFLOPS}} / {{BW_GBS}} ≈ **{{RIDGE}} FLOPs for every byte** it reads from DRAM, which is
about {{RIDGE_PER_FLOAT}} FLOPs per float. A naive matrix multiply does 2 FLOPs per 2 floats it reads. That
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
{{CC}}, the Ampere "GA10x" design) is split into **4 sub-partitions**. Each has its own warp scheduler, a
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
{{MAX_WARPS_SM}} warps ({{MAX_THREADS_SM}} threads). What a kernel actually gets is capped by whichever
resource runs out first:

| Resource (per SM on this GPU) | Example of it being the limit |
|---|---|
| {{MAX_THREADS_SM}} threads | a 1024-thread block: only one fits, so 1024/{{MAX_THREADS_SM}} = 67% occupancy |
| {{REGS_PER_SM}} 32-bit registers | 128 registers/thread × 256 threads = 32768 per block, so only 2 blocks fit |
| {{SMEM_PER_SM_KIB}} KiB shared memory | 48 KiB per block means at most 2 blocks |
| {{MAX_BLOCKS_SM}} blocks | very small blocks run out of block slots |

Higher occupancy gives the scheduler more warps to choose from, but **it is not a goal in itself**. Part 2
shows a kernel with lower occupancy beating one with higher occupancy by a wide margin, because each of its
warps has far more independent work. [`results/kernels.csv`](../results/kernels.csv) lists every kernel's
registers, shared memory and occupancy, and which resource limits it.

## 5. The memory hierarchy

| Level | Who can see it | Size on this GPU | Speed (typical, not measured here) | Managed by |
|---|---|---|---|---|
| **Registers** | one thread | {{REGS_PER_SM}} × 4 B = {{REGFILE_KIB}} KiB per SM, ≤ 255 per thread | operands of every instruction, no load needed | compiler |
| **Shared memory** | one block | up to {{SMEM_PER_SM_KIB}} KiB per SM | ~20–30 cycles | **you** (`__shared__`) |
| **L1 cache** | one SM | shares one 128 KiB SRAM per SM with shared memory | ~30 cycles | hardware |
| **L2 cache** | whole GPU | {{L2_MIB}} MiB | ~200 cycles | hardware |
| **Global memory (DRAM)** | whole GPU + host copies | {{MEM_GIB}} GiB, {{BW_GBS}} GB/s | hundreds of cycles | you (`cudaMalloc`) |

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
**ridge point**, {{RIDGE}} FLOP/byte on this GPU. A kernel to the left of the ridge is **memory-bound**:
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

{{PART2}}
