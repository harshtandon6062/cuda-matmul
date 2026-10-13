# SGEMM from scratch: optimizing matrix multiply on an RTX 3050 Laptop GPU

Six CUDA kernels for FP32 matrix multiplication (SGEMM, `C = αAB + βC`), written from scratch and
optimized step by step on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU**. Each one is benchmarked against
cuBLAS, profiled with Nsight Compute, and explained for someone learning CUDA.

**Headline (N = 4096):** the best kernel runs at **4,173 GFLOPS, 95.1% of cuBLAS** (4,388 GFLOPS) and
**56.6× faster than the naive kernel** (73.7 GFLOPS). Every output is verified element by element against
cuBLAS before it is timed.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU: 20 SMs × 128 FP32 lanes, compute capability 8.6, 5.7 GiB GDDR6 on a 96-bit bus, 1.5 MiB L2 ([results/device.md](results/device.md)) |
| Theoretical peak | 7,639 FP32 GFLOPS (20 × 128 × 2 × 1.492 GHz) · 168 GB/s DRAM · ridge point 45 FLOP/byte. Measured: 157 GB/s copy, FMA loop up to 9,126 GFLOPS when the GPU boosts above its rated clock |
| Toolchain | CUDA 13.0 (nvcc 13.0.48), driver 580.65.06, Ubuntu 24.04, gcc 13 |
| Docs | [docs/LEARN.md](docs/LEARN.md) (CUDA from zero, kernel by kernel) · [docs/INTERVIEW.md](docs/INTERVIEW.md) (20 Q&A with these numbers) · [docs/RESUME.md](docs/RESUME.md) |

## Results

Square N×N FP32 matrices, C = A·B. Times are the median of 10 runs after 3 warm-ups (CUDA events).
Every kernel's output was checked element by element against cuBLAS before it was timed (and against a
double-precision CPU reference for N ≤ 512). cuBLAS runs in its default FP32 mode (no TF32 tensor cores),
so it does the same arithmetic as our kernels.

<!-- RESULTS:BEGIN -->
### GFLOPS by matrix size (percent of cuBLAS in parentheses)

| Kernel | N = 256 | N = 512 | N = 1024 | N = 2048 | N = 4096 |
|---|---:|---:|---:|---:|---:|
| cuBLAS | **1,973** | **4,228** | **5,533** | **5,124** | **4,388** |
| 1 naive | 52.3 (2.6%) | 72.4 (1.7%) | 72.9 (1.3%) | 74.0 (1.4%) | 73.7 (1.7%) |
| 2 coalesced | 428 (22%) | 588 (14%) | 533 (9.6%) | 513 (10%) | 502 (11%) |
| 3a smem tiling, 16x16 | 655 (33%) | 771 (18%) | 743 (13%) | 681 (13%) | 664 (15%) |
| 3b smem tiling, 32x32 | 541 (27%) | 728 (17%) | 646 (12%) | 676 (13%) | 663 (15%) |
| 4 1D register blocking | 1,260 (64%) | 1,827 (43%) | 2,081 (38%) | 1,859 (36%) | 1,894 (43%) |
| 5 2D register blocking | 618 (31%) | 2,545 (60%) | 3,452 (62%) | 3,908 (76%) | 3,594 (82%) |
| 6 2D + float4 loads | 762 (39%) | 2,621 (62%) | 4,178 (76%) | 4,643 (91%) | 4,173 (95%) |

### Details at N = 4096

| Kernel | Time (ms) | GFLOPS | % of cuBLAS | % of peak | Max rel. error vs cuBLAS | SM clock, NVML (MHz, indicative) |
|---|---:|---:|---:|---:|---:|---:|
| cuBLAS | 31.32 | 4,388 | 100% | 57% | reference | 1920 |
| 1 naive | 1865.96 | 73.7 | 1.7% | 1.0% | 3.1e-06 | 1920 |
| 2 coalesced | 273.60 | 502 | 11% | 6.6% | 3.1e-06 | 1680 |
| 3a smem tiling, 16x16 | 206.88 | 664 | 15% | 8.7% | 3.1e-06 | 1605 |
| 3b smem tiling, 32x32 | 207.32 | 663 | 15% | 8.7% | 3.1e-06 | 1830 |
| 4 1D register blocking | 72.57 | 1,894 | 43% | 25% | 3.1e-06 | 1612 |
| 5 2D register blocking | 38.25 | 3,594 | 82% | 47% | 3.1e-06 | 1815 |
| 6 2D + float4 loads | 32.93 | 4,173 | 95% | 55% | 3.1e-06 | 1642 |
<!-- RESULTS:END -->

![GFLOPS vs N](results/gflops.png)

![Roofline](results/roofline.png)

All tables, including registers, occupancy and the Nsight Compute counters, are in
[results/results.md](results/results.md). Raw data: [results/results.csv](results/results.csv),
[results/kernels.csv](results/kernels.csv), [results/ncu/](results/ncu/),
[results/device.md](results/device.md).

## What limits each kernel

Measured with Nsight Compute at N = 4096 ([results/ncu/summary.md](results/ncu/summary.md)). The full
story is in [docs/LEARN.md](docs/LEARN.md#part-2-the-kernels).

| Kernel | GFLOPS | What limits it (evidence) |
|---|---:|---|
| 1 naive | 73.7 | **Uncoalesced loads.** 16.5 sectors per load request; warps stall ~212 cycles per instruction waiting to issue memory instructions (`lg_throttle`); FMA units 2% busy. DRAM is only 3% busy because the caches absorb the redundant traffic. |
| 2 coalesced | 502 | **Load/store pipeline inside the SM.** 2.5 sectors per request, but still 2 global loads per FMA: SM memory throughput at 94.5% of its limit while DRAM is at 20%. |
| 3a/3b smem tiling | 664 / 663 | **Shared-memory instruction throughput.** Global load requests fell 16–32×, but every FMA still needs 2 shared loads; top stall `mio_throttle` (~20 cycles per instruction), FMA units 8–10% busy. Tile 16 (100% occupancy) and tile 32 (half the DRAM traffic) end up equal. |
| 4 1D register blocking | 1,894 | **Still shared-memory heavy:** 9 shared loads per 8 FMAs. `mio_throttle` down to 6.7, FMA units 23% busy. |
| 5 2D register blocking | 3,594 | **Register pressure and bank conflicts.** At 145 registers only 1 block/SM fit; capped to 128 with `__launch_bounds__(256, 2)` for +24%. 268 M shared-load bank conflicts remain. Compute-bound side of the roofline (56.6 FLOP/byte). |
| 6 float4 | 4,173 | **Instruction issue and bank conflicts.** Issue slots 63% used, FMA pipe 58%; top stall `not_selected` (ready warps waiting for a slot). Same 268 M bank conflicts as kernel 5: the compiler had already vectorized kernel 5's shared loads, so the gain came from 4× fewer global load instructions. |
| cuBLAS | 4,388 | Issue slots 75%, FMA pipe 64%, **zero** bank conflicts, 2.5× fewer shared-memory wavefronts than kernel 6. The whole GPU is power-capped at ~60 W during sustained runs, which is why cuBLAS is at 57% of the theoretical peak at N = 4096 but 72% at N = 1024. |

At small sizes the 128×128-tile kernels launch too few blocks (4 blocks for 20 SMs at N = 256), so the
64×64-tile kernel 4 is the fastest of ours there.

## Build and run

Requirements: an NVIDIA GPU and driver, the CUDA toolkit (nvcc + cuBLAS; Nsight Compute is optional), and
Python 3 with matplotlib and pandas (`pip install -r requirements.txt`).

```bash
make bench        # build, device info, benchmark, Nsight Compute profile (if permitted), charts
```

That one command runs, in order:

| Step | Command | Output |
|---|---|---|
| Device info | `./bin/device_info results` | `results/device.md`, `results/device.json` |
| Benchmark | `./bin/sgemm` | `results/results.csv`, `results/kernels.csv` |
| Profiling | `scripts/profile.sh 4096` | `results/ncu/` (skipped with a message if not permitted) |
| Charts + tables | `scripts/report.py` | `results/gflops.png`, `results/roofline.png`, `results/results.md`, this README's results table |

Useful variations:

```bash
make                                          # build only
./bin/sgemm --sizes 2048 --kernels 5_regblock_2d,6_vectorized   # quick experiment
./bin/sgemm --help
make profile PROFILE_N=2048                   # profile at another size
make PYTHON=~/venv/bin/python report          # if your python3 lacks matplotlib/pandas
make info                                     # show detected nvcc, GPU architecture, python
```

The Makefile finds `nvcc` (on `PATH` or in `/usr/local/cuda*`) and compiles for the compute capability
`nvidia-smi` reports, here `sm_86`. Machine-specific overrides (e.g. `PYTHON := ...`) can go in a
git-ignored `local.mk`.

## Setup notes for this machine (Ubuntu 24.04, driver 580)

**CUDA toolkit.** The driver alone does not include `nvcc`. The toolkit was installed from NVIDIA's runfile
for CUDA 13.0.0, which ships with exactly this driver version (580.65.06). The `--toolkit` flag installs
only the toolkit (nvcc, cuBLAS, Nsight Compute) under `/usr/local/cuda-13.0` and leaves the display driver
alone:

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run
sudo sh cuda_13.0.0_580.65.06_linux.run --silent --toolkit
```

**Profiling permission.** By default only root may read GPU performance counters, and `ncu` then fails
with `ERR_NVGPUCTRPERM`. To allow normal users, run the following and then reboot:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u -k all
sudo reboot
# check: grep RmProfilingAdminOnly /proc/driver/nvidia/params   ->  RmProfilingAdminOnly: 0
```

Without it, `make bench` still works: it skips profiling and the roofline uses analytical arithmetic
intensities.

## Project layout

```text
src/
  kernels/01_naive.cu        1  one thread per output, uncoalesced mapping
  kernels/02_coalesced.cu    2  same, coalesced mapping
  kernels/03_smem_tiled.cu   3  shared-memory tiling, TILE = 16 and 32
  kernels/04_regblock_1d.cu  4  1D register blocking (8 outputs per thread)
  kernels/05_regblock_2d.cu  5  2D register blocking (8x8 outputs per thread)
  kernels/06_vectorized.cu   6  kernel 5 + float4 loads, transposed A tile
  bench.cu                   harness: inputs, cuBLAS/CPU references, checks, timing, CSV
  device_info.cu             STEP 1: device properties, theoretical peaks, microbenchmarks
  device_query.cuh  gpu_monitor.cuh  kernels.cuh  common.cuh
scripts/
  profile.sh                 Nsight Compute collection
  ncu_summary.py             condenses the ncu export
  report.py                  charts + tables
results/                     everything measured (committed)
docs/                        LEARN.md, INTERVIEW.md, RESUME.md
```

## Methodology and honesty notes

- **Correctness first.** Each kernel runs once with α = 0.75, β = 0.25 on a random C. Every element is
  compared with cuBLAS, and with a double-precision CPU loop for N ≤ 512. The tolerance is K·FLT_EPSILON
  relative to the largest reference value; measured errors are 4.9·10⁻⁷ to 3.1·10⁻⁶. Failing kernels are
  reported and not timed.
- **Timing.** Plain C = A·B (α = 1, β = 0). 3 warm-ups, then 10 launches each bracketed by CUDA events,
  with a short GPU busy-wait queued first so that launch overhead never shows up as idle time. The median
  is reported (min/max are in the CSV). A 1-second load before the first measurement brings the GPU up to
  its boost clock.
- **cuBLAS is called in its default FP32 mode** (no TF32 tensor cores), via the row-major trick: cuBLAS
  is column-major, so computing Bᵀ·Aᵀ gives row-major A·B without copying data.
- **Laptop clocks vary.** The GPU is power-capped at ~60 W, so repeated runs differ by a few percent.
  NVML's clock reading turned out to be indicative only: the FMA microbenchmark did more work per second
  than the NVML clock allows. The SM clock measured by Nsight Compute's cycle counters is in
  `results/ncu/summary.csv`.
- **"% of peak"** is against 7,639 GFLOPS at the 1,492 MHz clock CUDA reports. The GPU can boost higher
  (NVML maximum 2,100 MHz), so this figure is a convention, not a hard ceiling.
- **Profiling** runs each kernel once at N = 4096 under `ncu --clock-control none`. A single cold launch
  escapes the sustained power cap, so GFLOPS under the profiler are a little higher than the benchmark
  medians. Only the benchmark medians are quoted as performance.
- **Kernels 4–6 need N to be a multiple of their tile size** (64 or 128). All test sizes are powers of
  two; the harness skips sizes that don't fit.
- **Register experiment** (kernel 5 capped at 128 registers; kernel 6 forced to 80) and its numbers:
  [results/experiments.md](results/experiments.md).

## Acknowledgements

The kernel progression, and the tile shapes of kernels 4–6, follow Simon Boehm's article "How to Optimize
a CUDA Matmul Kernel for cuBLAS-like Performance". The concepts follow the CUDA C++ Programming Guide. The
code in this repository was written for this project, and every number was measured on the machine
described above.
