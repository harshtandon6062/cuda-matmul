// =============================================================================
// device_info.cu — STEP 1: which GPU is this, and what are its theoretical limits?
//
//   ./bin/device_info [output_dir]        writes <output_dir>/device.md and device.json
//
// Every number is queried at run time (cudaGetDeviceProperties, cudaDeviceGetAttribute, NVML).
// Two short microbenchmarks then measure how close simple code gets to the theoretical peaks:
//   * DRAM bandwidth: copy a buffer far larger than the L2 cache
//   * FP32 throughput: a kernel doing nothing but independent FMAs in registers
// =============================================================================
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "common.cuh"
#include "device_query.cuh"
#include "gpu_monitor.cuh"

// ---- Microbenchmark 1: achievable DRAM bandwidth ------------------------------------------
// Each thread copies 16-byte chunks (float4), the widest load/store a thread can issue.
__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = in[i];
}

// ---- Microbenchmark 2: achievable FP32 throughput -----------------------------------------
// One FMA (fused multiply-add, x = x*a + b) counts as 2 FLOPs. A result is ready only ~4 cycles
// after the FMA is issued, so each thread keeps kChains INDEPENDENT chains going: the scheduler
// can issue an FMA from another chain (or another warp) every cycle instead of waiting.
constexpr int kChains = 8;
constexpr int kUnroll = 16;

__global__ void fma_kernel(float* out, int iters, float a, float b) {
    float x[kChains];
#pragma unroll
    for (int j = 0; j < kChains; ++j) x[j] = threadIdx.x * 1e-3f + j;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
            for (int j = 0; j < kChains; ++j) x[j] = fmaf(x[j], a, b);
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < kChains; ++j) s += x[j];
    if (s == -1.0f) out[blockIdx.x] = s;  // (practically) never true; stops the compiler deleting the loop
}

namespace {

// Median of `reps` runs of f(), timed on the GPU with CUDA events, in milliseconds.
template <typename F>
float median_ms(F&& f, int reps) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> ms(reps);
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        f();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms[r], start, stop));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(ms.begin(), ms.end());
    return ms[reps / 2];
}

struct BandwidthResult {
    double gbs = 0;
    size_t buffer_bytes = 0;
};

BandwidthResult measure_copy_bandwidth() {
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    // Two 256 MiB buffers: ~100x larger than the L2 cache, so every access really goes to DRAM.
    size_t bytes = std::min<size_t>(size_t(256) << 20, free_b / 4);
    bytes &= ~size_t(15);
    const size_t n = bytes / sizeof(float4);

    float4 *in = nullptr, *out = nullptr;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemset(in, 0, bytes));

    const int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((n + threads - 1) / threads, 1u << 30));
    auto run = [&] { copy_kernel<<<blocks, threads>>>(in, out, n); };
    run();  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const float ms = median_ms(run, 7);

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    // Every byte is read once and written once.
    return {2.0 * bytes / (ms * 1e-3) / 1e9, bytes};
}

struct FmaResult {
    double gflops = 0;
    int clock_mhz = -1;  // median SM clock sampled while the benchmark was running
    int clock_min_mhz = -1, clock_max_mhz = -1;
};

FmaResult measure_fma_throughput(const DeviceSummary& d, const GpuMonitor& mon) {
    const int threads = 256;
    // Enough blocks to fill every SM several times over.
    const int blocks = d.prop.multiProcessorCount * (d.prop.maxThreadsPerMultiProcessor / threads) * 8;
    const int iters = 2048;
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, blocks * sizeof(float)));
    auto run = [&] { fma_kernel<<<blocks, threads>>>(out, iters, 0.999f, 0.001f); };

    run();  // warm-up: also lets the GPU ramp up to its boost clock
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Queue ~20 runs and read the clock every 10 ms while they execute; keep the median reading.
    for (int i = 0; i < 20; ++i) run();
    std::vector<int> clocks;
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (cudaStreamQuery(0) != cudaErrorNotReady) break;  // all queued runs finished
        clocks.push_back(mon.sm_clock_mhz());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    FmaResult r;
    if (!clocks.empty()) {
        std::sort(clocks.begin(), clocks.end());
        r.clock_mhz = clocks[clocks.size() / 2];
        r.clock_min_mhz = clocks.front();
        r.clock_max_mhz = clocks.back();
    }

    const float ms = median_ms(run, 7);
    CUDA_CHECK(cudaFree(out));
    const double flops = 2.0 * kChains * kUnroll * static_cast<double>(iters) * blocks * threads;
    r.gflops = flops / (ms * 1e-3) / 1e9;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out_dir = argc > 1 ? argv[1] : "results";

    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::fprintf(stderr, "No CUDA-capable GPU found.\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));
    const DeviceSummary d = query_device(0);
    const cudaDeviceProp& p = d.prop;
    GpuMonitor mon(0);

    int driver_ver = 0, runtime_ver = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_ver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_ver));

    if (d.fp32_lanes < 0) {
        std::fprintf(stderr, "Unknown compute capability %d.%d: add it to fp32_lanes_per_sm() in "
                             "src/device_query.cuh\n", p.major, p.minor);
        return 1;
    }

    std::printf("GPU: %s (compute capability %d.%d, %d SMs)\n", p.name, p.major, p.minor,
                p.multiProcessorCount);
    std::printf("Measuring DRAM copy bandwidth...\n");
    const BandwidthResult bw = measure_copy_bandwidth();
    std::printf("Measuring FP32 FMA throughput...\n");
    const FmaResult fma = measure_fma_throughput(d, mon);

    const int lanes_total = p.multiProcessorCount * d.fp32_lanes;
    const double sm_ghz = d.sm_clock_khz / 1e6;
    const double mem_ghz = d.mem_clock_khz / 1e6;
    const double peak = d.peak_gflops();
    const double peak_bw = d.peak_bandwidth_gbs();
    const int max_clock = mon.max_sm_clock_mhz();
    const double peak_at_max_clock = max_clock > 0 ? 2.0 * lanes_total * max_clock / 1e3 : 0.0;
    // Which clock would the measured FMA rate correspond to if every lane did one FMA per cycle?
    const double implied_clock_mhz = fma.gflops * 1e3 / (2.0 * lanes_total);
    const std::string driver = mon.driver_version();

    // ------------------------------------------------------------------ device.md
    const std::string md_path = out_dir + "/device.md";
    FILE* md = std::fopen(md_path.c_str(), "w");
    if (!md) {
        std::perror(md_path.c_str());
        return 1;
    }
    std::fprintf(md, "# GPU: %s\n\n", p.name);
    std::fprintf(md, "Generated by `bin/device_info` (source: `src/device_info.cu`). Every value is queried "
                     "from the driver at run time; no hardware number is typed in by hand.\n\n");
    std::fprintf(md, "| Property | Value | Where it comes from |\n|---|---|---|\n");
    std::fprintf(md, "| Compute capability | %d.%d (`sm_%d%d`) | `prop.major`, `prop.minor` |\n", p.major,
                 p.minor, p.major, p.minor);
    std::fprintf(md, "| Streaming multiprocessors (SMs) | %d | `prop.multiProcessorCount` |\n",
                 p.multiProcessorCount);
    std::fprintf(md, "| FP32 lanes per SM | %d | architecture table for CC %d.%d (`src/device_query.cuh`) |\n",
                 d.fp32_lanes, p.major, p.minor);
    std::fprintf(md, "| FP32 lanes in total (\"CUDA cores\") | %d | SMs x lanes per SM |\n", lanes_total);
    std::fprintf(md, "| SM clock, peak reported by CUDA | %d MHz | `cudaDevAttrClockRate` |\n",
                 d.sm_clock_khz / 1000);
    if (max_clock > 0)
        std::fprintf(md, "| SM clock, maximum reported by NVML | %d MHz | `nvmlDeviceGetMaxClockInfo` |\n",
                     max_clock);
    std::fprintf(md, "| Memory clock | %d MHz | `cudaDevAttrMemoryClockRate` |\n", d.mem_clock_khz / 1000);
    std::fprintf(md, "| Memory bus width | %d bits | `cudaDevAttrGlobalMemoryBusWidth` |\n", d.bus_width_bits);
    std::fprintf(md, "| Global memory (DRAM) | %.2f GiB | `prop.totalGlobalMem` |\n",
                 p.totalGlobalMem / double(1 << 30));
    std::fprintf(md, "| L2 cache | %.2f MiB | `prop.l2CacheSize` |\n", p.l2CacheSize / double(1 << 20));
    std::fprintf(md, "| Shared memory per SM | %zu KiB | `prop.sharedMemPerMultiprocessor` |\n",
                 p.sharedMemPerMultiprocessor / 1024);
    std::fprintf(md, "| Shared memory per block (default / opt-in maximum) | %zu KiB / %zu KiB | "
                     "`prop.sharedMemPerBlock`, `prop.sharedMemPerBlockOptin` |\n",
                 p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
    std::fprintf(md, "| 32-bit registers per SM / per block | %d / %d | `prop.regsPerMultiprocessor`, "
                     "`prop.regsPerBlock` |\n", p.regsPerMultiprocessor, p.regsPerBlock);
    std::fprintf(md, "| Max threads per SM / per block | %d / %d | `prop.maxThreadsPerMultiProcessor`, "
                     "`prop.maxThreadsPerBlock` |\n", p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock);
    std::fprintf(md, "| Max resident blocks per SM | %d | `prop.maxBlocksPerMultiProcessor` |\n",
                 p.maxBlocksPerMultiProcessor);
    std::fprintf(md, "| Warp size | %d threads | `prop.warpSize` |\n", p.warpSize);
    std::fprintf(md, "| Kernel run-time limit (display watchdog) | %s | `cudaDevAttrKernelExecTimeout` |\n",
                 d.kernel_timeout ? "yes" : "no");
    std::fprintf(md, "| Driver / CUDA runtime | %s (supports CUDA %d.%d) / CUDA %d.%d | NVML, "
                     "`cudaDriverGetVersion`, `cudaRuntimeGetVersion` |\n\n",
                 driver.c_str(), driver_ver / 1000, (driver_ver % 1000) / 10, runtime_ver / 1000,
                 (runtime_ver % 1000) / 10);

    std::fprintf(md, "## Theoretical peaks\n\n```text\n");
    std::fprintf(md, "Peak FP32 = SMs x FP32 lanes per SM x 2 FLOP per FMA x SM clock\n");
    std::fprintf(md, "          = %d x %d x 2 x %.3f GHz\n", p.multiProcessorCount, d.fp32_lanes, sm_ghz);
    std::fprintf(md, "          = %.1f GFLOPS\n\n", peak);
    std::fprintf(md, "Peak DRAM bandwidth = 2 transfers per clock (double data rate) x memory clock x bus width / 8\n");
    std::fprintf(md, "                    = 2 x %.3f GHz x %d bits / 8\n", mem_ghz, d.bus_width_bits);
    std::fprintf(md, "                    = %.1f GB/s\n\n", peak_bw);
    std::fprintf(md, "Ridge point = peak FP32 / peak bandwidth = %.1f FLOP per byte\n```\n\n", peak / peak_bw);
    std::fprintf(md, "A kernel that does fewer than ~%.0f floating-point operations per byte it moves to or "
                     "from DRAM cannot reach the FP32 peak on this GPU: it is **memory-bound**. Above that "
                     "it can be **compute-bound**.\n\n", peak / peak_bw);
    if (max_clock > 0)
        std::fprintf(md, "The same formula at the NVML maximum clock (%d MHz) gives %.1f GFLOPS. That is a ceiling "
                         "the GPU reaches only briefly when power and temperature allow. This project quotes "
                         "\"%% of peak\" against the %.1f GFLOPS figure above.\n\n",
                     max_clock, peak_at_max_clock, peak);

    std::fprintf(md, "## Measured with microbenchmarks\n\n");
    std::fprintf(md, "| Microbenchmark | Measured | %% of theoretical |\n|---|---|---|\n");
    std::fprintf(md, "| DRAM copy, 2 x %zu MiB buffers (read + write) | %.1f GB/s | %.1f%% |\n",
                 bw.buffer_bytes >> 20, bw.gbs, 100.0 * bw.gbs / peak_bw);
    std::fprintf(md, "| FP32 FMA loop in registers (no memory traffic) | %.1f GFLOPS | %.1f%% |\n\n", fma.gflops,
                 100.0 * fma.gflops / peak);
    if (fma.clock_mhz > 0)
        std::fprintf(md, "SM clock sampled by NVML every 10 ms while the FMA loop ran: median **%d MHz** "
                         "(range %d-%d MHz). ", fma.clock_mhz, fma.clock_min_mhz, fma.clock_max_mhz);
    std::fprintf(md, "The measured rate implies an average clock of at least %.0f MHz (measured GFLOPS / "
                     "(2 x %d lanes), the clock needed if every lane finished one FMA per cycle).\n\n",
                 implied_clock_mhz, lanes_total);
    if (fma.clock_mhz > 0 && implied_clock_mhz > fma.clock_mhz * 1.02)
        std::fprintf(md, "The implied clock is *higher* than NVML's reading, which is physically impossible if NVML "
                         "were exact: on this machine NVML's clock value does not follow the real boost clock "
                         "closely. Nsight Compute's cycle counter (`sm__cycles_elapsed.avg.per_second`, in "
                         "`results/ncu/`) is the trustworthy clock measurement; NVML readings in "
                         "`results.csv` are indicative only.\n\n");
    std::fprintf(md, "Laptop GPUs run above or below the reported clock depending on power and temperature, "
                     "so measured throughput can differ from the formula in either direction.\n");
    std::fclose(md);

    // ------------------------------------------------------------------ device.json
    const std::string json_path = out_dir + "/device.json";
    FILE* js = std::fopen(json_path.c_str(), "w");
    if (!js) {
        std::perror(json_path.c_str());
        return 1;
    }
    std::fprintf(js, "{\n");
    std::fprintf(js, "  \"name\": \"%s\",\n", p.name);
    std::fprintf(js, "  \"compute_capability\": \"%d.%d\",\n", p.major, p.minor);
    std::fprintf(js, "  \"sm_count\": %d,\n", p.multiProcessorCount);
    std::fprintf(js, "  \"fp32_lanes_per_sm\": %d,\n", d.fp32_lanes);
    std::fprintf(js, "  \"sm_clock_mhz\": %d,\n", d.sm_clock_khz / 1000);
    std::fprintf(js, "  \"max_sm_clock_mhz\": %d,\n", max_clock);
    std::fprintf(js, "  \"mem_clock_mhz\": %d,\n", d.mem_clock_khz / 1000);
    std::fprintf(js, "  \"bus_width_bits\": %d,\n", d.bus_width_bits);
    std::fprintf(js, "  \"global_mem_bytes\": %zu,\n", p.totalGlobalMem);
    std::fprintf(js, "  \"l2_bytes\": %d,\n", p.l2CacheSize);
    std::fprintf(js, "  \"peak_gflops\": %.3f,\n", peak);
    std::fprintf(js, "  \"peak_bandwidth_gbs\": %.3f,\n", peak_bw);
    std::fprintf(js, "  \"measured_copy_gbs\": %.3f,\n", bw.gbs);
    std::fprintf(js, "  \"measured_fma_gflops\": %.3f,\n", fma.gflops);
    std::fprintf(js, "  \"fma_sm_clock_mhz\": %d,\n", fma.clock_mhz);
    std::fprintf(js, "  \"driver\": \"%s\",\n", driver.c_str());
    std::fprintf(js, "  \"cuda_runtime\": \"%d.%d\"\n", runtime_ver / 1000, (runtime_ver % 1000) / 10);
    std::fprintf(js, "}\n");
    std::fclose(js);

    std::printf("  SMs %d x %d FP32 lanes @ %d MHz  ->  peak %.1f GFLOPS\n", p.multiProcessorCount,
                d.fp32_lanes, d.sm_clock_khz / 1000, peak);
    std::printf("  memory %d MHz x %d bits          ->  peak %.1f GB/s\n", d.mem_clock_khz / 1000,
                d.bus_width_bits, peak_bw);
    std::printf("  measured: copy %.1f GB/s (%.1f%%), FMA %.1f GFLOPS (%.1f%%, SM clock %d MHz)\n", bw.gbs,
                100.0 * bw.gbs / peak_bw, fma.gflops, 100.0 * fma.gflops / peak, fma.clock_mhz);
    std::printf("Wrote %s and %s\n", md_path.c_str(), json_path.c_str());
    return 0;
}
