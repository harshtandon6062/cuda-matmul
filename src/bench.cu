// =============================================================================
// bench.cu — STEP 2: correctness + timing harness for every SGEMM kernel and cuBLAS.
//
//   ./bin/sgemm                                       full run -> results/results.csv, results/kernels.csv
//   ./bin/sgemm --sizes 1024 --kernels 5_regblock_2d  quick experiment
//   ./bin/sgemm --profile --sizes 4096                run each kernel exactly once (for Nsight Compute)
//
// For every square size N (M = N = K):
//   1. A, B and an initial C are filled with uniform random floats in [-1, 1) from a fixed seed.
//   2. References: cuBLAS for every N; for N <= 512 also a double-precision CPU loop, which in turn
//      checks that cuBLAS is being called correctly.
//   3. Correctness: each kernel runs once with alpha = 0.75, beta = 0.25 (so the "+ beta * C" part is
//      tested too) and EVERY element is compared with the references. A kernel that fails is
//      reported and NOT timed.
//   4. Timing: plain C = A * B (alpha = 1, beta = 0, the usual benchmark setting). 3 warm-up
//      launches, then 10 launches each timed with CUDA events; the median is reported.
// =============================================================================
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <thread>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>

#include "common.cuh"
#include "device_query.cuh"
#include "gpu_monitor.cuh"
#include "kernels.cuh"

#define CUBLAS_CHECK(call)                                                                \
    do {                                                                                  \
        cublasStatus_t st__ = (call);                                                     \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                              \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %s\n  in: %s\n", __FILE__,       \
                         __LINE__, cublasGetStatusString(st__), #call);                   \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

// Keeps the GPU busy for about `cycles` clock cycles. It is queued in front of the timed launches
// so that the CPU can enqueue all of them before the first one starts. Without it, a very short
// kernel could finish before the CPU issues the next launch, and the idle gap would be timed too.
__global__ void busy_wait_kernel(long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

namespace {

constexpr float kCheckAlpha = 0.75f, kCheckBeta = 0.25f;  // correctness run: exercises alpha and beta
constexpr float kBenchAlpha = 1.0f, kBenchBeta = 0.0f;    // timed runs: plain C = A * B

struct Options {
    std::vector<int> sizes{256, 512, 1024, 2048, 4096};
    std::vector<std::string> only;  // kernel names to run (empty = all)
    int warmup = 3;
    int reps = 10;
    int cpu_max_n = 512;  // the CPU reference is O(N^3) on one core, so only for small N
    unsigned seed = 2024;
    bool profile = false;
    std::string csv = "results/results.csv";
    std::string kernel_info = "results/kernels.csv";
};

std::vector<KernelSpec> all_kernels() {
    return {naive_spec(),       coalesced_spec(),   smem_tile16_spec(), smem_tile32_spec(),
            regblock_1d_spec(), regblock_2d_spec(), vectorized_spec()};
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == sep) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

[[noreturn]] void usage(const char* prog, int code) {
    std::printf(
        "usage: %s [options]\n"
        "  --sizes N1,N2,...   square matrix sizes (default 256,512,1024,2048,4096)\n"
        "  --kernels a,b,...   run only these kernels (default: all; cuBLAS always runs)\n"
        "  --warmup W          untimed launches before timing (default 3)\n"
        "  --reps R            timed launches, the median is reported (default 10)\n"
        "  --csv FILE          results table (default results/results.csv)\n"
        "  --kernel-info FILE  registers / shared memory / occupancy per kernel (default results/kernels.csv)\n"
        "  --profile           run every kernel once per size, no checks or timing (for Nsight Compute)\n"
        "kernels:",
        prog);
    for (const auto& k : all_kernels()) std::printf(" %s", k.name);
    std::printf("\n");
    std::exit(code);
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0], 1);
            return argv[++i];
        };
        if (a == "--sizes") {
            o.sizes.clear();
            for (const auto& s : split(value(), ',')) o.sizes.push_back(std::atoi(s.c_str()));
        } else if (a == "--kernels") {
            o.only = split(value(), ',');
        } else if (a == "--warmup") {
            o.warmup = std::atoi(value().c_str());
        } else if (a == "--reps") {
            o.reps = std::atoi(value().c_str());
        } else if (a == "--csv") {
            o.csv = value();
        } else if (a == "--kernel-info") {
            o.kernel_info = value();
        } else if (a == "--profile") {
            o.profile = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0], 0);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(argv[0], 1);
        }
    }
    if (o.sizes.empty() || o.reps < 1 || o.warmup < 0) usage(argv[0], 1);
    for (int n : o.sizes)
        if (n <= 0) usage(argv[0], 1);
    return o;
}

// ---------------------------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------------------------

// cuBLAS uses the Fortran convention: matrices are COLUMN-major. A row-major matrix read as
// column-major is its transpose, and (A * B)^T = B^T * A^T. So asking cuBLAS for "B * A" on our
// row-major buffers produces exactly row-major C = A * B, without moving or transposing any data.
void cublas_sgemm(cublasHandle_t h, int M, int N, int K, float alpha, const float* A, const float* B,
                  float beta, float* C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

// Plain CPU reference in double precision. The i-k-j loop order makes the innermost loop walk
// B and the output row contiguously, which is much kinder to the CPU caches than i-j-k.
std::vector<double> cpu_sgemm(int M, int N, int K, double alpha, const std::vector<float>& A,
                              const std::vector<float>& B, double beta, const std::vector<float>& C0) {
    std::vector<double> C(static_cast<size_t>(M) * N), row(N);
    for (int i = 0; i < M; ++i) {
        std::fill(row.begin(), row.end(), 0.0);
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(i) * K + k];
            const float* b = &B[static_cast<size_t>(k) * N];
            for (int j = 0; j < N; ++j) row[j] += a * b[j];
        }
        for (int j = 0; j < N; ++j) {
            const size_t idx = static_cast<size_t>(i) * N + j;
            C[idx] = alpha * row[j] + beta * C0[idx];
        }
    }
    return C;
}

// Uniform random floats in [-1, 1). Fixed seed: every run multiplies exactly the same matrices.
void make_inputs(unsigned seed, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    for (auto& x : C0) x = dist(rng);
}

// ---------------------------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------------------------

struct ErrorStats {
    bool valid = false;    // false = this comparison was not performed
    double max_abs = 0;    // max over elements of |got - ref|
    double max_rel = 0;    // max |got - ref| divided by max |ref|
    size_t nonfinite = 0;  // NaN / Inf outputs
    bool pass = false;
};

// A float dot product of length K can pick up rounding error of order K * FLT_EPSILON relative to
// the size of the values involved. Correct kernels differ from the reference only in summation
// order and land around 1e-7..1e-5; an indexing or synchronization bug gives errors of order 1.
// The tolerance K * FLT_EPSILON (3e-5 at K = 256, 5e-4 at K = 4096) separates the two cleanly.
template <typename T>
ErrorStats compare(const std::vector<float>& got, const std::vector<T>& ref, int K) {
    ErrorStats e;
    e.valid = true;
    double max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double r = static_cast<double>(ref[i]);
        if (!std::isfinite(g)) {
            ++e.nonfinite;
            continue;
        }
        e.max_abs = std::max(e.max_abs, std::fabs(g - r));
        max_ref = std::max(max_ref, std::fabs(r));
    }
    e.max_rel = max_ref > 0 ? e.max_abs / max_ref : e.max_abs;
    e.pass = e.nonfinite == 0 && e.max_rel <= K * static_cast<double>(FLT_EPSILON);
    return e;
}

// ---------------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------------

struct Timing {
    double median_ms = 0, min_ms = 0, max_ms = 0;
    int sm_clock_mhz = -1;  // NVML reading taken while the timed launches were running
    int temp_c = -1;
};

template <typename F>
Timing time_launches(F&& launch, int warmup, int reps, long long spin_cycles, const GpuMonitor& mon) {
    for (int i = 0; i < warmup; ++i) launch();  // first launches pay one-time costs (module load, caches)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cudaEvent_t> start(reps), stop(reps);
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventCreate(&start[r]));
        CUDA_CHECK(cudaEventCreate(&stop[r]));
    }
    busy_wait_kernel<<<1, 1>>>(spin_cycles);
    // Each launch is bracketed by its own pair of events. The GPU writes a timestamp when it
    // reaches an event in the stream, so the difference is pure GPU execution time.
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start[r]));
        launch();
        CUDA_CHECK(cudaEventRecord(stop[r]));
    }
    CUDA_CHECK(cudaGetLastError());

    // Sample the SM clock every ~2 ms while the timed launches run; report the median sample.
    Timing t;
    std::vector<int> clocks{mon.sm_clock_mhz()};
    while (cudaEventQuery(stop[reps - 1]) == cudaErrorNotReady && clocks.size() < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        clocks.push_back(mon.sm_clock_mhz());
    }
    std::sort(clocks.begin(), clocks.end());
    t.sm_clock_mhz = clocks[clocks.size() / 2];
    t.temp_c = mon.temperature_c();
    CUDA_CHECK(cudaEventSynchronize(stop[reps - 1]));

    std::vector<double> ms(reps);
    for (int r = 0; r < reps; ++r) {
        float v = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&v, start[r], stop[r]));
        ms[r] = v;
        CUDA_CHECK(cudaEventDestroy(start[r]));
        CUDA_CHECK(cudaEventDestroy(stop[r]));
    }
    std::sort(ms.begin(), ms.end());
    t.median_ms = reps % 2 ? ms[reps / 2] : 0.5 * (ms[reps / 2 - 1] + ms[reps / 2]);
    t.min_ms = ms.front();
    t.max_ms = ms.back();
    return t;
}

// ---------------------------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------------------------

struct Row {
    std::string kernel;
    int n = 0;
    std::string status;
    bool timed = false;
    Timing t;
    double gflops = 0, pct_cublas = 0, pct_peak = 0;
    ErrorStats vs_cublas, vs_cpu;
};

void write_csv_header(FILE* f) {
    std::fprintf(f, "kernel,N,status,ms_median,ms_min,ms_max,gflops,pct_of_cublas,pct_of_peak,"
                    "max_abs_err_vs_cublas,max_rel_err_vs_cublas,max_abs_err_vs_cpu,max_rel_err_vs_cpu,"
                    "sm_clock_mhz,gpu_temp_c\n");
}

void write_csv_row(FILE* f, const Row& r) {
    std::fprintf(f, "%s,%d,%s,", r.kernel.c_str(), r.n, r.status.c_str());
    if (r.timed)
        std::fprintf(f, "%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,", r.t.median_ms, r.t.min_ms, r.t.max_ms, r.gflops,
                     r.pct_cublas, r.pct_peak);
    else
        std::fprintf(f, ",,,,,,");
    for (const ErrorStats* e : {&r.vs_cublas, &r.vs_cpu}) {
        if (e->valid)
            std::fprintf(f, "%.3e,%.3e,", e->max_abs, e->max_rel);
        else
            std::fprintf(f, ",,");
    }
    if (r.timed && r.t.sm_clock_mhz > 0)
        std::fprintf(f, "%d,", r.t.sm_clock_mhz);
    else
        std::fprintf(f, ",");
    if (r.timed && r.t.temp_c > 0)
        std::fprintf(f, "%d\n", r.t.temp_c);
    else
        std::fprintf(f, "\n");
}

void print_table_header() {
    std::printf("  %-15s %10s %9s %8s %7s %12s %12s %6s  %s\n", "kernel", "ms", "GFLOPS", "%cuBLAS", "%peak",
                "relerr/cuBLAS", "relerr/CPU", "MHz", "status");
}

void print_row(const Row& r) {
    auto err = [](const ErrorStats& e) {
        char buf[32];
        if (!e.valid)
            std::snprintf(buf, sizeof(buf), "-");
        else if (e.nonfinite)
            std::snprintf(buf, sizeof(buf), "%zu NaN/Inf", e.nonfinite);
        else
            std::snprintf(buf, sizeof(buf), "%.2e", e.max_rel);
        return std::string(buf);
    };
    if (r.timed)
        std::printf("  %-15s %10.3f %9.1f %7.1f%% %6.1f%% %12s %12s %6d  %s\n", r.kernel.c_str(), r.t.median_ms,
                    r.gflops, r.pct_cublas, r.pct_peak, err(r.vs_cublas).c_str(), err(r.vs_cpu).c_str(),
                    r.t.sm_clock_mhz, r.status.c_str());
    else
        std::printf("  %-15s %10s %9s %8s %7s %12s %12s %6s  %s\n", r.kernel.c_str(), "-", "-", "-", "-",
                    err(r.vs_cublas).c_str(), err(r.vs_cpu).c_str(), "-", r.status.c_str());
}

// Registers, shared memory and theoretical occupancy of every kernel, straight from the compiled
// binary (cudaFuncGetAttributes) and the CUDA occupancy calculator. Occupancy = resident warps per SM
// divided by the maximum the SM supports; it is capped by whichever resource runs out first.
void write_kernel_info(const std::vector<KernelSpec>& kernels, const cudaDeviceProp& p, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::perror(path.c_str());
        std::exit(EXIT_FAILURE);
    }
    std::fprintf(f, "kernel,threads_per_block,registers_per_thread,static_smem_bytes,local_mem_bytes,"
                    "blocks_per_sm,warps_per_sm,theoretical_occupancy_pct,occupancy_limited_by\n");
    std::printf("\nPer-kernel resources (from the compiled code):\n");
    std::printf("  %-15s %8s %5s %9s %9s %10s %10s  %s\n", "kernel", "threads", "regs", "smem(B)", "spill(B)",
                "blocks/SM", "occupancy", "limited by");

    const int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;
    for (const auto& k : kernels) {
        cudaFuncAttributes a{};
        CUDA_CHECK(cudaFuncGetAttributes(&a, k.kernel));
        int blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k.kernel, k.threads_per_block, 0));

        // Each resource caps how many blocks fit on one SM; the smallest cap wins.
        const int warps_per_block = ceil_div(k.threads_per_block, p.warpSize);
        const int by_threads = max_warps / warps_per_block;
        constexpr int kRegAllocUnit = 256;  // registers are handed out per warp in chunks of 256
        const int regs_per_warp = ceil_div(a.numRegs * p.warpSize, kRegAllocUnit) * kRegAllocUnit;
        const int by_regs = a.numRegs > 0 ? (p.regsPerMultiprocessor / regs_per_warp) / warps_per_block : INT_MAX;
        const size_t smem_per_block = a.sharedSizeBytes + p.reservedSharedMemPerBlock;
        const int by_smem =
            a.sharedSizeBytes > 0 ? static_cast<int>(p.sharedMemPerMultiprocessor / smem_per_block) : INT_MAX;
        const int by_slots = p.maxBlocksPerMultiProcessor;

        std::string limiter;
        const std::pair<int, const char*> caps[] = {
            {by_threads, "threads"}, {by_regs, "registers"}, {by_smem, "shared memory"}, {by_slots, "block slots"}};
        for (const auto& [cap, name] : caps)
            if (cap == blocks) limiter += (limiter.empty() ? "" : " + ") + std::string(name);
        if (limiter.empty()) limiter = "other (see ncu)";

        const int warps = blocks * warps_per_block;
        const double occupancy = 100.0 * warps / max_warps;
        std::fprintf(f, "%s,%d,%d,%zu,%zu,%d,%d,%.1f,%s\n", k.name, k.threads_per_block, a.numRegs,
                     a.sharedSizeBytes, a.localSizeBytes, blocks, warps, occupancy, limiter.c_str());
        std::printf("  %-15s %8d %5d %9zu %9zu %10d %9.1f%%  %s\n", k.name, k.threads_per_block, a.numRegs,
                    a.sharedSizeBytes, a.localSizeBytes, blocks, occupancy, limiter.c_str());
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// Profile mode: each kernel exactly once, so Nsight Compute sees one launch per kernel.
// ---------------------------------------------------------------------------------------------
int run_profile(const Options& opt, const std::vector<KernelSpec>& kernels, cublasHandle_t handle) {
    for (int n : opt.sizes) {
        const size_t count = static_cast<size_t>(n) * n;
        const size_t bytes = count * sizeof(float);
        std::vector<float> hA(count), hB(count), hC0(count);
        make_inputs(opt.seed, hA, hB, hC0);
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dC, 0, bytes));

        cublas_sgemm(handle, n, n, n, kBenchAlpha, dA, dB, kBenchBeta, dC);
        int launched = 0;
        for (const auto& k : kernels) {
            if (n % k.size_multiple != 0) continue;
            k.launch(n, n, n, kBenchAlpha, dA, dB, kBenchBeta, dC);
            CUDA_CHECK(cudaGetLastError());
            ++launched;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("profile run: N=%d, cuBLAS + %d kernels\n", n, launched);
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);

    std::vector<KernelSpec> kernels;
    for (const auto& k : all_kernels())
        if (opt.only.empty() || std::find(opt.only.begin(), opt.only.end(), k.name) != opt.only.end())
            kernels.push_back(k);
    for (const auto& name : opt.only) {
        const bool known = name == "cublas" || std::any_of(kernels.begin(), kernels.end(),
                                                           [&](const KernelSpec& k) { return name == k.name; });
        if (!known) {
            std::fprintf(stderr, "unknown kernel: %s\n", name.c_str());
            usage(argv[0], 1);
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    const DeviceSummary dev = query_device(0);
    GpuMonitor mon(0);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Default math mode = genuine FP32 on the CUDA cores, the same arithmetic our kernels do.
    // (CUBLAS_TF32_TENSOR_OP_MATH would let cuBLAS use tensor cores with reduced-precision TF32
    // inputs: faster, but not an apples-to-apples comparison.)
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    if (opt.profile) {
        const int rc = run_profile(opt, kernels, handle);
        CUBLAS_CHECK(cublasDestroy(handle));
        return rc;
    }

    const double peak = dev.peak_gflops();
    std::printf("GPU: %s | %d SMs | CC %d.%d | peak FP32 %.1f GFLOPS at %d MHz | NVML clock monitor: %s\n",
                dev.prop.name, dev.prop.multiProcessorCount, dev.prop.major, dev.prop.minor, peak,
                dev.sm_clock_khz / 1000, mon.ok() ? "on" : "off");
    std::printf("Timing: %d warm-up + %d timed launches per kernel, median reported. "
                "Correctness: every element vs cuBLAS (and vs CPU for N <= %d).\n",
                opt.warmup, opt.reps, opt.cpu_max_n);
    write_kernel_info(kernels, dev.prop, opt.kernel_info);

    FILE* csv = std::fopen(opt.csv.c_str(), "w");
    if (!csv) {
        std::perror(opt.csv.c_str());
        return EXIT_FAILURE;
    }
    write_csv_header(csv);

    // A laptop GPU idles at a few hundred MHz and needs a moment of real load to reach its boost
    // clock. Keep it busy for about a second before the first measurement, so the small sizes at
    // the start are not timed on a half-asleep GPU.
    {
        const int n = 1024;
        float *a, *c;
        CUDA_CHECK(cudaMalloc(&a, sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(&c, sizeof(float) * n * n));
        CUDA_CHECK(cudaMemset(a, 0, sizeof(float) * n * n));
        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventRecord(s));
        while (elapsed < 1000.0f) {
            for (int i = 0; i < 20; ++i) cublas_sgemm(handle, n, n, n, 1.0f, a, a, 0.0f, c);
            CUDA_CHECK(cudaEventRecord(e));
            CUDA_CHECK(cudaEventSynchronize(e));
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, s, e));
        }
        std::printf("GPU spin-up: 1 s of load, SM clock now %d MHz\n", mon.sm_clock_mhz());
        CUDA_CHECK(cudaEventDestroy(s));
        CUDA_CHECK(cudaEventDestroy(e));
        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(c));
    }

    const long long spin_cycles = 2LL * dev.sm_clock_khz;  // ~2 ms
    int failures = 0;
    for (int n : opt.sizes) {
        const size_t count = static_cast<size_t>(n) * n;
        const size_t bytes = count * sizeof(float);
        const double flops = 2.0 * n * n * static_cast<double>(n);  // n*n outputs x n multiply-adds x 2
        std::printf("\n=== N = %d  (%.2f GFLOP per multiply) ===\n", n, flops / 1e9);

        // Does this size fit? A, B, C, C_ref, plus headroom for cuBLAS's workspace.
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const size_t needed = 4 * bytes + (size_t(64) << 20);
        if (needed > free_b) {
            std::printf("  skipped: needs %.0f MiB of GPU memory, only %.0f MiB free\n", needed / 1048576.0,
                        free_b / 1048576.0);
            Row r;
            r.n = n;
            r.status = "skipped_memory";
            r.kernel = "cublas";
            write_csv_row(csv, r);
            for (const auto& k : kernels) {
                r.kernel = k.name;
                write_csv_row(csv, r);
            }
            continue;
        }

        std::vector<float> hA(count), hB(count), hC0(count), hRef(count), hOut(count);
        make_inputs(opt.seed, hA, hB, hC0);
        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMalloc(&dRef, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        // ---- References ------------------------------------------------------------------
        CUDA_CHECK(cudaMemcpy(dRef, hC0.data(), bytes, cudaMemcpyHostToDevice));
        cublas_sgemm(handle, n, n, n, kCheckAlpha, dA, dB, kCheckBeta, dRef);
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));  // waits for the GEMM

        const bool have_cpu = n <= opt.cpu_max_n;
        std::vector<double> hCpu;
        Row cublas_row;
        cublas_row.kernel = "cublas";
        cublas_row.n = n;
        if (have_cpu) {
            hCpu = cpu_sgemm(n, n, n, kCheckAlpha, hA, hB, kCheckBeta, hC0);
            cublas_row.vs_cpu = compare(hRef, hCpu, n);
            if (!cublas_row.vs_cpu.pass) {
                std::fprintf(stderr, "cuBLAS disagrees with the CPU reference (rel err %.2e): the harness is broken.\n",
                             cublas_row.vs_cpu.max_rel);
                return EXIT_FAILURE;
            }
        }

        // ---- cuBLAS timing ------------------------------------------------------------------
        cublas_row.t = time_launches([&] { cublas_sgemm(handle, n, n, n, kBenchAlpha, dA, dB, kBenchBeta, dRef); },
                                     opt.warmup, opt.reps, spin_cycles, mon);
        cublas_row.timed = true;
        cublas_row.status = "ok";
        cublas_row.gflops = flops / (cublas_row.t.median_ms * 1e-3) / 1e9;
        cublas_row.pct_cublas = 100.0;
        cublas_row.pct_peak = 100.0 * cublas_row.gflops / peak;
        print_table_header();
        print_row(cublas_row);
        write_csv_row(csv, cublas_row);

        // ---- Our kernels --------------------------------------------------------------------
        for (const auto& k : kernels) {
            Row r;
            r.kernel = k.name;
            r.n = n;
            if (n % k.size_multiple != 0) {
                r.status = "skipped_size";
                print_row(r);
                write_csv_row(csv, r);
                continue;
            }

            // Correctness run on a random C so that alpha and beta are both exercised.
            CUDA_CHECK(cudaMemcpy(dC, hC0.data(), bytes, cudaMemcpyHostToDevice));
            k.launch(n, n, n, kCheckAlpha, dA, dB, kCheckBeta, dC);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));
            r.vs_cublas = compare(hOut, hRef, n);
            if (have_cpu) r.vs_cpu = compare(hOut, hCpu, n);
            if (!r.vs_cublas.pass || (have_cpu && !r.vs_cpu.pass)) {
                r.status = "FAILED";  // a kernel that computes the wrong answer is not benchmarked
                ++failures;
                print_row(r);
                write_csv_row(csv, r);
                continue;
            }

            r.t = time_launches([&] { k.launch(n, n, n, kBenchAlpha, dA, dB, kBenchBeta, dC); }, opt.warmup,
                                opt.reps, spin_cycles, mon);
            r.timed = true;
            r.status = "ok";
            r.gflops = flops / (r.t.median_ms * 1e-3) / 1e9;
            r.pct_cublas = 100.0 * r.gflops / cublas_row.gflops;
            r.pct_peak = 100.0 * r.gflops / peak;
            print_row(r);
            write_csv_row(csv, r);
        }
        std::fflush(csv);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dRef));
    }
    std::fclose(csv);
    CUBLAS_CHECK(cublasDestroy(handle));

    std::printf("\nWrote %s and %s\n", opt.csv.c_str(), opt.kernel_info.c_str());
    if (failures)
        std::printf("WARNING: %d kernel/size combinations FAILED the correctness check and were not timed.\n",
                    failures);
    return 0;
}
