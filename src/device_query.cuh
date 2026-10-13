// device_query.cuh — read the GPU's properties at run time and derive its theoretical peaks.
// Nothing about the hardware is hard-coded in this project; it is all asked from the driver.
#pragma once

#include <cuda_runtime.h>

#include "common.cuh"

// How many FP32 fused multiply-adds one SM can finish per clock cycle ("CUDA cores per SM").
// CUDA has no API for this number: it is a fixed property of each architecture, listed in the
// "Arithmetic Instructions" throughput table of the CUDA C++ Programming Guide. We look it up
// from the compute capability that the driver reports.
inline int fp32_lanes_per_sm(int major, int minor) {
    switch (major) {
        case 3: return 192;                    // Kepler
        case 5: return 128;                    // Maxwell
        case 6: return minor == 0 ? 64 : 128;  // Pascal: GP100 = 64, GP10x = 128
        case 7: return 64;                     // Volta, Turing
        case 8: return minor == 0 ? 64 : 128;  // Ampere A100 = 64; GA10x (8.6), Orin (8.7), Ada (8.9) = 128
        case 9: return 128;                    // Hopper
        case 10:
        case 11:
        case 12: return 128;                   // Blackwell
        default: return -1;                    // unknown architecture: callers must say so
    }
}

struct DeviceSummary {
    int device = 0;
    cudaDeviceProp prop{};
    int sm_clock_khz = 0;     // SM clock the driver reports as the peak ("boost") clock
    int mem_clock_khz = 0;    // memory clock
    int bus_width_bits = 0;   // memory interface width
    int fp32_lanes = 0;       // FP32 lanes per SM (see table above)
    int kernel_timeout = 0;   // 1 if a display watchdog limits kernel run time

    // Peak FP32 = SMs x lanes/SM x 2 FLOP per FMA x clock.
    double peak_gflops() const {
        return 2.0 * prop.multiProcessorCount * fp32_lanes * (sm_clock_khz * 1e3) / 1e9;
    }
    // Peak DRAM bandwidth = 2 transfers per clock (double data rate) x clock x bus width in bytes.
    double peak_bandwidth_gbs() const {
        return 2.0 * (mem_clock_khz * 1e3) * (bus_width_bits / 8.0) / 1e9;
    }
};

inline DeviceSummary query_device(int device) {
    DeviceSummary d;
    d.device = device;
    CUDA_CHECK(cudaGetDeviceProperties(&d.prop, device));
    // Clocks and bus width come from cudaDeviceGetAttribute: CUDA 13 removed the old
    // cudaDeviceProp::clockRate / memoryClockRate fields.
    CUDA_CHECK(cudaDeviceGetAttribute(&d.sm_clock_khz, cudaDevAttrClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&d.mem_clock_khz, cudaDevAttrMemoryClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&d.bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&d.kernel_timeout, cudaDevAttrKernelExecTimeout, device));
    d.fp32_lanes = fp32_lanes_per_sm(d.prop.major, d.prop.minor);
    return d;
}
