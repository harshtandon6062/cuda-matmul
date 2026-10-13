// gpu_monitor.cuh — read the GPU's *current* clock and temperature through NVML, the library
// behind nvidia-smi. A laptop GPU changes its clock constantly (power and thermal limits), so
// the benchmark records the clock that was actually in effect while each kernel ran.
// If NVML is unavailable every query returns -1 and the rest of the program carries on.
#pragma once

#include <string>

#include <cuda_runtime.h>
#include <nvml.h>

class GpuMonitor {
public:
    explicit GpuMonitor(int cuda_device) {
        char bus_id[64] = {0};
        if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), cuda_device) != cudaSuccess) return;
        if (nvmlInit_v2() != NVML_SUCCESS) return;
        initialized_ = true;
        // Match the CUDA device to the NVML device by PCI address (robust with several GPUs).
        ok_ = nvmlDeviceGetHandleByPciBusId_v2(bus_id, &dev_) == NVML_SUCCESS;
    }
    ~GpuMonitor() {
        if (initialized_) nvmlShutdown();
    }
    GpuMonitor(const GpuMonitor&) = delete;
    GpuMonitor& operator=(const GpuMonitor&) = delete;

    bool ok() const { return ok_; }
    int sm_clock_mhz() const { return clock(NVML_CLOCK_SM, false); }
    int max_sm_clock_mhz() const { return clock(NVML_CLOCK_SM, true); }
    int mem_clock_mhz() const { return clock(NVML_CLOCK_MEM, false); }

    int temperature_c() const {
        unsigned int t = 0;
        if (!ok_ || nvmlDeviceGetTemperature(dev_, NVML_TEMPERATURE_GPU, &t) != NVML_SUCCESS) return -1;
        return static_cast<int>(t);
    }

    std::string driver_version() const {
        char buf[96] = {0};
        if (!initialized_ || nvmlSystemGetDriverVersion(buf, sizeof(buf)) != NVML_SUCCESS) return "unknown";
        return buf;
    }

private:
    int clock(nvmlClockType_t type, bool max) const {
        unsigned int mhz = 0;
        if (!ok_) return -1;
        const nvmlReturn_t r = max ? nvmlDeviceGetMaxClockInfo(dev_, type, &mhz)
                                   : nvmlDeviceGetClockInfo(dev_, type, &mhz);
        return r == NVML_SUCCESS ? static_cast<int>(mhz) : -1;
    }

    bool initialized_ = false;
    bool ok_ = false;
    nvmlDevice_t dev_{};
};
