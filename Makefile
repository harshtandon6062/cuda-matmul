# =============================================================================
# SGEMM from scratch — build, benchmark, profile, plot.
#
#   make            build bin/device_info and bin/sgemm
#   make bench      build + device info + benchmark + profiling (if allowed) + charts
#   make device     results/device.md, results/device.json      (STEP 1)
#   make run        results/results.csv, results/kernels.csv    (STEP 2/3)
#   make profile    results/ncu/  (Nsight Compute counters)     (STEP 4)
#   make report     charts + markdown tables from existing CSVs (STEP 4)
#   make clean      remove build outputs (results/ is kept)
#
# Overrides, on the command line or in a git-ignored local.mk:
#   NVCC=/path/to/nvcc  GPU_CC=86  PYTHON=/path/to/python  PROFILE_N=4096
# =============================================================================

-include local.mk

# ---- CUDA toolkit: nvcc on PATH, else the standard install locations ----------
ifndef NVCC
  NVCC := $(shell command -v nvcc 2>/dev/null)
  ifeq ($(NVCC),)
    NVCC := $(firstword $(wildcard /usr/local/cuda/bin/nvcc) \
                        $(lastword $(sort $(wildcard /usr/local/cuda-*/bin/nvcc))))
  endif
endif
CUDA_HOME := $(patsubst %/bin/nvcc,%,$(NVCC))
NCU       ?= $(firstword $(wildcard $(CUDA_HOME)/bin/ncu) $(shell command -v ncu 2>/dev/null))

# ---- Target GPU: compile for the compute capability of the GPU in this machine -
# nvidia-smi prints e.g. "8.6" -> GPU_CC=86 -> SASS for sm_86 only.
ifndef GPU_CC
  GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')
endif
ifneq ($(GPU_CC),)
  ARCH_FLAGS := -gencode arch=compute_$(GPU_CC),code=sm_$(GPU_CC)
else
  ARCH_FLAGS := -arch=native
endif

# -lineinfo maps SASS back to source lines in Nsight Compute; it does not slow the code down.
NVCCFLAGS := -O3 -std=c++17 $(ARCH_FLAGS) -lineinfo -Xcompiler -Wall
LDFLAGS   := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -Xlinker -rpath=$(CUDA_HOME)/lib64
LDLIBS    := -lcublas -lnvidia-ml

PYTHON    ?= python3
PROFILE_N ?= 4096

BUILD   := build
BIN     := bin
RESULTS := results

HEADERS     := $(wildcard src/*.cuh)
KERNEL_SRCS := $(sort $(wildcard src/kernels/*.cu))
SGEMM_OBJS  := $(BUILD)/bench.o $(patsubst src/%.cu,$(BUILD)/%.o,$(KERNEL_SRCS))

.PHONY: all bench device run profile report clean check-nvcc info

all: $(BIN)/device_info $(BIN)/sgemm

$(BIN)/sgemm: $(SGEMM_OBJS)
	@mkdir -p $(BIN)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN)/device_info: $(BUILD)/device_info.o
	@mkdir -p $(BIN)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/%.o: src/%.cu $(HEADERS) | check-nvcc
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

check-nvcc:
	@if [ -z "$(NVCC)" ] || [ ! -x "$(NVCC)" ]; then \
	  echo "ERROR: nvcc not found. Install the CUDA toolkit (README, 'Setup') or run: make NVCC=/path/to/nvcc"; \
	  exit 1; fi

# ---- The one command: everything, in order ------------------------------------
bench: all
	@mkdir -p $(RESULTS)
	./$(BIN)/device_info $(RESULTS)
	./$(BIN)/sgemm --csv $(RESULTS)/results.csv --kernel-info $(RESULTS)/kernels.csv
	@NCU="$(NCU)" PYTHON="$(PYTHON)" scripts/profile.sh $(PROFILE_N) || \
	  echo "NOTE: Nsight Compute profiling skipped (see message above); charts use analytical arithmetic intensity."
	$(PYTHON) scripts/report.py

device: $(BIN)/device_info
	@mkdir -p $(RESULTS)
	./$(BIN)/device_info $(RESULTS)

run: $(BIN)/sgemm
	@mkdir -p $(RESULTS)
	./$(BIN)/sgemm --csv $(RESULTS)/results.csv --kernel-info $(RESULTS)/kernels.csv

profile: $(BIN)/sgemm
	NCU="$(NCU)" PYTHON="$(PYTHON)" scripts/profile.sh $(PROFILE_N)

report:
	$(PYTHON) scripts/report.py

info:
	@echo "NVCC      = $(NVCC)"
	@echo "CUDA_HOME = $(CUDA_HOME)"
	@echo "NCU       = $(NCU)"
	@echo "GPU_CC    = $(GPU_CC)"
	@echo "ARCH      = $(ARCH_FLAGS)"
	@echo "PYTHON    = $(PYTHON)"

clean:
	rm -rf $(BUILD) $(BIN)
