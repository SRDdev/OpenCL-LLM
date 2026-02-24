# Genral Purpose GPU-Optimization : High-Performance Kernel Design

> "I have been fascinated by the challenge of model optimization moving beyond standard libraries to understand how hardware actually executes math. This repository is my deep dive into the 'mechanics' of performance."

This repository is a technical exploration of optimizing compute-bound and memory-bound kernels for mobile and embedded GPUs using OpenCL. It documents a step-by-step progression from hardware profiling to the implementation of state-of-the-art fused kernels like **Flash Attention**.

## Objectives

The goal is to demonstrate sophisticated resource management on **Unified Memory Architectures (UMA)**. By treating the GPU as a precision instrument, this project implements optimizations that bypass the "Memory Wall" in modern AI workloads.

## Project Roadmap

| Phase | Milestone | Key Technical Achievement |
| --- | --- | --- |
| **I** | **Hardware Handshake** | Mapping the memory hierarchy: Host RAM vs. SRAM (Local) vs. Registers. |
| **II** | **Tensor Foundations** | Implementation of robust Matrix Multiplication ($C = A \cdot B^T$) with arbitrary size handling. |
| **III** | **The Tiling Leap** | Developing **Tiled MatMul** and a standard Attention pipeline ($Softmax(QK^T)V$). |
| **IV** | **Kernel Fusion** | Implementing **Flash Attention 1 & 2** via Online Softmax and tiled accumulation. |

---

## Core Optimization Strategies

### 1. Manual Cache Management

Embedded GPUs often lack massive hardware-managed caches. This project focuses on **SRAM Tiling**—explicitly moving data into `__local` memory to serve as a software-managed cache, reducing high-latency Global Memory (DRAM) round-trips.

### 2. Zero-Copy Orchestration

On UMA systems, the CPU and GPU share physical silicon. We leverage page-aligned memory allocation to allow the GPU to operate directly on host-allocated pointers, removing the $O(N)$ latency and power cost of redundant memory copies.

### 3. Fused Operator Design

To solve the memory bottleneck in Attention mechanisms, we implement **Online Softmax**. This allows the kernel to compute normalization factors incrementally, enabling the fusion of multiple operations into a single, high-bandwidth GPU pass.

---

## Build & Run

### Prerequisites

* **OpenCL SDK** (Headers and ICD Loader).
* **C++17** compatible compiler.
* **CMake 3.10+**.

### Execution

```bash
mkdir build && cd build
cmake ..
make
./cl_info  # Run Phase I hardware diagnostics

```

---

## Why This Matters

As AI models move toward edge devices, the primary bottleneck is rarely raw TFLOPS—it is **memory bandwidth**. This repository serves as a blueprint for writing "bandwidth-aware" kernels that ensure maximum throughput on power-constrained hardware.
