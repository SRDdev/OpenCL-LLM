# OpenCL Matrix Multiplication: The Journey to 3.4 Teraflops

## The Benchmark Setup
We set out to optimize a standard 1024 × 1024 Matrix Multiplication kernel using OpenCL (C++17), evolving it from a naive implementation to a highly tuned, hardware-aware architecture. 

To prove that hardware dictates software performance, we ran the exact same unified benchmark suite across two completely different GPU architectures:
1.  **Integrated GPU:** AMD Radeon Graphics (Laptop iGPU)
2.  **Dedicated GPU:** NVIDIA GeForce RTX 3060 (Laptop dGPU)

Total Operations per run: **2.147 Billion Ops**

---

## Performance Summary: AMD vs. NVIDIA

The table below tracks the evolution of the 10 kernels and the resulting performance (in GFLOPS) on both architectures. 

*Note: The speedup multiplier is relative to the Naive (Base) kernel on each respective GPU.*

| # | Kernel Optimization Stage | AMD iGPU (GFLOPS) | AMD Speedup | NVIDIA RTX 3060 (GFLOPS) | NVIDIA Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | **Naive (Global Write in Loop)** | 12.73 | 1.0x | 67.92 | 1.0x |
| **02** | **Naive (Register Accumulation)** | 29.83 | 2.3x | 157.14 | 2.3x |
| **03** | **Global Memory Coalescing** | 125.74 | 9.8x | 664.71 | 9.7x |
| **04** | **Shared Memory (Tiles A & B)** | 227.61 | 17.8x | 840.21 | 12.3x |
| **05** | **Shared Memory (Tile B Only)** | 293.82 | 23.0x | 576.62 | 8.4x |
| **06** | **1D Block Tiling (WPT=4)** | 325.72 | 25.5x | 929.59 | 13.6x |
| **07** | **2D Block Tiling (Scalar)** | 666.87 | 52.3x | 2182.26 | 32.1x |
| **08** | **2D Block Tiling (Vectorized)** | 760.75 | 59.7x | 2304.56 | 33.9x |
| **09** | **Warp Tiling (Scalar)** | 336.89 | 26.4x | **3421.13** | **50.3x** |
| **10** | **Ultimate VecWarp (Vectorized)**| **1082.14** | **85.0x** | 3206.65 | 47.2x |

### The Brutal Reality of Hardware Tuning
1. **The Vectorization Trap:** Notice how Kernel 10 (Vectorized Warp) completely destroyed Kernel 9 on the AMD GPU, breaking the 1 Teraflop barrier. AMD's architecture loves explicit `float4` vectorization. However, on the NVIDIA RTX 3060, forcing vectorization actually *hurt* performance by 200 GFLOPS. NVIDIA executes in warps of 32 scalar threads; forcing wide vectors increased register pressure and choked the compiler. 
2. **Brute-Forcing Bad Code:** The RTX 3060 churned out 67 GFLOPS on the absolute worst Naive code—faster than a tuned, coalesced kernel on the AMD chip. Massive L2 caches and high-bandwidth GDDR6 VRAM act as a crutch, hiding terrible memory access patterns.

---

## Raw Benchmark Logs: NVIDIA RTX 3060 (Dedicated)

Below are the raw terminal outputs from the RTX 3060, peaking at an incredible **3.4 Teraflops** with a sub-millisecond execution time.

```text
================================================================================
 OPENCL PERFORMANCE: NAIVE MATMUL (GLOBAL WRITE)
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (Base)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 31.62 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 33.56 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 67.92 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: NAIVE MATMUL (REGISTER ACCUM)
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (Register Accum)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 13.67 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 15.60 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 157.14 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: GLOBAL COALESCING MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (Coalesced)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 3.23 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 5.17 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 664.71 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: SHARED MEMORY MATMUL (TILES A&B)
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (SRAM)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 2.56 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 4.50 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 840.21 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: SHARED MEMORY MATMUL (TILE B ONLY)
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (SRAM B)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 3.72 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 5.66 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 576.62 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: 1D BLOCK TILE MEMORY MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x4 (1D Block)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 2.31 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 4.25 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 929.59 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: 2D BLOCK TILE SCALAR MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 4x4 (2D Block)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 0.98 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 2.92 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 2182.26 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: 2D BLOCK TILE VECTOR MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 4x4 (Vec4)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 0.93 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 2.87 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 2304.56 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: WARP-TILED SCALAR MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 8x8 (Reg Scalar)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 0.63 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 2.57 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 3421.13 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: ULTIMATE VEC-WARP MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 8x8 (Reg Vec4)
Total Ops          : 2.15 Billion
CPU -> GPU         : 1.29 ms
Kernel Exec        : 0.67 ms
GPU -> CPU         : 0.65 ms
Total Latency      : 2.60 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 3206.65 GFLOPS
================================================================================
```

---

## Raw Benchmark Logs: AMD Radeon Graphics (Integrated)

Below are the raw terminal outputs from the AMD iGPU, peaking at **1.08 Teraflops** when hardware-aligned vectorization was implemented.

```text
================================================================================
 OPENCL PERFORMANCE: NAIVE MATMUL (GLOBAL WRITE)
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (Base)
Total Ops          : 2.15 Billion
CPU -> GPU         : 3.20 ms
Kernel Exec        : 168.71 ms
GPU -> CPU         : 0.46 ms
Total Latency      : 172.37 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 12.73 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: NAIVE MATMUL (REGISTER ACCUM)
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (Register Accum)
Total Ops          : 2.15 Billion
CPU -> GPU         : 3.24 ms
Kernel Exec        : 71.99 ms
GPU -> CPU         : 0.50 ms
Total Latency      : 75.73 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 29.83 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: GLOBAL COALESCING MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (Coalesced)
Total Ops          : 2.15 Billion
CPU -> GPU         : 3.23 ms
Kernel Exec        : 17.08 ms
GPU -> CPU         : 0.48 ms
Total Latency      : 20.79 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 125.74 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: SHARED MEMORY MATMUL (TILES A&B)
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (SRAM)
Total Ops          : 2.15 Billion
CPU -> GPU         : 3.21 ms
Kernel Exec        : 9.44 ms
GPU -> CPU         : 0.46 ms
Total Latency      : 13.11 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 227.61 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: SHARED MEMORY MATMUL (TILE B ONLY)
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x1 (SRAM B)
Total Ops          : 2.15 Billion
CPU -> GPU         : 3.19 ms
Kernel Exec        : 7.31 ms
GPU -> CPU         : 0.44 ms
Total Latency      : 10.94 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 293.82 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: 1D BLOCK TILE MEMORY MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 1x4 (1D Block)
Total Ops          : 2.15 Billion
CPU -> GPU         : 2.44 ms
Kernel Exec        : 6.59 ms
GPU -> CPU         : 0.47 ms
Total Latency      : 9.50 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 325.72 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: 2D BLOCK TILE SCALAR MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 4x4 (2D Block)
Total Ops          : 2.15 Billion
CPU -> GPU         : 3.49 ms
Kernel Exec        : 3.22 ms
GPU -> CPU         : 0.47 ms
Total Latency      : 7.18 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 666.87 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: 2D BLOCK TILE VECTOR MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 4x4 (Vec4)
Total Ops          : 2.15 Billion
CPU -> GPU         : 3.11 ms
Kernel Exec        : 2.82 ms
GPU -> CPU         : 0.48 ms
Total Latency      : 6.42 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 760.75 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: WARP-TILED SCALAR MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 8x8 (Reg Scalar)
Total Ops          : 2.15 Billion
CPU -> GPU         : 2.87 ms
Kernel Exec        : 6.37 ms
GPU -> CPU         : 0.44 ms
Total Latency      : 9.68 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 336.89 GFLOPS
================================================================================

================================================================================
 OPENCL PERFORMANCE: ULTIMATE VEC-WARP MATMUL
================================================================================
Metric             | Value
-------------------| -----------------------------------------------------------
Matrix Size        : 1024 x 1024
Work/Thread        : 8x8 (Reg Vec4)
Total Ops          : 2.15 Billion
CPU -> GPU         : 3.58 ms
Kernel Exec        : 1.98 ms
GPU -> CPU         : 0.44 ms
Total Latency      : 6.00 ms
Validation         : PASS
--------------------------------------------------------------------------------
PERFORMANCE        : 1082.14 GFLOPS
================================================================================
```