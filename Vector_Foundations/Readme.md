# OpenCL Matrix Multiplication Benchmarks

## 🔴 Device 1: Advanced Micro Devices, Inc. (gfx90c)
* **OpenCL Version:** OpenCL 2.0 AMD-APP (3302.6)
* **Max Compute Units:** 8
* **Max Clock Freq:** 2000 MHz
* **Max Work-Group:** 256 threads per group
* **Max Work-Item Dims:** (1024, 1024, 1024)
* **Global Memory:** 6234 MB
* **Local Memory (SRAM):** 32 KB per Compute Unit

| Size | Kernel | Kernel(ms) | HRAM In(ms) | HRAM Out(ms) | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **128** | Naive | 0.162 | 0.034 | 0.014 | PASS |
| | SRAM Tiled | 0.081 | - | 0.015 | Speedup: **2.00x** |
| | Register Tiled | 0.085 | - | 0.013 | Speedup: **1.90x** |
| **256** | Naive | 0.961 | 0.093 | 0.113 | PASS |
| | SRAM Tiled | 0.443 | - | 0.105 | Speedup: **2.16x** |
| | Register Tiled | 0.469 | - | 0.100 | Speedup: **2.04x** |
| **512** | Naive | 6.311 | 0.299 | 0.413 | PASS |
| | SRAM Tiled | 3.249 | - | 0.407 | Speedup: **1.94x** |
| | Register Tiled | 2.186 | - | 0.407 | Speedup: **2.88x** |
| **1024** | Naive | 104.945 | 4.391 | 1.004 | PASS |
| | SRAM Tiled | 26.127 | - | 1.035 | Speedup: **4.01x** |
| | Register Tiled | 17.254 | - | 0.986 | Speedup: **6.08x** |
| **2048** | Naive | 985.969 | 3.981 | 2.553 | PASS |
| | SRAM Tiled | 206.188 | - | 2.444 | Speedup: **4.78x** |
| | Register Tiled | 134.761 | - | 2.581 | Speedup: **7.31x** |
| **4096** | Naive | 6051.134 | 20.954 | 7.365 | PASS |
| | SRAM Tiled | 599.123 | - | 7.805 | Speedup: **10.0x** |
| | Register Tiled | 424.538 | - | 7.577 | Speedup: **14.2x** |

---

## 🟢 Device 2: NVIDIA Corporation (GeForce RTX 3060 Laptop GPU)
* **OpenCL Version:** OpenCL 3.0 CUDA
* **Max Compute Units:** 30
* **Max Clock Freq:** 1425 MHz
* **Max Work-Group:** 1024 threads per group
* **Max Work-Item Dims:** (1024, 1024, 64)
* **Global Memory:** 6143 MB
* **Local Memory (SRAM):** 48 KB per Compute Unit

| Size | Kernel | Kernel(ms) | HRAM In(ms) | HRAM Out(ms) | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **128** | Naive | 0.015 | 0.022 | 0.012 | PASS |
| | SRAM Tiled | 0.017 | - | 0.012 | Speedup: **0.88x** |
| | Register Tiled | 0.018 | - | 0.011 | Speedup: **0.83x** |
| **256** | Naive | 0.062 | 0.087 | 0.041 | PASS |
| | SRAM Tiled | 0.054 | - | 0.041 | Speedup: **1.15x** |
| | Register Tiled | 0.061 | - | 0.041 | Speedup: **1.01x** |
| **512** | Naive | 0.531 | 0.322 | 0.157 | PASS |
| | SRAM Tiled | 0.341 | - | 0.157 | Speedup: **1.55x** |
| | Register Tiled | 0.318 | - | 0.158 | Speedup: **1.66x** |
| **1024** | Naive | 4.453 | 1.419 | 0.769 | PASS |
| | SRAM Tiled | 2.559 | - | 0.776 | Speedup: **1.74x** |
| | Register Tiled | 2.334 | - | 0.709 | Speedup: **1.90x** |
| **2048** | Naive | 96.791 | 6.064 | 3.271 | PASS |
| | SRAM Tiled | 20.235 | - | 3.082 | Speedup: **4.78x** |
| | Register Tiled | 18.065 | - | 3.015 | Speedup: **5.35x** |
| **4096** | Naive | 981.545 | 25.497 | 209.977 | PASS |
| | SRAM Tiled | 158.052 | - | 12.292 | Speedup: **6.21x** |
| | Register Tiled | 134.388 | - | 12.168 | Speedup: **7.30x** |