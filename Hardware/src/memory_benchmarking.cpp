#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include "utils/utils.hpp"
using namespace std;

int main(){
    //1. Platform setup
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    //2. Load Kernels
    string sourceCode = readKernelFile("Hardware\\kernels\\memory_bench.cl");
    cl::Program::Sources sources(1, {sourceCode.c_str(), sourceCode.length()});
    cl::Program program(context, sources);

    if (program.build("-cl-std=CL2.0") != CL_SUCCESS) {
        cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return 1;
    }

    cl::Kernel k_global(program,"benchmark_global");
    cl::Kernel k_local(program,"benchmark_local");
    cl::Kernel k_private(program,"benchmark_private");

    // 3. Setup Data (10 Million Floats = ~40MB)
    int numElements = 10000000;
    int iterations = 10000;
    size_t bufferSize = numElements*sizeof(float);
    vector<float> hostData(numElements, 1.0f);
    cl::Buffer deviceBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufferSize, hostData.data());

    // 4. Execution Configuration
    cl::NDRange globalSize(numElements);
    cl::NDRange localSize(256);

    // --- Dynamically Query Device Memory Sizes ---
    cl_ulong globalMem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    cl_ulong localMem  = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    double globalMemMB = globalMem / (1024.0 * 1024.0);
    double localMemKB  = localMem / 1024.0;
    double bufferMB    = bufferSize / (1024.0 * 1024.0);

    // --- Print Beautiful Header & Specs ---
    cout << "=====================================================================" << endl;
    cout << "       PHASE 1.5: THE MEMORY WALL BENCHMARK                          " << endl;
    cout << "=====================================================================" << endl;
    cout << "  Device       : " << device.getInfo<CL_DEVICE_NAME>() << endl;
    cout << "  Array Size   : " << numElements / 1000000 << " Million elements (" << bufferMB << " MB)" << endl;
    cout << "  Iterations   : " << iterations << " reads per element" << endl;
    cout << "---------------------------------------------------------------------" << endl;
    cout << "  [1] HRAM (Global) : ~" << (int)globalMemMB << " MB  | Slowest, Off-Chip PCIe/Bus" << endl;
    cout << "  [2] SRAM (Local)  : ~" << localMemKB << " KB    | Fast, Shared On-Chip Compute Unit" << endl;
    cout << "  [3] Registers     : ~Bytes    | Instant, Per-Thread ALU Storage" << endl;
    cout << "=====================================================================\n" << endl;

    // --- Updated Lambda: Now returns the time for analysis ---
    auto run_and_time = [&](cl::Kernel& kernel, string name, string desc, bool use_local = false) -> double {
        cl::Event event;
        kernel.setArg(0, deviceBuffer);
        
        if (use_local) {
            kernel.setArg(1, cl::Local(256 * sizeof(float)));
            kernel.setArg(2, iterations);
        } else {
            kernel.setArg(1, iterations);
        }

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
        queue.finish();

        // Extract nanoseconds from OpenCL profiling
        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double duration_ms = (end - start) / 1000000.0;
        
        // Print nicely formatted row
        cout << left << setw(22) << name << " | " 
             << right << setw(8) << fixed << setprecision(2) << duration_ms << " ms" 
             << "  ->  " << desc << endl;
             
        return duration_ms;
    };

    // --- Run Benchmarks & Build Table ---
    cout << left << setw(22) << "MEMORY TIER" << " | " << setw(11) << "TIME (ms)" << " | " << "DESCRIPTION" << endl;
    cout << "-----------------------|-------------|--------------------------------" << endl;
    
    double t_hram = run_and_time(k_global,  "[1] HRAM (Global)",   "Baseline (Off-Chip VRAM)");
    double t_sram = run_and_time(k_local,   "[2] SRAM (Local)",    "Shared Memory (On-Chip)", true);
    double t_priv = run_and_time(k_private, "[3] Registers (Priv)","ALU Registers (No Fetch)");

    // --- Print Dynamic Performance Analysis ---
    cout << "\n=====================================================================" << endl;
    cout << "  PERFORMANCE ANALYSIS:" << endl;
    cout << "  > SRAM is ~" << fixed << setprecision(1) << (t_hram / t_sram) << "x faster than HRAM." << endl;
    cout << "  > Registers are ~" << fixed << setprecision(1) << (t_hram / t_priv) << "x faster than HRAM." << endl;
    cout << "=====================================================================" << endl;

    return 0;
}

