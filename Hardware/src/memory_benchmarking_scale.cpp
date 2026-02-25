#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include "utils/utils.hpp"

using namespace std;

int main() {
    // 1. Platform & Device Setup
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // 2. Load Kernels
    string sourceCode = readKernelFile("Hardware\\kernels\\memory_bench.cl");
    cl::Program::Sources sources(1, {sourceCode.c_str(), sourceCode.length()});
    cl::Program program(context, sources);

    if (program.build("-cl-std=CL2.0") != CL_SUCCESS) {
        cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return 1;
    }

    cl::Kernel k_global(program, "benchmark_global");
    cl::Kernel k_local(program, "benchmark_local");
    cl::Kernel k_private(program, "benchmark_private");

    // 3. Setup Experiment Parameters
    int iterations = 10000;
    // We use multiples of 256 to align with the local workgroup size perfectly
    vector<int> testSizes = {256000, 1024000, 2560000, 5120000, 10240000, 20480000}; 
    cl::NDRange localSize(256);

    // 4. Setup CSV Output
    ofstream csvFile("benchmark_results.csv");
    csvFile << "Elements,HRAM_ms,SRAM_ms,Registers_ms\n";

    cout << "=====================================================================" << endl;
    cout << "      PHASE 2: SCALING THE MEMORY WALL                               " << endl;
    cout << "=====================================================================" << endl;
    cout << " Device     : " << device.getInfo<CL_DEVICE_NAME>() << endl;
    cout << " Iterations : " << iterations << " per element" << endl;
    cout << "=====================================================================\n" << endl;

    // Updated Lambda: Now accepts the buffer and global size so it can be reused in the loop
    auto run_and_time = [&](cl::Kernel& kernel, cl::Buffer& buffer, cl::NDRange globalSize, bool use_local = false) -> double {
        cl::Event event;
        kernel.setArg(0, buffer);
        
        if (use_local) {
            kernel.setArg(1, cl::Local(256 * sizeof(float)));
            kernel.setArg(2, iterations);
        } else {
            kernel.setArg(1, iterations);
        }

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
        queue.finish();

        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        return (end - start) / 1000000.0; // Return ms
    };

    // 5. Run the Loop
    for (int numElements : testSizes) {
        size_t bufferSize = numElements * sizeof(float);
        double bufferMB = bufferSize / (1024.0 * 1024.0);
        
        cout << "Testing Size: " << numElements / 1000000.0 << " Million (" << fixed << setprecision(2) << bufferMB << " MB)..." << flush;

        // Allocate host and device memory for this specific size
        vector<float> hostData(numElements, 1.0f);
        cl::Buffer deviceBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufferSize, hostData.data());
        cl::NDRange globalSize(numElements);

        // Run benchmarks
        double t_hram = run_and_time(k_global, deviceBuffer, globalSize);
        double t_sram = run_and_time(k_local, deviceBuffer, globalSize, true);
        double t_priv = run_and_time(k_private, deviceBuffer, globalSize);

        cout << " Done. (HRAM: " << t_hram << "ms | SRAM: " << t_sram << "ms)" << endl;

        // Write to CSV
        csvFile << numElements << "," << t_hram << "," << t_sram << "," << t_priv << "\n";
    }

    csvFile.close();
    cout << "\n> Experiments complete. Data saved to 'benchmark_results.csv'." << endl;

    return 0;
}