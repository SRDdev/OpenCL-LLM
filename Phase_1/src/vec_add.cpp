    #define CL_HPP_TARGET_OPENCL_VERSION 300
    #include <CL/opencl.hpp>
    #include <iostream>
    #include <fstream>
    #include <vector>
    #include <sstream>
    using namespace std;


    int main(){
        // --- 1. SETUP (Standard Boilerplate) ---
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.empty()) return 1;
        vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if(devices.empty()) { cerr << "No GPUs found." << endl; return 1; }
        cl::Device device = devices[0];
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        std::cout << "Target Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // --- 2. LOAD KERNEL SOURCE ---
        std::ifstream file("kernels/vec_add.cl");
        if (!file.is_open()){
            std::cerr << "Error: Could not open kernel file!" << std::endl;
            return 1;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string sourceCode = buffer.str();

        // --- 3. BUILD THE PROGRAM (JIT Compilation) ---
        cl::Program::Sources sources;
        sources.push_back({sourceCode.c_str(), sourceCode.length()});
        cl::Program program(context, sources);

        // Attempt to compile
        if(program.build({device}) != CL_SUCCESS){
            std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            return 1;
        }
        else{
            cout << "JIT Compiled !!!"<< endl;;
        }

        // --- 4. PREPARE DATA ---
        int N = 100;
        vector<float> A(N,1.0f);
        vector<float> B(N,2.0f);
        vector<float> C(N);

        // Create Buffers for each.
        cl::Buffer bufA(context, CL_MEM_COPY_HOST_PTR, sizeof(float)*N, A.data());
        cl::Buffer bufB(context, CL_MEM_COPY_HOST_PTR, sizeof(float)*N, B.data());
        cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(float)*N);


        // --- 5. SETUP KERNEL ARGUMENTS ---
        // Extract the specific function "vec_add" from the compiled program
        cl::Kernel kernel(program, "vec_add");
        kernel.setArg(0,bufA);
        kernel.setArg(1,bufB);
        kernel.setArg(2,bufC);
        kernel.setArg(3,N);

        // --- 6. EXECUTE (The Launch) ---
        // Global Size: Total threads needed (N)
        // Local Size: Threads per group (usually null here to let driver decide, or explicit like 64)
        cl::NDRange globalSize(N);
        cl::NDRange localSize(64);
        queue.enqueueNDRangeKernel(kernel,cl::NullRange,globalSize,cl::NullRange);
        queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(float)*N, C.data());

        //Verify
        std::cout << "Result C[0] = " << C[0] << " (Expect 3)" << std::endl;
        std::cout << "Result C[99] = " << C[99] << " (Expect 3)" << std::endl;
        return 0;
    }


// Performace Mode
// #define CL_HPP_TARGET_OPENCL_VERSION 300
// #include <CL/opencl.hpp>

// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <sstream>
// #include <chrono>
// #include <iomanip>

// using namespace std;
// using namespace std::chrono;

// int main() {
//     // ===================== 1. PLATFORM / DEVICE =====================
//     vector<cl::Platform> platforms;
//     cl::Platform::get(&platforms);
//     if (platforms.empty()) {
//         cerr << "No OpenCL platforms found\n";
//         return 1;
//     }

//     vector<cl::Device> devices;
//     platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
//     if (devices.empty()) {
//         cerr << "No GPU devices found\n";
//         return 1;
//     }

//     cl::Device device = devices[0];
//     cl::Context context(device);

//     // Enable profiling explicitly
//     cl::CommandQueue queue(
//         context,
//         device,
//         CL_QUEUE_PROFILING_ENABLE
//     );

//     cout << "Target Device: "
//          << device.getInfo<CL_DEVICE_NAME>() << endl;

//     // ===================== 2. LOAD KERNEL =====================
//     ifstream file("kernels/vec_add.cl");
//     if (!file.is_open()) {
//         cerr << "Failed to open kernel file\n";
//         return 1;
//     }

//     stringstream ss;
//     ss << file.rdbuf();
//     string source = ss.str();

//     cl::Program::Sources sources;
//     sources.push_back({ source.c_str(), source.size() });

//     cl::Program program(context, sources);
//     if (program.build({ device }) != CL_SUCCESS) {
//         cerr << "Build error:\n"
//              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
//              << endl;
//         return 1;
//     }

//     cout << "JIT Compiled Successfully\n";

//     // ===================== 3. EXTREME DATA =====================
//     const size_t N = 50'000'000;     // 50 million
//     const int ITERATIONS = 5;

//     vector<float> A(N, 1.0f);
//     vector<float> B(N, 2.0f);
//     vector<float> C_cpu(N);
//     vector<float> C_gpu(N);

//     size_t bytes = N * sizeof(float);

//     cout << "\nData Size: " << N << " floats ("
//          << fixed << setprecision(2)
//          << bytes / (1024.0 * 1024.0) << " MB per vector)\n";

//     // ===================== 4. CPU (SERIAL) =====================
//     double cpu_time_ms = 0.0;

//     for (int it = 0; it < ITERATIONS; it++) {
//         auto t0 = high_resolution_clock::now();

//         for (size_t i = 0; i < N; i++) {
//             C_cpu[i] = A[i] + B[i];
//         }

//         auto t1 = high_resolution_clock::now();
//         cpu_time_ms += duration<double, milli>(t1 - t0).count();
//     }

//     cpu_time_ms /= ITERATIONS;

//     double cpu_bandwidth =
//         (3.0 * bytes) / (cpu_time_ms / 1000.0) / 1e9;

//     double cpu_gflops =
//         (double(N) / (cpu_time_ms / 1000.0)) / 1e9;

//     // ===================== 5. GPU BUFFERS =====================
//     cl::Buffer bufA(context,
//                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//                     bytes,
//                     A.data());

//     cl::Buffer bufB(context,
//                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//                     bytes,
//                     B.data());

//     cl::Buffer bufC(context,
//                     CL_MEM_WRITE_ONLY,
//                     bytes);

//     cl::Kernel kernel(program, "vec_add");
//     kernel.setArg(0, bufA);
//     kernel.setArg(1, bufB);
//     kernel.setArg(2, bufC);
//     kernel.setArg(3, (int)N);

//     // IMPORTANT: global size must be multiple of local size
//     size_t localSize = 256;
//     size_t globalSize =
//         ((N + localSize - 1) / localSize) * localSize;

//     // ===================== 6. GPU EXECUTION =====================
//     double gpu_kernel_ms = 0.0;
//     double gpu_total_ms = 0.0;

//     for (int it = 0; it < ITERATIONS; it++) {
//         cl::Event kernelEvent;

//         auto t0 = high_resolution_clock::now();

//         queue.enqueueNDRangeKernel(
//             kernel,
//             cl::NullRange,
//             cl::NDRange(globalSize),
//             cl::NDRange(localSize),
//             nullptr,
//             &kernelEvent
//         );

//         queue.enqueueReadBuffer(
//             bufC,
//             CL_TRUE,
//             0,
//             bytes,
//             C_gpu.data()
//         );

//         queue.finish();

//         auto t1 = high_resolution_clock::now();

//         cl_ulong start =
//             kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//         cl_ulong end =
//             kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

//         gpu_kernel_ms += (end - start) * 1e-6;
//         gpu_total_ms += duration<double, milli>(t1 - t0).count();
//     }

//     gpu_kernel_ms /= ITERATIONS;
//     gpu_total_ms /= ITERATIONS;

//     double gpu_bandwidth =
//         (3.0 * bytes) / (gpu_kernel_ms / 1000.0) / 1e9;

//     double gpu_gflops =
//         (double(N) / (gpu_kernel_ms / 1000.0)) / 1e9;

//     // ===================== 7. VERIFY =====================
//     cout << "\nVerification:\n";
//     cout << "C_cpu[0]     = " << C_cpu[0]
//          << ", C_gpu[0]     = " << C_gpu[0] << endl;
//     cout << "C_cpu[N-1]   = " << C_cpu[N-1]
//          << ", C_gpu[N-1]   = " << C_gpu[N-1] << endl;

//     // ===================== 8. METRICS =====================
//     cout << fixed << setprecision(6);
//     cout << "\n========== PERFORMANCE METRICS ==========\n";

//     cout << "\n[CPU - Non Parallel]\n";
//     cout << "Avg Time      : " << cpu_time_ms << " ms\n";
//     cout << "Bandwidth     : " << cpu_bandwidth << " GB/s\n";
//     cout << "Throughput    : " << cpu_gflops << " GFLOPS\n";

//     cout << "\n[GPU - Parallel (OpenCL)]\n";
//     cout << "Kernel Time   : " << gpu_kernel_ms << " ms\n";
//     cout << "Total Time    : " << gpu_total_ms << " ms\n";
//     cout << "Bandwidth     : " << gpu_bandwidth << " GB/s\n";
//     cout << "Throughput    : " << gpu_gflops << " GFLOPS\n";

//     cout << "\nSpeedup (Kernel-only): "
//          << cpu_time_ms / gpu_kernel_ms << "x\n";

//     cout << "========================================\n";

//     return 0;
// }
