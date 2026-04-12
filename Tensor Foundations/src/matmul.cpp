#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <thread> 
#include <chrono> 
#include "C:/Users/ASUS/Desktop/Shreyas/OpenCL-LLM/utils/utils.hpp"

using namespace std;

// --- Kernel Profile Definition ---
struct KernelProfile {
    string displayName;
    string fileName;
    string functionName;
    int wpt_x;         
    int wpt_y;         
    int local_x;       
    int local_y;       
    string wptString;  
};

// --- CPU Ground Truth Function ---
bool verify_results(const vector<float>& A, const vector<float>& B, const vector<float>& GPU_C, int M, int N, int K) {
    float epsilon = 1e-2f;
    for (int i = 0; i < 64; i++) { 
        float cpu_sum = 0.0f;
        int row = i / N;
        int col = i % N;
        for (int k = 0; k < K; k++) {
            cpu_sum += A[row * K + k] * B[k * N + col];
        }
        if (std::abs(GPU_C[i] - cpu_sum) > epsilon) {
            cout << "\n[!] DEBUG Mismatch at index " << i << ": GPU=" << GPU_C[i] << " CPU=" << cpu_sum << endl;
            return false;
        }
    }
    return true; 
}

// --- Benchmark Runner Function ---
// UPDATED: Removed Context & Queue from arguments so we can create them locally.
void runBenchmark(const KernelProfile& profile, cl::Device& device,
                  vector<float>& h_A, vector<float>& h_B, vector<float>& h_C, 
                  int M, int N, int K, size_t matrixSize) {
    
    // 0. ISOLATED CONTEXT CREATION (The Fix)
    // Creates a brand new execution state on the GPU. No memory bleed-over.
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // 1. Build Program
    string filePath = "Tensor Foundations/kernels/" + profile.fileName;
    string src;
    try {
        src = readKernelFile(filePath.c_str());
    } catch (...) {
        cout << "[!] Error: Could not read " << filePath << ". Skipping..." << endl;
        return;
    }

    cl::Program program(context, src);
    if (program.build("-cl-std=CL2.0") != CL_SUCCESS) {
        cout << "[!] Build Error in " << profile.fileName << ":\n" 
             << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return;
    }
    
    cl::Kernel kernel(program, profile.functionName.c_str());

    // 2. Setup Device Buffers (Allocated fresh on the new context)
    cl::Buffer d_A(context, CL_MEM_READ_ONLY, sizeof(float) * matrixSize);
    cl::Buffer d_B(context, CL_MEM_READ_ONLY, sizeof(float) * matrixSize);
    cl::Buffer d_C(context, CL_MEM_WRITE_ONLY, sizeof(float) * matrixSize);
    cl::Event evInA, evInB, evExec, evOut;

    std::fill(h_C.begin(), h_C.end(), 0.0f);
    queue.enqueueWriteBuffer(d_C, CL_FALSE, 0, sizeof(float) * matrixSize, h_C.data(), nullptr, nullptr);

    // 3. Data Transfer & Kernel Execution
    queue.enqueueWriteBuffer(d_A, CL_FALSE, 0, sizeof(float) * matrixSize, h_A.data(), nullptr, &evInA);
    queue.enqueueWriteBuffer(d_B, CL_FALSE, 0, sizeof(float) * matrixSize, h_B.data(), nullptr, &evInB);

    kernel.setArg(0, d_A); kernel.setArg(1, d_B); kernel.setArg(2, d_C);
    kernel.setArg(3, M);   kernel.setArg(4, N);   kernel.setArg(5, K);
    
    cl::NDRange globalSize(N / profile.wpt_x, M / profile.wpt_y); 
    cl::NDRange localSize(profile.local_x, profile.local_y); 
    
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &evExec);
    queue.enqueueReadBuffer(d_C, CL_TRUE, 0, sizeof(float) * matrixSize, h_C.data(), nullptr, &evOut);

    // 4. Timing Helper
    auto get_ms = [](cl::Event& e) {
        e.wait();
        cl_ulong start, end;
        e.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        e.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        return (double)(end - start) * 1.0e-6;
    };

    double t_in    = get_ms(evInA) + get_ms(evInB);
    double t_exec  = get_ms(evExec);
    double t_out   = get_ms(evOut);
    double t_total = t_in + t_exec + t_out;

    bool passed = verify_results(h_A, h_B, h_C, M, N, K);
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (t_exec > 0) ? (ops / (t_exec * 1.0e-3 * 1.0e9)) : 0.0;

    // 5. Formatted Output
    cout << "\n" << string(80, '=') << endl;
    cout << " OPENCL PERFORMANCE: " << profile.displayName << endl;
    cout << string(80, '=') << endl;
    
    cout << left << setw(18) << "Metric" << " | " << "Value" << endl;
    cout << string(19, '-') << "| " << string(59, '-') << endl;
    
    cout << left << setw(18) << "Matrix Size"    << " : " << N << " x " << M << endl;
    cout << left << setw(18) << "Work/Thread"    << " : " << profile.wptString << endl; 
    cout << left << setw(18) << "Total Ops"      << " : " << (ops / 1.0e9) << " Billion" << endl;
    cout << left << setw(18) << "CPU -> GPU"     << " : " << fixed << setprecision(2) << t_in << " ms" << endl;
    cout << left << setw(18) << "Kernel Exec"    << " : " << fixed << setprecision(2) << t_exec << " ms" << endl;
    cout << left << setw(18) << "GPU -> CPU"     << " : " << fixed << setprecision(2) << t_out << " ms" << endl;
    cout << left << setw(18) << "Total Latency"  << " : " << fixed << setprecision(2) << t_total << " ms" << endl;
    cout << left << setw(18) << "Validation"     << " : " << (passed ? "PASS" : "FAIL") << endl;
    
    cout << string(80, '-') << endl;
    cout << left << setw(18) << "PERFORMANCE"    << " : " << setprecision(2) << gflops << " GFLOPS" << endl;
    cout << string(80, '=') << "\n" << endl;

    // 6. DESTRUCTIVE CLEANUP
    // Force the queue to empty. When this function returns, `context` is destroyed,
    // explicitly forcing the GPU driver to hard-flush VRAM.
    queue.flush();
    queue.finish(); 
}


int main() {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) { cout << "No OpenCL platforms found." << endl; return 1; }
    
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) { cout << "No GPUs found." << endl; return 1; }
    
    cl::Device device = devices[0];

    int N = 1024, M = 1024, K = 1024;
    size_t matrixSize = (size_t)N * N;
    vector<float> h_A(matrixSize, 1.1f), h_B(matrixSize, 2.2f), h_C(matrixSize, 0.0f);

    vector<KernelProfile> suite = {
        {"NAIVE MATMUL (GLOBAL WRITE)",       "naive1.cl",      "NaiveMatMul",              1, 1, 16, 16, "1x1 (Base)"},
        {"NAIVE MATMUL (REGISTER ACCUM)",     "naive2.cl",      "NaiveMatMul",              1, 1, 16, 16, "1x1 (Register Accum)"},
        {"GLOBAL COALESCING MATMUL",          "coalesced.cl",   "GlobalCoalescing_MatMul",  1, 1, 16, 16, "1x1 (Coalesced)"},
        {"SHARED MEMORY MATMUL (TILES A&B)",  "shared1.cl",     "SharedMem_MatMul",         1, 1, 16, 16, "1x1 (SRAM)"},
        {"SHARED MEMORY MATMUL (TILE B ONLY)","shared2.cl",     "SharedMem_MatMul",         1, 1, 16, 16, "1x1 (SRAM B)"},
        {"1D BLOCK TILE MEMORY MATMUL",       "block1d.cl",     "BlockTile1D_MatMul",       4, 1, 16, 16, "1x4 (1D Block)"},
        {"2D BLOCK TILE SCALAR MATMUL",       "block2d.cl",     "BlockTile2D_MatMul",       4, 4, 16, 16, "4x4 (2D Block)"},
        {"2D BLOCK TILE VECTOR MATMUL",       "block2d_vec.cl", "BlockTile2D_Vload_MatMul", 4, 4, 16, 16, "4x4 (Vec4)"},
        {"WARP-TILED SCALAR MATMUL",          "warp_scalar.cl", "WarpTile_MatMul",          8, 8, 16,  8, "8x8 (Reg Scalar)"},
        {"ULTIMATE VEC-WARP MATMUL",          "warp_vec.cl",    "WarpTile_Vec_MatMul",      8, 8, 16, 16, "8x8 (Reg Vec4)"}
    };

    cout << "Starting OpenCL Benchmark Suite..." << endl;
    for (const auto& profile : suite) {
        
        // Context is spawned uniquely for each run here.
        runBenchmark(profile, device, h_A, h_B, h_C, M, N, K, matrixSize);
        
        // HARDWARE COOLDOWN: Bumped to 2.5 seconds to clear thermal load
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    }

    return 0;
}