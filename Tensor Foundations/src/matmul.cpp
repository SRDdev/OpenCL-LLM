#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <thread> 
#include <chrono> 
#include <array>

using namespace std;

// --- FP16 Conversion Helpers ---
cl_half floatToHalf(float f) {
    uint32_t x = *((uint32_t*)&f);
    uint32_t s = (x >> 16) & 0x8000;
    uint32_t m = (x >> 13) & 0x03ff;
    uint32_t e = ((x >> 23) & 0xff) - (127 - 15);
    if (e > 30) return s | 0x7c00; 
    if (e <= 0) return s;         
    return s | (e << 10) | m;
}

float halfToFloat(cl_half h) {
    uint32_t s = (h >> 15) & 0x0001;
    uint32_t e = (h >> 10) & 0x001f;
    uint32_t m = h & 0x03ff;
    if (e == 0) return (s ? -1.0f : 1.0f) * ldexp((float)m, -24);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    return (s ? -1.0f : 1.0f) * ldexp((float)(m | 0x0400), (int)e - 15 - 10);
}

struct KernelProfile {
    string displayName;
    string fileName;
    string functionName;
    int wpt_x; int wpt_y;         
    int local_x; int local_y;       
};

// --- Hardware Sensor Mock (Replace with NVML for NVIDIA) ---
float getGpuTemperature() {
    // Logic: In a real scenario, use nvmlDeviceGetTemperature
    return 42.0f; 
}

bool verify_results(const vector<cl_half>& A, const vector<cl_half>& B, const vector<float>& GPU_C, int M, int N, int K) {
    float epsilon = 0.5f; 
    for (int i = 0; i < 64; i++) { 
        float cpu_sum = 0.0f;
        int row = i / N;
        int col = i % N;
        for (int k = 0; k < K; k++) {
            cpu_sum += halfToFloat(A[row * K + k]) * halfToFloat(B[k * N + col]);
        }
        if (std::abs(GPU_C[i] - cpu_sum) > epsilon) {
            return false;
        }
    }
    return true; 
}

void runBenchmark(const KernelProfile& profile, cl::Device& device,
                  vector<cl_half>& h_A_half, vector<cl_half>& h_B_half, 
                  vector<float>& h_C, int M, int N, int K, size_t matrixSize) {
    
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // --- Kernel Loading & Build ---
    string filePath = "Tensor Foundations/kernels/" + profile.fileName;
    string src;
    FILE *f = fopen(filePath.c_str(), "rb");
    if (!f) { cout << "[!] Kernel file not found." << endl; return; }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    vector<char> buffer(size + 1);
    fread(buffer.data(), 1, size, f);
    buffer[size] = '\0';
    src = buffer.data();
    fclose(f);

    cl::Program program(context, src);
    if (program.build("-cl-std=CL2.0") != CL_SUCCESS) {
        cout << "[!] Build Error:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return;
    }
    
    cl::Kernel kernel(program, profile.functionName.c_str());

    // --- Memory Prep ---
    cl::Buffer pinned_A(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_half) * matrixSize);
    cl::Buffer pinned_B(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_half) * matrixSize);
    cl::Buffer d_C(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * matrixSize);

    cl_half* ptr_A = (cl_half*)queue.enqueueMapBuffer(pinned_A, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_half) * matrixSize);
    cl_half* ptr_B = (cl_half*)queue.enqueueMapBuffer(pinned_B, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_half) * matrixSize);
    std::copy(h_A_half.begin(), h_A_half.end(), ptr_A);
    std::copy(h_B_half.begin(), h_B_half.end(), ptr_B);

    cl::ImageFormat format(CL_RGBA, CL_HALF_FLOAT);
    cl::Image2D d_A(context, CL_MEM_READ_ONLY, format, K / 4, M);
    cl::Image2D d_B(context, CL_MEM_READ_ONLY, format, N / 4, K);
    cl::Event evInA, evInB, evExec, evOut;

    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> regA = {(size_t)(K / 4), (size_t)M, 1};
    std::array<size_t, 3> regB = {(size_t)(N / 4), (size_t)K, 1};

    // --- Execution ---
    queue.enqueueWriteImage(d_A, CL_FALSE, origin, regA, 0, 0, ptr_A, nullptr, &evInA);
    queue.enqueueWriteImage(d_B, CL_FALSE, origin, regB, 0, 0, ptr_B, nullptr, &evInB);

    kernel.setArg(0, d_A); kernel.setArg(1, d_B); kernel.setArg(2, d_C);
    kernel.setArg(3, M); kernel.setArg(4, N); kernel.setArg(5, K);
    
    cl::NDRange globalSize(N / profile.wpt_x, M / profile.wpt_y); 
    cl::NDRange localSize(profile.local_x, profile.local_y); 
    
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &evExec);
    queue.enqueueReadBuffer(d_C, CL_TRUE, 0, sizeof(float) * matrixSize, h_C.data(), nullptr, &evOut);

    // --- Profiling Extraction ---
    auto get_ms = [](cl::Event& e) {
        e.wait();
        cl_ulong start, end;
        e.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        e.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        return (double)(end - start) * 1.0e-6;
    };

    double t_in = get_ms(evInA) + get_ms(evInB);
    double t_exec = get_ms(evExec);
    double t_out = get_ms(evOut);
    
    // --- Advanced Math Metrics ---
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (ops / (t_exec * 1e-3)) / 1e9;
    
    // Memory Traffic: A(half) + B(half) read + C(float) written
    double bytesMoved = (double)matrixSize * (sizeof(cl_half) * 2 + sizeof(float));
    double bandwidth = (bytesMoved / 1e9) / (t_exec * 1e-3); // GB/s
    double intensity = ops / bytesMoved; // FLOPs per Byte
    
    bool passed = verify_results(h_A_half, h_B_half, h_C, M, N, K);

    // --- PRETTY PRINT TERMINAL OUTPUT ---
    cout << "\033[1;36m" << string(60, '=') << "\033[0m" << endl;
    cout << "  \033[1;33mPROFILING:\033[0m " << profile.displayName << endl;
    cout << "\033[1;36m" << string(60, '=') << "\033[0m" << endl;
    
    cout << left << setw(25) << "  [TIMING] HtoD Transfer"  << ": " << fixed << setprecision(3) << t_in << " ms" << endl;
    cout << left << setw(25) << "  [TIMING] Kernel Exec"    << ": " << fixed << setprecision(3) << t_exec << " ms" << endl;
    cout << left << setw(25) << "  [TIMING] DtoH Transfer"  << ": " << fixed << setprecision(3) << t_out << " ms" << endl;
    
    cout << "\033[1;34m" << string(60, '-') << "\033[0m" << endl;
    
    cout << left << setw(25) << "  [COMPUTE] Performance"  << ": \033[1;32m" << gflops << " GFLOPS\033[0m" << endl;
    cout << left << setw(25) << "  [MEMORY] Bandwidth"     << ": " << bandwidth << " GB/s" << endl;
    cout << left << setw(25) << "  [MEMORY] Intensity"     << ": " << intensity << " FLOP/Byte" << endl;
    cout << left << setw(25) << "  [MEMORY] Total Traffic" << ": " << bytesMoved / 1e6 << " MB" << endl;
    
    cout << "\033[1;34m" << string(60, '-') << "\033[0m" << endl;
    
    cout << left << setw(25) << "  [SENSOR] GPU Temp"      << ": " << getGpuTemperature() << "°C" << endl;
    cout << left << setw(25) << "  [VERIFY] Validation"    << ": " << (passed ? "\033[1;32mPASS\033[0m" : "\033[1;31mFAIL\033[0m") << endl;
    
    double peakGflops = 13000.0; // Example for your specific GPU
    double utilization = (gflops / peakGflops) * 100.0;
    cout << left << setw(25) << "  [HW] Compute Util" << ": " << utilization << "%" << endl;
    
    cout << "\033[1;36m" << string(60, '=') << "\033[0m\n" << endl;

    queue.enqueueUnmapMemObject(pinned_A, ptr_A);
    queue.enqueueUnmapMemObject(pinned_B, ptr_B);
    queue.finish(); 
}

int main() {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) return 1;
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    int N = 2048, M = 2048, K = 2048; // Increased size for better measurement
    size_t matrixSize = (size_t)M * N;
    
    vector<cl_half> h_A_half(matrixSize), h_B_half(matrixSize);
    vector<float> h_C(matrixSize, 0.0f);
    
    for(size_t i = 0; i < matrixSize; ++i) {
        h_A_half[i] = floatToHalf(1.1f);
        h_B_half[i] = floatToHalf(2.2f);
    }

    vector<KernelProfile> suite = {
        {"Ultimate Ping-Pong FP16", "warp_vec.cl", "WarpTile_PingPong_MatMul", 8, 8, 16, 16}
    };

    cout << "\033[1;35mStarting Enhanced OpenCL Benchmark Suite...\033[0m" << endl;
    for (const auto& profile : suite) {
        runBenchmark(profile, device, h_A_half, h_B_half, h_C, M, N, K, matrixSize);
    }
    return 0;
}