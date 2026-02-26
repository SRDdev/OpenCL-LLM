#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath> 
#include "C:/Users/ASUS/Desktop/Shreyas/OpenCL-LLM/utils/utils.hpp"

using namespace std;

int main() {
    // --- Device Setup ---
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // --- Device Information ---
    string devName = device.getInfo<CL_DEVICE_NAME>();
    string devVendor = device.getInfo<CL_DEVICE_VENDOR>();
    string devVersion = device.getInfo<CL_DEVICE_VERSION>();
    cl_uint maxComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    cl_uint maxClockFreq = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    size_t maxWGSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    vector<size_t> maxWISizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    cl_ulong globalMem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    cl_ulong localMem = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    cout << "Details\n-----------------------------------------------------------" << endl;
    cout << "Device Name        : " << devName << endl;
    cout << "Vendor             : " << devVendor << endl;
    cout << "OpenCL Version     : " << devVersion << endl;
    cout << "Max Compute Units  : " << maxComputeUnits << endl;
    cout << "Max Clock Freq     : " << maxClockFreq << " MHz" << endl;
    cout << "Max Work-Group     : " << maxWGSize << " threads per group" << endl;
    cout << "Max Work-Item Dims : (" << maxWISizes[0] << ", " << maxWISizes[1] << ", " << maxWISizes[2] << ")" << endl;
    cout << "Global Memory      : " << globalMem / (1024 * 1024) << " MB" << endl;
    cout << "Local Memory (SRAM): " << localMem / 1024 << " KB per Compute Unit" << endl;
    cout << "-----------------------------------------------------------\n" << endl;

    // --- Build Section ---
    string naiveSrc = readKernelFile("Vector_Foundations\\kernels\\naive_matmul.cl") + "\n";
    string sramSrc = readKernelFile("Vector_Foundations\\kernels\\matmul.cl") + "\n";
    string regSrc = readKernelFile("Vector_Foundations\\kernels\\register_matmul.cl") + "\n";
    
    cl::Program::Sources sources;
    sources.push_back({naiveSrc.c_str(), naiveSrc.length()});
    sources.push_back({sramSrc.c_str(), sramSrc.length()});
    sources.push_back({regSrc.c_str(), regSrc.length()});

    cl::Program program(context, sources);
    if (program.build("-cl-std=CL2.0") != CL_SUCCESS) {
        cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return 1;
    }

    cl::Kernel kernelNaive(program, "naive_matmul");
    cl::Kernel kernelSRAM(program, "matmul");       
    cl::Kernel kernelReg(program, "register_matmul");    

    // --- Table Header ---
    cout << "Table\n-----------------------------------------------------------" << endl;
    cout << left << setw(6)  << "Size" 
         << setw(16) << "Kernel"
         << setw(14) << "Kernel(ms)" 
         << setw(14) << "HRAM In(ms)"
         << setw(14) << "HRAM Out(ms)" 
         << setw(10) << "Result" << endl;
    cout << string(85, '-') << endl;

    vector<int> testSizes = {128, 256, 512, 1024, 2048, 4096};

    for (int N : testSizes) {
        int Arow = N, Bcol = N, Brow = N;
        vector<float> A(N * N, 1.0f), B(N * N, 2.0f);
        
        vector<float> C_naive(N * N, 0.0f);
        vector<float> C_sram(N * N, 0.0f);
        vector<float> C_reg(N * N, 0.0f);

        cl::Buffer bufA(context, CL_MEM_READ_ONLY, sizeof(float)*A.size());
        cl::Buffer bufB(context, CL_MEM_READ_ONLY, sizeof(float)*B.size());
        cl::Buffer bufC_naive(context, CL_MEM_WRITE_ONLY, sizeof(float)*C_naive.size());
        cl::Buffer bufC_sram(context, CL_MEM_WRITE_ONLY, sizeof(float)*C_sram.size());
        cl::Buffer bufC_reg(context, CL_MEM_WRITE_ONLY, sizeof(float)*C_reg.size());

        cl::Event evInA, evInB;
        cl::Event evKNaive, evOutNaive;
        cl::Event evKSRAM, evOutSRAM;
        cl::Event evKReg, evOutReg;

        queue.enqueueWriteBuffer(bufA, CL_FALSE, 0, sizeof(float)*A.size(), A.data(), nullptr, &evInA);
        queue.enqueueWriteBuffer(bufB, CL_FALSE, 0, sizeof(float)*B.size(), B.data(), nullptr, &evInB);

        // --- 1. Run Naive ---
        kernelNaive.setArg(0, bufA); kernelNaive.setArg(1, bufB); kernelNaive.setArg(2, bufC_naive);
        kernelNaive.setArg(3, Arow); kernelNaive.setArg(4, Bcol); kernelNaive.setArg(5, Brow);
        queue.enqueueNDRangeKernel(kernelNaive, cl::NullRange, cl::NDRange(Bcol, Arow), cl::NullRange, nullptr, &evKNaive);
        queue.enqueueReadBuffer(bufC_naive, CL_TRUE, 0, sizeof(float)*C_naive.size(), C_naive.data(), nullptr, &evOutNaive);

        // --- 2. Run SRAM Tiled ---
        kernelSRAM.setArg(0, bufA); kernelSRAM.setArg(1, bufB); kernelSRAM.setArg(2, bufC_sram);
        kernelSRAM.setArg(3, Arow); kernelSRAM.setArg(4, Bcol); kernelSRAM.setArg(5, Brow);
        cl::NDRange localWorkSizeSRAM(16, 16); 
        queue.enqueueNDRangeKernel(kernelSRAM, cl::NullRange, cl::NDRange(Bcol, Arow), localWorkSizeSRAM, nullptr, &evKSRAM);
        queue.enqueueReadBuffer(bufC_sram, CL_TRUE, 0, sizeof(float)*C_sram.size(), C_sram.data(), nullptr, &evOutSRAM);

        // --- 3. Run Register Tiled (WPT = 4) ---
        // FIXED: Explicitly set the arguments for the register kernel
        kernelReg.setArg(0, bufA); kernelReg.setArg(1, bufB); kernelReg.setArg(2, bufC_reg);
        kernelReg.setArg(3, Arow); kernelReg.setArg(4, Bcol); kernelReg.setArg(5, Brow);
        
        cl::NDRange localWorkSizeReg(16, 16);
        int WPT = 4;
        cl::NDRange globalWorkSizeReg(Bcol / WPT, Arow); // Shrink the grid horizontally
        
        queue.enqueueNDRangeKernel(kernelReg, cl::NullRange, globalWorkSizeReg, localWorkSizeReg, nullptr, &evKReg);
        queue.enqueueReadBuffer(bufC_reg, CL_TRUE, 0, sizeof(float)*C_reg.size(), C_reg.data(), nullptr, &evOutReg);

        // Helper to extract timings safely
        auto get_ms = [](cl::Event& e) {
            e.wait(); 
            cl_ulong start, end;
            e.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            e.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            return (end - start) * 1.0e-6;
        };

        // Calculate transfer time
        double tIn = get_ms(evInA) + get_ms(evInB);
        
        // Calculate kernel times (FIXED: actually timing evKReg now)
        double tKNaive = get_ms(evKNaive);
        double tKSRAM = get_ms(evKSRAM);
        double tKReg = get_ms(evKReg);

        // Calculate output times
        double tOutNaive = get_ms(evOutNaive);
        double tOutSRAM = get_ms(evOutSRAM);
        double tOutReg = get_ms(evOutReg);

        // Verification logic
        bool matchSRAM = true, matchReg = true;
        for(size_t i=0; i<C_naive.size(); ++i) {
            if(std::abs(C_naive[i] - C_sram[i]) > 1e-3) { matchSRAM = false; }
            if(std::abs(C_naive[i] - C_reg[i]) > 1e-3) { matchReg = false; }
        }

        // --- Clean Output ---
        cout << left << setw(6)  << N 
             << setw(16) << "Naive"
             << setw(14) << fixed << setprecision(3) << tKNaive 
             << setw(14) << tIn 
             << setw(14) << tOutNaive 
             << setw(10) << "PASS" << endl;
             
        cout << left << setw(6)  << "" 
             << setw(16) << "SRAM Tiled"
             << setw(14) << fixed << setprecision(3) << tKSRAM 
             << setw(14) << "-" 
             << setw(14) << tOutSRAM 
             << setw(10) << (matchSRAM ? "Speedup: " + to_string(tKNaive/tKSRAM).substr(0,4) + "x" : "FAIL") << endl;

        cout << left << setw(6)  << "" 
             << setw(16) << "Register Tiled"
             << setw(14) << fixed << setprecision(3) << tKReg 
             << setw(14) << "-" 
             << setw(14) << tOutReg 
             << setw(10) << (matchReg ? "Speedup: " + to_string(tKNaive/tKReg).substr(0,4) + "x" : "FAIL") << endl;
             
        cout << string(85, '-') << endl;
    }

    return 0;
}