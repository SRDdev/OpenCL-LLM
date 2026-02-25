#define CL_HPP_TARGET_OPENCL_VERSION 200 
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

using namespace std;

// Utility for clean formatting
void printRow(string label, string value) {
    cout << left << setw(32) << label << ": " << value << endl;
}

int main() {
    // 1. Setup Platform
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        cout << "Error: No OpenCL Platform Found!" << endl;
        return 1;
    }

    // 2. Setup Device (Selecting the first available GPU)
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        cout << "Error: No GPU Device Found on this platform!" << endl;
        return 1;
    }

    cl::Device device = devices[0];

    // --- Header ---
    cout << "==========================================================" << endl;
    cout << "      PHASE I: HARDWARE HANDSHAKE & MEMORY MAPPING       " << endl;
    cout << "      Device: " << device.getInfo<CL_DEVICE_NAME>() << endl;
    cout << "==========================================================" << endl;

    // --- [1] EXECUTION RESOURCES (The "Workers") ---
    size_t maxWGSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    cl_uint computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    cl_uint maxDims = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    vector<size_t> maxItems = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    cout << "\n[1] EXECUTION RESOURCES" << endl;
    printRow("Compute Units (Cores/SMs)", to_string(computeUnits));
    printRow("Max Work-Group Size", to_string(maxWGSize) + " threads");
    string dims = to_string(maxItems[0]) + " x " + to_string(maxItems[1]) + " x " + to_string(maxItems[2]);
    printRow("Max Work-Item Sizes (X/Y/Z)", dims);

    // --- [2] MEMORY HIERARCHY (HRAM -> SRAM -> Registers) ---
    cl_ulong globalMem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    cl_ulong cacheSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
    cl_ulong localMem  = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    cl_ulong maxAlloc  = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    
    // Resilient check for Unified Memory (UMA)
    cl_bool unifiedMem = 0;
    try {
        device.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &unifiedMem);
    } catch (...) {
        // If the driver throws an error, we assume it's a modern UMA device 
        // where this specific flag is deprecated or handled via SVM.
    }

    cout << "\n[2] MEMORY HIERARCHY" << endl;
    printRow("HRAM (Global Memory)", to_string(globalMem / (1024 * 1024)) + " MB");
    printRow("Global Mem Cache (L2)", to_string(cacheSize / 1024) + " KB");
    printRow("SRAM (Local Memory/Tile)", to_string(localMem / 1024) + " KB");
    printRow("Max Single Buffer Size", to_string(maxAlloc / (1024 * 1024)) + " MB");
    printRow("Unified Memory (UMA)", (unifiedMem ? "YES (Shared Host RAM)" : "NO (Discrete VRAM)"));

    // --- [3] VECTORIZATION & DATA TYPES ---
    cl_uint prefVecFloat = device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
    cl_uint prefVecHalf  = device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>();
    string extensions    = device.getInfo<CL_DEVICE_EXTENSIONS>();
    bool hasFP16 = (extensions.find("cl_khr_fp16") != string::npos);

    cout << "\n[3] VECTORIZATION & PRECISION" << endl;
    printRow("Preferred Float Vector Width", to_string(prefVecFloat) + " (float" + to_string(prefVecFloat) + ")");
    printRow("Preferred Half Vector Width", to_string(prefVecHalf) + " (half" + to_string(prefVecHalf) + ")");
    printRow("Native FP16 Support", (hasFP16 ? "YES" : "NO"));

    cout << "\n==========================================================" << endl;
    cout << "OPTIMIZATION HINT:" << endl;
    if (localMem < 32768) {
        cout << ">> Small SRAM detected. Use aggressive tiling and small block sizes." << endl;
    } else {
        cout << ">> Large SRAM available. Ideal for high-performance fused kernels." << endl;
    }
    cout << "==========================================================" << endl;

    return 0;
}