#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
using namespace std;

int main(){
    // 1. BOILERPLATE: Get Platform and Device
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) { std::cerr << "No platforms found!" << std::endl; return 1; }
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU,&devices);
    if(devices.empty()) { std::cerr << "No GPU found!" << std::endl; return 1; }
    cl::Device device = devices[0];
    std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // 2. THE CONTEXT
    cl::Context context(device);

    // 3. The Queue
    cl::CommandQueue queue(context, device);

    // 4. DATA PREPARATION (Host Side)
    const int N = 10;
    vector<int> hostInput = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    vector<int> hostOutput(N);

    // 5. MEMORY ALLOCATION (Device Side)
    // We allocate memory on the GPU VRAM.
    // CL_MEM_READ_WRITE: The kernel can both read and write to this spot.
    // CL_MEM_COPY_HOST_PTR: Optimization! Allocates AND copies data from 'hostInput' in one go.
    cl::Buffer deviceBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*N, hostInput.data());
    std::cout << "-> Buffer created and data copied to GPU..." << std::endl;

    // 6. READ BACK (Device -> Host)
    // Now we want to verify the data is there by reading it back into 'hostOutput'.
    // CL_TRUE = Blocking Read. The CPU will PAUSE here until the copy is finished.
    queue.enqueueReadBuffer(deviceBuffer, CL_TRUE, 0, sizeof(int)*N, hostOutput.data());
    std::cout << "-> Data read back from GPU." << std::endl;

    // Verify
    bool correct = true;
    std::cout << "Result: ";
    for(int i = 0; i < N; i++) {
        std::cout << hostOutput[i] << " ";
        if(hostOutput[i] != hostInput[i]) correct = false;
    }
    std::cout << std::endl;

    if(correct) std::cout << "SUCCESS! Host <-> Device memory link established." << std::endl;
    else std::cout << "FAILURE! Data mismatch." << std::endl;

    return 0;
}