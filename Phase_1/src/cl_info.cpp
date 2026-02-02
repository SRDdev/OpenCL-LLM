// This file contains the code to get started with OpenCL and getting to know your device.
#define CL_HPP_TARGET_OPENCL_VERSION 200 
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
using namespace std;

int main(){
    // Setup platform
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()){
        cout << "No Platform Found !!"; 
        return 1;
    } 

    // Setup device
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU,&devices);
    if(devices.empty()){
        cout << "No Device Found !!"; 
        return 1;
    } 

    cl::Device device = devices[0];
    std::cout << "===============================================" << std::endl;
    std::cout << "DEVICE LIMITS: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "===============================================" << std::endl;
    size_t maxWorkerGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    cl_uint maxDimensions = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    vector<size_t> maxItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    cl_uint computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    
    // PRINTING THE LIMITS
    std::cout << "1. Max Threads per Group (Local Size): " << maxWorkerGroupSize << std::endl;
    std::cout << "2. Max Dimensions supported          : " << maxDimensions << std::endl;
    std::cout << "3. Max Threads per Dimension (X/Y/Z) : ";
    for(int i=0; i<maxDimensions; i++) std::cout << maxItemSizes[i] << (i < maxDimensions-1 ? " / " : "");
    std::cout << std::endl;
    std::cout << "4. Number of Compute Units (SMs)     : " << computeUnits << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
}