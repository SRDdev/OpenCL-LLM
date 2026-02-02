#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
using namespace std;

int main(){
    // 1. Setup Platform and Devices
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU,&devices);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context,device);
    cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl;

    // 2. Build Kernel
    ifstream file("kernels/sum_reduce.cl");
    if (!file.is_open()){cerr << "Kernel file not found!" << endl; return 1; }
    string source(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));

    cl::Program program(context,source);
    if(program.build({device}) != CL_SUCCESS){
        cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl; 
        return 1;
    }

    // 3. Prepare Data
    int N =  1024;
    int localSize = 256;
    int numGroups = N / localSize; //  calculation : (1024/256) = 4 Groups
    cout << "Elements: " << N << " | Groups: " << numGroups << endl;
    
    // Input: Fill with 1.0f. The Sum should be 1024.0f.
    vector<float> inputData(N,1.0f);
    // Output: One partial sum per group
    vector<float> partialSums(numGroups);
    
    cl::Buffer bufIn(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N, inputData.data());
    cl::Buffer bufOut(context, CL_MEM_WRITE_ONLY, sizeof(float)*numGroups, nullptr);
    

    //Execute
    cl::Kernel kernel(program,"sum_reduce");
    kernel.setArg(0,bufIn);
    kernel.setArg(1,bufOut);

    // KEY STEP: Allocating "Local Memory"
    // We don't pass data here, we just tell OpenCL: "Reserve this much space in L1 cache"
    // We need 1 float per thread in the group.
    kernel.setArg(2,cl::Local(sizeof(float)*localSize));

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(localSize));
    queue.enqueueReadBuffer(bufOut, CL_TRUE, 0, sizeof(float)*numGroups, partialSums.data());

    float totalSum = 0;
    cout << "Partial Sums from GPU Groups: ";
    for(float x : partialSums) {
        cout << x << " ";
        totalSum += x;
    }
    cout << endl;

    cout << "Total Sum: " << totalSum << " (Expected: " << N << ")" << endl;

    return 0;
}