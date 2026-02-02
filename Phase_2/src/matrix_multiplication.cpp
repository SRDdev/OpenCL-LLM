#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;

int main(){
    // 1. Setup Platforms and Device
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU,&devices);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context,device);

    // 2. Kernel
    ifstream file("kernels/mm_naive.cl");
    if (!file.is_open()) { cerr << "Kernel file missing!" << endl; return 1; }
    string source(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
    cl::Program program(context, source);
    if(program.build({device}) != CL_SUCCESS){
        cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return 1;
    }

    // 3. Data
    // --- DIMENSIONS ---
    int M = 64;
    int N = 64;
    int K = 64;

    vector<float> A(M*K, 1.0f);
    vector<float> B(K*N, 2.0f);
    vector<float> C(M*N, 0.0f);

    // 4. Pipeline 
    cl::Buffer bufA(context, CL_MEM_COPY_HOST_PTR, sizeof(float)*A.size(), A.data());
    cl::Buffer bufB(context, CL_MEM_COPY_HOST_PTR, sizeof(float)*B.size(), B.data());
    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(float)*C.size(), nullptr);

    // 5. Execute
    cl::Kernel kernel(program,"mm_naive");
    kernel.setArg(0,bufA);
    kernel.setArg(1,bufB);
    kernel.setArg(2,bufC);
    kernel.setArg(3,M);
    kernel.setArg(4,N);
    kernel.setArg(5,K);
    
    cl::NDRange globalSize(N, M);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NullRange);
    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(float)*C.size(), C.data());

    cout << "C[0] = " << C[0] << " (Expected: " << 1.0f * 2.0f * K << ")" << endl;
    cout << "C[last] = " << C.back() << endl;
    return 0;
}