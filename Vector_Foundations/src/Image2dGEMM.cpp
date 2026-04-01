#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <array>
#include <iomanip>
#include <cstdlib>

using namespace std;

string loadKernel(string file) {
    ifstream f(file);
    if (!f.is_open()) {
        cerr << "\n[CRITICAL ERROR] Failed to open kernel file: " << file << endl;
        cerr << "Please check your file path and working directory." << endl;
        exit(EXIT_FAILURE); // Prevent silent failures!
    }
    return string(istreambuf_iterator<char>(f), istreambuf_iterator<char>());
}

// Function to visualize the linear memory layout interpreted as a matrix
void printMatrix(const string& name, const vector<float>& mat, int rows, int cols, bool isColMajor) {
    cout << "\n--- " << name << " (" << rows << "x" << cols << ") " << (isColMajor ? "[Col-Major]" : "[Row-Major]") << " ---" << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Row-Major index: row * width + col
            // Col-Major index: col * height + row
            int idx = isColMajor ? (j * rows + i) : (i * cols + j);
            cout << setw(5) << (int)mat[idx] << " ";
        }
        cout << endl;
    }
}

int main() {
    int M = 8, N = 8, K = 8;
    int TR = 4, TC = 2; 
    const int Vec = 4;

    // 1. Setup OpenCL
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device dev = devices[0];
    cl::Context ctx(dev);
    cl::CommandQueue q(ctx, dev, CL_QUEUE_PROFILING_ENABLE);

    // 2. Initialize Data
    vector<float> h_A(M * K);      
    vector<float> h_B(K * N);      
    vector<float> h_C(M * N, 0.0f);
    vector<float> h_Ref(M * N, 0.0f);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)i; 
    for (int i = 0; i < K * N; i++) h_B[i] = (float)i;

    // Visual Proof of Layout
    printMatrix("Matrix A", h_A, M, K, true);
    printMatrix("Matrix B", h_B, K, N, false);

    cout << "\n============================================================" << endl;
    cout << "                    GEMM Execution Info" << endl;
    cout << "============================================================" << endl;
    cout << "[Inputs]" << endl;
    cout << " Matrix A       : " << M << " x " << K << " | Size: " << M*K << " elements | Layout: Col-Major" << endl;
    cout << " Matrix B       : " << K << " x " << N << " | Size: " << K*N << " elements | Layout: Row-Major" << endl;
    cout << "\n[Workgroup & Thread Info]" << endl;
    cout << " Tile Size      : " << TR << " x " << TC << endl;
    cout << " Vector Size    : " << Vec << endl;
    cout << " Global Threads : " << M << " x " << N/Vec << " (Dim0 x Dim1)" << endl;
    cout << " Local Threads  : " << TR << " x " << TC << " (Dim0 x Dim1)" << endl;
    cout << "\n[Image2D Configurations]" << endl;
    cout << " Image A Dims   : Width = " << M << ", Height = " << K << " | Format: CL_R, Float" << endl;
    cout << " Image B Dims   : Width = " << N/Vec << ", Height = " << K << " | Format: CL_RGBA, Float4" << endl;
    cout << " Image C Dims   : Width = " << N/Vec << ", Height = " << M << " | Format: CL_RGBA, Float4" << endl;
    cout << "============================================================\n" << endl;

    // 3. Image2D Setup
    cl_int err;
    cl::ImageFormat fmtR(CL_R, CL_FLOAT);
    cl::ImageFormat fmtRGBA(CL_RGBA, CL_FLOAT);

    vector<float> h_Bvec(K * (N / Vec) * 4);
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N / Vec; n++) {
            for (int v = 0; v < 4; v++) {
                h_Bvec[(k * (N / Vec) + n) * 4 + v] = h_B[k * N + n * 4 + v];
            }
        }
    }

    cl::Image2D imgA(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fmtR, M, K, 0, h_A.data(), &err);
    cl::Image2D imgB(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fmtRGBA, N/Vec, K, 0, h_Bvec.data(), &err);
    cl::Image2D imgC(ctx, CL_MEM_WRITE_ONLY, fmtRGBA, N/Vec, M, 0, nullptr, &err);

    // 4. Kernel Execution
    string src = loadKernel("Vector_Foundations\\kernels\\SharedMatMul_Image2D.cl");
    cl::Program prog(ctx, src);
    
    // Add build error checking
    if (prog.build() != CL_SUCCESS) {
        cerr << "\n[CRITICAL ERROR] Kernel compilation failed:\n";
        cerr << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev) << endl;
        exit(EXIT_FAILURE);
    }

    cl::Kernel kernel(prog, "SharedMatMul_Image2D");

    kernel.setArg(0, imgA); kernel.setArg(1, imgB); kernel.setArg(2, imgC);
    kernel.setArg(3, M); kernel.setArg(4, N); kernel.setArg(5, K);
    kernel.setArg(6, TR); kernel.setArg(7, TC);

    q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M, N/Vec), cl::NDRange(TR, TC));
    q.finish(); // Ensure kernel completes before reading and printing
    
    // 5. Read and Unpack
    vector<float> h_Cvec(M * (N / Vec) * 4);
    q.enqueueReadImage(imgC, CL_TRUE, {0,0,0}, {(size_t)N/Vec, (size_t)M, 1}, 0, 0, h_Cvec.data());
    for (int i = 0; i < M; i++) 
        for (int j = 0; j < N; j++) 
            h_C[i * N + j] = h_Cvec[(i * (N/Vec) + j/Vec) * 4 + (j%Vec)];

    // 6. CPU Reference
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += h_A[k * M + i] * h_B[k * N + j];
            }
            h_Ref[i * N + j] = sum;
        }
    }

    printMatrix("GPU Result", h_C, M, N, false);
    printMatrix("CPU Ref", h_Ref, M, N, false);

    // Verification
    int correct_count = 0;
    int total_elements = M * N;
    float epsilon = 1e-4f;

    for (int i = 0; i < total_elements; i++) {
        if (fabs(h_C[i] - h_Ref[i]) < epsilon) {
            correct_count++;
        }
    }

    cout << "\n============================================================" << endl;
    cout << "                    Verification Results" << endl;
    cout << "============================================================" << endl;
    cout << " Elements Checked : " << total_elements << endl;
    cout << " Elements Correct : " << correct_count << " / " << total_elements << endl;
    
    if (correct_count == total_elements) {
        cout << " Final Status     : [ PASS ] All values match!" << endl;
    } else {
        cout << " Final Status     : [ FAIL ] Discrepancies found!" << endl;
    }
    cout << "============================================================\n" << endl;

    return 0;
}