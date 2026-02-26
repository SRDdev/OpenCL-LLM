// This file contains the code for naive matrix multiplication.

__kernel void naive_matmul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K){
    int row = get_global_id(1);
    int col = get_global_id(0);
    float sum = 0.0f;
    if(row < M && col < N){
        for(int i=0; i<K; i++){
            float a = A[row * K + i];
            float b = B[i * N + col];
            sum+= a*b;
        }
    C[row * N + col] = sum;
    }
}