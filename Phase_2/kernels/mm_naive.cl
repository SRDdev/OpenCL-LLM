__kernel void mm_naive(__global const float* A, __global const float* B, __global float* C, const int N, const int M, const int K){
    int row = get_global_id(1);
    int col = get_global_id(0);
    if(row<M && col < N){
        float sum = 0.0f;
        for(int i = 0; i<K; i++){
            float a = A[row * K + i];
            float b = B[i * N + col];
            sum+= a*b;
        }   
        C[row * N + col] = sum;
    }
}