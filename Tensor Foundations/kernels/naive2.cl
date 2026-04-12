__kernel void NaiveMatMul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            // Memory access: Row of A, Column of B
            sum += A[row * K + i] * B[i * N + col];
        }
        // WRITE ONLY ONCE AFTER THE LOOP
        C[row * N + col] = sum;
    }
}