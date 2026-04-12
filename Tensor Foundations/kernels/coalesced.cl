__kernel void GlobalCoalescing_MatMul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    // Mapping ID(0) to Column for coalesced memory access
    const int col = get_global_id(0); 
    const int row = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            // A[row * K + i]: Row-based access (stays same for all 'col' in a warp)
            // B[i * N + col]: Column-based access (contiguous for adjacent 'col')
            sum += A[row * K + i] * B[i * N + col];
        }
        // Write once to global memory
        C[row * N + col] = sum;
    }
}