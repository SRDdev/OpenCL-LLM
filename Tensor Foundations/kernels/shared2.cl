#define TS 16 // Tile Size (16x16 = 256 threads)
__kernel void SharedMem_MatMul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    // Global position in C
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    // Local position within the work-group
    const int local_col = get_local_id(0);
    const int local_row = get_local_id(1);

    // Only Tile B is stored in SRAM
    __local float TileB[TS][TS];

    float sum = 0.0f;

    // Loop over tiles of K
    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) {

        // 1. Cooperative Load: Only load Matrix B into Local Memory
        // Every thread loads one element of B
        int b_row = t * TS + local_row;
        int b_col = col; // Same as global col
        
        if (b_row < K && b_col < N) {
            TileB[local_row][local_col] = B[b_row * N + b_col];
        } else {
            TileB[local_row][local_col] = 0.0f;
        }

        // 2. Sync: Wait for Tile B to be fully loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Compute: Fetch A from Global Mem, B from Local Mem
        // All threads in a work-group row will access the same A element (Broadcast)
        for (int k = 0; k < TS; k++) {
            float valA = A[row * K + (t * TS + k)];
            float valB = TileB[k][local_col];
            sum += valA * valB;
        }

        // 4. Sync: Ensure calculation is done before loading the next tile of B
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}