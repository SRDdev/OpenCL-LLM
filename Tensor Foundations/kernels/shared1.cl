#define MAX_TILE 16
#define DEBUG 1
__kernel void SharedMem_MatMul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int col_local = get_local_id(0);
    int row_local = get_local_id(1);

    __local float TileA[MAX_TILE][MAX_TILE];
    __local float TileB[MAX_TILE][MAX_TILE];

    float sum = 0.0f;
    const int numTiles = (K + MAX_TILE - 1) / MAX_TILE;
    for (int t = 0; t < numTiles; t++) {
        
        // 1. Cooperative Load: Each thread loads one element of A and B into SRAM
        // Load Tile A
        if (row < M && (t * MAX_TILE + col_local) < K) {
            TileA[row_local][col_local] = A[row * K + (t * MAX_TILE + col_local)];
        } else {
            TileA[row_local][col_local] = 0.0f;
        }

        // Load Tile B
        if (col < N && (t * MAX_TILE + row_local) < K) {
            TileB[row_local][col_local] = B[(t * MAX_TILE + row_local) * N + col];
        } else {
            TileB[row_local][col_local] = 0.0f;
        }

        // 2. Synchronize: Wait for all threads in the work-group to finish loading
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Compute: Multiply the tiles currently in local memory
        for (int i = 0; i < MAX_TILE; i++) {
            sum += TileA[row_local][i] * TileB[i][col_local];
        }

        // 4. Synchronize: Wait for all threads to finish computing before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 5. Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}