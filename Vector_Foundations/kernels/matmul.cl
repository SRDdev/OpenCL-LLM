#define TileSize 16

__kernel void matmul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    // 1. Thread Identifiers
    int row = get_global_id(1);
    int col = get_global_id(0);
    int localrow = get_local_id(1);
    int localcol = get_local_id(0);

    // 2. Allocate ultra-fast SRAM (Local Memory) for our tiles
    __local float TileA[TileSize][TileSize];
    __local float TileB[TileSize][TileSize];
    
    float acc = 0.0f;
    int numTiles = (K + TileSize - 1) / TileSize;

    for (int i = 0; i < numTiles; i++) {
        // Load TileA
        if (row < M && (i * TileSize + localcol) < K) {
            TileA[localrow][localcol] = A[row * K + (i * TileSize + localcol)];
        } else {
            TileA[localrow][localcol] = 0.0f;
        }

        // Load TileB
        if (col < N && (i * TileSize + localrow) < K) {
            TileB[localrow][localcol] = B[(i * TileSize + localrow) * N + col];
        } else {
            TileB[localrow][localcol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_inner = 0; k_inner < TileSize; ++k_inner) {
            acc += TileA[localrow][k_inner] * TileB[k_inner][localcol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
