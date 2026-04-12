#define TS 16  // Tile Size
#define WPT 4  // Work Per Thread (Each thread makes 4 sandwiches)
__kernel void BlockTile1D_MatMul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    // Row is standard, but Col identifies the START of our 4-column block
    const int row = get_global_id(1);
    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);
    
    // Which block of 4 columns is this work-group handling?
    const int group_col_start = get_group_id(0) * TS * WPT;

    // Local memory for Matrix B (Needs to be wider to hold 4 tiles worth of columns)
    __local float TileB[TS][TS * WPT];

    // Registers (Private memory) to hold our 4 results
    float sum[WPT];
    for (int w = 0; w < WPT; w++) {
        sum[w] = 0.0f;
    }

    // Loop through the K-dimension in tiles
    for (int t = 0; t < (K / TS); t++) {

        // 1. COOPERATIVE LOAD: Pull 4 elements of B into SRAM
        // Each thread loads 4 values to fill the wider TileB
        for (int w = 0; w < WPT; w++) {
            int tiled_col = local_col + (w * TS);
            int global_col = group_col_start + tiled_col;
            
            TileB[local_row][tiled_col] = B[(t * TS + local_row) * N + global_col];
        }

        // Wait for everyone to finish loading the table
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2. COMPUTE: One Bread (A), Four Jams (B)
        for (int k = 0; k < TS; k++) {
            // Fetch A once from Global Memory (broadcast to the row)
            float valA = A[row * K + (t * TS + k)];

            // Use that same valA for all 4 partial sums
            for (int w = 0; w < WPT; w++) {
                sum[w] += valA * TileB[k][local_col + (w * TS)];
            }
        }

        // Wait before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. STORE: Write the 4 results back to Matrix C
    for (int w = 0; w < WPT; w++) {
        int global_col = group_col_start + local_col + (w * TS);
        if (row < M && global_col < N) {
            C[row * N + global_col] = sum[w];
        }
    }
}