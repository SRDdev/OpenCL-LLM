#define TS 16   // The dimension we split K by
#define WPT 4    // Each thread handles 4x4 results
__kernel void BlockTile2D_MatMul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    const int local_col = get_local_id(0); // 0 to 15
    const int local_row = get_local_id(1); // 0 to 15
    
    // Each work-group handles a 64x64 block of C
    const int group_row = get_group_id(1) * TS * WPT;
    const int group_col = get_group_id(0) * TS * WPT;

    // Registers for the 4x4 tray (16 elements total)
    float sum[WPT][WPT];
    for(int r=0; r<WPT; r++) 
        for(int c=0; c<WPT; c++) 
            sum[r][c] = 0.0f;

    // SRAM Tiles: Each thread will load 4 elements of A and 4 elements of B
    // to fill a 64x16 block of A and a 16x64 block of B
    __local float TileA[TS * WPT][TS]; // 64 x 16
    __local float TileB[TS][TS * WPT]; // 16 x 64

    for (int t = 0; t < (K / TS); t++) {
        
        // 1. Cooperative Load: Fill the larger SRAM Tiles
        for (int w = 0; w < WPT; w++) {
            // Load A: Row is spread across WPT, Column is the K-tile
            int a_row = group_row + local_row + (w * TS);
            int a_col = t * TS + local_col;
            TileA[local_row + (w * TS)][local_col] = A[a_row * K + a_col];

            // Load B: Row is the K-tile, Column is spread across WPT
            int b_row = t * TS + local_row;
            int b_col = group_col + local_col + (w * TS);
            TileB[local_row][local_col + (w * TS)] = B[b_row * N + b_col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2. Compute 4x4 Block
        for (int k = 0; k < TS; k++) {
            // Pre-load B values into registers for this k-step
            float b_vals[WPT];
            for (int c = 0; c < WPT; c++) {
                b_vals[c] = TileB[k][local_col + (c * TS)];
            }

            for (int r = 0; r < WPT; r++) {
                float a_val = TileA[local_row + (r * TS)][k];
                for (int c = 0; c < WPT; c++) {
                    sum[r][c] += a_val * b_vals[c];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. Write 4x4 results back to C
    for (int r = 0; r < WPT; r++) {
        for (int c = 0; c < WPT; c++) {
            int cur_r = group_row + local_row + (r * TS);
            int cur_c = group_col + local_col + (c * TS);
            if (cur_r < M && cur_c < N) {
                C[cur_r * N + cur_c] = sum[r][c];
            }
        }
    }
}