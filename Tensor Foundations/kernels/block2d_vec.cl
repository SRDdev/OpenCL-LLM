#define TS 16   // Tile Size for K
#define WPT 4    // Each thread handles 4x4 block
__kernel void BlockTile2D_Vload_MatMul(__global const float* A, __global const float* B, __global float* C,  int M, int N, int K) {
    const int tx = get_local_id(0); // 0-15
    const int ty = get_local_id(1); // 0-15
    
    // Each group handles 64x64 block of C
    const int group_row = get_group_id(1) * 64;
    const int group_col = get_group_id(0) * 64;

    // Registers for 4x4 sub-block
    float4 sum[WPT];
    for(int i=0; i<WPT; i++) sum[i] = (float4)(0.0f);

    // SRAM Tiles: TileA(64x16), TileB(16x64)
    __local float TileA[64][16];
    __local float TileB[16][64];

    for (int t = 0; t < K; t += TS) {
        // 1. Cooperative Load
        // Each thread loads 4 elements of A (row-wise) and 4 of B (row-wise)
        for (int i = 0; i < WPT; i++) {
            // Load A: Each thread handles a row, and we distribute across 4 rows
            int a_row = group_row + ty + i * TS;
            int a_col = t + tx;
            TileA[ty + i * TS][tx] = A[a_row * K + a_col];

            // Load B: Use vload4 for coalesced global read
            int b_row = t + ty;
            int b_col_vec = (group_col + tx * 4) / 4;
            vstore4(vload4(b_col_vec, &B[b_row * N]), tx, &TileB[ty][0]);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2. Compute 4x4 block
        for (int k = 0; k < TS; k++) {
            // Broadcast 4 columns of B into a register
            float4 b_vec = vload4(tx, &TileB[k][0]);

            for (int r = 0; r < WPT; r++) {
                float a_val = TileA[ty * WPT + r][k];
                sum[r] += (float4)(a_val) * b_vec;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. Vectorized Write back
    for (int r = 0; r < WPT; r++) {
        int cur_r = group_row + ty * WPT + r;
        int cur_c_vec = (group_col + tx * 4) / 4;
        if (cur_r < M && (group_col + tx * 4) < N) {
            vstore4(sum[r], cur_c_vec, &C[cur_r * N]);
        }
    }
}