#define TS 16       // Tile size for K-dimension
#define WPT_M 8     // 8 rows per thread
#define WPT_N 2     // 2 float4s (8 columns) per thread

__kernel void WarpTile_Vec_MatMul(__global const float* A, 
                                      __global const float* B, 
                                      __global float* C, 
                                      int M, int N, int K) {
    
    // 16x16 threads = 256 threads per group
    const int tx = get_local_id(0); // 0..15
    const int ty = get_local_id(1); // 0..15
    
    // Group handles a 128x128 block of the output matrix C
    const int group_row = get_group_id(1) * 128;
    const int group_col = get_group_id(0) * 128; 

    // REGISTER CACHING using vectors (8x2 float4s = 8x8 floats)
    float4 sum[WPT_M][WPT_N];
    for(int i = 0; i < WPT_M; i++) {
        for(int j = 0; j < WPT_N; j++) {
            sum[i][j] = (float4)(0.0f);
        }
    }

    // LOCAL MEMORY
    __local float TileA[128][16];
    __local float TileB[16][128];

    for (int t = 0; t < K; t += TS) {

        // 1. COOPERATIVE LOAD WITHOUT MODULO
        // Tile A: 128x16. 256 threads load 8 elements each.
        // tx (0-15) cleanly maps to the 16 columns. ty handles the rows.
        for (int i = 0; i < 8; i++) {
            int a_row = group_row + ty + (i * 16);
            int a_col = t + tx;
            TileA[ty + (i * 16)][tx] = A[a_row * K + a_col];
        }

        // Tile B: 16x128. 256 threads load 8 elements (2 float4s) each.
        // ty (0-15) cleanly maps to the 16 rows. tx handles the columns.
        for (int i = 0; i < 2; i++) {
            int b_row = t + ty;
            // Vectorized global offset
            int b_col_vec = (group_col / 4) + tx + (i * 16);
            // Vectorized local offset
            int local_col_vec = tx + (i * 16);
            
            vstore4(vload4(b_col_vec, &B[b_row * N]), local_col_vec, &TileB[ty][0]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 2. COMPUTE
        for (int k = 0; k < TS; k++) {
            
            // Pre-load a row of TileB into registers (vectorized)
            float4 b_vec[WPT_N];
            for (int j = 0; j < WPT_N; j++) {
                b_vec[j] = vload4(tx + (j * 16), &TileB[k][0]);
            }

            // Multiply against TileA (Broadcast from Shared Memory)
            for (int i = 0; i < WPT_M; i++) {
                float a_val = TileA[ty + (i * 16)][k];
                float4 a_vec = (float4)(a_val); // Broadcast scalar to vector
                
                for (int j = 0; j < WPT_N; j++) {
                    sum[i][j] += a_vec * b_vec[j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. STORE (Vectorized)
    for (int i = 0; i < WPT_M; i++) {
        for (int j = 0; j < WPT_N; j++) {
            int cur_r = group_row + ty + (i * 16);
            int cur_c_vec = (group_col / 4) + tx + (j * 16);
            
            if (cur_r < M) {
                vstore4(sum[i][j], cur_c_vec, &C[cur_r * N]);
            }
        }
    }
}