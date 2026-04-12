#define TS 16       // Tile size for K-dimension
#define WPT 8       // Work Per Thread (Each thread handles 8x8 block)
#define RTS 16      // Row Tile Size (Threads in X)
#define CTS 8       // Column Tile Size (Threads in Y)

__kernel void WarpTile_MatMul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    const int tx = get_local_id(0); // 0..15
    const int ty = get_local_id(1); // 0..7
    
    // Y-dimension block size is 64 (8 threads * 8 elements)
    // X-dimension block size is 128 (16 threads * 8 elements)
    const int group_row = get_group_id(1) * 64;
    const int group_col = get_group_id(0) * 128;

    // REGISTER CACHING
    float sum[WPT][WPT];
    for(int r=0; r<WPT; r++) {
        for(int c=0; c<WPT; c++) {
            sum[r][c] = 0.0f;
        }
    }

    // LOCAL MEMORY (Sized specifically for our 64x128 output block)
    __local float TileA[64][TS];  // 64 rows x 16 cols
    __local float TileB[TS][128]; // 16 rows x 128 cols

    // Flat thread ID (0 to 127) for cooperative memory loading
    const int tid = ty * RTS + tx; 

    for (int t = 0; t < K; t += TS) {

        // COOPERATIVE LOAD
        // Tile A: 64x16 = 1024 elements. 128 threads load 8 elements each.
        for(int i = 0; i < 8; i++) {
            int idx = i * 128 + tid;
            int rowA = idx / TS;
            int colA = idx % TS;
            int global_row = group_row + rowA;
            int global_col = t + colA;
            TileA[rowA][colA] = (global_row < M && global_col < K) ? 
                                A[global_row * K + global_col] : 0.0f;
        }

        // Tile B: 16x128 = 2048 elements. 128 threads load 16 elements each.
        for(int i = 0; i < 16; i++) {
            int idx = i * 128 + tid;
            int rowB = idx / 128;
            int colB = idx % 128;
            int global_row = t + rowB;
            int global_col = group_col + colB;
            TileB[rowB][colB] = (global_row < K && global_col < N) ? 
                                B[global_row * N + global_col] : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // COMPUTE: Warp-level Tiling
        for (int k = 0; k < TS; k++) {
            float regB[WPT];
            for (int c = 0; c < WPT; c++) {
                regB[c] = TileB[k][tx + (c * RTS)];
            }

            for (int r = 0; r < WPT; r++) {
                float regA = TileA[ty + (r * CTS)][k];
                for (int c = 0; c < WPT; c++) {
                    sum[r][c] += regA * regB[c];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // STORE
    for (int r = 0; r < WPT; r++) {
        for (int c = 0; c < WPT; c++) {
            int cur_r = group_row + ty + (r * CTS);
            int cur_c = group_col + tx + (c * RTS);
            if (cur_r < M && cur_c < N) {
                C[cur_r * N + cur_c] = sum[r][c];
            }
        }
    }
}
