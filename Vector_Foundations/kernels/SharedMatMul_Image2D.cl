//============================================================
// Shared Memory Image2D GEMM
//============================================================

#define Vec 4
#define MAX_TILE 32
#define DEBUG 1

__kernel void SharedMatMul_Image2D(__read_only image2d_t A,__read_only image2d_t B,__write_only image2d_t C,int M,int N,int K,int TileR,int TileC)
{
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    int row = get_global_id(0);
    int col = get_global_id(1);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    // Shared Memory (Only caching Row-Major B)
    __local float4 TileB[MAX_TILE * MAX_TILE];

    float4 acc = (float4)(0.0f);

#if DEBUG
    if (row == 0 && col == 0) {
        printf("\n================ Kernel Execution Start ================\n");
        printf("A Layout : Col-Major (Direct Global Memory Access)\n");
        printf("B Layout : Row-Major (Shared via Local Memory Tile)\n");
        printf("Thread Grid : %d x %d\n", M, N / Vec);
        printf("========================================================\n");
    }
#endif

    // Tile Loop
    for (int kt = 0; kt < K; kt += TileR) {

        #if DEBUG
                if (row == 0 && col == 0) {
                    printf("\n---------------------------------------------\n");
                    printf("Tile Pass Start | K Range : %d -> %d\n", kt, kt + TileR - 1);
                    printf("---------------------------------------------\n");
                }
        #endif

        // Load Tile From B (Row-Major) into Shared Local Memory
        int kLoad = kt + local_row;
        float4 b_val = (float4)(0.0f);

        if (kLoad < K && col < (N / Vec)) {
            b_val = read_imagef(B, smp, (int2)(col, kLoad));
        }

        TileB[local_row * TileC + local_col] = b_val;

        #if DEBUG
                if (row == 0 && col == 0 && local_row == 0 && local_col == 0) {
                    printf("\nStep 1 : Load B Tile into Local Mem\n");
                    printf("B Coord : (%d , %d) | Value : [%6.2f %6.2f %6.2f %6.2f]\n",
                        col, kLoad, b_val.x, b_val.y, b_val.z, b_val.w);
                }
        #endif

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute Phase
        int kMax = min(TileR, K - kt);

        #if DEBUG
                if (row == 0 && col == 0) {
                    printf("\nStep 2 : Compute Phase (Global A * Shared B)\n");
                }
        #endif

        for (int k = 0; k < kMax; k++) {
            // FIX: Because A is Col-Major and mapped to width=M, height=K
            // The X coordinate is the Row index, and the Y coordinate is the Col index (k).
            float a_val = read_imagef(A, smp, (int2)(row, kt + k)).x;
            
            // Read B from Shared Local Memory
            float4 b_tile = TileB[k * TileC + local_col];
            float4 prev = acc;

            acc += a_val * b_tile;

            #if DEBUG
                        if (row == 0 && col == 0) {
                            printf("\n >> Iteration k=%d\n", k);
                            printf("    Fetch A (Col-Major, Unshared) : A[%d,%d] = %6.2f\n", row, kt + k, a_val);
                            printf("    Fetch B (Row-Major, Shared)   : TileB[%d,%d] = [%6.2f %6.2f %6.2f %6.2f]\n", 
                                k, local_col, b_tile.x, b_tile.y, b_tile.z, b_tile.w);
                            printf("    New  ACC = [%6.2f %6.2f %6.2f %6.2f]\n",
                                acc.x, acc.y, acc.z, acc.w);
                        }
            #endif
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write Output
    if (row < M && col < (N / Vec)) {
        write_imagef(C, (int2)(col, row), acc);
    }
}