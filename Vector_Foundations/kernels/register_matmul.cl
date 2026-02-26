// Work Per Thread (Register Tile) Matrix Multiplication.
#define TileSize 16
#define WPT 4
__kernel void register_matmul(__global const float *A, __global const float *B,
                              __global float *C, int M, int N, int K) {
  // 1. Identifiers
  int localrow = get_local_id(1);
  int localcol = get_local_id(0);

  // The global grid is smaller now! Each group handles WPT times more columns.
  int row = get_group_id(1) * TileSize + localrow;
  int col = get_group_id(0) * TileSize * WPT + localcol;

  // 2. Local Memory (SRAM)
  __local float TileA[TileSize][TileSize];
  __local float TileB[TileSize][TileSize * WPT];

  // 3. Private Memory (Registers)
  // We store 4 separate running totals in ultra-fast registers
  float acc[WPT];
  for (int w = 0; w < WPT; w++) {
    acc[w] = 0.0f;
  }

  int numTiles = (K + TileSize - 1) / TileSize;
  for (int t = 0; t < numTiles; t++) {
    if (row < M && (t * TileSize + localcol) < K) {
      TileA[localrow][localcol] = A[row * K + (t * TileSize + localcol)];
    } else
      TileA[localrow][localcol] = 0.0f;

    for (int w = 0; w < WPT; w++) {
      int globalColB = col + w * TileSize;
      if (globalColB < N && (t * TileSize + localrow) < K) {
        TileB[localrow][localcol + w * TileSize] =
            B[(t * TileSize + localrow) * N + globalColB];
      } else {
        TileB[localrow][localcol + w * TileSize] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k_inner = 0; k_inner < TileSize; ++k_inner) {
      // Load ONE value from Local Memory into a Register
      float valA = TileA[localrow][k_inner];

      // Multiply that ONE value against 4 different B values! (Register Reuse)
      for (int w = 0; w < WPT; w++) {
        acc[w] += valA * TileB[k_inner][localcol + w * TileSize];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // --- Write the 4 resulTileSize back to Global Memory ---
  if (row < M) {
    for (int w = 0; w < WPT; w++) {
      int globalColOut = col + w * TileSize;
      if (globalColOut < N) {
        C[row * N + globalColOut] = acc[w];
      }
    }
  }
}