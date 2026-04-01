#define TileSize 16
#define VEC 4

__kernel void matmul_vectorized(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
    // 1. Local Memory for B (16 rows, 4 float4 columns = 16x16 floats)
    __local float4 TileB[TileSize][TileSize / VEC];
    
    // 2. IDs
    int col = get_global_id(0); 
    int row = get_global_id(1); 
    
    int localcol = get_local_id(0); 
    int localrow = get_local_id(1); 

    float4 acc = (float4)0.0f;
    int numTiles = (K + TileSize - 1) / TileSize;

    for (int k = 0; k < numTiles; k++) {
        
        // --- LOAD TILE B INTO LOCAL MEM ---
        // Each thread in the (4, 16) group loads one float4 from Global B
        int tiled_B_col = k * TileSize + (localcol * VEC);
        int tiled_B_row = k * TileSize + localrow;

        if (tiled_B_row < K && (col * VEC) < N) {
            // vload4 is optimized for Adreno's 128-bit bus
            TileB[localrow][localcol] = vload4(0, &B[tiled_B_row * N + (col * VEC)]);
        } else {
            TileB[localrow][localcol] = (float4)0.0f;
        }

        // Wait for all 64 threads to finish loading TileB
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- COMPUTE ---
        // We iterate through the 'K' within the current tile
        for (int k_inner = 0; k_inner < TileSize; k_inner++) {
            int k_idx = k * TileSize + k_inner;
            
            if (row < M && k_idx < K) {
                // Read a single scalar from A
                float a_val = A[row * K + k_idx];
                
                // Multiply scalar by the float4 from our Local TileB
                // acc.x += a * b.x, acc.y += a * b.y, etc.
                acc += a_val * TileB[k_inner][localcol];
            }
        }

        // Synchronize before loading the next tile of B
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 4. Store the result vector back to Global Memory
    if (row < M && (col * VEC) < N) {
        vstore4(acc, 0, &C[row * N + (col * VEC)]);
    }
}

//--------------------------------------------------------------------//
// We define a sampler to handle coordinate wrapping and filtering
const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | 
                      CLK_ADDRESS_CLAMP_TO_EDGE | 
                      CLK_FILTER_NEAREST;

#define TileSize 16
#define VEC 4

__kernel void matmul_image(
    __read_only image2d_t imgA, 
    __read_only image2d_t imgB, 
    __write_only image2d_t imgC, 
    int M, int N, int K) 
{
    // Local Memory for a Tile of B (16x16 floats stored as 16x4 float4s)
    __local float4 TileB[TileSize][TileSize / VEC];
    
    // IDs
    int col_vec = get_global_id(0); // This is in "pixels" (groups of 4 floats)
    int row = get_global_id(1); 
    
    int localcol = get_local_id(0); 
    int localrow = get_local_id(1); 

    float4 acc = (float4)0.0f;
    int numTiles = (K + TileSize - 1) / TileSize;

    for (int t = 0; t < numTiles; t++) {
        
        // --- 1. LOAD TILE B FROM IMAGE ---
        // Image coordinates are (x, y) -> (column, row)
        // Since B stores 4 floats per pixel, x is the column index / 4
        int2 coordsB = (int2)(col_vec, t * TileSize + localrow);

        // read_imagef returns a float4 automatically. No vload4 needed!
        // The sampler handles out-of-bounds by clamping or returning 0.
        TileB[localrow][localcol] = read_imagef(imgB, smp, coordsB);

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- 2. COMPUTE ---
        for (int k_inner = 0; k_inner < TileSize; k_inner++) {
            int k_idx = t * TileSize + k_inner;
            
            // Read scalar from A. 
            // We assume A is also an image for consistency.
            // If A is row-major, x=k_idx, y=row
            float4 a_pixel = read_imagef(imgA, smp, (int2)(k_idx, row));
            float a_val = a_pixel.x; // Use the first channel
                
            acc += a_val * TileB[k_inner][localcol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 3. STORE TO IMAGE ---
    // Output C also stores 4 floats per pixel (RGBA)
    if (row < M && col_vec < (N / VEC)) {
        write_imagef(imgC, (int2)(col_vec, row), acc);
    }
}