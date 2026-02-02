__kernel void sum_reduce(__global const float* input, 
                         __global float* output, 
                         __local float* sharedData) { // <--- Magic happens here
    
    // IDs
    int gid = get_global_id(0); // Global ID (0 to TotalThreads)
    int lid = get_local_id(0);  // Local ID within group (0 to 255)
    int groupSize = get_local_size(0);

    // 1. LOAD: Copy from Slow Global Memory to Fast Local Memory
    sharedData[lid] = input[gid];
    
    // WAIT! Make sure everyone in the group has finished loading.
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. REDUCE: The Tree Algorithm
    // We fold the array in half repeatedly.
    // Example: Size 256 -> 128 -> 64 -> 32 ... -> 1
    for (int stride = groupSize / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            sharedData[lid] += sharedData[lid + stride];
        }
        // WAIT! Ensure all additions for this round are done before next round.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. STORE: The single winner (Thread 0) writes the group's result
    if (lid == 0) {
        output[get_group_id(0)] = sharedData[0];
    }
}