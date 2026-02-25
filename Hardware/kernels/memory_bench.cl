// Notice the 'volatile' keyword added to the pointers

// 1. HRAM (Global)
__kernel void benchmark_global(__global volatile float* data, int iterations) {
    int id = get_global_id(0);
    int neighbor_id = (id + 1) % get_global_size(0); 

    for(int i = 0; i < iterations; i++) {
        float val1 = data[id];
        float val2 = data[neighbor_id];
        data[id] = val1 + val2;
    }
}

// 2. SRAM (Local)
__kernel void benchmark_local(__global volatile float* data, __local volatile float* local_data, int iterations) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int neighbor_lid = (lid + 1) % get_local_size(0); 

    local_data[lid] = data[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < iterations; i++) {
        float val1 = local_data[lid];
        float val2 = local_data[neighbor_lid];
        local_data[lid] = val1 + val2;
    }
    data[id] = local_data[lid];
}

// 3. Registers (Private)
__kernel void benchmark_private(__global volatile float* data, int iterations) {
    int id = get_global_id(0);
    int neighbor_id = (id + 1) % get_global_size(0);

    float val1 = data[id];
    float val2 = data[neighbor_id];
    float new_location = 0.0f;

    for(int i = 0; i < iterations; i++) {
        new_location = val1 + val2;
        // Volatile forces the register to actually update
        volatile float force_math = new_location; 
        val1 = force_math; 
    }
    data[id] = new_location;
}