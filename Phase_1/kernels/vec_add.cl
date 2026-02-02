// Phase1_Foundation/kernels/vec_add.cl

__kernel void vec_add(__global const float* A,
                      __global const float* B,
                      __global float* C, 
                      const int N)
{
        int id = get_global_id(0);
        if(id < N){
            C[id] = A[id] + B[id];
        }
}