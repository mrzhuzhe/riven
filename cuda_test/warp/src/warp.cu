#include <iostream>

__global__ void print_index_kernel(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x & (warpSize-1);
    
    if ((lane_idx & (warpSize/2-1)) == 0)
        printf(" %5d\t%5d\t %2d\t%2d\n", idx, blockIdx.x, warp_idx, lane_idx);
}


int main(){
    int gridDim = 4, blockDim = 128;
    puts("thread, block, warp, lane");
    print_index_kernel<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    return 0;
}