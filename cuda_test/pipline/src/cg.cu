/*
    Time= 0.092800 msec, bandwidth= 723.155884 GB/s
    host 16777216.000000, device 16777216.000000
    // this implement is much faster
*/
#include <iostream>
#include <cooperative_groups.h>
#include <helper_timer.h>
#include "common.h"

#define NUM_LOAD 4

namespace cg = cooperative_groups;
__device__ void Block_reduction(float *out, float *in, float *s_data, int active_size, int size, const cg::grid_group &grid, const cg::thread_block &block){
    int tid = block.thread_rank();  
    s_data[tid] = 0.f;

    for (int i = grid.thread_rank(); i< size; i+=active_size)
        s_data[tid] += in[i];
    
    block.sync();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>=1){
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        block.sync();
    }
    
    if (block.thread_rank() == 0)
        out[block.group_index().x] = s_data[0];

}

__global__ void reduction_kernel(float *data_out, float *data_in, int size){
    
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    
    extern __shared__ float s_data[];
    
    Block_reduction(data_out, data_in, s_data, grid.size(), size, grid, block);

    grid.sync();

    if (block.group_index().x == 0){
        Block_reduction(data_out, data_out, s_data, block.size(), gridDim.x, grid, block);
    }
}

int reduction_grid_sync(float *g_outPtr, float *g_inPtr, int size, int n_threads){
        
    int num_blocks_per_sm;    
    cudaDeviceProp deviceProp;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, n_threads, n_threads*sizeof(float));
    cudaGetDeviceProperties(&deviceProp, 0);
    int num_sms = deviceProp.multiProcessorCount;
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1)/ n_threads);
    
    void *params[3];
    params[0] = (void*)&g_outPtr;
    params[1] = (void*)&g_inPtr;
    params[2] = (void*)&size;
    
    // much much faster
    cudaLaunchCooperativeKernel((void*)reduction_kernel, n_blocks, n_threads, params, n_threads * sizeof(float), NULL);

    return n_blocks;
}


void run_benchmark(int (*reduce)(float *, float *, int, int), 
float *d_outPtr, float *d_inPtr, int size){
    int num_threads = 256;
    int test_iter = 100;

    reduce(d_outPtr, d_inPtr, size, num_threads);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < test_iter; i++){
        reduce(d_outPtr, d_inPtr, size, num_threads);               
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    double elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = size * sizeof(float ) / elapsed_time_msed / 1e6;
    printf("Time= %f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

}

int check_cooperative_launch_support(){
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.cooperativeLaunch == 0)
        return 0;
    return 1;
}

int main(){
    float *h_inPtr;
    float *d_inPtr, *d_outPtr;

    unsigned int size = 1 << 24;
    
    float result_host, result_gpu;
    //int mode = 0;

    srand(2019);

    if (check_cooperative_launch_support() == 0){
        printf("CPU does not support cooperative kernel");
    }


    h_inPtr = (float *)malloc(size*sizeof(float));

    init_input(h_inPtr, size);

    cudaMalloc((void **)&d_inPtr, size*sizeof(float));
    cudaMalloc((void **)&d_outPtr, size*sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size*sizeof(float), cudaMemcpyHostToDevice);

    run_benchmark(reduction_grid_sync, d_outPtr, d_inPtr, size);
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);

    result_host = get_cpu_result(h_inPtr, size);
    printf("host %f, device %f\n", result_host, result_gpu);

    cudaFree(d_outPtr);
    cudaFree(d_inPtr);
    free(h_inPtr);

    return 0;
}