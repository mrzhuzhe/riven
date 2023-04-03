/*
    Time= 0.251090 msec, bandwidth= 267.270172 GB/s
    host 16777216.000000, device 16777216.00000
*/

#include <iostream>
#include <helper_timer.h>
#include "utils.h"

#define NUM_LOAD 4


/**
    Two warp level primitives are used here for this example
    https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
    https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
 */
__inline__ __device__ float warp_reduce_sum(float val)
{
    #pragma unroll 5
    for (int offset = warpSize / 2; offset > 0; offset >>=1){
        unsigned int mask = __activemask();
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val){
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    if (wid == 0){
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void reduction_kernel(float *data_out, float *data_in, int size){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;    

    float sum[NUM_LOAD] = {0.f};
    for (int i = idx; i < size; i += blockDim.x * gridDim.x * NUM_LOAD){
        for (int step = 0; step < NUM_LOAD; step ++ ){
            int _cur = i + step * blockDim.x * gridDim.x;
            sum[step] += (_cur < size ) ? data_in[_cur] : 0.f;
        }        
    }
    for (int i = 1; i < NUM_LOAD; i++){
        sum[0] += sum[i];
    }
    sum[0] = block_reduce_sum(sum[0]);

    if (threadIdx.x == 0){
        data_out[blockIdx.x] = sum[0];
    }
}

int reduction(float *d_out, float *d_in, int size, int n_threads){
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, n_threads, n_threads*sizeof(float));
    //printf("num_blocks_per_sm %d \n", num_blocks_per_sm);
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1)/ n_threads);
    //printf("n_blocks %d \n", n_blocks);
    //reduction_kernel<<<n_blocks, n_threads, n_threads*sizeof(float), 0>>>(d_out, d_in, size);
    //reduction_kernel<<<1, n_threads, n_threads*sizeof(float), 0>>>(d_out, d_in, n_blocks);
    reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, size);
    reduction_kernel<<<1, n_threads>>>(d_out, d_in, n_blocks);
    

    return 1;
}

void run_benchmark(int (*reduce)(float *, float *, int, int), 
float *d_outPtr, float *d_inPtr, int size){
    int num_threads = 256;
    int test_iter = 100;

    reduce(d_outPtr, d_inPtr, size, num_threads);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    int _size;
    for (int i = 0; i < test_iter; i++){
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
        _size = size;
        while (_size > 1){
            _size = reduce(d_outPtr, d_outPtr, size, num_threads);
        }        
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    double elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = size * sizeof(float ) / elapsed_time_msed / 1e6;
    printf("Time= %f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

}


int main(){
    float *h_inPtr;
    float *d_inPtr, *d_outPtr;

    unsigned int size = 1 << 24;
    
    float result_host, result_gpu;
    //int mode = 0;

    srand(2023);

    h_inPtr = (float *)malloc(size*sizeof(float));

    init_input(h_inPtr, size);

    cudaMalloc((void **)&d_inPtr, size*sizeof(float));
    cudaMalloc((void **)&d_outPtr, size*sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size*sizeof(float), cudaMemcpyHostToDevice);

    run_benchmark(reduction, d_outPtr, d_inPtr, size);
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);

    result_host = get_cpu_result(h_inPtr, size);
    printf("host %f, device %f\n", result_host, result_gpu);

    cudaFree(d_outPtr);
    cudaFree(d_inPtr);
    free(h_inPtr);

    return 0;
}