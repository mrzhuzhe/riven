/*
Time= 0.418960 msec, bandwidth= 160.179642 GB/s
host 16777216.000000, device 16777216.000000
*/

#include <iostream>
#include <helper_timer.h>
#include "utils.h"

__global__ void shared_reduction_kernel(float *data_out, float *data_in, int size){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx < size) ? data_in[idx] : 0.f;

    __syncthreads();
    
    // interleaved addressing
    for (unsigned int stride =1; stride < blockDim.x; stride *= 2 ){
        if ((idx % (stride *2)) == 0)
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        data_out[blockIdx.x] = s_data[0];
}


__global__ void shared_reduction_kernel_without_divergence(float *data_out, float *data_in, int size){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx < size) ? data_in[idx] : 0.f;

    __syncthreads();

    // interleaved addressing
    // 这个并不是线程分歧 是增加存储数据的局部性
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride) 
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        data_out[blockIdx.x] = s_data[0];
}

void global_reduction(float *d_out, float *d_in, int n_threads, int size){
    cudaMemcpy(d_out, d_in, size*sizeof(float), cudaMemcpyDeviceToDevice);

    while (size > 1){
        int n_blocks =(size + n_threads -1) / n_threads;
        // wrong
        //shared_reduction_kernel_without_divergence<<<n_blocks, n_threads, n_threads*sizeof(float), 0>>>(d_out, d_in, size);
        /*
            Time= 0.514730 msec, bandwidth= 130.376831 GB/s
            host 16777216.000000, device 16777216.000000
        */

        //shared_reduction_kernel<<<n_blocks, n_threads, n_threads*sizeof(float), 0>>>(d_out, d_out, size);
        /*
            Time= 0.419230 msec, bandwidth= 160.076477 GB/s
            host 16777216.000000, device 16777216.000000
        */
        shared_reduction_kernel_without_divergence<<<n_blocks, n_threads, n_threads*sizeof(float), 0>>>(d_out, d_out, size);
        size = n_blocks;
    }
}

void run_benchmark(void (*reduce)(float *, float *, int, int), 
float *d_outPtr, float *d_inPtr, int size){
    int num_threads = 256;
    int test_iter = 100;

    reduce(d_outPtr, d_inPtr, num_threads, size);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < test_iter; i++){
        reduce(d_outPtr, d_inPtr, num_threads, size);
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

    srand(2019);

    h_inPtr = (float *)malloc(size*sizeof(float));

    init_input(h_inPtr, size);

    cudaMalloc((void **)&d_inPtr, size*sizeof(float));
    cudaMalloc((void **)&d_outPtr, size*sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size*sizeof(float), cudaMemcpyHostToDevice);

    run_benchmark(global_reduction, d_outPtr, d_inPtr, size);
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);

    result_host = get_cpu_result(h_inPtr, size);
    printf("host %f, device %f\n", result_host, result_gpu);

    cudaFree(d_outPtr);
    cudaFree(d_inPtr);
    free(h_inPtr);

    return 0;
}