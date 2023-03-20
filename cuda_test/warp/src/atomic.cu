/*
    Time= 27.767040 msec, bandwidth= 2.416853 GB/s
    host 16777216.000000, device 16777216.000000
*/

#include <iostream>
#include <helper_timer.h>
#include "utils.h"

__global__ void reduction_kernel(float *data_out, float *data_in, int size){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&data_out[0], data_in[idx]);
}

void reduction(float *d_out, float *d_in, int size, int n_threads){
    int n_blocks = (size + n_threads - 1)/n_threads;
    reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, size);    
}

void run_benchmark(void (*reduce)(float *, float *, int, int), 
float *d_outPtr, float *d_inPtr, int size){
    int num_threads = 256;
    int test_iter = 100;

    reduce(d_outPtr, d_inPtr, size, num_threads);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < test_iter; i++){
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
        reduce(d_outPtr, d_outPtr, size, num_threads);               
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