/*
    Time= 0.246060 msec, bandwidth= 272.733734 GB/s
    host 16777216.000000, device 16777216.00000
*/

#include <iostream>
#include <helper_timer.h>

#define NUM_LOAD 4
#include "utils.h"

__global__ void reduction_kernel_opt(float *data_out, float *data_in, int size){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    float input[NUM_LOAD] = {0.f};
    for (int i = idx; i < size; i += blockDim.x * gridDim.x * NUM_LOAD){
        for (int step = 0; step < NUM_LOAD; step ++ ){
            int _cur = i + step * blockDim.x * gridDim.x;
            input[step] += (_cur < size ) ? data_in[_cur] : 0.f;
        }        
    }
    for (int i = 1; i < NUM_LOAD; i++){
        input[0] += input[i];
    }
    s_data[threadIdx.x] = input[0];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        data_out[blockIdx.x] = s_data[0];
    }

}

int reduction(float *d_out, float *d_in, int size, int n_threads){
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel_opt, n_threads, n_threads*sizeof(float));
    
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1)/ n_threads);
    
    reduction_kernel_opt<<<n_blocks, n_threads, n_threads*sizeof(float), 0>>>(d_out, d_in, size);
    reduction_kernel_opt<<<1, n_threads, n_threads*sizeof(float), 0>>>(d_out, d_in, n_blocks);
    
    /*
    float result_gpu;
    cudaMemcpy(&result_gpu, &d_out[0], sizeof(float), cudaMemcpyDeviceToHost);
    printf(" %f \n", result_gpu);
    */
    return 1;
}


void
run_reduction(int (*reduce)(float*, float*, int, int), 
              float *d_outPtr, float *d_inPtr, int size, int n_threads)
{
    while(size > 1) 
    {
        size = reduce(d_outPtr, d_inPtr, size, n_threads);
    }
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
        _size = size;   // reset size
        while (_size > 1){
            _size = reduce(d_outPtr, d_outPtr, size, num_threads);            
        }        
        
        //run_reduction(reduce, d_outPtr, d_outPtr, size, num_threads);
        /*
        float result_gpu = 0;
        cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
        printf(" %f\n", result_gpu);
        */
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

    srand(2019);

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