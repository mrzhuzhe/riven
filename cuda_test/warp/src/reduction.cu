#include <iostream>
#include <helper_timer.h>

__global__ void global_reduction_kernel(float *data_out, float *data_in, int stride, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + stride < size){
        data_out[idx] += data_in[idx + stride];
    }
}

void global_reduction(float *d_out, float *d_in, int n_threads, int size){
    int n_blocks = (size+n_threads-1) / n_threads;

    for (int stride =1; stride <size; stride *= 2 ){

        global_reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, stride, size);
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
        cudaMemcpy(d_outPtr, d_inPtr, size*sizeof(float), cudaMemcpyDeviceToDevice);
        reduce(d_outPtr, d_inPtr, num_threads, size);
        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    float elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = size * sizeof(float ) / elapsed_time_msed / 1e6;
    printf("Time= %3.f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

}

void init_input(float *data, int size){
    for (int i = 0; i< size; i++){
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float get_cpu_result(float *data, int size){
    double result = 0.f;
    for (int i = 0; i< size; i++)
        result += data[i];
    return (float)result;
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