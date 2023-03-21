/*
    compared a sample result...
    host: 2.000000, device 2.000000
    Time=17.972000 msec, bandwidth=44.808945 GB/s

    */
#include <iostream>
#include "common.h"
#include <helper_timer.h>

int main(){
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);

    cudaMallocHost((void **)&h_a, bufsize);
    cudaMallocHost((void **)&h_b, bufsize);
    cudaMallocHost((void **)&h_c, bufsize);

    srand(2023);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    //init_buffer(h_c, size);


    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
       
    sdkStartTimer(&timer);
    cudaEventRecord(start);

    
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<<dimGrid, dimBlock>>>(d_c, d_a, d_b);

    cudaEventRecord(stop);  // record event on kernel finished

    cudaEventSynchronize(stop); // sync based on cuda event

    sdkStopTimer(&timer);

    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost);

    int print_idx = 512;
    printf("compared a sample result...\n");
    printf("host: %f, device %f\n", h_a[print_idx]+h_b[print_idx], h_c[print_idx]);

    double elasped_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elasped_time_msed / 1e6;
    printf("Time=%f msec, bandwidth=%f GB/s\n", elasped_time_msed, bandwidth);

    sdkDeleteTimer(&timer);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}