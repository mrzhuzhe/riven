#include <iostream>
#include "common.h"

__global__ void scan_v1_kernel(float *d_out, float *d_inp, int length){

}

void scan_v1(float *d_out, float *d_in, int length){
    dim3 dimBlock(Block_DIM);
    dim3 dimGrid(length + BLOCK_DIM -1) / BLOCK_DIM;
    scan_v1_kernel<<< dimGrid, dimBlock >>>(d_out, d_in, length);
}

void scan_host(float *h_out, float *h_in, int length, int version){
    for (int i =0; i< length; i++){
        for (int j =0; j < length; j++){
            if (i >=j){
                h_out[i] += h_in[i-j];
            }
        }
    }
}

void scan_host_v2(float *h_out, float *h_in, int length, int version){
    for (int i =0; i< length; i++){        
        h_out[i] = h_in[i];                    
    }
    int offset = 1;
    while( offset < length){
        for (int i = 0; i < length; i++){
            int idx_a = offset * (2 *i+1) -1;
            int idx_b = offset * (2 *i+2) -1;
            if (idx_a>=0 && idx_b < length)
                h_out[idx_b] += h_out[idx_a];
        }
        offset <<= 1;
    }
    offset >>= 1;
    while (offset>0){
        for (int i = 0; i < length; i++){
            int idx_a = offset * (2 *i+2) -1;
            int idx_b = offset * (2 *i+3) -1;
            if (idx_a>=0 && idx_b < length)
                h_out[idx_b] += h_out[idx_a];
        }
        offset <<= 1;
    }
}

int main(){
    srand(2023);
    float *h_input, *h_output_host, *h_output_gpu;
    float *d_input, *d_output;
    int length = BLOCK_DIM * 2;

    h_input = (float *)malloc(sizeof(float)*length);
    h_output_host = (float *)malloc(sizeof(float)*length);
    h_output_gpu = (float *)malloc(sizeof(float)*length);

    cudaMalloc((void **)&d_input, sizeof(float)*length);
    cudaMalloc((void **)&d_output, sizeof(float)*length);

    generate_data(h_input, length);
    print_val(h_input, 1, "input ::");

    scan_host(h_output_host, h_input, length, 1);
    print_val(h_output_host, DEBUG_OUTPUT_NUM, "result[cpu]   ::");

    cudaMemcpy(d_input, h_input, sizeof(float) * length, cudaMemcpyHostToDevice);
    scan_v1(d_output, d_input, length);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    print_val(h_output_gpu, DEBUG_OUTPUT_NUM, "result[gpu_v1]::");
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");

    
    cudaMemcpy(d_input, h_input, sizeof(float) * length, cudaMemcpyHostToDevice);
    scan_v2(d_output, d_input, length);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    print_val(h_output_gpu, DEBUG_OUTPUT_NUM, "result[gpu_v2]::");
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");


    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // free host memory
    free(h_input);
    free(h_output_host);
    free(h_output_gpu);

    return 0;

}