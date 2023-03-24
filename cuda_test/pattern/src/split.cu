#include <iostream>
#include "common.h"
#include <cuda_runtime.h>
#include "cuda_profiler_api.h"

#define FLT_ZERO 0.f
#define GRID_DIM 1
#define BLOCK_DIM 512

__global__ void scan_v2_kernel(float *d_out, float *d_in, int length){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_buffer[];
    s_buffer[threadIdx.x] = d_in[idx];
    s_buffer[threadIdx.x + BLOCK_DIM] = d_in[idx+BLOCK_DIM];

    int offset = 1;

    while (offset < length)
    {
        __syncthreads();
        int idx_a = offset * (2*tid + 1) - 1;
        int idx_b = offset * (2*tid + 2) - 1;

        if (idx_a >=0 && idx_b <2 * BLOCK_DIM){
            s_buffer[idx_b] += s_buffer[idx_a];
        }

        offset <<= 1;
    }

    offset >>= 1;
    while (offset > 0){
        __syncthreads();
        int idx_a = offset * (2 * tid + 2) - 1;
        int idx_b = offset * (2 * tid + 3) - 1;
        if (idx_a >=0 && idx_b <2 * BLOCK_DIM){
            s_buffer[idx_b] += s_buffer[idx_a];
        }

        offset >>= 1;
    }

    __syncthreads();
    d_out[idx] = s_buffer[tid];
    d_out[idx + BLOCK_DIM] = s_buffer[tid + BLOCK_DIM];

}


void scan_v2(float *d_out, float *d_in, int length){
    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid((length + (2 * BLOCK_DIM) -1)/(2 * BLOCK_DIM));
    scan_v2_kernel<<<dimGrid, dimBlock, sizeof(float)*BLOCK_DIM*2>>>(d_out, d_in, length);
    cudaDeviceSynchronize();
}


__global__ void predicate_kernel(float *d_predict, float *d_in, int length){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= length) return;
    d_predict[idx] = d_in[idx] > FLT_ZERO;
}

__global__ void pack_kernel(float *d_out, float *d_in, float *d_pre, float *d_scan, int length){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= length) return;
    if (d_pre[idx] != 0.f){
        int address = d_scan[idx] - 1;
        d_out[address] = d_in[idx];
    }
}

__global__ void split_kernel(float *d_out, float *d_in, float *d_pre, float *d_scan, int length){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= length) return;
    if (d_pre[idx] != 0.f){
        int address = d_scan[idx] - 1;
        d_out[idx] = d_in[address];
    }
}

void pack_host(float *h_out, float *h_in, int length){
    int idx_output = 0;
    for (int i = 0; i< length; i++){
        if (h_in[i] > FLT_ZERO){           
            h_out[idx_output] = h_in[i];
            idx_output++;
        }
    }    
}

void split_host(float *h_out, float *h_in, int length){
    for (int i = 0; i< length; i++){
        if (h_in[i] > FLT_ZERO){           
            h_out[i] = h_in[i];
        } else {
            h_out[i] = 0.f;
        }
    }    
}

int main(){
    
    float *h_input, *h_output_host, *h_output_gpu;
    float *d_input, *d_output;
    float *d_predicates, *d_scanned; // for temporarly purpose operation
    float length = BLOCK_DIM;

    srand(2023);

    // allocate host memory
    h_input = (float *)malloc(sizeof(float) * length);
    h_output_host = (float *)malloc(sizeof(float) * length);
    h_output_gpu = (float *)malloc(sizeof(float) * length);

    // allocate device memory
    cudaMalloc((void**)&d_input, sizeof(float) * length);
    cudaMalloc((void**)&d_output, sizeof(float) * length);
    cudaMalloc((void**)&d_predicates, sizeof(float) * length);
    cudaMalloc((void**)&d_scanned, sizeof(float) * length);

    random_init(h_input, length);
    cudaMemcpy(d_input, h_input, sizeof(float) * length, cudaMemcpyHostToDevice);

    print_val(h_input, 16, "input    ::");

    cudaProfilerStart();

    predicate_kernel<<< GRID_DIM, BLOCK_DIM>>>(d_predicates, d_input, length);

    // scan
    scan_v2(d_scanned, d_predicates, length);

    //cudaMemcpy(h_output_gpu, d_scanned, sizeof(float) * length, cudaMemcpyDeviceToHost);
    //print_val(h_output_gpu, 16, "d_predicates[gpu]::");

    // addressing & gather (pack)
    pack_kernel<<< GRID_DIM, BLOCK_DIM >>>(d_output, d_input, d_predicates, d_scanned, length);
    cudaDeviceSynchronize();

    // validation the result (compack)
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    pack_host(h_output_host, h_input, length);

    print_val(h_output_host, 16, "pack[cpu]::");
    print_val(h_output_gpu, 16, "pack[gpu]::");

    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");
    else
        printf("Something wrong..\n");

    /********************************
     * Split                        *
     ********************************/
    cudaMemcpy(d_input, d_output, sizeof(float) * length, cudaMemcpyDeviceToDevice);
    cudaMemset(d_output, 0, sizeof(float) * length);
    split_kernel<<<GRID_DIM, BLOCK_DIM>>>(d_output, d_input, d_predicates, d_scanned, length);
    cudaDeviceSynchronize();
    cudaProfilerStop();

    // validation the result (split)
    cudaMemcpy(h_output_gpu,  d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    split_host(h_output_host, h_input, length); // notice: we just generate desired output for the evaluation purpose

    print_val(h_output_gpu, 16, "split[gpu]");
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");
    else
        printf("Something wrong..\n");

    // finalize
    cudaFree(d_predicates);
    cudaFree(d_scanned);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output_gpu);
    free(h_output_host);
    free(h_input);

    return 0;
}