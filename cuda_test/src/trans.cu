#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

#define N 16
#define BLOCK_SIZE 4

void host_add(int *a, int *b, int *c) {
    for (int idx=0;idx<N;idx++)
        c[idx] = a[idx] + b[idx];
}

void fill_array(int *data) {
    for (int idx=0;idx<N;idx++)
        data[idx] = idx;
}

void fill_2darray(int *data) {
    for (int idy=0;idy<N;idy++){
        for (int idx=0;idx<N;idx++){
            data[idy * N + idx] = idy * N + idx;
        }
    }        
}

void print_output(int *a, int *b, int *c) {
    for (int idx=0;idx<N;idx++)
        printf("\n %d = %d + %d", c[idx], a[idx], b[idx]);
}

void print_2doutput(int *a) {
     for (int idy=0;idy<N;idy++){
        for (int idx=0;idx<N;idx++){
            printf(" %d ", a[idy * N + idx]);
        }
        printf("\n"); 
    }    
    
}

/*
__global__ void device_add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
*/
__global__ void device_add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

// shared
__global__ void matrix_traspose_navie(int *input, int *output) {
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;
    int index = indexY * N + indexX;
    int transposedIndex = indexX * N + indexY;
    output[index] = input[transposedIndex];
}

__global__ void matrix_traspose_shared(int *input, int *output) {
    __shared__ int sharedMemory [BLOCK_SIZE] [BLOCK_SIZE];
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;

    int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
    int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

    int localIndexX = threadIdx.x;
    int localIndexY = threadIdx.y;
    int index = indexY * N + indexX;
    int transposedIndex = tindexY * N + tindexX;

    sharedMemory[localIndexX][localIndexY] = input[index];
    __syncthreads();
    output[transposedIndex] = sharedMemory[localIndexY][localIndexX];

}

int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    int size = N * sizeof(int);
    int qsize = size * size;
    a = (int *)malloc(qsize); fill_2darray(a);
    b = (int *)malloc(qsize); fill_2darray(b);
    c = (int *)malloc(qsize);

    cudaMalloc((void **)&d_a, qsize);
    cudaMalloc((void **)&d_b, qsize);
    cudaMalloc((void **)&d_c, qsize);

    cudaMemcpy(d_a, a, qsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, qsize, cudaMemcpyHostToDevice);

    //device_add<<<1,N>>>(d_a,d_b,d_c);
    //int thread_per_block = 8;
    
    //int thread_per_block = dim3(4, 4, 1);
    //int no_of_blocks = dim3(N / thread_per_block.x, N / thread_per_block.y, 1);
    
    dim3 thread_per_block(4, 4, 1);
    dim3 no_of_blocks(N / thread_per_block.x, N / thread_per_block.y, 1);
    //device_add<<<no_of_blocks, thread_per_block>>>(d_a,d_b,d_c);

    
    //matrix_traspose_navie<<<no_of_blocks, thread_per_block>>>(a, c);

    //matrix_traspose_navie<<<no_of_blocks, thread_per_block>>>(d_a, d_c);
    matrix_traspose_shared<<<no_of_blocks, thread_per_block>>>(d_a, d_c);

    cudaMemcpy(c, d_c, qsize, cudaMemcpyDeviceToHost);

    print_2doutput(a);
    printf("\n\n\n");
    print_2doutput(c);

    //  host_add(a, b, c);
    //print_output(a, b, c);
    
    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}