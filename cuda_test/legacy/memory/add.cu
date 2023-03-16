/*
nvcc -ccbin \
g++ -I../include \
-gencode arch=compute_75,code=sm_75 \
-o outputs/add \
add.cu
*/

#include<stdio.h>
#include<stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c){
    for (int idx=0;idx<N;idx++)
        c[idx] = a[idx] + b[idx];
}

__global__ void device_add(int *a, int *b, int *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
    //printf("%f", index);
}

void fill_array(int *data){
    for (int idx=0;idx<N;idx++)
        data[idx] = idx;
}

void print_output(int *a, int *b, int *c){
    for (int idx=0;idx<N;idx++)
        printf("\n %d + %d = %d", a[idx], b[idx], c[idx]);
}

int main(void){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int threads_per_block = 0, no_of_blocks=0;
    int size = N * sizeof(int);

    a = (int *)malloc(size); fill_array(a);
    b = (int *)malloc(size); fill_array(b);
    c = (int *)malloc(size);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    threads_per_block = 16;
    no_of_blocks = N/threads_per_block;

    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    device_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    //device_add<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    print_output(a, b, c);

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}