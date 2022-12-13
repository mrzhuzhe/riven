#include <stdio.h>
#include <stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c) {
    for (int idx=0;idx<N;idx++)
        c[idx] = a[idx] + b[idx];
}

void fill_array(int *data) {
    for (int idx=0;idx<N;idx++)
        data[idx] = idx;
}

void print_output(int *a, int *b, int *c) {
    for (int idx=0;idx<N;idx++)
        printf("\n %d = %d + %d", c[idx], a[idx], b[idx]);
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

int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    int size = N * sizeof(int);
    a = (int *)malloc(size); fill_array(a);
    b = (int *)malloc(size); fill_array(b);
    c = (int *)malloc(size);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    //device_add<<<1,N>>>(d_a,d_b,d_c);
    //int thread_per_block = 8;
    int thread_per_block = 4;
    int no_of_blocks = N / thread_per_block;
    device_add<<<no_of_blocks, thread_per_block>>>(d_a,d_b,d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    

    //  host_add(a, b, c);
    print_output(a, b, c);
    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}