#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_gpu(void) {
    printf("Hello world from thread [%d, %d] from devices\n ", threadIdx.x, blockIdx.x);
}

int main(void) {
    printf("Hello world from host! \n");
    print_from_gpu<<<1,2>>>();
    cudaDeviceSynchronize();
    return 0;
}