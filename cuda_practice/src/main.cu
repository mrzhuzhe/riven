#include <cstdio>
#include <cuda_runtime.h>

__device__ __inline__ void say_hello(){
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tnum = blockDim.x * gridDim.x;
    printf("\
    Hello World Blick %d of %d, threadIdx %d of %d \
    Flattened Thread %d of %d \n \
    ",
     blockIdx.x, gridDim.x, threadIdx.x, blockDim.x, tid, tnum);
}

__global__ void kernel() {
    say_hello();
}

__global__ void kernel2(){
    printf("Block (%d,%d,%d) of (%d,%d,%d), Thread (%d,%d,%d) of (%d,%d,%d)\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        gridDim.x, gridDim.y, gridDim.z,
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockDim.x, blockDim.y, blockDim.z
    );
}

__global__ void another() {
    printf("another: Thread %d of %d\n", threadIdx.x, blockDim.x);
}

//  kernel launch from __device__ or __global__ functions requires separate compilation mode
__global__ void kernel3(){
    printf("kernel3: Thread %d of %d \n", threadIdx.x, blockDim.x);
    int numthreads = threadIdx.x * threadIdx.x + 1;
    another<<<1, numthreads>>>();
    printf("kernel3: called another with %d threads\n", numthreads);
}

int main(){
    //kernel<<<2, 3>>>();
    //kernel2<<<dim3(2, 1, 1), dim3(2, 2, 2)>>>();
    kernel3<<<1, 3>>>();
    cudaError_t err = cudaDeviceSynchronize();
    printf("error code: %d\n", err);
    printf("error name %s\n", cudaGetErrorName(err));
    //printf("%d\n", ret);
    return 0;
}