#include <iostream>
#include <math.h>

__global__ void init(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+= stride ){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
}

__global__ void add(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+= stride ){
        y[i] = x[i] + y[i];
    }
}

int main(){
    printf("hello pinmem\n");
    int N = 1 << 20;
    float *x, *y;
    int device = -1;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));


    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    init<<<numBlocks, blockSize>>>(N, x, y);

    cudaGetDevice(&device);
    cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);

    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaMemPrefetchAsync(y, N*sizeof(float), cudaCpuDeviceId, NULL);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "\n Max error " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}