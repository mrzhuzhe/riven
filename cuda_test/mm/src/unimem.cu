#include <iostream>
#include <math.h>

//  stide loop https://zhuanlan.zhihu.com/p/571320529
__global__ void add(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (index == 0)
        printf("%d %d", blockDim.x , gridDim.x);
    for (int i = index; i < n; i+= stride ){
        //int i = index;
        y[i] = x[i] + y[i];
    }
}

int main(){
    printf("hello pinmem\n");
    int N = 1 << 20;
    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    printf("numSMs %d \n", numSMs);

    // if block and thread is smaller than total still work
    add<<<32*numSMs, 256>>>(N, x, y);

    //add<<<numBlocks, blockSize>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}