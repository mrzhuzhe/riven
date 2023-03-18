#include <iostream>
#include <math.h>


#define STRIDE_64K 65536

__global__ void init(int n, float *x, float *y){
    int lane_id = threadIdx.x & 31;
    size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
    size_t warps_per_grid = (blockDim.x * gridDim.x) >> 5;
    size_t warp_total = ((sizeof(float)*n) + STRIDE_64K-1)/STRIDE_64K;

    
    if(blockIdx.x == 0 && threadIdx.x == 0){
    //if(blockIdx.x == 0){        
        printf("\n TID[%d]", threadIdx.x);
        printf("\n WID[%d]", warp_id);
        printf("\n LANEID[%d]", lane_id);
        printf("\n warps_per_grid[%d]", warps_per_grid);
        printf("\n warp_total[%d]", warp_total);
        printf("\n rep[%d]", STRIDE_64K/sizeof(float) >> 5);
    }

    /*
        TID[0]
        WID[0]
        LANEID[0]
        warps_per_grid[64]
        warp_total[64]
        rep[512]
        Max error 0 
    */
    

    for (; warp_id < warp_total; warp_id += warps_per_grid){
        #pragma unroll
        for (int rep = 0; rep < STRIDE_64K / sizeof(float)/32; rep++){
            size_t ind = warp_id * STRIDE_64K/ sizeof(float) + rep * 32 + lane_id;
            if (ind < n){
                x[ind] = 1.0f;
                /*
                if (blockIdx.x == 0 && threadIdx.x == 0){
                    printf("\n ind[%d]", ind);
                }
                */
                y[ind] = 2.0f;
            }
        }
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

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));


    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    size_t warp_total = ((sizeof(float)*N) + STRIDE_64K - 1) / STRIDE_64K;
    int numBlocksInit = (warp_total*32) / blockSize;    // 8?
    printf("numBlocksInit %d\n", numBlocksInit);

    init<<<numBlocksInit, blockSize>>>(N, x, y);
    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "\n Max error " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}