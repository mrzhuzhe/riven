#include <cstdio>
#include <cuda_runtime.h>
#include "allocator.h"
#include "TikTok.h"

__global__ void parallel_sum(int *sum, int const *arr, int n){
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n / 1024; i += blockDim.x * gridDim.x){
        int local_sum = 0;
        for (int j = i * 1024; j < i * 1024 + 1024; j++){
            local_sum += arr[j];
        }
        sum[i] = local_sum;
    }
}


__global__ void parallel_sum2(int *sum, int const *arr, int n){
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n / 1024; i += blockDim.x * gridDim.x){        
        int local_sum[1024];
        for (int j = 0; j < 1024; j++){
            local_sum[j] = arr[i*1024+j];
        }
        for (int j = 0; j < 512; j++){
            local_sum[j] += local_sum[j + 512];
        }
        for (int j = 0; j < 256; j++){
            local_sum[j] += local_sum[j + 256];
        }
        for (int j = 0; j < 128; j++){
            local_sum[j] += local_sum[j + 128];
        }
        for (int j = 0; j < 64; j++){
            local_sum[j] += local_sum[j + 64];
        }
        for (int j = 0; j < 32; j++){
            local_sum[j] += local_sum[j + 32];
        }
        for (int j = 0; j < 16; j++){
            local_sum[j] += local_sum[j + 16];
        }
        for (int j = 0; j < 8; j++){
            local_sum[j] += local_sum[j + 8];
        }
        for (int j = 0; j < 4; j++){
            local_sum[j] += local_sum[j + 4];
        }
        for (int j = 0; j < 2; j++){
            local_sum[j] += local_sum[j + 2];
        }
        for (int j = 0; j < 1; j++){
            local_sum[j] += local_sum[j + 1];
        }
        sum[i] = local_sum[0];
    }
}

__global__ void parallel_sum3(int *sum, int const *arr, int n){
    //__shared__ volatile int local_sum[1024];
    __shared__ int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    local_sum[j] = arr[i*1024 +j];
    if (j < 512){
        local_sum[j] += local_sum[j + 512];
    }
    if (j < 256){
        local_sum[j] += local_sum[j + 256];
    }
    if (j < 128){
        local_sum[j] += local_sum[j + 128];
    }
    if (j < 64){
        local_sum[j] += local_sum[j + 64];
    }
    if (j < 32){
        local_sum[j] += local_sum[j + 32];
    }
    if (j < 16){
        local_sum[j] += local_sum[j + 16];
    }
    if (j < 8){
        local_sum[j] += local_sum[j + 8];
    }
    if (j < 4){
        local_sum[j] += local_sum[j + 4];
    }
    if (j < 2){
        local_sum[j] += local_sum[j + 2];
    }
    if (j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}

__global__ void parallel_sum4(int *sum, int const *arr, int n){
    __shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    local_sum[j] = arr[i*1024 +j];
    __syncthreads();
    if (j < 512){
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256){
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128){
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64){
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32){
        local_sum[j] += local_sum[j + 32];
    }
    //__syncthreads();
    if (j < 16){
        local_sum[j] += local_sum[j + 16];
    }
    //__syncthreads();
    if (j < 8){
        local_sum[j] += local_sum[j + 8];
    }
    //__syncthreads();
    if (j < 4){
        local_sum[j] += local_sum[j + 4];
    }
    //__syncthreads();
    if (j < 2){
        local_sum[j] += local_sum[j + 2];
    }
    //__syncthreads();
    if (j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}

// warp divergence
__global__ void parallel_sum5(int *sum, int const *arr, int n){
    __shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    local_sum[j] = arr[i*1024 +j];
    __syncthreads();
    if (j < 512){
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256){
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128){
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64){
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32){
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}

// 网格跨步循环
__global__ void parallel_sum6(int *sum, int const *arr, int n){
    // shared on same block
    __shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    int temp_sum = 0;
    for (int t = i * 1024 + j; t < n; t += 1024 * gridDim.x){
        temp_sum += arr[t];        
    }
    // less j more read from share memory
    local_sum[j] = temp_sum;
    __syncthreads();
    if (j < 512){
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256){
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128){
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64){
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32){
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}

// BLS block-local-storage
template <int blockSize, class T>
__global__ void parallel_sum_kernel(T *sum, T const * arr, int n){
    __shared__ volatile int local_sum[blockSize];
    int j = threadIdx.x;
    int i = blockIdx.x;
    T temp_sum = 0;
    for (int t = i * blockSize + j; t < n; t += blockSize * gridDim.x){
        temp_sum += arr[t];
    }
    local_sum[j] = temp_sum;
    __syncthreads();
    // this part canbe elimilate in complie optimize phase
    if constexpr (blockSize >= 1024){
        if (j < 512)
            local_sum[j] += local_sum[j + 512];
        __syncthreads();
    }
    if constexpr (blockSize >= 512){
        if (j < 256)
            local_sum[j] += local_sum[j + 256];
        __syncthreads();
    }
    if constexpr (blockSize >= 256){
        if (j < 128)
            local_sum[j] += local_sum[j + 128];
        __syncthreads();
    }
    if constexpr (blockSize >= 128){
        if (j < 64)
            local_sum[j] += local_sum[j + 64];
        __syncthreads();
    }
    if (j < 32){
        if constexpr (blockSize >= 64)
            local_sum[j] += local_sum[j + 32];
        if constexpr (blockSize >= 32)
            local_sum[j] += local_sum[j + 16];
        if constexpr (blockSize >= 16)
            local_sum[j] += local_sum[j + 8];
        if constexpr (blockSize >= 8)
            local_sum[j] += local_sum[j + 4];
        if constexpr (blockSize >= 4)
            local_sum[j] += local_sum[j + 2];
        if (j == 0){
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}


template <int reduceScale = 4096, int blockSize = 256, class T>
int parallel_sum7(T const *arr, int n){
    std::vector<int, CudaAllocator<int>> sum(n / reduceScale);
    parallel_sum_kernel<blockSize><<<n / reduceScale, blockSize>>>(sum.data(), arr, n);
    cudaDeviceSynchronize();
    T final_sum = 0;
    for (int i = 0; i < n / reduceScale; i++){
        final_sum += sum[i];
    }
    return final_sum;
}

template <int reduceScale = 4096, int blockSize = 256, int cutoffSize = reduceScale * 2, class T>
int parallel_sum8(T const *arr, int n){

    if (n > cutoffSize) {
        std::vector<int, CudaAllocator<int>> sum(n / reduceScale);
        parallel_sum_kernel<blockSize><<<n / reduceScale, blockSize>>>(sum.data(), arr, n);
        return parallel_sum8(sum.data(), n / reduceScale);
    } else {
        cudaDeviceSynchronize();
        T final_sum = 0;
        for (int i = 0; i < n ; i++){
            final_sum += arr[i];
        }
        return final_sum;
    }    
}


int main() {
    int n = 1 << 24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n/1024);


    for (int i = 0; i <n ; i++){
        arr[i] = std::rand() % 4;
    }

    Tik();
    //parallel_sum<<<n/1024/128, 128>>>(sum.data(), arr.data(), n);
    //parallel_sum3<<<n/1024/128, 128>>>(sum.data(), arr.data(), n);
    //parallel_sum4<<<n/1024, 1024>>>(sum.data(), arr.data(), n);
    //parallel_sum5<<<n/1024, 1024>>>(sum.data(), arr.data(), n);
    parallel_sum6<<<n/4096, 1024>>>(sum.data(), arr.data(), n);
    cudaDeviceSynchronize();
    int final_sum = 0;
    for (int i = 0 ; i<n/4096; i++){
        final_sum += sum[i];
    }
    Tok("sum");

    printf("result: %d\n", final_sum);
    
    
    Tik();
    final_sum = parallel_sum8(arr.data(), n);
    Tok("parallel sum kernel");
    printf("BLS result: %d\n", final_sum);
    

    return 0;
}