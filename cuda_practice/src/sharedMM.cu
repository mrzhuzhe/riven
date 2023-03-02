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
        //int local_sum = 0;
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
    __shared__ int local_sum[1024];
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
    __syncthreads();
    if (j < 16){
        local_sum[j] += local_sum[j + 16];
    }
    __syncthreads();
    if (j < 8){
        local_sum[j] += local_sum[j + 8];
    }
    __syncthreads();
    if (j < 4){
        local_sum[j] += local_sum[j + 4];
    }
    __syncthreads();
    if (j < 2){
        local_sum[j] += local_sum[j + 2];
    }
    __syncthreads();
    if (j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}


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
    //parallel_sum4<<<n/1024/128, 128>>>(sum.data(), arr.data(), n);
    parallel_sum5<<<n/1024/128, 128>>>(sum.data(), arr.data(), n);
    cudaDeviceSynchronize();
    int final_sum = 0;
    for (int i = 0; i < n/1024; i++){
        final_sum += sum[i];
    }
    Tok("sum");

    printf("result: %d\n", final_sum);

    return 0;
}