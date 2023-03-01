#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>



__global__ void kernel(int *pret){
    *pret = 42;
}

/*
__global__ void kernel2(int *pret2){
    *pret2 = 42;
}
*/

__global__ void kernel(int *arr, int n){
    /*
    for (int i = 0; i < n; i++){
        arr[i] = i;
    }
    */
   //int i = threadIdx.x;   
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i > n) return;   
   arr[i] = i;
}

__global__ void kernel2(int *arr, int n){
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i+= blockDim.x *gridDim.x){
        arr[i] = i;
    }
}

template <class T>
struct CudaAllocator {
    using value_type = T;

    T *allocate(size_t size) {
        T *ptr = nullptr;
        cudaMallocManaged(&ptr, size * sizeof(T));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        cudaFree(ptr);
    }
};


int main(){
    // simple copy
    int *pret;
    cudaMalloc(&pret, sizeof(int));
    kernel<<<1, 1>>>(pret);

    int ret;
    cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost);
    printf("result: %d\n", ret);
    cudaFree(pret);

    // unified memmory
    int *pret2;
    cudaMallocManaged(&pret2, sizeof(int));
    kernel<<<1, 1>>>(pret2);
    cudaDeviceSynchronize();
    printf("result: %d\n", *pret2);
    cudaFree(pret2);


    //  unified array
    int n = 32;
    //int *arr;
    //cudaMallocManaged(&arr, n * sizeof(int));
    //kernel<<<1, 1>>>(arr, n);
    //kernel<<<1, n>>>(arr, n);

    n = 128;
    //int nthreads = 128;
    //int nblocks = n / nthreads;
    //int nblocks = (n + nthreads + 1) / nthreads;
    //kernel<<<nblocks, nthreads>>>(arr, n);
    std::vector<int, CudaAllocator<int>> arr(n);
    //kernel2<<<32, 128>>>(arr, n);
    kernel2<<<32, 128>>>(arr.data(), n);

    cudaDeviceSynchronize();
    for (int i = 0 ; i < n; i++){
        printf("arr[%d]: %d\n", i, arr[i]);
    }
    //cudaFree(arr);
    
    return 0;
}