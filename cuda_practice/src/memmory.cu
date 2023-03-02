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

__global__ void kernel2(float *arr, int n){
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

    template <class ...Args>
    void construct(T *p, Args &&...args){
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))::new((void *)p) T(std::forward<Args>(args)...);        
    }
};

template <class Func>
__global__ void parallel_for(int n, Func func){
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x ){
        func(i);
    }
}

struct MyFunctor {
    __device__ void operator()(int i) const {
        printf("number %d\n", i);
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
    //std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<float, CudaAllocator<float>> arr(n);
    //kernel2<<<32, 128>>>(arr, n);
    kernel2<<<32, 128>>>(arr.data(), n);

    cudaDeviceSynchronize();
    for (int i = 0 ; i < n; i++){
        printf("arr[%d]: %f\n", i, arr[i]);
    }
    //cudaFree(arr);
    
    //int n = 1024;
    //parallel_for<<<32, 128>>>(n, MyFunctor{});

    //  parallel_for<<<32, 128>>>(n, [&] __device__ (int i) {
    //int *arr_data = arr.data();
    //parallel_for<<<32, 128>>>(n, [=] __device__ (int i) {
    parallel_for<<<32, 128>>>(n, [arr = arr.data()] __device__ (int i) {
        //printf("number %d\n", i);
        //arr_data[i] = i;
        //arr[i] = sinf(i);
        arr[i] = __sinf(i); // less precise __expf __logf __cosf __powf
    });
    for (int i = 0 ; i < n; i++){
        printf("arr[%d]: %f\n", i, arr[i] - sinf(i));
    }
    cudaDeviceSynchronize();
    
    return 0;
}