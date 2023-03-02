#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>



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

__global__ void parallel_sum(int *sum, int const *arr, int n){
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x ){
        //sum[0] += arr[i];
        atomicAdd(&sum[0], arr[i]);        
        //atomicCAS(&sum[0], arr[i], &sum[0] + arr[i]);
        //atomicCAS(sum[0], arr[i]);
    }
}

int main(){
    int n = 65536;
    float a = 3.14f;
    /*
    thrust::universal_vector<float> x(n);
    thrust::universal_vector<float> y(n);
    */
    thrust::host_vector<float> x_host(n);
    thrust::host_vector<float> y_host(n);
    
    /*
    for (int i = 0;i<n; i++){
        /*
        x[i] = std::rand() * (1.f/RAND_MAX);
        y[i] = std::rand() * (1.f/RAND_MAX);
        * /
        x_host[i] = std::rand() * (1.f/RAND_MAX);
        y_host[i] = std::rand() * (1.f/RAND_MAX);
    }
    */

    auto float_rand = [] {
        return std::rand() * (1.f/RAND_MAX); 
    };
    thrust::generate(x_host.begin(), x_host.end(), float_rand);
    thrust::generate(y_host.begin(), y_host.end(), float_rand);

    thrust::device_vector<float> x_dev = x_host;
    thrust::device_vector<float> y_dev = y_host;

    thrust::for_each(x_dev.begin(), x_dev.end(), [] __device__ (float &x){
        x += 100.f;
    });

    thrust::for_each(x_dev.cbegin(), x_dev.cend(), [] __device__ (float const &x){
        printf("%f\n", x);
    });


    //  zip iterator
    /*
    thrust::for_each(
        thrust::make_zip_iterator(x_dev.begin(), y_dev.cbegin()),
        thrust::make_zip_iterator(x_dev.end(), y_dev.cend()),
        [a] __device__ (auto const &tup) {
            auto &x = thrust::GET<0>(tup);
            auto const &y = thrust::get<1>(tup);
            x = a * x + y;            
        }
    );
    */

    parallel_for<<<n/512, 128>>>(n, [a, x_dev=x_dev.data(), y_dev = y_dev.data()]
    __device__ (int i) {
        x_dev[i] = a * x_dev[i] + y_dev[i];
    });

    //cudaDeviceSynchronize();
    x_host = x_dev;

    for (int i =0; i<n; i++){
        printf("x[%d] = %f\n", i, x_host[i]);
    }

    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(1);

    for (int i = 0; i< n; i++){
        arr[i] = std::rand() % 4;
    }

    parallel_sum<<<n/128, 128>>>(sum.data(), arr.data(), n);
    cudaDeviceSynchronize();
    printf("result: %d\n", sum[0]);

    return 0;
}