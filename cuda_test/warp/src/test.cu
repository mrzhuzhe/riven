//  https://zhuanlan.zhihu.com/p/572820783 warp mask test

#include <iostream>
#define N 128

void init_data(float *data, int size){
    for (int i=0; i < size; i++ ){
        data[i] = i;
    }
}

void print_warp(float *data, int size){
    // print by warp
    int count = 0;
    for (int i=0; i < size; i++ ){        
        if (i % 32 == 0){            
            printf("\n warp %d \n", count);
            count++;
        } 
        printf(" %f ", data[i]);
    }
    printf("\n");
}

// kernel 是否可以指定返回值
__global__ void test_kernel(float *a, float *b, int n){
    int tid = threadIdx.x;
    //printf("----- tid: %d\n", tid);
    if (tid > n)
        return;
    float temp = a[tid];
    //printf(" %d-%f ", tid, temp);
    b[tid] = __all_sync(0xffffffff, temp > 32); 
}

__global__ void test_kernel2(float *a, float *b, int n){
    int tid = threadIdx.x;
    if (tid > n)
        return;
    float temp = a[tid];
    b[tid] = __any_sync(0xffffffff, temp > 32); 
}

__global__ void test_kernel3(float *a, float *b, int n){
    int tid = threadIdx.x;
    if (tid > n)
        return;
    float temp = a[tid];
    b[tid] = __uni_sync(0xffffffff, temp > 48 && temp < 53); 
}

__global__ void test_kernel4(float *a, float *b, int n){   
    int tid = threadIdx.x;
    if (tid > n)
        return;
    float temp = a[tid];
    b[tid] = __ballot_sync(0xffffffff, temp > 31  && temp < 34); 
}

__global__ void test_kernel5(float *a, float *b, int n){   
    int tid = threadIdx.x;
    if (tid > n)
        return;
    float temp = a[tid];
    b[tid] = __shfl_sync(0xffffffff, temp, 3); 
}

__global__ void test_kernel6(float *a, float *b, int n){   
    int tid = threadIdx.x;
    if (tid > n)
        return;
    float temp = a[tid];
    //b[tid] = __shfl_up_sync(0xffffffff, temp, 2); 
    b[tid] = __shfl_down_sync(0xffffffff, temp, 2); 
}

__device__ float warpSum(float val){
    for (int i = warpSize/2; i > 0; i = i / 2){
        unsigned int mask = __activemask();
        val += __shfl_down_sync(mask, val, i); 
    }
    return val;
}

__global__ void test_kernel7(float *a, float *b, int n){   
    int tid = threadIdx.x;
    int lanid = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;
    float val = a[tid];
    __shared__ float smm[32];
    val = warpSum(val);
    if (lanid == 0){
        smm[warpid] = warpSum(val); // smaller than 32 ?
    }
    __syncthreads();
    if (warpid == 0){
        b[tid] = smm[tid];
    } else {
        b[tid] = 0;
    }
}



int main(){
    const int n_blocks = 1;
    const int n_threads = N / n_blocks;
    const int size = N;
    const int m_size = size * sizeof(float);
    
    float *h_a, *h_b;    
    float *d_a, *d_b;
    int n;

    n = size;
    h_a = (float *)malloc(m_size);
    h_b = (float *)malloc(m_size);

    init_data(h_a, size);

    cudaMalloc((void **)&d_a, m_size);
    cudaMalloc((void **)&d_b, m_size);
    cudaMemcpy(d_a, h_a, m_size, cudaMemcpyHostToDevice);

    test_kernel<<<n_blocks, n_threads>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, m_size, cudaMemcpyDeviceToHost);
        
    cudaDeviceSynchronize();
    
    print_warp(h_b, size);

    // test2
    test_kernel2<<<n_blocks, n_threads>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, m_size, cudaMemcpyDeviceToHost);
        
    cudaDeviceSynchronize();
    
    print_warp(h_b, size);
    
    // test3
    test_kernel3<<<n_blocks, n_threads>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, m_size, cudaMemcpyDeviceToHost);
        
    cudaDeviceSynchronize();
    
    print_warp(h_b, size);
    
    // test4
    printf("this is ballout \n");
    test_kernel4<<<n_blocks, n_threads>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, m_size, cudaMemcpyDeviceToHost);
        
    cudaDeviceSynchronize();
    
    print_warp(h_b, size);

    // test5
    printf("\n --- shfl sync ---- ");
    test_kernel5<<<n_blocks, n_threads>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, m_size, cudaMemcpyDeviceToHost);
        
    cudaDeviceSynchronize();
    
    print_warp(h_b, size);

    // test6
    printf("\n --- shfl up sync ---- ");
    test_kernel6<<<n_blocks, n_threads>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, m_size, cudaMemcpyDeviceToHost);
        
    cudaDeviceSynchronize();
    
    print_warp(h_b, size);

    // test7

    printf("\n --- shfl up reduce ---- ");
    test_kernel7<<<n_blocks, n_threads>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, m_size, cudaMemcpyDeviceToHost);
        
    cudaDeviceSynchronize();
    
    print_warp(h_b, size);


    
    return 0;
}