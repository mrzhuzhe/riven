//  https://zhuanlan.zhihu.com/p/572820783 warp mask test

#include <iostream>
#define N 128

// kernel 是否可以指定返回值
__global__ void test_kernel(float *a, float *b, int n){
    int tid = threadIdx.x;
    //printf("----- tid: %d\n", tid);
    if (tid > n)
        return;
    float temp = a[tid];
    printf(" %d-%f ", tid, temp);
    b[tid] = __all_sync(0xffffffff, temp > 48); 
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

    for (int i=0; i < size; i++ ){
        h_a[i] = i;
    }

    cudaMalloc((void **)&d_a, m_size);
    cudaMalloc((void **)&d_b, m_size);
    cudaMemcpy(d_a, h_a, m_size, cudaMemcpyHostToDevice);

    test_kernel<<<n_blocks, n_threads>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, m_size, cudaMemcpyDeviceToHost);
    printf("\n");

    for (int i=0; i < size; i++ ){
        printf(" %f ", h_b[i]);
        if ((i+1) % 10 == 0){
            printf("\n");
        } 
    }
    printf("\n");
    cudaDeviceSynchronize();

    return 0;
}