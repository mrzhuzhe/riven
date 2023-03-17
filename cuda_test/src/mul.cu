//  ======== Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
//                 Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
//                  Refer https://developer.nvidia.com/tools-overview for more details.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <helper_functions.h> // for benchmark purpose

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

__global__ void sgemm_gpu_kernel(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.f;
    for (int  i = 0; i < K; ++i){
        sum += A[row*K+i] * B[i*K+col];
    }

    C[row*M+col] = alpha * sum + beta * C[row*M + col];
}

void sgemm_gpu(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 dimGrid(M/dimBlock.x, N/dimBlock.y);
    sgemm_gpu_kernel<<<dimGrid, dimBlock>>>(A, B, C, N, M, K, alpha, beta); 
}


void performance_estimation(void(*sgemm)(const float *, const float *, float *, int, int, int, float, float),
const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    int test_iterations = 100;

    StopWatchInterface *timer = 0;
    
    // 
    sgemm(A, B, C, N, M, K, alpha, beta);

    //
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    //  
    for (int i = 0; i< test_iterations; i++){
        sgemm(A, B, C, N, M, K, alpha, beta);
    }
    
    //
    sdkStopTimer(&timer);

    //
    float operation_time = sdkGetAverageTimerValue(&timer);
    float operation_time_1_epoch = operation_time / test_iterations;

    printf("Operation Time = %.4f msec\n", operation_time_1_epoch);

    sdkDeleteTimer(&timer);
}


void random_init(float *data, int size)
{
    for (int i = 0; i<size; ++i){
        data[i] = (float)i;
        //data[i] = 1.f;
        //data[i] = (rand() & 0xFF) / (float)RAND_MAX;       
    }    
}

void print_output(float *a, float *b, float *c, int m, int n, int k) {
    int count = 10;
    int begin = 100;
    for (int idx=begin;idx<begin+count;idx++){
        int col = idx % n; // mod
        int row = idx / n; // residual
        printf("%f =", c[idx]);
        for (int i = 0; i < k; i++){
            if (i > 0)
                printf(" +");
            printf(" %f * %f", a[row*k+i], b[i*k+col]);
        }
        printf("\n");
    }
}



int main()
{
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    int N, M, K;
    float alpha = 1.f;
    float beta = 0.f;
    //  N = M = K = 2048 * 2048;    // out of range
    N = M = K = 128; 
    
    //
    A = (float *)malloc(N * K *sizeof(float));
    B = (float *)malloc(K * M *sizeof(float));
    C = (float *)malloc(N * M *sizeof(float));

    //
    cudaMalloc((void **)&d_A, N * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * M * sizeof(float));
    cudaMalloc((void **)&d_C, N * M * sizeof(float));

    //    
    random_init(A, N * K);
    random_init(B, K * M);
    //random_init(C, N * M);

       
    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice);

    //
    performance_estimation(sgemm_gpu, d_A, d_B, d_C, N, M, K, alpha, beta);

    cudaMemcpy(C, d_C, N * M *sizeof(float), cudaMemcpyDeviceToHost);

    //print_output(A, B, C, M, N, K);    
    
    //
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //
    free(A);
    free(B);
    free(C);

    return 0;
}