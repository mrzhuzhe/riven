#include <iostream>
#include <helper_timer.h>
#include <cuda_profiler_api.h>
#include "common.h"
#include "cuda_runtime.h"

#define BLOCK_DIM 16


__global__ void sgemm_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    float _c = 0.f;
    for (int i =0; i < K; i++){
        _c += A[row*K +i] * B[i*K+col];
    }

    C[row*N+col] = alpha * _c + beta * C[row*N + col];
}


__global__ void sgemm_memory(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta){
    int bid_x = blockIdx.x * blockDim.x;
    int bid_y = blockIdx.y * blockDim.y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
   
    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];
    float _c = 0.f;
    for (int k = 0; k < K; k += BLOCK_DIM){        
        /*
        s_tile_A[tid_y][tid_x] = A[(bid_y + tid_y)*K + tid_x + k];
        s_tile_B[tid_y][tid_x] = B[(k*BLOCK_DIM+tid_y)*N + bid_x + tid_x];
        */
        s_tile_A[tid_y][tid_x] = A[(bid_y + tid_y)*K + tid_x + k];
        s_tile_B[tid_y][tid_x] = B[(k+tid_y)*N + bid_x + tid_x];

        //printf(" 1 %d %f %d %d", k, _c, (bid_y + tid_y)*K + tid_x + k, (k+tid_y)*N + bid_x + tid_x);
        __syncthreads();
        
        for (int e =0; e < BLOCK_DIM; e++){
            _c += s_tile_A[tid_y][e] * s_tile_B[e][tid_x];
            //_c[e] = s_tile_A[tid_y][e] * s_tile_B[e][tid_x];
            //printf("111 %f ", _c);
            //printf("%f ", _mid);
        }
        __syncthreads();
    }
    
    //printf("%f ", _c);
    C[(bid_y + tid_y)*N + bid_x + tid_x] = alpha * _c + beta * C[(bid_y + tid_y)*N + bid_x + tid_x];
}

void sgemm(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta){
    for (int row = 0; row < M; row ++){
        for (int col =0; col < N; col++){
            float _c = 0.f;
            for (int e = 0; e < K; e++){
                _c += A[row*K + e] * B[e*N + col];
            }
            C[row*N+col] = alpha * _c + beta * C[row*N + col];
        }
    }
}

int main(){
    float *A, *B, *C_host, *C_gpu;
    float *d_A, *d_B, *d_C;
    int M, N, K;
    float alpha = 1.f;
    float beta = 0.f;
    int n_iter = 1;
    M = N = K = 2048;

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    A = (float *)malloc(M*K*sizeof(float));
    B = (float *)malloc(K*N*sizeof(float));
    C_host = (float *)malloc(M*N*sizeof(float));
    C_gpu = (float *)malloc(M*N*sizeof(float));

    cudaMalloc((void **)&d_A, M*K*sizeof(float));
    cudaMalloc((void **)&d_B, N*K*sizeof(float));
    cudaMalloc((void **)&d_C, M*N*sizeof(float));

    random_init(A, M*K);
    random_init(B, K*N);

   

    cudaMemcpy((void **)d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)d_B, B, N*K*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM -1)/BLOCK_DIM, (M + BLOCK_DIM -1)/BLOCK_DIM);

    cudaProfilerStart();

    sdkStartTimer(&timer);
    
    for (int i = 0; i < n_iter; i ++ ){
        sgemm_kernel<<< gridDim, blockDim >>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    
    sdkStopTimer(&timer);
    double elapsed_time_msed = sdkGetTimerValue(&timer);
    printf("Time= %f msec\n", elapsed_time_msed);

    sdkResetTimer(&timer);

    for (int i = 0; i < n_iter; i ++ ){
        sgemm_memory<<< gridDim, blockDim >>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    
    sdkStopTimer(&timer);
    elapsed_time_msed = sdkGetTimerValue(&timer);
    printf("Time= %f msec \n", elapsed_time_msed);

    cudaProfilerStop();

    cudaDeviceSynchronize();

    cudaMemcpy(C_gpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    sgemm(A, B, C_host, M, N, K, alpha, beta);

    if (value_test(C_host, C_gpu, M*N)) {
        printf("ok\n");
    } else {
        printf("bad\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C_host);
    free(C_gpu);

    return 0;
}