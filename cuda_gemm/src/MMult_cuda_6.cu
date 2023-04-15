// 
#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
//#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * vertical Y horizon X
 */
template <int BLOCK, int STRIDE>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  const int STEP = BLOCK * STRIDE;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
     
  float *abeg = a + by * STEP * k;
  float *bbeg = b + bx * STEP;
  float *end_a = abeg + k;

  float sum[STRIDE][STRIDE] = { 0.f };

  for (float *a_ptr = abeg, *b_ptr = bbeg; a_ptr < end_a; a_ptr += STEP, b_ptr += STEP * n){
    //__shared__ __align__(16 * 1024) float ashare[STEP][STEP];
    //__shared__ __align__(16 * 1024) float bshare[STEP][STEP]; // memory align why ?
    __shared__ __align__(16) float ashare[STEP][STEP];
    __shared__ __align__(16) float bshare[STEP][STEP]; // memory align why ?

    for (int i = 0; i < STRIDE; i++){
      for (int j = 0; j < STRIDE; j++){
          ashare[ty * STRIDE + i][tx * STRIDE + j] = a_ptr[(ty * STRIDE + i)*k + tx * STRIDE + j];
          bshare[ty * STRIDE + i][tx * STRIDE + j] = b_ptr[(ty * STRIDE + i)*n + tx * STRIDE + j];
      }
    }   
    __syncthreads();
    for (int i = 0; i < STRIDE; i++){
      for (int j = 0; j < STRIDE; j++){
        for (int kk = 0; kk < STEP; kk++){
          sum[i][j] += ashare[ty * STRIDE + i][kk] * bshare[kk][tx * STRIDE + j];
        }
      }
    }
    
    __syncthreads();
  }
  
  for (int i = 0; i < STRIDE; i++){
    for (int j = 0; j < STRIDE; j++){

      const int _m = STEP * by + ty * STRIDE + i;
      const int _n = STEP * bx + tx * STRIDE + j;
      if (_n < n && _m < m){
        c[_m * n + _n] = sum[i][j]; 
      }
    }
  }
    

}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 8;
  constexpr int STRIDE = 4;
  // subm, subn, subk
  dim3 block(BLOCK, BLOCK);
  // notice
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
  //printf("# m, n grid %d %d \n", (m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}