#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
//#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * vertical Y horizon X
 */
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  const int _m = blockIdx.x * BLOCK + threadIdx.x;
  const int _n = blockIdx.y * BLOCK + threadIdx.y;
   
  float *abeg = a + by * BLOCK * k;
  float *bbeg = b + bx * BLOCK;
  float *end_a = abeg + k;

  float sum = 0.f;

  for (float *a_ptr = abeg, *b_ptr = bbeg; a_ptr < end_a; a_ptr += BLOCK, b_ptr += BLOCK * n){
    __shared__ float ashare[BLOCK][BLOCK];
    __shared__ float bshare[BLOCK][BLOCK];

    //printf("%d %d %d %d\n", _m, _n, m, n);
    //  [TODO] seems something went wrong
    //ashare[ty][tx] = (_m < k && _n < m) ? a_ptr[ty*k + tx] : 0.0;  // this should be 
    //bshare[ty][tx] = (_m < n && _n < k) ? b_ptr[ty*n + tx] : 0.0;

    ashare[ty][tx] = a_ptr[ty*k + tx];  // this should be wrong 
    bshare[ty][tx] = b_ptr[ty*n + tx];
    //if (ty == 15 ){
    //if (_n == m && _m == m) {
      //printf("%f %f %d %d %d \n", a_ptr[ty*k + tx], b_ptr[ty*n + tx], m, n, k);
    //printf("%f %f %d %d \n", a_ptr[ty*k + tx], b_ptr[ty*n + tx], tx, ty);
    //}
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BLOCK; kk++){
      sum += ashare[ty][kk] * bshare[kk][tx];
    }
    __syncthreads();
  }
  if (_n < n && _m < m){
    c[_n * n + _m] = sum; 
  } 
  //else {
    //printf(" %d %d %d %d %d \n", _m, _n, m ,n, k);
  //}
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  // subm, subn, subk
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  //printf("# m, n grid %d %d \n", (m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}