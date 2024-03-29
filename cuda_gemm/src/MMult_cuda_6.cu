#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cuda_runtime.h>

#define BLOCKSIZE 32
#define SMEM_LDA (BLOCKSIZE)
#define SMEM_LDB (BLOCKSIZE)

__global__ __launch_bounds__(256, 2) void sgemm_32x32x8(int m, int n, int k,
                                                          const float *a,
                                                          const float *b,
                                                          float *c) {

  __shared__ __align__(
      16 * 1024) char smem[24 * 1024]; // 16KB shared memory for buffer
  float *ashare = reinterpret_cast<float *>(smem);
  float *bshare =
      reinterpret_cast<float *>(smem + 16 * 1024); // 8k shared mem for B
  float sum[2][2] = {0};
  float panelA[2] = {0}, panelB[2] = {0};

  int from_a = (blockIdx.y * 32 + threadIdx.x / 8) * k + threadIdx.x % 8;
  int from_b = (threadIdx.x / 32) * n + blockIdx.x * 32 + threadIdx.x % 32;

  for (int loop = 0; loop < k; loop += 8) {
    // part1: gmem to smem
    // load gmem to smem for ashare
    int to_a = (threadIdx.x % 8) * SMEM_LDA +
               (threadIdx.x / 8); // 连续的地址不能给同一个 thread 用

    /*
    for (int i = 0; i < 4; ++i) {
      ashare[to_a + i] = a[from_a + i * k];
    }
    */
    ashare[to_a] = a[from_a];

    // load gmem to smem for bshare
    int to_b = (threadIdx.x / 32) * SMEM_LDB + (threadIdx.x % 32);

    /*
    for (int i = 0; i < 4; ++i) {
      bshare[to_b + i * 32] =
          b[from_b + i * 32]; // 32 thread 合并访问。 thread i 访问  [i, i+32,
                              // i+64, i+96]
    }
    */
    bshare[to_b] = b[from_b];

    __syncthreads();
    from_a += 8;
    from_b += 8 * n;

    // part2: calculation
    // 计算 2x2 个 4x4
    int aidx0 = (threadIdx.x / 16);
    int bidx0 = (threadIdx.x % 16);

    for (int subk = 0; subk < 8; ++subk) {
      float *ptrA = ashare + aidx0 + subk * SMEM_LDA;

      /*
      for (int i = 0; i < 4; ++i) {
        panelA[i] = ptrA[i];
        panelA[i + 4] = ptrA[i + 16 * 4];
      }      
      */
      panelA[0] = ptrA[0];
      panelA[1] = ptrA[0 + 16];
     
      const float *ptrB = bshare + bidx0 + subk * SMEM_LDB;

      /*
      for (int i = 0; i < 4; ++i) {
        panelB[i] = ptrB[i];
        panelB[i + 4] = ptrB[i + 16 * 4];
      }
      */
      panelB[0] = ptrB[0];
      panelB[1] = ptrB[0 + 16];

      
      
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          sum[i][j] += panelA[i] * panelB[j];
        }
      }          
    }
    __syncthreads();
  }

  // part3: save to C
  int write_offset = (blockIdx.y * 32 + (threadIdx.x / 16) ) * n +
                     blockIdx.x * 32 + (threadIdx.x % 16) ;
  /*
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      c[write_offset + i * n + j] = sum[i][j];     
    }
  }
  */
  /*
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      c[write_offset + i * n + j] = sum[i][j];
      c[write_offset + i * n + j + 64] = sum[i][j + 4];
      c[write_offset + (i + 64) * n + j] = sum[i + 4][j];
      c[write_offset + (i + 64) * n + j + 64] = sum[i + 4][j + 4];
    }
  }
  */
  
  c[write_offset] = sum[0][0];
  c[write_offset + 16] = sum[0][0 + 1];
  c[write_offset + (16) * n ] = sum[0 + 1][0];
  c[write_offset + (16) * n + 16] = sum[0 + 1][0 + 1];

}

#undef SMEM_LDA
#undef SMEM_LDB

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = BLOCKSIZE;
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm_32x32x8<<<grid, 256>>>(m, n, k, d_A, d_B, d_C);
}