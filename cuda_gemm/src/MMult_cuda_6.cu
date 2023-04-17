// 
#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
//#include <cublas_v2.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_MP     2

#define SMEM_LDA (128)
#define SMEM_LDB (128)

#define BLOCK_DIM 128

/**
 * vertical Y horizon X
 */
//  __launch_bounds__ https://stackoverflow.com/questions/44704506/limiting-register-usage-in-cuda-launch-bounds-vs-maxrregcount
template <int BLOCK, int STRIDE>
__global__ void 
//__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP) // seems of no use
sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  __shared__ __align__(16) char smem[(8 + 16) * 1024];
  float *ashare = reinterpret_cast<float *>(smem);
  float *bshare = reinterpret_cast<float *>(smem + 16 * 1024);  // move pointer ?
  
  float sum[STRIDE][STRIDE] = {0};
  float panelA[STRIDE] = {0}, panelB[STRIDE] = {0};

  const int warpSize = 32;
  const int warpId = threadIdx.x / warpSize;
  const int warpLane = threadIdx.x % warpSize;
  const int roll = 1;
  int from_a = (blockIdx.y * warpSize * roll + threadIdx.x / 8 * roll ) * k + threadIdx.x % 8;
  int from_b = (warpId * n + blockIdx.x *  warpSize * roll + warpLane);

  for (int loop = 0; loop < k; loop += STRIDE){
    int to_a = (threadIdx.x % 8) * warpSize * roll + (threadIdx.x % 8) * roll;
    ashare[to_a] = a[from_a];
        
    int to_b = warpId * warpSize * roll + warpLane;
    bshare[to_b] = b[from_b];
  
    __syncthreads();
    from_a += STRIDE;
    from_b += STRIDE * n;
  

    int aidx0 = (threadIdx.x / 16) * roll;
    int bidx0 = (threadIdx.x % 16) * roll;
    
    for (int subk = 0; subk < 8; subk++){
      float *ptrA = ashare + aidx0 + subk * warpSize * roll;
      panelA[i] = ptrA[i];
      panelA[i] = ptrA[i + 16];
      
      const float *ptrB = bshare + bidx0 +subk * warpSize * roll;
      panelB[i] = ptrB[i];
      panelB[i] = ptrB[i + 16];

    

      for (int i = 0; i< 8; i++){
        for (int j = 0; j<8; j++){
          sum[i][j] += panelA[i] * panelB[j];
        }
      }
    }
    __syncthreads();
  }

  int write_offset = (blockIdx.y * warpSize * roll + threadIdx.x / 16) * n + blockIdx.x * warpSize * roll + threadIdx.x % 16;

  c[]


}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = BLOCK_DIM;
  constexpr int STRIDE = 8;
 
  dim3 block(32 * STRIDE);  
  dim3 grid((m + BLOCK - 1) / BLOCK , (n + BLOCK - 1) / BLOCK );
  sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}