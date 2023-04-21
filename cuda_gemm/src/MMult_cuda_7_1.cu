#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <vector>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define SMEM_LDA (128)
#define SMEM_LDB (128)

// remove original guard
__device__ __forceinline__ void ldg32_nc_0(float &reg, const void *ptr) {
  asm volatile("{.reg .pred p;\n"
               "mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 &&                 \
    __CUDA_ARCH__ >= 750
               "ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
               "ld.global.nc.f32 %0, [%1];}\n"
#endif
               : "=f"(reg)
               : "l"(ptr));
}

__device__ __forceinline__ uint32_t smem_u32addr(const void *smem_ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(smem_ptr));

  return addr;
}

__device__ __forceinline__ void lds128(float &reg0, float &reg1, float &reg2,
                                       float &reg3, const uint32_t &addr) {
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
               : "r"(addr));
}

__device__ __forceinline__ void sts128(const float &reg0, const float &reg1,
                                       const float &reg2, const float &reg3,
                                       const uint32_t &addr) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ void sts32(const float &reg, const uint32_t &addr) {
  asm volatile("st.shared.f32 [%0], %1;\n" : : "r"(addr), "f"(reg));
}

// 256 threads  perblock, 2 blocks per multiprocessor
/**
 * version 10 相对于 version 9 的特点是
 * 1. 用了 uint32_t 代替 64bit 的 smem 地址，然后用 lds128 来加载数据，3080
 * 上实跑没卵用。
 * 2. gmem 到 smem 时，用 reg 做搬运, 3080/Tesla T4  实跑都没有卵用。
 */
__global__ __launch_bounds__(256, 2) void sgemm_128x128x8(int m, int n, int k,
                                                          const float *a,
                                                          const float *b,
                                                          float *c) {

  __shared__ __align__(
      16 * 1024) char smem[24 * 1024]; // 16KB shared memory for buffer
  float *ashare = reinterpret_cast<float *>(smem);
  float *bshare =
      reinterpret_cast<float *>(smem + 16 * 1024); // 8k shared mem for B
  float sum[8][8] = {0};
  float panelA[8] = {0}, panelB[8] = {0};

  int from_a = (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8;
  int from_b = (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32;

  float a_ldg_reg[4], b_ldg_reg[4];

  uint32_t a_sts_addr = smem_u32addr(ashare + (threadIdx.x % 8) * SMEM_LDA +
                                     (threadIdx.x / 8) * 4);
  uint32_t b_sts_addr =
      smem_u32addr(bshare + (threadIdx.x / 32) * SMEM_LDB + (threadIdx.x % 32));

  uint32_t aptr_base = smem_u32addr(ashare + (threadIdx.x / 16) * 4);
  uint32_t bptr_base = smem_u32addr(bshare + (threadIdx.x % 16) * 4);

  for (int loop = 0; loop < k; loop += 8) {
// part1: gmem to smem
// load gmem to smem for ashare
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldg32_nc_0(a_ldg_reg[i],
                 (const char *)(a + from_a) + i * k * sizeof(float));
    }
    sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr);

// load gmem to smem for bshare
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldg32_nc_0(b_ldg_reg[i],
                 (const char *)(b + from_b) + i * 32 * sizeof(float));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts32(b_ldg_reg[i], b_sts_addr + i * 32 * sizeof(float));
    }

    __syncthreads();
    from_a += 8;
    from_b += 8 * n;

// part2: calculation
// 计算 2x2 个 4x4
#pragma unroll
    for (int subk = 0; subk < 8; ++subk) {
      lds128(panelA[0], panelA[1], panelA[2], panelA[3],
             aptr_base + (subk * SMEM_LDA) * sizeof(float));
      lds128(panelA[4], panelA[5], panelA[6], panelA[7],
             aptr_base + (subk * SMEM_LDA + 64) * sizeof(float));

      lds128(panelB[0], panelB[1], panelB[2], panelB[3],
             bptr_base + (subk * SMEM_LDB) * sizeof(float));
      lds128(panelB[4], panelB[5], panelB[6], panelB[7],
             bptr_base + (subk * SMEM_LDB + 64) * sizeof(float));

#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sum[i][j] += panelA[i] * panelB[j];
        }
      }
    }
    __syncthreads();
  }

  // part3: save to C
  int write_offset = (blockIdx.y * 128 + (threadIdx.x / 16) * 4) * n +
                     blockIdx.x * 128 + (threadIdx.x % 16) * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      c[write_offset + i * n + j] = sum[i][j];
      c[write_offset + i * n + j + 64] = sum[i][j + 4];
      c[write_offset + (i + 64) * n + j] = sum[i + 4][j];
      c[write_offset + (i + 64) * n + j + 64] = sum[i + 4][j + 4];
    }
  }
}

#undef SMEM_LDA
#undef SMEM_LDB

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 128;
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm_128x128x8<<<grid, 256>>>(m, n, k, d_A, d_B, d_C);
}