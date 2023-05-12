#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA and CUBLAS functions
#define A(i,j) a[ (j)*lda + (i) ]
#define C(i,j) c[ (j)*lda + (i) ]
#define KERNEL(i,j) kernel[ (j)*kw + (i) ]

__constant__ float c_kernel[3][3];

#define BLOCK 16
#define STRIDES 2

__global__ void Conv_kernel(int m,  int k,  float *a, int lda, 
                                    int kw, int kh, float *kernel,                                    
                                    float *c, int ldc, int stride){    
    int i, j, w, h;
    int s_i, s_j;
    i = (blockIdx.x * BLOCK + threadIdx.x) * STRIDES;
    j = (blockIdx.y * BLOCK + threadIdx.y) * STRIDES;
    float *a_ptr = &A( i * stride, j * stride);
    float *c_ptr = &C( i * stride, j * stride);
    
    for (int idy = 0; idy < STRIDES; idy++){
      a_ptr += idy * lda;
      c_ptr += idy * lda;
      for (int idx = 0; idx < STRIDES; idx++){
        float sum = 0;
        s_i = i + idx;
        s_j = j + idy;    
        if ( s_i < m && s_j < k){     
          // column major  
          for (h = 0; h < kh; h++){ 
            for (w = 0; w < kw; w++ ){                
                sum += *(a_ptr + idx + h * lda + w ) * c_kernel[h][w];          
            }
          } 
          *(c_ptr + idx) = sum; 
        }
      
      }   
    }     
}

void MY_MMult( int m,  int k,  float *a, int lda, 
                                    int kw, int kh, float *kernel,                                    
                                    float *c, int ldc, int stride )
{
  //  multi channel ? multi batch ?
  //  img2col how to do img2features how to map result back
  
  int Wo = (m - kw) / stride + 1;
  int Ho = (k - kh) / stride + 1;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((Wo + BLOCK - 1) / BLOCK / STRIDES, (Ho + BLOCK - 1)/ BLOCK / STRIDES);

  // constant memory
  cudaMemcpyToSymbol(c_kernel, kernel, kw * kh * sizeof(float));
  
  Conv_kernel<<<grid, block>>>(Wo, Ho, a, lda, kw, kh, kernel, c, lda, stride);  
}