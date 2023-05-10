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

#define BLOCK 16
__global__ void Conv_kernel(int m,  int k,  float *a, int lda, 
                                    int kw, int kh, float *kernel,                                    
                                    float *c, int ldc, int stride){
    
    int i, j, w, h;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    i = blockIdx.x * BLOCK + tidx;
    j = blockIdx.y * BLOCK + tidy;
    //const int sw = BLOCK + kw - 1;
    //const int sh = BLOCK + kh - 1;
    __shared__ float smm[18][18]; // todo static and dynamic memory allowcate
    smm[tidx][tidy] = A(i * stride , j * stride);   
    // border of kernel  
    if ((tidx == BLOCK -1) && i < m){
      for (int idx = 1; idx < kw; idx++){
        smm[tidx + idx][tidy] = A(i * stride + idx, j * stride);
      }
    }
    if ((tidy == BLOCK - 1) && j < k ){
      for (int idx = 1; idx < kh; idx++){
        smm[tidx][tidy + idx] = A(i * stride, j * stride + idx);
      }
    }
    __syncthreads(); 
    
    float sum = 0;
    if ( i < m && j < k){
      for (w = 0; w < kw; w++ ){
        for (h = 0; h < kh; h++){              
            //sum += A( i * stride + w, j * stride + h) * KERNEL(w, h);          
            sum += smm[tidx + w][tidy + h] * KERNEL(w, h);
        }
      } 
      C( i,j ) = sum; 
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
  dim3 grid((Wo + BLOCK - 1) / BLOCK, (Ho + BLOCK - 1)/ BLOCK);
  //printf(" %d ", (Wo + BLOCK - 1) / BLOCK);
  Conv_kernel<<<grid, block>>>(Wo, Ho, a, lda, kw, kh, kernel, c, lda, stride);


  
}