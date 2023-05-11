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
__global__ void Conv_kernel(int m,  int k,  cudaTextureObject_t texObj_a, int lda, 
                                    int kw, int kh, float *kernel,                                    
                                    float *c, int ldc, int stride){      
    int i, j, w, h;
    i = blockIdx.x * BLOCK + threadIdx.x;
    j = blockIdx.y * BLOCK + threadIdx.y;
    /*
    if (i == 0 && j == 0){
      for (h = 0; h < kh; h++){ 
        for (w = 0; w < kw; w++ ){
            printf(" %f ", tex2D<float>(texObj_a, (i * stride + w), (j * stride + h)));                   
        }
        printf("\n");
      }       
    }
    */
    float sum = 0;
    if ( i < m && j < k){     
      // column major  
      for (h = 0; h < kh; h++){ 
        for (w = 0; w < kw; w++ ){
            //sum += A( i * stride + w, j * stride + h) * c_kernel[h][w];     
            // TODO col major ?     
            sum += tex2D<float>(texObj_a, (i * stride + w), (j * stride + h)) * c_kernel[h][w];
        }
      } 
      C( i,j ) = sum; 
    }
}

void MY_MMult( int m,  int k,  cudaTextureObject_t texObj_a, int lda, 
                                    int kw, int kh, float *kernel,                                    
                                    float *c, int ldc, int stride )
{
  //  multi channel ? multi batch ?
  //  img2col how to do img2features how to map result back
  
  int Wo = (m - kw) / stride + 1;
  int Ho = (k - kh) / stride + 1;

  // copy A to texture memory e

  dim3 block(BLOCK, BLOCK);
  //dim3 grid((Wo + BLOCK - 1) / BLOCK, (Ho + BLOCK - 1)/ BLOCK);
  dim3 grid((Wo + BLOCK - 1) / BLOCK, (Ho + BLOCK - 1)/ BLOCK);

  
  // constant memory
  cudaMemcpyToSymbol(c_kernel, kernel, kw * kh * sizeof(float));
  
  //Conv_kernel<<<grid, block>>>(Wo, Ho, texObj, lda, kw, kh, kernel, c, lda, stride);  
  Conv_kernel<<<grid, block>>>(Wo, Ho, texObj_a, lda, kw, kh, kernel, c, lda, stride);  

}