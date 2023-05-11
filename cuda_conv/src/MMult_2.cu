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
__global__ void Conv_kernel(int m,  int k,  float *a, int lda, 
                                    int kw, int kh, float *kernel,                                    
                                    float *c, int ldc, int stride){    
    int i, j, w, h;
    i = blockIdx.x * BLOCK + threadIdx.x;
    j = blockIdx.y * BLOCK + threadIdx.y;

    //printf("\n");
    /*
    if (i == 0 && j ==0){
      printf("--- 1 %f %f %f \n", c_kernel[0][0], c_kernel[0][1], c_kernel[0][2]);
      printf("--- 2 %f %f %f \n", c_kernel[1][0], c_kernel[1][1], c_kernel[1][2]);
      printf("--- 3 %f %f %f \n", c_kernel[2][0], c_kernel[2][1], c_kernel[2][2]);
    }
    */

    float sum = 0;
    if ( i < m && j < k){     
      // column major  
      for (h = 0; h < kh; h++){ 
        for (w = 0; w < kw; w++ ){
            sum += A( i * stride + w, j * stride + h) * c_kernel[h][w];          
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
  // constant memory
  cudaMemcpyToSymbol(c_kernel, kernel, kw * kh * sizeof(float));
  /*
  float h_kernel[3][3];
  cudaMemcpyFromSymbol(h_kernel, c_kernel, kw * kh * sizeof(float));  
  printf("\n");
  printf("--- %f %f %f \n", h_kernel[0][0], h_kernel[0][1], h_kernel[0][2]);
  printf("--- %f %f %f \n", h_kernel[1][0], h_kernel[1][1], h_kernel[1][2]);
  printf("--- %f %f %f \n", h_kernel[2][0], h_kernel[2][1], h_kernel[2][2]);
  */
  Conv_kernel<<<grid, block>>>(Wo, Ho, a, lda, kw, kh, kernel, c, lda, stride);  
}