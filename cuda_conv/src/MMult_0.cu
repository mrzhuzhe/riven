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
    //for ( i=0; i< Wo; i+=1 ){
    //  for ( j=0; j< Ho; j+=1 ){
        int i, j, w, h;
        i = blockIdx.x * BLOCK + threadIdx.x;
        j = blockIdx.y * BLOCK + threadIdx.y;
        for (w = 0; w < kw; w++ ){
          for (h = 0; h < kh; h++){
            C( i,j ) = C( i,j ) + A( i * stride + w, j * stride + h) * KERNEL(w, h);          
          }
        }  
    //  }
    //}
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
  dim3 grid(Wo / BLOCK, Ho / BLOCK);
  Conv_kernel<<<grid, block>>>(m, k, a, lda, kw, kh, kernel, c, lda, stride);


  
}