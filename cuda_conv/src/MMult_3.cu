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
  
    float sum = 0;
    if ( i < m && j < k){     
      // column major  
      for (h = 0; h < kh; h++){ 
        for (w = 0; w < kw; w++ ){
            //sum += A( i * stride + w, j * stride + h) * c_kernel[h][w];          
            sum += tex2D<float>(texObj_a, (j * stride + h)/ (float)k, (i * stride + w)/ (float)m) * c_kernel[h][w];
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


  // copy A to texture memory s
  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray_t cuArray;
  cudaMallocArray(&cuArray, &channelDesc, m, k);

  // Set pitch of the source (the width in memory in bytes of the 2D array pointed
  // to by src, including padding), we dont have any padding
  const size_t spitch = m * sizeof(float);
  // Copy data located at address h_data in host memory to device memory
  cudaMemcpy2DToArray(cuArray, 0, 0, a, spitch, m * sizeof(float),
                      k, cudaMemcpyHostToDevice);

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;
  
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // copy A to texture memory e

  dim3 block(BLOCK, BLOCK);
  dim3 grid((Wo + BLOCK - 1) / BLOCK, (Ho + BLOCK - 1)/ BLOCK);

  
  // constant memory
  cudaMemcpyToSymbol(c_kernel, kernel, kw * kh * sizeof(float));
  
  Conv_kernel<<<grid, block>>>(Wo, Ho, texObj, lda, kw, kh, kernel, c, lda, stride);  


  // Destroy texture object
  cudaDestroyTextureObject(texObj);

  // Free device memory
  cudaFreeArray(cuArray);

}