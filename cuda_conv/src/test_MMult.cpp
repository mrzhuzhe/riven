#include <stdio.h>
#include "parameters.h"
#include <stdlib.h>
#include <string.h>

// CUDA runtime
#include "helper.h"
#include <cuda_runtime.h>

void REF_MMult_CPU(int, int, float *, int, int, int, float *, float *, int, int );
void REF_MMult_GPU(int, int, float *, int, int, int, float *, float *, int, int );
void MY_MMult(int, int, float *, int, int, int, float *, float *, int, int );
void random_matrix(int, int, float *, int, int);
float compare_matrices( int, int, float *, int, float *, int );
void print_matrix( int, int, float *, int );

double dclock();

int main() {
  const int debug = 0;
  // print gpu info
  cudaDeviceProp deviceProp;
  int devID = 0;
  checkCudaErrors(cudaSetDevice(devID));
  auto error = cudaGetDeviceProperties(&deviceProp, devID);
  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }
  printf("#GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
         deviceProp.name, deviceProp.major, deviceProp.minor);
  printf("# maxGridSize: %n maxBlocksPerMultiProcessor: %d maxThreadsPerBlock: %d \n", deviceProp.maxGridSize, deviceProp.maxBlocksPerMultiProcessor, deviceProp.maxThreadsPerBlock);
  
  int p, 
  m, k, 
  kw, kh,
  rep;

  double dtime, dtime_best, gflops;
  float diff;

  float *a, *c, *cref, *cold, *kernel;

  printf("MY_MMult = [\n");

  kw = 3; 
  kh = 3;
  const int padding = 1;
  const int stride = 1;

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));
  // checkCudaErrors(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  /* Time the "optimized" implementation */
  cudaEvent_t start, stop;
  // Allocate CUDA events that we'll use for timing
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // printf( "create Handle\n");

  for (p = PFIRST; p <= PLAST; p += PINC) {
    if (debug){
      p = 1024;
    }

    m = (M == -1 ? p : M);
    k = (K == -1 ? p : K);

    gflops = 2.0 * m  * k * 1.0e-09;

    const int lda = k;

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_C = m * k * sizeof(float);
    const size_t mem_size_kernel = kw * kh * sizeof(float);
    a = (float *)malloc(mem_size_A);
    kernel = (float *)malloc(mem_size_kernel);
    c = (float *)malloc(mem_size_C);
    cold = (float *)malloc(mem_size_C);
    cref = (float *)malloc(mem_size_C);

    /* Generate random matrices A, B, Cold */
    random_matrix( m, k, a, lda, padding);
    random_matrix( kw, kh, kernel, kw, 0);

    memset(cold, 0, mem_size_C);
    memset(cref, 0, mem_size_C);
    
    //printf("%d %d %d \n", m, n, k);

    /* Init device matrix*/
    float *d_A, *d_C, *d_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMemcpy(d_A, a, mem_size_A, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_kernel, mem_size_kernel));
    checkCudaErrors(cudaMemcpy(d_kernel, kernel, mem_size_kernel, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

    /* Run the reference implementation so the answers can be compared */
    // printf( "init\n");
    if (debug){

      REF_MMult_CPU( m, k, a, lda, kw, kh, kernel, cref, lda, stride );

      MY_MMult( m, k, d_A, lda, kw, kh, d_kernel, d_C, lda, stride);
      
      checkCudaErrors(cudaMemcpy(cold, d_C, mem_size_C, cudaMemcpyDeviceToHost));

      //print_matrix(m, k, a, lda);
      print_matrix(kw, kh, kernel, kw);
      
      print_matrix(m, k, cref, lda);
      printf("\n");
      print_matrix(m, k, cold, lda);
      
      diff = compare_matrices(m, k, cold, lda, cref, lda);

      free( a );
      free( c );
      free( cold );
      free( cref );

      checkCudaErrors(cudaFree(d_A));
      checkCudaErrors(cudaFree(d_kernel));
      checkCudaErrors(cudaFree(d_C));

      return 0;
    }    

    REF_MMult_CPU( m, k, a, lda, kw, kh, kernel, cref, lda, stride );
    
    //REF_MMult_GPU( m, k, d_A, lda, kw, kh, d_kernel, d_C, lda, stride);
    //checkCudaErrors(cudaMemcpy(cref, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    // printf( "benchmark\n");

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    for (rep = 0; rep < NREPEATS; rep++) {      
      /* Time your implementation */
      MY_MMult( m, k, d_A, lda, kw, kh, d_kernel, d_C, lda, stride);
    }

    // printf( "mymmult\n");

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / NREPEATS;
    double flopsPerMatrixMul = 5.0 * m * k;
    double gflops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1e4f);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(cold, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    diff = compare_matrices(m, k, cold, lda, cref, lda);
    //printf("\n # ------ %f \n ", diff);

    /*
    if (diff > 0.5f || diff < -0.5f) {
      printf("diff too big !\n");
      exit(-1);
    }
    */

    printf("%d %.2f %le %f \n", p, gflops, diff, msecPerMatrixMul);

    free(a);
    free(c);
    free(cold);
    free(cref);

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_kernel));
    checkCudaErrors(cudaFree(d_C));
  }

  // Destroy the handle
  checkCudaErrors(cublasDestroy(handle));

  printf("];\n");
  return 0;
}