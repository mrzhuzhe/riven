# Riven 

> Parallel programing with Cuda

## Notice

1. all application is compiler with cuda Arch 80 (RTX 3090) you can change it on CMakeLists

## Applications
### Gemm

```
// x86 gemm
cd /gemm/
// nowaday the best result /MMult22_avx.c it's about 60GFlops (corresponding openBLAS is about 75GFLOPS )
// [TODO]use core-avx2 is much better than mavx
// [TODO] 8x6_avx is only about 40GFLOPS  MMult22_avx3_8x6.c
// [TODO] inline volatile seems not work

// cuda gemm
cd /cuda_gemm/
//  best result is /MMult_cuda_6_1.cu 

```


### CUDA
```
cd /cuda_test/

// subfolders with corrosponding apps

/mm  // shared and texture memory

/nn  // a neural network like caffe

/warp   // cuda concept about grid block warp and cooperate group

/pipline    // cuda stream events 

/pattern    // application like convulition


# build
cmake -S src -B build
cmake --build build


```

### Arm Neon

Arm neon  intrinsi on Mac M1

There is a bug that this cannot be build with cmake.


### Legacy

1. /RayTracing learn ray tracing in one week
2. /openmp_test 
3. /llvm llvm totorial
3. /cuda_fluid 