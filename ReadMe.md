# Riven 

> Parallel programing with Cuda

## Clone with submodule
> git clone --recursive git@github.com:mrzhuzhe/riven.git

Submodules list:

- Eigen

## Notice

1. all application is compiler with cuda Arch 80 (RTX 3090) you can change it on CMakeLists

## Applications


### Linear equation Solver

```
cd solver

# build
cmake -S src -B build
cmake --build build

# test_case
# 1. simple guassion elimination solver (as same as middle school)
test_simp

# 2. LU factor, PLU factor, PLU linear equation solver 
test_lu

# 3. power method for eigen value, household process for simillar matrix, qr factor for eigen values
test_eigen

# 4. jacobian iteration, guassion_seidel iteration, multi grid method, conjugate gradient
test_iteration 32 1

# 5. conjugate gradient, biconjugate gradient, preconditioner(jacobian and Incomplete_Cholesky_factorization) conjugate gradient, GMRES, biconjugate gradient stablized
test_cg 32 1

```


### Gemm

both for CPU and CUDA

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




### Legacy

1. /RayTracing learn ray tracing in one week
2. /openmp_test 
3. /llvm llvm totorial
3. /cuda_fluid 