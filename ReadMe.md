# Riven 

> Parallel programing with Cuda

## Notice

1. all application is compiler with cuda Arch 80 (RTX 3090) you can change it on CMakeLists

## Applications

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