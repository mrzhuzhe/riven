/*
FMA, FLOPS = 60079558623232.000000 GFLops, Operation Time= 1.117000 msec
Success!!
*/
#include <cuda_fp16.h>
#include <helper_timer.h>
#include <cstdio>
#include <sm_61_intrinsics.h>
#include "cb.h"

// FMA numerical arithmetic function in GPU @FP32
// y = x * y + z
// in this kernel, assuming we have transposed matrix 
__global__ void fmaf_kernel(float *d_x, float *d_y, float *d_z, int size){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx_x; i < size; i += stride){
        d_z[i] = fmaf(d_x[i], d_y[i], 0.f);
    }
}

void fmaf_host(float *h_x, float *h_y, float *h_z, int size){
    #pragma omp parallel
    {
    #pragma omp for
        for (int i = 0; i < size; i++){
            h_z[i] = h_x[i] * h_y[i] + 0.f;
        }
    }
}



int main() {
    CBuffer<float> X, Y, Z;

    int size = 1 << 26;
    srand(2023);

    X.init(size, true);
    Y.init(size, true);
    Z.init(size, true);

    X.cuda();
    Y.cuda();
    Z.cuda();

    int n_threads = 256;
    int num_sms;
    int num_blocks_per_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sms, fmaf_kernel, n_threads, n_threads*sizeof(float2));
    int n_blocks = min(num_blocks_per_sms * num_sms, (size/2 + n_threads -1) / n_threads);
    

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    fmaf_kernel<<< n_blocks, n_threads, n_threads * sizeof(float) >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, size);
    
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    double elapsed_time_msed = sdkGetTimerValue(&timer);
    float ops = size  / elapsed_time_msed * 1e6;
    printf("FMA, FLOPS = %f GFLops, Operation Time= %f msec\n", ops, elapsed_time_msed);

    fmaf_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? printf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    sdkDeleteTimer(&timer);    

    return 0;
}
