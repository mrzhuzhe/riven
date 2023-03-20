/*
    FMA, FLOPS = 228261439209472.000000 GFLops, Operation Time= 0.294000 msec
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
__global__ void dp4a_kernel(char *d_x, char *d_y, int *d_z, int size){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

#if __CUDA_ARCH__ > 610
    char4 *quad_x = (char4 *)d_x;
    char4 *quad_y = (char4 *)d_y;

    for (int i = idx_x; i < size; i += stride)
        d_z[i] = __dp4a(quad_y[i], quad_x[i], 0);
#else 
    for (int i = idx_x; i < size; i += 4 * stride){
        int sum = 0;
        for (int j = 0; j < 4; j++ ){
            sum += d_y[4*i+j] * d_x[4*i+j];            
        }
        d_z[i] = sum + 0;
    }
#endif

}

void dp4a_host(char *h_x, char *h_y, int *h_z, int size)
{
    #pragma omp parallel 
    {
    #pragma omp for
        for  (int i = 0; i < size; i++){
            int sum = 0;
            for (int j=0; j<4; j++)
                sum += (int)h_y[4*i+j]*(int)h_x[4*i+j];
            h_z[i] = sum;
        }
    }
}



int main() {    
    CBuffer<char> X, Y;
    CBuffer<int> Z;

    int size = 1 << 26;
    srand(2023);

    X.init(size, true);
    Y.init(size, true);
    Z.init(size/4, true);

    X.cuda();
    Y.cuda();
    Z.cuda();

    int n_threads = 256;
    int num_sms;
    int num_blocks_per_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sms, dp4a_kernel, n_threads, n_threads*sizeof(int));
    int n_blocks = min(num_blocks_per_sms * num_sms, (size/4 + n_threads -1) / n_threads);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    dp4a_kernel<<< n_blocks, n_threads, n_threads * sizeof(int) >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, size/4);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    double elapsed_time_msed = sdkGetTimerValue(&timer);
    float ops = size  / elapsed_time_msed * 1e6;
    printf("FMA, FLOPS = %f GFLops, Operation Time= %f msec\n", ops, elapsed_time_msed);

    dp4a_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size/4);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? printf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    sdkDeleteTimer(&timer);    

    return 0;
}
