#include <cuda_fp16>
#include <helper_timer.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <sm_61_intrinsics.h>


__global__ void hfma_kernel(char *d_x, char *d_y, int *d_z, int size){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    half2 *dual_x = reinterpret_cast<half2*>(d_x);
    half2 *dual_y = reinterpret_cast<half2*>(d_y);
    float2 *dual_z = reinterpret_cast<float2*>(d_z);

    extern __shared__ float2 s_data[];

#if __CUDA_ARCH__ > 530
    for (int i = idx_x; i < size; i += stride){
        dual_z[i] = __half22float2(dual_x[i]), dual_y[i]);
    }
#else 
    for (int i = idx_x; i < size; i += stride){
        dual_z[i] = __half22float2(dual_x[i]) * __half22float2(dual_y[i]);
    }
#endif

}


void fhma_host(float *h_x, float *h_y, float *h_z, int size){
    #pragma omp parallel
    {
    #pragma omp for
        for (int i = 0; i < size; i++){
            h_z[i] = __half2float(h_x[i]) * __half2float(h_y[i]);
        }
    }
}


int main() {

    CBuffer<half> X, Y;
    CBuffer<float> Z;

    int size = 1 << 26;
    srand(2019);

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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sms, hfma_kernel, n_threads, n_threads*sizeof(float2));
    int n_blocks = min(num_blocks_per_sms * num_sms, (size/2 + n_threads -1) / n_threads);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    hfma_kernel<<< n_blocks, n_threads, n_threads * sizeof(float2) >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, size/2);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    double elapsed_time_msed = sdkGetTimerValue(&timer);
    float ops = (float)size  / elapsed_time_msed * 1e6;
    printf("FMA, FLOPS = %f GFLops, Operation Time= %f msec\n", ops, elapsed_time_msed);

    fhma_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? prinf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    sdkDeleteTimer(&timer);    

    return 0;
}
