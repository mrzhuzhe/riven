#include <cuda_fp16>
#include <helper_timer.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <sm_61_intrinsics.h>

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


void fmaf_host(float *h_x, float *h_y, float *h_z, int size){
    #pragma omp parallel
    {
    #pragma omp for
        for (int i = 0; i < size; i++){
            h_z[i] = h_x[i] * h_y[i] + 0.f;
        }
    }
}

void dp4a_host(char *h_x, char *h_y, char *h_z, int size)
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
    //CBuffer<float> X, Y, Z;

    CBuffer<char> X, Y;
    CBuffer<int> Z;

    int size = 1 << 26;
    srand(2019);

    X.init(size, true);
    Y.init(size, true);
    //Z.init(size, true);
    Z.init(size/4, true);

    X.cuda();
    Y.cuda();
    Z.cuda();

    int n_threads = 256;
    int num_sms;
    int num_blocks_per_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sms, fmaf_kernel, n_threads, n_threads*sizeof(float2));
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sms, dp4a_kernel, n_threads, n_threads*sizeof(int));
    //int n_blocks = min(num_blocks_per_sms * num_sms, (size/2 + n_threads -1) / n_threads);
    int n_blocks = min(num_blocks_per_sms * num_sms, (size/4 + n_threads -1) / n_threads);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    //fmaf_kernel<<< n_blocks, n_threads, n_threads * sizeof(float) >>>(X.d_ptr, Y.d_ptr, Z.d_ptr, size);
    dp4a_kernel<<< n_blocks, n_threads, n_threads * sizeof(int) >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, size/4);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    double elapsed_time_msed = sdkGetTimerValue(&timer);
    float ops = size  / elapsed_time_msed * 1e6;
    printf("FMA, FLOPS = %f GFLops, Operation Time= %f msec\n", ops, elapsed_time_msed);

    //fmaf_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size);
    dp4a_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size/4);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? prinf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    sdkDeleteTimer(&timer);    

    return 0;
}
