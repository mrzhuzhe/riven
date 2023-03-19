#include <cuda_fp16>
#include <helper_timer.h>
#include <cooperative_groups.h>
#include <cstdio>

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
    cudaDeviceGetAttribute

    return 0;
}
