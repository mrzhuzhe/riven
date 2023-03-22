#include <iostream>
#include <helper_timer.h>
#include <cuda_profiler_api.h>
#include "common.h"

#define BLOCK_DIM 16
#define MAX_FILTER_LENGTH 128

__global__ void conv_kernel_v1(float *d_out, float *d_in, float *d_filter, int n_r, int n_c, int filter_size)
{
    int i_x = blockDim.x * blockIdx.x + threadIdx.x;
    int i_y = blockDim.y * blockIdx.y + threadIdx.y;

    float res = 0.f;
    for (int f_row = -filter_size/2; f_row <= filter_size/2; ++f_row){
        for (int f_col = -filter_size/2; f_col <= filter_size/2; ++f_col){
            int img_r = i_y + f_row;
            int img_c = i_x + f_col;
            float img_v = (img_r >= 0 && img_r < n_r && img_c >=0 && img_c < n_c) ? 
                d_in[img_r * n_c + img_c] : 0.f;
            float filter_v = d_filter[(f_row+filter_size/2)*filter_size + f_col + filter_size/2];
            res += img_v * filter_v;
        }
    }

    d_out[i_y * n_c + i_x] = res;
}

__constant__ float c_filter[MAX_FILTER_LENGTH*MAX_FILTER_LENGTH];
__global__ void conv_kernel_v2(float *d_out, float *d_in, float *d_filter, int n_r, int n_c, int filter_size)
{
    int i_x = blockDim.x * blockIdx.x + threadIdx.x;
    int i_y = blockDim.y * blockIdx.y + threadIdx.y;

    float res = 0.f;
    for (int f_row = -filter_size/2; f_row <= filter_size/2; ++f_row){
        for (int f_col = -filter_size/2; f_col <= filter_size/2; ++f_col){
            int img_r = i_y + f_row;
            int img_c = i_x + f_col;
            float img_v = (img_r >= 0 && img_r < n_r && img_c >=0 && img_c < n_c) ? 
                d_in[img_r * n_c + img_c] : 0.f;
            float filter_v = c_filter[(f_row+filter_size/2)*filter_size + f_col + filter_size/2];
            res += img_v * filter_v;
        }
    }

    d_out[i_y * n_c + i_x] = res;
}

int main(){
    return 0;
}