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


__global__ void conv_kernel_v3(float *d_out, float *d_in, float *d_filter, int n_r, int n_c, int filter_size)
{
    int i_x = blockDim.x * blockIdx.x + threadIdx.x;
    int i_y = blockDim.y * blockIdx.y + threadIdx.y;

    int pad_size = filter_size / 2;
    int tile_size = BLOCK_DIM + 2 * pad_size;

    extern __shared__ float s_input[];

    
    for (int row  = 0 ; row <= tile_size / BLOCK_DIM; row++){
        for (int col = 0; col <= tile_size/ BLOCK_DIM; col++){
            int idx_row = i_y + BLOCK_DIM * row - pad_size;
            int idx_col = i_x + BLOCK_DIM * col - pad_size;
            int fid_row = threadIdx.y + BLOCK_DIM * row;
            int fid_col = threadIdx.x + BLOCK_DIM * col;

            if ( fid_row >= tile_size || fid_col >= tile_size) continue;

            s_input[tile_size*fid_row + fid_col] = \
            //(idx_row >=0 &&idx_row <= n_r && idx_col >= 0 && idx_col <= n_c) ?   // wrong
            (idx_row >=0 &&idx_row < n_r && idx_col >= 0 && idx_col < n_c) ? 
            d_in[n_c * idx_row + idx_col] : 0.f;

        }
    }    

    __syncthreads();
    /*
    if (i_x == BLOCK_DIM && i_y == BLOCK_DIM){
        for (int row = 0; row < 2 * pad_size + BLOCK_DIM; row++){
            for (int col = 0; col < 2 * pad_size + BLOCK_DIM; col++){
                printf("%f ", s_input[tile_size * row + col]);
            }
            printf("\n");
        }
    }
    */

    
    float res = 0.f;
    for (int f_row = -filter_size/2; f_row <= filter_size/2; ++f_row){
        for (int f_col = -filter_size/2; f_col <= filter_size/2; ++f_col){
            int img_r = threadIdx.y + pad_size + f_row;
            int img_c = threadIdx.x + pad_size + f_col;
            float img_v = s_input[img_r * tile_size + img_c];
            float filter_v = c_filter[(f_row+filter_size/2)*filter_size + f_col + filter_size/2];
            res += img_v * filter_v;
        }
    }

    d_out[i_y * n_c + i_x] = res;
}


void conv_gpu(int version, float *d_out, float *d_in, float *d_filter, int num_row, int num_col, int filter_size){
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((num_col + BLOCK_DIM - 1)/BLOCK_DIM, (num_row+BLOCK_DIM-1)/BLOCK_DIM);
    if (version == 1){
        conv_kernel_v1<<< dimGrid, dimBlock >>>(d_out, d_in, d_filter, num_row, num_col, filter_size);
    } else if (version == 2){
        conv_kernel_v2<<< dimGrid, dimBlock >>>(d_out, d_in, d_filter, num_row, num_col, filter_size);
    } else if (version == 3 ){
        int shared_mem_size = (2*filter_size + BLOCK_DIM) * (2*filter_size + BLOCK_DIM) * sizeof(float);
        conv_kernel_v3<<< dimGrid, dimBlock, shared_mem_size, 0 >>>(d_out, d_in, d_filter, num_row, num_col, filter_size);
    }


    cudaGetLastError();
}
 
void conv_host(float *h_out, float *h_in, float *h_filter, int n_r, int n_c, int filter_size)
{
    for (int row = 0; row < n_r; ++row){
        for (int col = 0; col < n_c; ++col){
            float res = 0.f;
            for (int f_row = -filter_size/2; f_row <= filter_size/2; ++f_row){
                for (int f_col = -filter_size/2; f_col <= filter_size/2; ++f_col){
                    int img_r = row + f_row;
                    int img_c = col + f_col;
                    float img_v = (img_r >= 0 && img_r < n_r && img_c >=0 && img_c < n_c) ? 
                        h_in[img_r * n_c + img_c] : 0.f;
                    float filter_v = h_filter[(f_row+filter_size/2)*filter_size + f_col + filter_size/2];
                    res += img_v * filter_v;
                }
            }

            h_out[row * n_c + col] = res;
        }    
    }

    
}

int main(){
    int num_row = 1 << 11;
    int num_col = 1 << 11;
    int filter_size = 9;
    int buf_size = num_row * num_col * sizeof(float);

    float *h_input, *d_input;
    float *h_output_host, *h_output_gpu, *d_output;
    float *h_filter, *d_filter;

    float elapsed_time_gpu;

    StopWatchInterface *timer_host, *timer_gpu;
    sdkCreateTimer(&timer_host);
    sdkCreateTimer(&timer_gpu);

    srand(2023);
    
    h_input = (float *)malloc(buf_size);
    h_output_host = (float *)malloc(buf_size);
    h_output_gpu = (float *)malloc(buf_size);
    h_filter = (float *)malloc(filter_size * filter_size * sizeof(float));
    
    cudaMalloc((void **)&d_input, buf_size);
    cudaMalloc((void **)&d_output, buf_size);
    cudaMalloc((void **)&d_filter, filter_size * filter_size * sizeof(float));

    generate_data(h_input, num_row, num_col);
    generate_filter(h_filter, filter_size);

    cudaMemcpy(d_input, h_input, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_filter, h_filter, filter_size * filter_size * sizeof(float));


    sdkStartTimer(&timer_host);
    conv_host(h_output_host, h_input, h_filter, num_row, num_col, filter_size);
    sdkStopTimer(&timer_host);
    float elapsed_time_host = sdkGetTimerValue(&timer_host);
    printf("Cpu conv %f\n", elapsed_time_host);


    sdkStartTimer(&timer_gpu);
    cudaProfilerStart();
    for (int i = 1; i<= 3; i++){
        sdkResetTimer(&timer_gpu);
        sdkStartTimer(&timer_gpu);
        conv_gpu(i, d_output, d_input, d_filter, num_row, num_col, filter_size);
        cudaDeviceSynchronize();
        sdkStopTimer(&timer_gpu);
        elapsed_time_gpu = sdkGetTimerValue(&timer_gpu);
        printf("GPU conv %d %f \n", i, elapsed_time_gpu);
        
        cudaMemcpy(h_output_gpu, d_output, buf_size, cudaMemcpyDeviceToHost);
        if (value_test(h_output_host, h_output_gpu, num_row * num_col)){
            printf("ok\n");
        } else {
            printf("bad %f %f\n", h_output_host[128], h_output_gpu[128]);
        }

    }
    cudaProfilerStop();

    

    


    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    free(h_input);
    free(h_output_host);
    free(h_output_gpu);
    free(h_filter);

    return 0;
}