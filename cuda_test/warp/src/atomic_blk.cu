#include <iostream>
#include <cooperative_group.h>
#include <helper_timer.h>

template <typename group_t>
__inline__ __device__ float warp_reduce_sum(group_t group, float val)
{
    //#pragma unroll
    for (int offset = group.size() / 2; offset > 0; offset >>=1){        
        val += group.shfl_down(val, offset);
    }
    return val;
}


__inline__ __device__ float block_reduce_sum(thread_block block,float val){
    __shared__ float shared[32];
    int warp_idx = threadIdx.x / warpSize;

    thread_block_tile<32> tile32 = tiled_partition<32>(block);
    val = warp_reduce_sum(tile32, val);

    if (tile32.thread_rank() == 0){
        shared[warp_idx] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[tile32.thread_rank()] : 0;

    if (warp_idx == 0){        
        val = warp_reduce_sum(tile32, val);
    }
    return val;
}

__global__ void reduction_kernel(float *data_out, float *data_in, int size){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    thread_block block = this_thread_block();
    
    float sum[NUM_LOAD] = {0.f};
    for (int i = idx; i < size; i += blockDim.x * gridDim.x * NUM_LOAD){
        for (int step = 0; step < NUM_LOAD; step ++ ){
            int _cur = i + step * blockDim.x * gridDim.x;
            sum[step] += (_cur < size ) ? data_in[_cur] : 0.f;
        }        
    }
    for (int i = 1; i < NUM_LOAD; i++){
        sum[0] += sum[i];
    }
    sum[0] = block_reduce_sum(block, sum[0]);

    if (block.thread_rank() == 0){
        atomicAdd(&data_out[0], sum[0]);
    }
}

void reduction(float *d_out, float *d_in, int size, int n_threads){
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlockPerMultiprocessor(&num_block_per_sm, shared_reduction_kernel, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1)/ n_threads);
    
    reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, size);
}

void run_benchmark(void (*reduce)(float *, float *, int, int), 
float *d_outPtr, float *d_inPtr, int size){
    int num_threads = 256;
    int test_iter = 100;

    reduce(d_outPtr, d_inPtr, size, num_threads);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < test_iter; i++){
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
        reduce(d_outPtr, d_inPtr, size, num_threads);               
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    double elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = size * sizeof(float ) / elapsed_time_msed / 1e6;
    printf("Time= %f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

}

void init_input(float *data, int size){
    for (int i = 0; i< size; i++){
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float get_cpu_result(float *data, int size){
    double result = 0.f;
    for (int i = 0; i< size; i++)
        result += data[i];
    return (float)result;
}

int main(){
    float *h_inPtr;
    float *d_inPtr, *d_outPtr;

    unsigned int size = 1 << 24;
    
    float result_host, result_gpu;
    int mode = 0;

    srand(2019);

    h_inPtr = (float *)malloc(size*sizeof(float));

    init_input(h_inPtr, size);

    cudaMalloc((void **)&d_inPtr, size*sizeof(float));
    cudaMalloc((void **)&d_outPtr, size*sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size*sizeof(float), cudaMemcpyHostToDevice);

    run_benchmark(reduction, d_outPtr, d_inPtr, size);
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);

    result_host = get_cpu_result(h_inPtr, size);
    printf("host %f, device %f\n", result_host, result_gpu);

    cudaFree(d_outPtr);
    cudaFree(d_inPtr);
    free(h_inPtr);

    return 0;
}