#include "loss.h"
#include "helper.h"

#include <cuda_runtime.h>

CrossEntropyLoss::CrossEntropyLoss()
{
    cudaMalloc((void**)&d_loss_, sizeof(float));
}

CrossEntropyLoss::~CrossEntropyLoss()
{
    if (d_loss_ != nullptr){
        cudaFree(d_loss_);
        d_loss_  = nullptr;
    }

    if (d_workspace_ != nullptr){
        cudaFree(d_workspace_);
    }
}

__device__ float clip(float prediction, float ep=1e-12){
    return fmin(fmax(prediction, ep), 1.f - ep);
}

__global__ void softmax_loss_kernel(float *reduced_loss, float *predict, float *target, float *workspace, int batch_size, int num_outputs){
    int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float s_data[];
    float loss = 0.f;

    for (int c = 0; c < num_outputs; c++){
        loss += target[batch_idx*num_outputs+c] * logf(predict[batch_idx*num_outputs+c]);        
    }
    //printf("%f %f\n", target[batch_idx*num_outputs+0], predict[batch_idx*num_outputs+0]);
    workspace[batch_idx] = -loss;
    if (blockIdx.x > 0) return;

    s_data[threadIdx.x] = 0.f;
    for (int i = 0; i < batch_size; i += blockDim.x)
    {
        s_data[threadIdx.x] += workspace[threadIdx.x + i];
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride >0; stride >>=1){
        if (threadIdx.x + stride < batch_size){
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        reduced_loss[blockIdx.x] = s_data[0];
    }
}

void CrossEntropyLoss::init_workspace(int batch_size){
    if (d_workspace_ == nullptr)
        cudaMalloc((void**)&d_workspace_, sizeof(float)*batch_size);
}

float CrossEntropyLoss::loss(Blob<float> *predict, Blob<float> *target){
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, softmax_loss_kernel, BLOCK_DIM_1D, BLOCK_DIM_1D*sizeof(float));
    int batch_size = target->n();
    int num_outputs = target->c();

    init_workspace(batch_size);

    //std::cout << "[[ LOSS ]]" << std::endl;
    //predict->print("predict", true);
    //target->print("target", true);

    int num_blocks = min(num_blocks_per_sm * num_sms, \
        (target->size() + BLOCK_DIM_1D -1) / BLOCK_DIM_1D);
    softmax_loss_kernel<<< num_blocks , BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float), 0 >>>
    (d_loss_, predict->cuda(), target->cuda(), d_workspace_, batch_size, num_outputs);

    cudaMemcpy(&h_loss_, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);

    return h_loss_ / float(batch_size);
}