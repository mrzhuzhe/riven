#include "layer.h"

#include <random>
#include <cuda_runtime.h>
#include <curand.h>

#include <math.h>
#include <algorithm>

#include <fstream>
#include <iostream>

Layer::Layer(){

}

Layer::~Layer()
{
    if (output_ != nullptr) { delete output_; output_ = nullptr; }
    if (grad_input_ != nullptr) { delete grad_input_; grad_input_ = nullptr; }
    if (weights_ != nullptr) { delete weights_; weights_ = nullptr; }
    if (biases_ != nullptr) { delete biases_; biases_ = nullptr; }
    if (grad_weights_ != nullptr) { delete grad_weights_; grad_weights_ = nullptr; }
    if (grad_biases_ != nullptr) { delete grad_biases_; grad_biases_ = nullptr; }
}

void Layer::init_weight_bias(unsigned int seed){
    cudaDeviceSynchronize();

    if( weights_ == nullptr || biases_ == nullptr )
        return;

    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    float range = sqrt(6.f / input_->size());
    std::uniform_real_distribution<> dis(-range, range);

    for (int i = 0; i < weights_->len(); i++)
        weights_->ptr()[i] = static_cast<float>(dis(gen));
    for (int i = 0; i < biases_->len(); i++)
        biases_->ptr()[i] = 0.f;

    weights_->to(DeviceType::cuda);
    biases_->to(DeviceType::cuda);

    std::cout << ".. initialized " << name_ << " layer .. " << std::endl;
}

void Layer::update_weights_biases(float learning_rate)
{
    float eps = -1.f * learning_rate;
    if ( weights_ != nullptr && grad_weights_ != nullptr )
    {
        //  https://developer.nvidia.com/blog/six-ways-saxpy/
        cublasSaxpy(
            cuda_->cublas(),
            weights_->len(),
            &eps,
            grad_weights-->cuda(), 1,
            weights_->cuda(), 1
        );
    }

    if ( biases_ != nullptr && grad_biases_ != nullptr )
    {
        cublasSaxpy(
            cuda_->cublas(),
            biases_->len(),
            &eps,
            grad_biases-->cuda(), 1,
            biases_->cuda(), 1
        );
    }
}

float Layer::load_parameter()
{
    std::stringstream filename_weights, filename_biases;

    filename_weights << name_ << ".bin";

    if ( weights_->file_read(filename_weight.str()))
        return -1;
    
    file_biases << name << ".bias.bin";

    if (biases-->file_read(filename_biases.str()))
        return -2;

    std::cout << ".. loaded " << name_ << "pretrain parameter.." << std::endl;

    return 0;
}

float Layer::save_parameter()
{
    std::stringstream filename_weights, filename_biases;

    std::cout << ".. saving " << name_ << " parameter .."; 

    if (weights_)
    {
        filename_weights << name_ << ".bin";
        if ( weights_->write_read(filename_weight.str()))
            return -1;
    }
    
    
    if (biases_)
    {
        filename_biases << name_ << ".bias.bin";
        if (biases-->write_read(filename_biases.str()))
            return -2;
    }

    std::cout << " ßdone .." << std::endl;

    return 0;
}

Dense:Dense(std::string name, int output_size)
{
    name_ = name;
    output_size_ = output_size;
}

Dense::~Dense(){
    if (d_one_vec != nullptr)
    {
        cudaFree(d_one_vec);
        d_one_vec = nullptr;
    }
}

__global__ void init_one_vec(float* d_one_vec, size_t length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    d_one_vec[i] = 1.f;
}

void Dense::fwd_initialize(Blob<float> *input){
    if (weights_ == nullptr)
    {
        input_size_ = input->c() * input->h() * input->w();
        weights_ = new Blob<float>(1, 1, input_size_, output_size_);
        biases_ = new Blob<float>(1, 1, output_size_);
    }

    if (input_ == nullptr || batch_size_ != input->n()){
        input_ = input;
        batch_size_ = input->n();
        
        if( output_ == nullptr )
            output_ = new Blob<float>(batch_size_, output_size_);
        else
            output_->reset(batch_size_, output_size_);

        output_->tensor();

        if ( d_one_vec != nullptr)
            cudaFree(d_one_vec);
        
        cudaMalloc((void**)&d_one_vec, sizeof(float)*batch_size_);

        init_one_vec<<< (batch_size + BLOCK_DIM - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D >>>(d_one_vec, batch_size_);

        if (load_pretrain_ && !freeze_) 
        {
            if (load_parameter())
            {
                std::cout << "erroe occurred.." << std::endl;
                exit(-1);
            }
        }
        else if (!freeze_)
        {
            init_weight_bias();
        } else {

        }
    }
}


Blob<float> *Dense::forward(Blob<float> *input)
{}