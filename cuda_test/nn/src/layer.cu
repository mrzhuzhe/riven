#include "layer.h"

#include <random>
#include <cuda_runtime.h>
#include <curand.h>

#include <math.h>
#include <algorithm>


#include <sstream>
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
            grad_weights_->cuda(), 1,
            weights_->cuda(), 1
        );
    }

    if ( biases_ != nullptr && grad_biases_ != nullptr )
    {
        cublasSaxpy(
            cuda_->cublas(),
            biases_->len(),
            &eps,
            grad_biases_->cuda(), 1,
            biases_->cuda(), 1
        );
    }
}

int Layer::load_parameter()
{
    std::stringstream filename_weights, filename_biases;

    filename_weights << name_ << ".bin";

    if ( weights_->file_read(filename_weights.str()))
        return -1;
    
    filename_biases << name_ << ".bias.bin";

    if (biases_->file_read(filename_biases.str()))
        return -2;

    std::cout << ".. loaded " << name_ << "pretrain parameter.." << std::endl;

    return 0;
}

int Layer::save_parameter()
{
    std::stringstream filename_weights, filename_biases;

    std::cout << ".. saving " << name_ << " parameter .."; 

    if (weights_)
    {
        filename_weights << name_ << ".bin";
        if ( weights_->file_write(filename_weights.str()))
            return -1;
    }
    
    
    if (biases_)
    {
        filename_biases << name_ << ".bias.bin";
        if (biases_->file_write(filename_biases.str()))
            return -2;
    }

    std::cout << " ßdone .." << std::endl;

    return 0;
}

Dense::Dense(std::string name, int output_size)
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

        init_one_vec <<< (batch_size_ + BLOCK_DIM - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D >>>(d_one_vec, batch_size_);

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
{
    // output = weight^T * input 
    cublasSgemm(
        cuda_->cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N, output_size_, batch_size_, input_size_,
        &cuda_->one, 
        weights_->cuda(), 
        input_size_,
        input_->cuda(),
        input_size_,
        &cuda_->zero,
        output_->cuda(),
        output_size_
    );
    // output += biases * d_one_vec^T
    cublasSgemm(
        cuda_->cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N, output_size_, batch_size_, 1,
        &cuda_->one, 
        biases_->cuda(), 
        output_size_,
        d_one_vec,
        1,
        &cuda_->one,
        output_->cuda(),
        output_size_
    );

    return output_;
}

void Dense::bwd_initialize(Blob<float> *grad_output)
{
    if (grad_weights_ == nullptr)
    {
        grad_weights_ = new Blob<float>(weights_->shape());
        grad_biases_ = new Blob<float>(biases_->shape());
    }

    if (grad_input_ == nullptr || batch_size_ != grad_output->n())
    {
        grad_output_ = grad_output;
        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }
}

Blob<float> *Dense::backward(Blob<float> *grad_output)
{   
    // db = (dy) * d_one_vec
    cublasSgemv(
        cuda_->cublas(),
        CUBLAS_OP_N,
        output_size_,
        batch_size_,
        &cuda_->one,
        grad_output_->cuda(),
        output_size_,
        d_one_vec,
        &cuda_->zero,
        grad_biases_->cuda(),
        1
    );

    // dw = x * (dy)^T
    cublasSgemm(
        cuda_->cublas(),
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        input_size_,
        output_size_,
        batch_size_,
        &cuda_->one,
        input_->cuda(),
        input_size_,
        grad_output_->cuda(),
        output_size_,
        &cuda_->zero,
        grad_weights_->cuda(),
        input_size_
    );

    // dx = W * dy
    if (!gradient_stop_)
        cublasSgemm(
            cuda_->cublas(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            input_size_,
            batch_size_,
            output_size_,
            &cuda_->one,
            weights_->cuda(),
            input_size_,
            grad_output_->cuda(),
            output_size_,
            &cuda_->zero,
            grad_input_->cuda(),
            input_size_
        );

    return grad_input_;

}

Activation::Activation(std::string name, cudnnActivationMode_t mode, float coef)
{
    name_ = name;
    act_mode_ = mode;
    act_coef_ = coef;

    cudnnCreateActivationDescriptor(&act_desc_);
    cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
}

Activation::~Activation(){
    cudnnDestroyActivationDescriptor(act_desc_);
}

void Activation::fwd_initialize(Blob<float> *input)
{
    if (input_ == nullptr || batch_size_ != input->n())
    {
        input_ = input;
        input_desc_ = input->tensor();
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Blob<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor();
    }
}

Blob<float> *Activation::forward(Blob<float> *input)
{
    cudnnActivationForward(cuda_->cudnn(),
        act_desc_,
        &cuda_->one,
        input_desc_,
        input->cuda(),
        &cuda_->zero,
        output_desc_,
        output_->cuda()
    );
    return output_;
}

void Activation::bwd_initialize(Blob<float> *grad_output)
{
    if (grad_input_ == nullptr || batch_size_ != grad_output_->n())
    {
        grad_output_ = grad_output;
        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
            
    }
}

Blob<float> *Activation::backward(Blob<float> *grad_output)
{
    cudnnActivationBackward(
        cuda_->cudnn(),
        act_desc_,
        &cuda_->one,
        output_desc_,
        output_->cuda(),
        output_desc_,
        grad_output_->cuda(),
        input_desc_,
        input_->cuda(),
        &cuda_->zero,
        input_desc_,
        grad_input_->cuda()
    );
    return grad_input_;
}

Softmax::Softmax(std::string name){
    name_ = name;
}

Softmax::~Softmax()
{

}

void Softmax::fwd_initialize(Blob<float> *input)
{
    if (input_ == nullptr || batch_size_ != input->n())
    {
        input_ = input;
        input_desc_ = input->tensor();
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Blob<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor();
    }
}

Blob<float> *Softmax::forward(Blob<float> *input)
{
    cudnnSoftmaxForward(cuda_->cudnn(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &cuda_->one,
        input_desc_,
        input->cuda(),
        &cuda_->zero,
        output_desc_,
        output_->cuda()
    );
    return output_;
}

void Softmax::bwd_initialize(Blob<float> *target)
{
    if (grad_input_ == nullptr || batch_size_ != target->n())
    {
        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }
} 

Blob<float> *Softmax::backward(Blob<float> *target)
{
    cudaMemcpyAsync(grad_input_->cuda(),
        output_->cuda(),
        output_->buf_size(),
        cudaMemcpyHostToDevice
    );

    cublasSaxpy(cuda_->cublas(),
        target->len(),
        &cuda_->minus_one,
        target->cuda(),
        1,
        grad_input_->cuda(),
        1
    );

    int grad_output_size = target->n() * target->c() * target->h() * target->w();
    float scale = 1.f / static_cast<float>(target->n());
    cublasSscal(
        cuda_->cublas(),
        grad_output_size,
        &scale,
        grad_input_->cuda(),
        1
    );
    return grad_input_;
}

float Softmax::get_loss(Blob<float> *target){
    return loss_.loss(output_, target);
}

int Softmax::get_accuracy(Blob<float> *target)
{
    int batch_size = output_->n();
    int output_size = output_->size();

    float *h_output, *h_target;
    int idx_output, idx_target;
    int hit_count = 0;

    h_output = output_->to(host);
    h_target = target->to(host);

    for (int b =0 ; b < batch_size; b++){
        idx_output = 0;
        idx_target = 0;

        for (int i = 0; i< 10; i++){
            if(h_output[b * output_size + i] > h_output[b * output_size + idx_output])
                idx_output = i;
            if(h_target[b * output_size + i] > h_output[b * output_size + idx_target])
                idx_target = i;
        }
        

        if (idx_output == idx_target)
            hit_count++;
    }
    return hit_count;    
}

/*
Conv2D::Conv2D(std::string name,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation
    ):
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    dilation_(dilation)
{
    name_ = name;
    cudnnCreateFilterDescriptor(&filter_desc_);

    cudnnCreateConvolutionDescriptor(&conv_desc_);
    cudnnSetConvolution2dDescriptor(conv_desc_,
    padding_, 
    padding_,
    stride_,
    stride_,
    dilation_,
    dilation_,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_FLOAT
    );

    cudnnSetConvolutionMathType(conv_desc_, CUDNN_DEFAULT_MATH);

    d_workspace_ = nullptr;
}

Conv2D::~Conv2D()
{
    cudnnDestroyFilterDescriptor(filter_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);

    if (d_workspace_ != nullptr) {
        cudaFree(d_workspace_);
        d_workspace_ = nullptr;
    }
}

void Conv2D::set_workspace()
{
    size_t temp_size = 0;

    // forward
    std::vector<cudnnConvolutionFwdAlgoPerf_t> 		 fwd_algo_perf_results(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
	std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algo_perf_results(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
	std::vector<cudnnConvolutionBwdDataAlgoPerf_t>	 bwd_data_algo_perf_results(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);

	int algo_max_count;
	int returnedAlgoCount = 0;
	cudnnGetConvolutionForwardAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count);
    
    cudnnGetConvolutionForwardAlgorithm_v7(cuda_->cudnn(),
		input_desc_, filter_desc_, conv_desc_, output_desc_,
		algo_max_count, &returnedAlgoCount, &fwd_algo_perf_results[0]);

    conv_fwd_algo_ = fwd_algo_perf_results[0].algo;

    cudnnGetConvolutionForwardWorkspaceSize(
        cuda_->cudnn(),
        input_desc_,
        filter_desc_,
        conv_desc_,
        output_desc_,
        conv_fwd_algo,
        &temp_size
    );
    workspace_size_ = std::max(workspace_size_, temp_size);
    
    // filter 
    cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count);
    
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(cuda_->cudnn(),
		input_desc_, output_desc_, conv_desc_, filter_desc_,
		algo_max_count, &returnedAlgoCount, &bwd_filter_algo_perf_results[0]);
    
    conv_bwd_filter_algo_ = bwd_filter_algo_perf_results[0].algo;

    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cuda->cudnn(),
        input_desc_,
        output_desc_,
        conv_desc_,
        filter_desc_,
        conv_bwd_filter_algo_,
        &temp_size
    );
    workspace_size_ = std::max(workspace_size_, temp_size);

    // data
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count);
    
    cudnnGetConvolutionBackwardDataAlgorithm_v7(cuda_->cudnn(),
		filter_desc_, output_desc_, conv_desc_, input_desc_,
		algo_max_count, &returnedAlgoCount, &bwd_data_algo_perf_results[0]);

    conv_bwd_data_algo_ = bwd_data_algo_perf_results[0].algo;
    
    cudnnGetConvolutionBackwardDataWorkspaceSize(cuda_->cudnn(),
		filter_desc_, output_desc_, conv_desc_, input_desc_,
		conv_bwd_data_algo_, &temp_size);

    workspace_size_ = std::max(workspace_size_, temp_size);

    if (workspace_size_ > 0){
        if (d_workspace_ != nullptr)
            cudaFree(d_workspace_);
        cudaMalloc((void**)&d_workspace_, workspace_size_);
    }
}

void Conv2D::fwd_initialize(Blob<float> *input){
    if (weights_ == nullptr)
    {
        cudnnSetFilter4dDescriptor(filter_desc_,
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels_, input->c(), kernel_size_, kernel_size_
        );
        weights_ = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
        biases_ = new Blob<float>(1, out_channels_);
        bias_desc_ = biases_->tensor();
    }

    if (input_ == nullptr || batch_size != input->n()){
        input_ = input;
        input_desc = input->tensor();
        batch_size_ = input->n();

        cudnnGetConvolution2dForwardOutputDim(
            conv_desc_, input_desc_, filter_desc_,
            &output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]
        );

        if (output == nullptr)
            output_ = new Blob<float>(output_size_);
        else
            output_->reset(output_size_);

        output_desc_ = output_->tensor();

        set_workspace();

        if (load_pretrain_ && ! freeze){
            if (load_parameter()){
                std::cout << "error occurred.. " << std::endl;
                exit(-1);
            }
        } else if (!freeze)
        {
            init_weight_bias();
        } else {

        }
    }
}

Blob<float> *Conv2D::forward(Blob<float> *input){
    cudnnConvolutionForward(cuda_->cudnn(),
        &cuda_->one,
        input_desc_,
        input_->cuda(),
        filter_desc_,
        weights_->cuda(),
        conv_desc_,
        conv_fwd_algo_,
        d_workspace_,
        workspace_size_,
        &cuda_->zero,
        output_desc_,
        output_->cuda()
    );

    cudnnAddTensor(cuda_->cudnn(),
        &cuda_->one,
        bias_desc_,
        biases_->cuda(),
        &cuda_->one,
        output_desc_,
        output_->cuda()
    );

    return output_;
}

void Conv2D::bwd_initialize(Blob<float> *grad_output){
    if (grad_weights_ == nullptr){
        grad_weights_ = new Blob<float>(weights_->shape());
        grad_biases_ = new Blob<float>(1, biases_->c());
    }

    if (grad_input == nullptr || batch_size_ != grad_output->n()){
        grad_output_ = grad_output;
        if (grad_input == nullptr)
            grad_input = new Blob<float>(input_->shape());
        else
            grad_input->reset(input_->shape());
    }
}

Blob<float> *Conv2D::backward(Blob<float> *grad_output)
{
    cudnnConvolutionBackwardBias(cuda_->cudnn(),
        &cuda_->one,
        output_desc_, grad_output->cuda(),
        &cuda_->zero(),
        bias_desc_, grad_biases_->cuda()
    );

    cudnnConvolutionBackwardFilter(cuda_->cudnn(),
        &cuda_->one,
        input_desc_, input_->cuda(),
        output_desc_, grad_output_->cuda(),
        conv_desc_, conv_bwd_filter_algo_, d_workspace, workspace_size_,
        &cuda_->zero,
        filter_desc_, grad_weights-->cuda()
    );

    if (!gradient_stop_){
        cudnnConvolutionBackwardData(cuda_->cudnn(),
            &cuda_->one,
            filter_desc_, weights_->cuda(),
            output_desc_, grad_output->cuda(),
            conv_desc_, conv_bwd_data_algo_, d_workspace_, workspace_size_,
            &cuda_->zero,
            input_desc_, grad_input_->cuda()
        );
    }
    return grad_input_;
}
*/

/*
Pooling:Pooling(std::string name,
    int kernel_size,
    int padding,
    int stride,
    cudnnPoolingMode_t mode)
    kernel_size_(kernel_size),
    padding_(padding),
    stride_(stride),
    mode_(mode)
{
    name_ = name;
    cudnnCreatePoolingDescriptor(&pool_desc_);
    cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN,
        kernel_size_, kernel_size_, padding_, padding_, stride_, stride_);
}

Pooling::~Pooling(){
    cudnnDestroyPoolingDescriptor(pool_desc_);
}

void Pooling::fwd_initialize(Blob<float> *input){
    if (input_ == nullptr || batch_size_ != input->n()){
        input_ = input;
        input_desc_ = input_->tensor();
        batch_size_ = input->n();

        cudnnGetPooling2dForwardOutputDim(pool_desc_, input_desc_, &output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]);
        if (output_ == nullptr)
            output_ = new Blob<float>(output_size_);
        else
            output_->reset(output_size_);

        output_desc_ = output_->tensor();
    }
}

Blob<float> *Pooling::forward(Blob<float> *input){
    cudnnPoolingForward(cuda_->cudnn(), pool_desc_,
        &cuda_->one, input_desc_, input_->cuda(),
        &cuda_->zero, output_desc_, output_->cuda()
    );
    return output_;
}

void Pooling::bwd_initialize(Blob<float> *grad_output){
    if (grad_input == nullptr || batch_size_ != grad_output->n())
    {
        grad_output_ = grad_output;
        if (grad_input == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else 
            grad_input_->reset(input_->shape());
    }
}


Blob<float> *Pooling::backward(Blob<float> *grad_output){
    cudnnPoolingBackward(cuda_->cudnn(),
        pool_desc_,
        &cuda_->one,
        output_desc_, output_->cuda(),
        output_desc_, grad_output->cuda(),
        input_desc_, input_->cuda(),
        &cuda_->zero,
        input_desc_, grad_input_->cuda()
    );
    return grad_input_;
}
*/