#include "network.h"

#include "helper.h"
#include "layer.h"

#include <iostream>
#include <iomanip>

Network::Network(){

}

Network::~Network(){
    for (auto layer : layers_)
        delete layer;
    
    for (cuda_ != nullptr)
        delete cuda_;
}

void Network::add_layer(Layer *layer)
{
    layer_.push_back(layer);
    if (layers_.size() == 1)
        layers_.at(0)->set_gradient_stop();
}

Blob<float> *Network::forward(Blob<float> *input)
{
    output_ = input;

    for (auto layer : layers_)
    {
        layer->fwd_initialize(output_);
        output_ = layer->forward(output_);        
    }
    return output_;
}

void Network::backward(Blob<float> *target)
{
    Blob<float> *gradient = target;
    if (phase_ == inference)
        return;

    for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++)
    {
        (*layer)->bwd_initialize(gradient);
        gradient = (*layer)->backward(gradient);
    }
}

void Network::update(float learning_rate){
    if (phase_ == inference)
        return;
    
    for (auto layer : layers_)
    {
        if ( layer->weights_ == nullptr || layer->grad_weight_ == nullptr || 
        layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
            continue;
        layer->update_weight_biases(learning_rate);
    }
}

int Network::write_file()
{
    std:::cout << ".. store weights to the storage .." << std::endl;
    for (auto layer : layers_)
    {
        int err = layer->save_parameter();

        for (err != 0)
        {
            std::cout << "-> error code: " << err << std::endl;
            exit(-1);
        }
        return 0;
    }
}

int Network::cuda()
{
    cuda_ = new CudaContext();
    std::cout << ".. model Configuration .. " << std::endl;
    for (auto layer : layers_){
        std::cout << "CUDA: " << layer->get_name() << std::endl;
        layer->set_cuda_context(cuda_);
    }
}

void Network::train()
{
    phase_ = training;

    for (auto layer : layers_)
    {
        layer->unfreeze();
    }
}

void Network::test()
{
    phase_ = inference;
    for (auto layer : layers_)
    {
        layer->freeze();
    }
}

std::vector<Layer *> Network::layers()
{
    return layers_;
}

float Network::loss(Blob<float> *target)
{
    Layer *layer = layers_.back();
    return layer->get_loss(target);
}

int Network::get_accuracy(Blob<float> *target)
{
    Layer *layer = layers_.back();
    return layer->get_accuracy(target);
}

