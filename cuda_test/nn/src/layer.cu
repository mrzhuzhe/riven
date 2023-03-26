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
    
}