#include "mnist.h"
#include <iostream>

MNIST::~MNIST()
{
    delete data_;
    delete target_;
}

void MNIST::create_shared_space(){
    data_ = new Blob<float>(batch_size_, channels_, height_, width_);
    data->tensor();
    target_ = new Blob<float>(batch_size_, num_classes_);
}