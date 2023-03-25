#ifndef __BLOB_H_
#define _BLOB_H_

#include <array>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

typedef enum {
    host,
    cuda
} DeviceType;

template <typename ftype>
class Blob {
    public:
    Blob(int n = 1, int c = 1, int h = 1, int w = 1): n_(n), c_(c), h_(h), w_(w)
    {
        h_ptr_ = new float[n_c * c_ + h_ * w_];
    }
    Blob(std::array<int, 4>): n_(size[0]), c_(size[1]), h_(size[2]), w_(size[3])
    {
        h_ptr_ = new float[n_*c_*h_*w_];
    }
    ~Blob()
    {
        if (h_ptr_ != nullptr)
            delete [] h_ptr_;
        if (d_ptr_ != nullptr)
            cudaFree(d_ptr_);
        if (is_tensor_)
            cudnnDestroyTensorDescriptor(tensor_desc_);
    }

    void reset(int n = 1, int c = 1, int h = 1, int w = 1): n_(n), c_(c), h_(h), w_(w)
    {
        if (h_ptr != nullptr)
        {
            delete [] h_ptr_;
            h_ptr_ = nullptr;
        }
        if (d_ptr_ != nullptr)
        {
            cudaFree(d_ptr_);
            d_ptr_ = nullptr;
        }
        h_ptr_ = new float[n_*c_*h_*w_];
        cuda();
        if (is_tensor_)
        {
            cudnnDestroyTensorDescriptor(tensor_desc_);
            is_tensor = false;
        }

    }
    void reset(std::array<int, 4> size){
        reset(size[0], size[1], size[2], size[3]);
    }

    std::array<int, 4> shape() { 
        return std::array<int, 4>(n_, c_, h_, w_);
    }

    int size() { return c_ * h_ * w_; }

    int len() { return n_ * c_ * h_ * w_; }

    int buf_size() { return sizeof(ftype) * len(); }

    int n() const { return n_; }
    int c() const { return c_; }
    int h() const { return h_; }
    int w() const { return w_; }

    bool is_tensor = false;
    cudnnTensorDescriptor_t tensor_desc_;
    cudnnTensorDescriptor_t tensor(){
        if (is_tensor_)
            return tensor_desc_;
        cudnnCreateTensorDescriptor(&tensor_desc_);
        cudnnSetTensor4Descriptot(tensor_desc_,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        n_, c_, h_, w_);

        is_tensor_ = true;
        return tensor_desc_;
    }

}


#endif