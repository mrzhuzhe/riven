#ifndef _MNIST_H_
#define _MNIST_H_

#include <string>
#include <fstream>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

#include "blob.h"

#define MNIST_CLASS 10

class MNIST
{
    public:
    MNIST(): dataset_dir_("./") {}
    MNIST(std::string dataset_dir): dataset_dir_(dataset_dir){}
    ~MNIST();

    void train(int batch_size = 1, bool shuffle = false );

    void test(int batch_size = 1);
    
    void get_batch();

    int next();

    Blob<float>* get_data() { return data_; }
    Blob<float>* get_target() { return target_; }

    private:

    std::string dataset_dir_;
#ifdef __linux__
    std::string train_dataset_file_ = "train-images-idx3-ubyte";
    std::string train_label_file_   = "train-labels-idx1-ubyte";
    std::string test_dataset_file_  = "t10k-images-idx3-ubyte";
    std::string test_label_file_    = "t10k-labels-idx1-ubyte";
#elif _WIN32
    std::string train_dataset_file_ = "train-images.idx3-ubyte";
    std::string train_label_file_ 	= "train-labels.idx1-ubyte";
    std::string test_dataset_file_ 	= "t10k-images.idx3-ubyte";
    std::string test_label_file_ 	= "t10k-labels.idx1-ubyte";
#endif

    std::vector<std::vector<float>> data_pool_;
    std::vector<std::array<float, MNIST_CLASS>> target_pool_;
    Blob<float>* data_ = nullptr;
    Blob<float>* target_ = nullptr;

    void load_data(std::string &image_file_path);
    void load_target(std::string &image_file_path);

    void normalize_data();
    int to_int(uint8_t *ptr);

    int step_ = -1;
    bool shuffle_;
    int batch_size_ = 1;
    int channels_ = 1;
    int height_ = 1;
    int width_ = 1;
    int num_classes_ = 10;
    int num_steps_ = 0;

    void create_shared_space();
    void shuffle_dataset();

};



#endif