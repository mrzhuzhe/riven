#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iostream>
#include <iomanip>
//  https://zhuanlan.zhihu.com/p/526508882

int main(){
    int batch_size_train = 256;
    int num_step_train = 1600;
    int monitoring_step = 200;

    double learning_rate = 0.02f;
    double lr_decay = 5e-5f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_step_test = 1000;

    std::cout << "MNIST training with cuDnn" << std::endl;

    std::cout << "[TRAIN]" << std::endl;

    MNIST train_data_loader = MNIST("./dataset");
    train_data_loader.train(batch_size_train, true);

    Network model;
    model.add_layer(new Dense("densel", 500));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    model.add_layer(new Dense("dense2", 10));
    model.add_layer(new SOftmax("softmax"));
    model.cuda();

    if (load_pretrain)
        model.load_pretrain();
    model.train();

    // cudaProfileStart();
    int step = 0;
    Blob<float> *train_data = train_data_loader.get_data();
    Blob<float> *train_target = train_data_loader.get_target();
    train_data_loader.get_batch();
    int tp_count = 0;
    while (step < num_step_train){
        train_data->to(cuda);
        train_target->to(cuda);

        model.forward(train_data);
        tp_count += model.get_accuracy(train_target);

        model.backward(train_target);

        learning_rate *= 1.f / (1.f + lr_decay *step);

        model.update(learning_rate);


        step = train_data_loader.next();


        if (step % monitoring_step == 0)
        {
            float loss = model.locc(train_target);
            float accuracy = 100.f * tp_count / monitoring_step / batch_size_train;
            std::cout << "step " << std::right << std::setw(4) << step << \
                ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                ", accuracy: " << accuracy << "%" << std::endl;
            tp_count = 0;
        }
    }

    if (file_save)
            model.write_file();

    std::cout << "[INFERENCE]" << std::endl;
    MNIST test_data_loader = MNIST("./dataset");
    test_data_loader.test(batch_size_test);

    model.test();

    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();

    test_data_loader.get_batch();
    tp_count = 0;
    step = 0;
    while (step < num_step_test){
        test_data->to(cuda);
        test_target->to(cuda);

        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);

        step = model.get_accuracy(test_target);    
    }

    float loss = model.locc(test_target);
    float accuracy = 100.f * tp_count / num_step_test / batch_size_test;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;

    std::cout << "Done.  "  << std::endl;

    return 0;

}