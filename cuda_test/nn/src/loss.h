#ifndef _LOSS_H_
#define _LOSS_H_

#include "blob.h"

class CrossEntropyLoss
{
    public:
        CrossEntropyLoss();
        ~CrossEntropyLoss();

        float loss(Blob<float> *predict, Blob<float> *target);
        float accuracy(Blob<float> *predict, Blob<float> *target);
    private:
        float h_loss_ = 0.f;
        float *d_loss_ = nullptr;

        float *d_workspace = nullptr;
        void init_workspace(int batch_size);
};

#endif