#include <iostream>
#include <helper_functions.h>

void init_buffer(float *data, const int size){
    for (int i =0 ;i < size; i++)
        data[i] = rand() / (float)RAND_MAX;
}

__global__ void vecAdd_kernel(float *c, const float *a, const float *b){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i< 500; i++)
        c[idx] = a[idx] + b[idx];
}

class Operator
{
    private:
        int index;
        cudaStream_t stream;
    public:
        Operator(){
            cudaStreamCtreate(&stream);
        }
        ~Operator(){
            cudaStreamDestroy(&stream);
        }
        void set_index(int idx) { index = idx; }
        void async_operation(float *h_c, const float *h_a, const float *h_b,
            float *d_c, float *d_a, float *d_b,
            const int size, const int bufsize
        );
}

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
            float *d_c, float *d_a, float *d_b,
            const int size, const int bufsize
        ){
            cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

            dim3 dimBlock(256);
            dim3 dimGrid(size / dimBlock.x);
            vecAdd_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_c, d_a, d_b);

            cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

            cudaStreamSynchronize(stream);
            printf("Launched GPU task %d\n", index);
        }

int main(){
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsizec = size * sizeof(float);
    int num_operator = 4;

    cudaMallocHost((void **)&h_a, bufsize);
    cudaMallocHost((void **)&h_b, bufsize);
    cudaMallocHost((void **)&h_c, bufsize);

    srand(2019);

    return 0;
}