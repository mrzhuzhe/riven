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
        StopWatchInterface *p_timer;

        static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userDate);
        void print_time();
    public:
        Operator(){
            cudaStreamCtreate(&stream);
            sdkCreateTimer(&p_timer);
        }
        ~Operator(){
            cudaStreamDestroy(&stream);
            sdkDeleteTimer(&p_timer);
        }
        void set_index(int idx) { index = idx; }
        void async_operation(float *h_c, const float *h_a, const float *h_b,
            float *d_c, float *d_a, float *d_b,
            const int size, const int bufsize
        );
};

void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData){
    Operator* this_ = (Operator*) userData;
    this_->print_time();
}

void Operator::print_time(){
    sdkStopTimer(&p_timer);
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %d - elapsed %f ms \n", _index, elasped_time_msed);
}

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
            float *d_c, float *d_a, float *d_b,
            const int size, const int bufsize
        ){
            sdkStartTimer(&p_timer);
            cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

            dim3 dimBlock(256);
            dim3 dimGrid(size / dimBlock.x);
            vecAdd_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_c, d_a, d_b);

            cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

            cudaStreamAddCallback(stream, Operator::Callback, this, 0);
            //cudaStreamSynchronize(stream);
            //printf("Launched GPU task %d\n", index);
        }

int main(){
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsizec = size * sizeof(float);
    int num_operator = 4;

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    cudaMallocHost((void **)&h_a, bufsize);
    cudaMallocHost((void **)&h_b, bufsize);
    cudaMallocHost((void **)&h_c, bufsize);

    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    Operator *ls_operator = new Operator[num_operator];

    sdkStartTimer(&timer);

    for (int i =0; i < num_operator; i++){
        int offset - i * size / num_operator;
        ls_operator[i].set_index(i);
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset]
        &d_c[offset] &d_a[offset], &d_b[offset],
        size / num_operator, bufsize / num_operator
        );
    }

    cudaDeviceSynchronize();

    sdkStopTimer(&timer);

    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %f, device f\n", h_a[print_idx]+h_b[print_idx], h_c[print_idx]);

    double elasped_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elasped_time_msed / 1e6;
    printf("Time=%f msec, bandwidth=%f GB/s\n", elasped_time_msed, bandwidth);

    sdkDeleteTimer($timer);

    delete [] ls_operator;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}