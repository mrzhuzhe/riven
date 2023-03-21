/*
    Priority Range: low(0), high(-5)
    stream 0 - elapsed 6.777000 ms 
    stream 1 - elapsed 9.221000 ms 
    stream 3 - elapsed 13.221000 ms 
    stream 2 - elapsed 14.702000 ms 
    kernel stream 0 - elapsed 2.966144 ms 
    kernel stream 1 - elapsed 3.849600 ms 
    kernel stream 2 - elapsed 7.147520 ms 
    kernel stream 3 - elapsed 2.940768 ms 
    compared a sample result...
    host: 2.000000, device 2.000000
    Time=15.009000 msec, bandwidth=53.654900 GB/s
*/

#include <iostream>
#include "common.h"
#include <helper_timer.h>


class Operator
{
    private:
        int _index;
        //cudaStream_t stream;
        StopWatchInterface *p_timer;

        static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userDate);
        void print_time();
        cudaEvent_t start, stop;
    protected:
        cudaStream_t stream = nullptr;
    public:
        Operator(bool create_stream = true){
            if (create_stream)
                cudaStreamCreate(&stream);
            sdkCreateTimer(&p_timer);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }
        ~Operator(){
            if (stream != nullptr)
                cudaStreamDestroy(stream);
            sdkDeleteTimer(&p_timer);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        void set_index(int idx) { _index = idx; }
        void async_operation(float *h_c, const float *h_a, const float *h_b,
            float *d_c, float *d_a, float *d_b,
            const int size, const int bufsize
        );
        void print_kernel_time();
};

void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData){
    Operator* this_ = (Operator*) userData;
    this_->print_time();
}

void Operator::print_time(){    
    sdkStopTimer(&p_timer);
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %d - elapsed %f ms \n", _index, elapsed_time_msed);
}

void Operator::print_kernel_time(){
    float elapsed_time_msed = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed, start, stop);
    printf("kernel stream %d - elapsed %f ms \n", _index, elapsed_time_msed);
}


void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
            float *d_c, float *d_a, float *d_b,
            const int size, const int bufsize
        )
{
    sdkStartTimer(&p_timer);
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    cudaEventRecord(start, stream);

    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_c, d_a, d_b);

    cudaEventRecord(stop, stream);

    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    //cudaEventSynchronize(stop); // what happen ?
    //  少了一个 stream sync 变快
    cudaStreamAddCallback(stream, Operator::Callback, this, 0);
}

class Operator_with_priority: public Operator {
    public:
        Operator_with_priority(): Operator(false) {}
    void set_priority(int priority){
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
    }
};


int main(){
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);
    int num_operator = 4;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.streamPrioritiesSupported == 0){
        printf("This device does not support priority streams");
        return 1;
    }

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    cudaMallocHost((void **)&h_a, bufsize);
    cudaMallocHost((void **)&h_b, bufsize);
    cudaMallocHost((void **)&h_c, bufsize);

    srand(2023);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    Operator_with_priority *ls_operator = new Operator_with_priority[num_operator];

    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    printf("Priority Range: low(%d), high(%d)\n ", priority_low, priority_high);

    sdkStartTimer(&timer);

    for (int i =0; i < num_operator; i++){        
        ls_operator[i].set_index(i);
        if (i+1 == num_operator)
            ls_operator[i].set_priority(priority_high); // 3 
        else
            ls_operator[i].set_priority(priority_low);
    }


    for (int i =0; i < num_operator; i++){
        int offset = i * size / num_operator;
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
        &d_c[offset], &d_a[offset], &d_b[offset],
        size / num_operator, bufsize / num_operator
        );
    }

    cudaDeviceSynchronize();

    sdkStopTimer(&timer);

    for (int i =0; i < num_operator; i++){
        ls_operator[i].print_kernel_time();
    }


    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %f, device %f\n", h_a[print_idx]+h_b[print_idx], h_c[print_idx]);

    double elasped_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elasped_time_msed / 1e6;
    printf("Time=%f msec, bandwidth=%f GB/s\n", elasped_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

    delete [] ls_operator;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
