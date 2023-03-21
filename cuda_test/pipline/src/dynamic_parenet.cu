#include <iostream>

#define BUF_SIZE (1 << 10)
#define BLOCKDIM 256

__global__ void child_kernel(int *data, int seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&data[idx], seed);
}

__global__ void parent_kernel(int *data){
    if (threadIdx.x == 0){
        int child_size = BUF_SIZE / gridDim.x;
        child_kernel<<< child_size/BLOCKDIM, BLOCKDIM >>>(&data[child_size*blockDim.x], blockDim.x+1);
    }
    cudaDeviceSynchronize();
}

int main(){
    int *data;
    int num_child = 2;

    cudaMallocManaged((void**)&data, BUF_SIZE*sizeof(int));
    cudaMemset(data, 0, BUF_SIZE * sizeof(int));

    parent_kernel<<<num_child, 1>>>(data);

    cudaDeviceSynchronize();

    int counter = 0;
    for (int i = 0; i < BUF_SIZE; i++){
        counter += data[i];
    }

    int counter_h = 0;
    for (int i = 0; i < num_child; i++){
        counter_h += (i+1);
    }

    counter_h *= BUF_SIZE / num_child;

    if (counter_h == counter)
        printf("Correct!!\n");
    else 
        printf("Error, %d %d", counter, counter_h);

    cudaFree(data);

    return 0;
}