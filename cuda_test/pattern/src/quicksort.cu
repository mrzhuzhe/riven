#include <iostream>
#include "cuda_runtime.h"
#include "common.h"

int main(){
    int num_items = 2048;

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);

    unsigned int *h_data = 0;
    unsigned int *d_data = 0;

    h_data = (unsigned int *)malloc(num_items * sizeof(unsigned int));
    random_init(h_data, num_items);

    

}