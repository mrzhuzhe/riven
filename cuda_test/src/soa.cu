#include <iostream>
#include <string.h>

#define NUM_THREADS 256
#define IMGSIZE 1048576

struct Coefficients_SOA {
    int* r;
    int* b;
    int* g;
    int* hue;
    int* saturation;
    int* maxVal;
    int* minVal;
    int* finalVal;
};

__global__ void complicateCalculation(Coefficients_SOA data){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int grayscale = (data.r[i] + data.g[i] + data.b[i])/data.maxVal[i];
    int hue_sat = data.hue[i] * data.saturation[i] / data.minVal[i];
    data.finalVal[i] = grayscale * hue_sat;
}

void complicateCalculation(){
    Coefficients_SOA d_x;

    cudaMalloc(&d_x.r, IMGSIZE*sizeof(int));
    cudaMalloc(&d_x.g, IMGSIZE*sizeof(int));
    cudaMalloc(&d_x.b, IMGSIZE*sizeof(int));
    cudaMalloc(&d_x.hue, IMGSIZE*sizeof(int));
    cudaMalloc(&d_x.saturation, IMGSIZE*sizeof(int));
    cudaMalloc(&d_x.maxVal, IMGSIZE*sizeof(int));
    cudaMalloc(&d_x.minVal, IMGSIZE*sizeof(int));
    cudaMalloc(&d_x.finalVal, IMGSIZE*sizeof(int));

    int num_blocks = IMGSIZE/NUM_THREADS;
    complicateCalculation<<<num_blocks, NUM_THREADS>>>(d_x);
    // some methods to get all keys() ?
    cudaFree(d_x.r);
    cudaFree(d_x.g);
    cudaFree(d_x.b);
    cudaFree(d_x.hue);
    cudaFree(d_x.saturation);
    cudaFree(d_x.maxVal);
    cudaFree(d_x.minVal);
    cudaFree(d_x.finalVal);
}

int main(int argc, char* argv[]){
    printf("sizeof(Coefficients_SOA) %ld\n", sizeof(Coefficients_SOA));   // why 64 ?
    Coefficients_SOA d_x;
    printf("sizeof(d_x.r) %ld\n", sizeof(d_x.r));   // why 8 
    complicateCalculation();
    return 0;
}