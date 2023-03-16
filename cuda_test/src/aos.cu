#include <iostream>
#include <string.h>

#define NUM_THREADS 256
#define IMGSIZE 1048576

struct Coefficients_AOS {
    int r;
    int b;
    int g;
    int hue;
    int saturation;
    int maxVal;
    int minVal;
    int finalVal;
};

__global__ void complicateCalculation(Coefficients_AOS* data){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int grayscale = (data[i].r + data[i].g + data[i].b)/data[i].maxVal;
    int hue_sat = data[i].hue * data[i].saturation / data[i].minVal;
    data[i].finalVal = grayscale * hue_sat;
}

void complicateCalculation(){
    Coefficients_AOS* d_x;
    cudaMalloc(&d_x, IMGSIZE*sizeof(Coefficients_AOS));
    int num_blocks = IMGSIZE/NUM_THREADS;
    complicateCalculation<<<num_blocks, NUM_THREADS>>>(d_x);
    cudaFree(d_x);
}

int main(int argc, char* argv[]){
    printf("sizeof(Coefficients_AOS) %ld", sizeof(Coefficients_AOS));   // 8 * 4 = 32
    complicateCalculation();
    return 0;
}