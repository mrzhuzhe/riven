#include <iostream>
#include "test_bandwidth.h"

int main(){

//  https://en.wikipedia.org/wiki/X86_calling_conventions
//  System V AMD64 ABI
//  RDI, RSI, RDX, RCX, R8, R9

    printf("Intel ASM test go \n");  
    float sum = 0;
    int BUFF_SIZE = 320 * 320;
    float *buff = (float*)malloc(BUFF_SIZE*sizeof(float));
    sum = 0;
    for (int i = 0; i < BUFF_SIZE; i++){
        buff[i] = 1.3;
    }
    for (int i = 0; i < BUFF_SIZE; i++){
        sum += buff[i];
    }
    std::cout << "sum " << sum << std::endl;

    sum = 0;
    test_bandwidth(&sum, buff, BUFF_SIZE);
    std::cout << "sum2 " << sum << std::endl;

    return 0;
}