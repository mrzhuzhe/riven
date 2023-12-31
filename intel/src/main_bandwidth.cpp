#include <iostream>
#include "test_bandwidth.h"
#include "test_bandwidth_2p.h"

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
        buff[i] = 1.1;
    }
    float start_time = clock();
    for (int i = 0; i < BUFF_SIZE; i++){
        sum += buff[i];
    }
    float end_time = clock();
    std::cout << "sum " << sum << std::endl;
    std::cout << "cost " << (end_time - start_time) << std::endl;

    float *sum2 = (float*)malloc(4*sizeof(float));
    start_time = clock();
    //  this is a ieee 754 bug
    test_bandwidth(sum2, buff, BUFF_SIZE);     
    std::cout << "sum2 " << (sum2[0]+ sum2[1] +sum2[2] +sum2[3])  << std::endl;
    // std::cout << "sum2 0 " << (sum2[0])  << std::endl;
    // std::cout << "sum2 1 " << (sum2[1])  << std::endl;
    // std::cout << "sum2 2 " << (sum2[2])  << std::endl;
    // std::cout << "sum2 3 " << (sum2[3])  << std::endl;
    end_time = clock();
    std::cout << "cost 2 " << (end_time - start_time) << std::endl;


    start_time = clock();
    //  this is a ieee 754 bug
    test_bandwidth(sum2, buff, BUFF_SIZE);     
    std::cout << "sum 2p " << (sum2[0]+ sum2[1] +sum2[2] +sum2[3])  << std::endl;
    // std::cout << "sum2 0 " << (sum2[0])  << std::endl;
    // std::cout << "sum2 1 " << (sum2[1])  << std::endl;
    // std::cout << "sum2 2 " << (sum2[2])  << std::endl;
    // std::cout << "sum2 3 " << (sum2[3])  << std::endl;
    end_time = clock();
    std::cout << "cost 2p " << (end_time - start_time) << std::endl;

    free(buff);
    buff = nullptr;

    return 0;
}