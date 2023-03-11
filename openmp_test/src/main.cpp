#include <iostream>
#include "TikTok.h"
//#include <omp.h>

float cpuPi(int num) {
    float sum = 0.0;
    float temp;
    for (int i = 0; i< num; i++ ){
        temp = (i + 0.5f)/num;
        sum += 4 / ( 1 + temp * temp);
    }
    return sum/num;
};

float cpuPiCriticNaive(int num) {
    float sum = 0.0;
#pragma omp parallel for default(none) shared(num, sum)
    for (int i = 0; i< num; i++ ){
        float temp = (i + 0.5f)/num;
    #pragma omp critical
    {
        sum += 4 / ( 1 + temp * temp);
    }        
    }
    return sum/num;
};



float cpuPiCritic(int num) {
    float sum = 0.0;
#pragma omp parallel for default(none) shared(num, sum)
    for (int i = 0; i< num; i++ ){
        float temp = (i + 0.5f)/num;
        temp = 4 / ( 1 + temp * temp);
    #pragma omp critical
    {
        sum += temp;
    }        
    }
    return sum/num;
};

float cpuPiAtomic(int num) {
    float sum = 0.0;
#pragma omp parallel for default(none) shared(num, sum)
    for (int i = 0; i< num; i++ ){
        float temp = (i + 0.5f)/num;
        temp = 4 / ( 1 + temp * temp);
    #pragma omp atomic
        sum += temp;
    }      
    return sum/num;
};

float cpuPiReduction(int num) {
    float sum = 0.0;
#pragma omp parallel for default(none) shared(num) reduction(+:sum)
    for (int i = 0; i< num; i++ ){
        float temp = (i + 0.5f)/num;
        temp = 4 / ( 1 + temp * temp);
        sum += temp;   
    }
    return sum/num;
};

float cpuPiSIMDReduction(int num) {
    float sum = 0.0;
#pragma omp parallel for simd default(none) shared(num) reduction(+:sum)
    for (int i = 0; i< num; i++ ){
        float temp = (i + 0.5f)/num;
        temp = 4 / ( 1 + temp * temp);
        sum += temp;   
    }
    return sum/num;
};

int main(){
    const int num = 100000;
    long s, e;
    float pi;
    Tik();
    printf("cpuPi: %f\n", cpuPi(num));
    Tok("cpuPi");

    Tik();
    printf("cpuPiCriticNaive: %f\n", cpuPiCriticNaive(num));
    Tok("cpuPiCriticNaive");

    Tik();
    printf("cpuPiCritic: %f\n", cpuPiCritic(num));
    Tok("cpuPiCritic");

    Tik();
    printf("cpuPiAtomic: %f\n", cpuPiAtomic(num));
    Tok("cpuPiAtomic");

    Tik();
    printf("cpuPiReduction: %f\n", cpuPiReduction(num));
    Tok("cpuPiReduction");

    Tik();
    printf("cpuPiSIMDReduction: %f\n", cpuPiSIMDReduction(num));
    Tok("cpuPiSIMDReduction");

    return 0;
}


/*
M1 result

cpuPi: 3.141532
cpuPi time elapsed: 0.1485 ms
cpuPiCriticNaive: 3.141596
cpuPiCriticNaive time elapsed: 160.101 ms
cpuPiCritic: 3.141572
cpuPiCritic time elapsed: 115.227 ms
cpuPiAtomic: 3.141444
cpuPiAtomic time elapsed: 16.0028 ms
cpuPiReduction: 3.141593
cpuPiReduction time elapsed: 0.819416 ms
cpuPiSIMDReduction: 3.141593
cpuPiSIMDReduction time elapsed: 0.092542 ms

*/