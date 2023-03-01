//  g++ loopop.cpp -o outputs/loopop.o -fopenmp 
//  must with openmp

#include <iostream>
#include <vector>
#include "TikTok.h"
#include <cmath>
#include <omp.h>
#include <x86intrin.h>

constexpr size_t n = 1<<26;
constexpr size_t steps = 32;

std::vector<float> a(n);
std::vector<float> b(n);

int main(){
#pragma omp parallel for 
    for (size_t i = 0; i < n; i++){
        a[i] = std::sin(i*0.1f);
    }
    Tik();
    for (int step = 0; step < steps; step++){
#pragma omp parallel for
        for (size_t i =1; i< n-1; i++){
            b[i] = (a[i-1] + a[i+1]) * 0.5f;
        }
        std::swap(a, b);
    }
    Tok("iter");
    float loss = 0;
#pragma omp parallel for reduction(+:loss)
    for (size_t i = 1; i< n-1; i++){
        loss += std::pow(a[i-1] + a[i+1] - a[i]*2, 2);
    }
    loss = std::sqrt(loss);
    std::cout << "loss: " << loss << std::endl;
    return 0;
}