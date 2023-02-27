//  g++ benchmark.cpp  -o outputs/benchmark.o -pthread -std=c++17 -lbenchmark -isystem benchmark/include  -Lbenchmark/build/src

#include <iostream>
#include <vector>
#include <cmath>
#include <benchmark/benchmark.h>

constexpr size_t n = 1 << 26;
std::vector<float> a(n);

void BM_fill(benchmark::State &bm){
    for (auto _: bm){
        for (size_t i = 0; i<n; i++){
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill);

BENCHMARK_MAIN();
/*
int main() {
    return 0;
}
*/