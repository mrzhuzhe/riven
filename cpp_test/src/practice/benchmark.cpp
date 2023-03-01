//  g++ benchmark.cpp  -o outputs/benchmark.o -pthread -std=c++17 -lbenchmark -isystem benchmark/include  -Lbenchmark/build/src -fopenmp

#include <iostream>
#include <vector>
#include <cmath>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h> 
// https://curc.readthedocs.io/en/latest/programming/OpenMP-C.html

constexpr size_t n = 1 << 27;
std::vector<float> a(n);


static uint32_t randomize(uint32_t i) {
	i = (i ^ 61) ^ (i >> 16);
	i *= 9;
	i ^= i << 4;
	i *= 0x27d4eb2d;
	i ^= i >> 15;
    return i;
}

/*
void BM_fill(benchmark::State &bm){
    for (auto _: bm){
        for (size_t i = 0; i<n; i++){
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill);

void BM_parallel_fill(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n; i++) {
            a[i] = 1;
        }        
    }
}
BENCHMARK(BM_parallel_fill);

void BM_sine(benchmark::State &bm){
    for (auto _: bm){
        for (size_t i = 0; i < n; i++){
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_sine);

void BM_parallel_sine(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i =0; i< n; i++){
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_parallel_sine);

void BM_serial_add(benchmark::State &bm){
    for (auto _: bm){
        for (size_t i =0; i< n;i++){
            a[i] = a[i] + 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_serial_add);

void BM_parallel_add(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i =0; i< n; i++){
            a[i] = a[i] + 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_parallel_add);

static float func(float x) {
    return x * (x*x+x*3.14f - 1/(x+1)) + 42 / (2.718f -x);
}

void BM_serial_func(benchmark::State &bm){
    for (auto _:bm){
        for (size_t i = 0; i < n; i++){
            a[i] = func(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_serial_func);

void BM_parallel_func(benchmark::State &bm){
    for (auto _:bm){
#pragma omp parallel for 
        for (size_t i = 0; i<n;i++){
            a[i] = func(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_parallel_func);
*/
void BM_skip1(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n;  i +=1) {
            a[i] = 1;
        }        
    }
}
BENCHMARK(BM_skip1);

void BM_skip2(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n;  i +=2) {
            a[i] = 1;
        }        
    }
}
BENCHMARK(BM_skip2);

void BM_skip4(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n;  i +=4) {
            a[i] = 1;
        }        
    }
}
BENCHMARK(BM_skip4);

void BM_skip8(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n;  i +=8) {
            a[i] = 1;
        }        
    }
}
BENCHMARK(BM_skip8);


void BM_aos(benchmark::State &bm){
    struct MyClass {
        float x;
        float y;
        float z;
    };
    std::vector<MyClass> mc(n);

    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n;  i +=1) {
            mc[i].x = mc[i].x + mc[i].y;
        }       
        benchmark::DoNotOptimize(mc);
    }
}
BENCHMARK(BM_aos);

void BM_soa(benchmark::State &bm){
    std::vector<float> mc_x(n);
    std::vector<float> mc_y(n);
    std::vector<float> mc_z(n);

    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i =0; i < n; i++){
            mc_x[i] = mc_x[i] + mc_y[i];
        }
        benchmark::DoNotOptimize(mc_x);
        benchmark::DoNotOptimize(mc_y);
        benchmark::DoNotOptimize(mc_z);

    }
}
BENCHMARK(BM_soa);
// aos and soa seems not different 


void BM_aosoa(benchmark::State &bm){
    struct MyClass {
        float x[1024];
        float y[1024];
        float z[1024];
    };
    std::vector<MyClass> mc(n/1024);

    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n/1024;  i +=1) {
#pragma omp simd
            for (size_t j =0; j < 1024; j++){
                mc[i].x[j] = mc[i].x[j] + mc[i].y[j];
            }
        }       
        benchmark::DoNotOptimize(mc);
    }
}
BENCHMARK(BM_aosoa);

void BM_random_64B(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n / 16; i++){
            size_t r = randomize(i) % (n / 16);
            for (size_t j = 0; j < 16; j++){
                benchmark::DoNotOptimize(a[r*16 + j]);
            }
        benchmark::DoNotOptimize(a);
        }
    }
}
BENCHMARK(BM_random_64B);

void BM_random_64B_prefetch(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for 
        for (size_t i = 0; i < n / 16; i++){
            size_t next_r = randomize(i + 64) % (n / 16);
            _mm_prefetch(&a[next_r*16], _MM_HINT_T0);
            size_t r = randomize(i) % (n /16);
            for (size_t j = 0; j < 16; j++){
                benchmark::DoNotOptimize(a[r*16 + j]);
            }
        benchmark::DoNotOptimize(a);
        }
    }
}
BENCHMARK(BM_random_64B_prefetch);


BENCHMARK_MAIN();


