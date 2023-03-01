//  g++ stencil.cpp  -o outputs/stencil.o -pthread -std=c++17 -lbenchmark -isystem benchmark/include  -Lbenchmark/build/src -fopenmp

#include <iostream>
#include <vector>
#include <cmath>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h> 
#include "narray.h"

size_t nx = 1 << 13;
size_t ny = 1 << 13; 
//std::vector<int> a(nx*ny);
ndarray<2, float, 16> a(nx, ny);
ndarray<2, float, 16> b(nx, ny);
size_t nblur = 4;

void BM_x_blur_prefetched(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++){
            for (int x = 0; x < nx; x++){
                _mm_prefetch(&a(x+32, y), _MM_HINT_T0);
                float res = 0;
                for (int t = -nblur; t <= nblur; t++){
                    res += a(x+t, y);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_x_blur_prefetched);

void BM_x_blur_tiled_prefetched(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++){
            for (int xBase = 0; xBase < nx; xBase += 16) {
                _mm_prefetch(&a(xBase + 16, y), _MM_HINT_T0);
                for (int x = xBase; x < xBase+16; x++){
                    float res = 0;
                    for (int t = -nblur; t <= nblur; t++){
                        res += a(x+t, y);
                    }
                    b(x, y) = res;
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_x_blur_tiled_prefetched);


void BM_x_blur_tiled_prefetched_streamed(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++){
            for (int xBase = 0; xBase < nx; xBase += 16) {
                _mm_prefetch(&a(xBase + 16, y), _MM_HINT_T0);
                for (int x = xBase; x < xBase+16; x += 4){
                    __m128  res = _mm_setzero_ps();
                    for (int t = -nblur; t <= nblur; t++){
                        res = _mm_add_ps(res, _mm_loadu_ps(&a(x+t, y)));
                    }
                    _mm_stream_ps(&b(x, y), res);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_x_blur_tiled_prefetched_streamed);

void BM_transpose(benchmark::State &bm){
    for (auto _: bm){
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++){
            for (int x =0; x< nx; x++){
                b(x, y) = a(y, x);
            }
        }
        benchmark::DoNotOptimize(b);
    }

}
BENCHMARK(BM_transpose);


void BM_transpose_tiled(benchmark::State &bm){
    for (auto _: bm){
        constexpr int blockSize = 64;
#pragma omp parallel for collapse(2)
        for (int yBase = 0; yBase < ny; yBase += blockSize){
            for (int xBase =0; xBase < nx; xBase += blockSize){
                for (int y = yBase; yBase < ny + blockSize; yBase++){
                    for (int x = xBase; xBase < nx+blockSize; xBase++){                        
                        b(x, y) = a(y, x);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(b);
    }

}
BENCHMARK(BM_transpose_tiled);

BENCHMARK_MAIN();