#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h> 
#include <iostream>
#include <memory>


int main(){
    //__m256d v0 = _mm256_set_pd(1.0f, 2.0f, 3.0f, 4.0f);
    //__m256d v1 = _mm256_set_pd(8.5f, 7.f, 6.f, 5.f);
    double _v0[] = {1.0f, 2.0f, 3.0f, 4.0f , 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f , 1.0f, 2.0f, 3.0f, 4.0f};
    double _v1[] = {8.5f, 7.f, 6.f, 5.f, 8.5f, 7.f, 6.f, 5.f, 8.5f, 7.f, 6.f, 5.f, 8.5f, 7.f, 6.f, 5.f};

    __m256h v0 =_mm256_load_ph(_v0);
    __m256h v1 =_mm256_load_ph(_v1);

    __m256h result0 = _mm256_load_ph(v0, v1);

    return 0;
}