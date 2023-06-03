

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>  // avx
#include <iostream>
#include <memory>

int main()
{
    //__m256d v0 = _mm256_set_pd(1.0f, 2.0f, 3.0f, 4.0f);
    //__m256d v1 = _mm256_set_pd(8.5f, 7.f, 6.f, 5.f);
    double _v0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    double _v1[] = {8.5f, 7.f, 6.f, 5.f};

    __m256d v0 =_mm256_load_pd(_v0);
    __m256d v1 =_mm256_load_pd(_v1);

    __m256d result0 = _mm256_add_pd(v0, v1);
    double a0[4];
    _mm256_store_pd(a0, result0);
    std::cout << a0[0] << " " << a0[1] << " " << a0[2] << " " << a0[3] << " " << std::endl;

    //https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/mm256-shuffle-pd.html
    
    // Shuffle vb (b1,b0,b3,b2)
    __m256d r0 = _mm256_shuffle_pd( v0, v0, 0x5 );
    _mm256_store_pd(a0, r0);
    std::cout << a0[0] << " " << a0[1] << " " << a0[2] << " " << a0[3] << " " << std::endl;

    // Permute vb (b3,b2,b1,b0)
    __m256d r1 = _mm256_permute2f128_pd( r0, r0, 0x1 );
    _mm256_store_pd(a0, r1);
    std::cout << a0[0] << " " << a0[1] << " " << a0[2] << " " << a0[3] << " " << std::endl;

    // Shuffle vb (b2,b3,b0,b1) 
    __m256d r2 = _mm256_shuffle_pd( r1, r1, 0x5 );
    _mm256_store_pd(a0, r2);
    std::cout << a0[0] << " " << a0[1] << " " << a0[2] << " " << a0[3] << " " << std::endl;

    //
    __m256d r3 = _mm256_blend_pd( v0, v1, 0x6 );
    _mm256_store_pd(a0, r3);
    std::cout << "bend" << std::endl;
    std::cout << a0[0] << " " << a0[1] << " " << a0[2] << " " << a0[3] << " " << std::endl;
    
}