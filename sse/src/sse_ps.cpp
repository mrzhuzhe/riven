//  https://zhuanlan.zhihu.com/p/55327037

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <iostream>
#include <memory>

int main()
{
    __m128 v0 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
    __m128 v1 = _mm_set_ps(4.5f, 3.0f, 2.0f, 1.0f);

    __m128 result0 = _mm_add_ps(v0, v1);
    float a0[4];
    _mm_store_ps(a0, result0);
    std::cout << a0[0] << a0[1] << a0[2] << a0[3] << std::endl;

    //__declspec(align(16)) float p0[] = {1.1f, 2.1f, 3.1f, 4.1f};
    float p0[] = {1.1f, 2.1f, 3.1f, 4.1f};
    float p1[] = {4.5f, 3.0f, 2.0f, 1.0f};
    __m128 v2 = _mm_load_ps(p0);
    __m128 v3 = _mm_load_ps(p1);
    __m128 result1 = _mm_add_ps(v2, v3);
    float a1[4];
    _mm_store_ps(a1, result1);
    std::cout << a1[0] << a1[1] << a1[2] << a1[3] << std::endl;

}