#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>


int main(){

    /*
    float x = -3.14;
    //printf("%f\n", (x));
    //printf("%f\n", std::abs(x));
    int size = 1000;
    int *a = (int *)malloc(size);
    a[0] = 123;
    a[1001] = 456;
    printf("%d %d %d %d\n", a[1001], a[0], a[1], a[1002]);
    //char str[10];
    //scanf("%10s", str);
    //printf("%s\n", str);
    */


    static_assert(std::is_same<decltype(0x7fffffff), int>::value, "小彭老师的断言");
    static_assert(std::is_same<decltype(0xffffffff), unsigned int>::value, "小彭老师的断言1");
    static_assert(std::is_same<decltype(0x100000000), int64_t>::value, "小彭老师的断言2");
    static_assert(std::is_same<decltype(0xffffffff'ffffffff), uint64_t>::value, "小彭老师的断言3");
    return 0;
}
 