#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>

float SqrtByCarmack( float number )
{
    int i;
    float x2, y;
    const float threehalfs = 1.5F;
    x2 = number * 0.5F;
    y = number;
    i = *(int *) &y;
    i = 0x5f375a86 - ( i>>1 );
    y = * (float *) &i;
    y = y * ( threehalfs - (x2 * y * y ));
    y = y * ( threehalfs - (x2 * y * y ));
    y = y * ( threehalfs - (x2 * y * y ));
    return number * y;
}

int func(int& second){
    second = 2;
    return 1;
}

auto makearr() {    
    //int a[1024];
    //int* a = (int*)malloc(1024*sizeof(int));
    std::vector<int> a(1024);
    for (int i=0;i<1024;i++)
        a[i] = i;
    return a;
}


void func(int arr[6], int size){
    printf("sizeof(arr) %ld\n", sizeof(arr));
    for (int i = 0; i < 6; i++ ) {
        printf("%d\n", arr[i]);
    }
}

int main(){

    
    float x = -3.14;
    //printf("%f\n", (x));
    printf("%f\n", std::abs(x));
    /*
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

    //static_assert(sizeof(intptr_t)==sizeof(void *)==sizeof(uintptr_t));
    static_assert(sizeof(intptr_t)==sizeof(void *));
    static_assert(sizeof(intptr_t)==sizeof(uintptr_t));

    printf("uint8_t = %ld\n", sizeof(uint8_t));
    printf("uint16_t = %ld\n", sizeof(uint16_t));
    printf("uint32_t = %ld\n", sizeof(uint32_t));
    printf("uint64_t = %ld\n", sizeof(uint64_t));
    printf("uintptr_t = %ld\n", sizeof(uintptr_t));

    printf("int8_t = %ld\n", sizeof(int8_t));
    printf("int16_t = %ld\n", sizeof(int16_t));
    printf("int32_t = %ld\n", sizeof(int32_t));
    printf("int64_t = %ld\n", sizeof(int64_t));

    printf("intptr_t = %ld\n", sizeof(intptr_t));
    printf("pointer = %ld\n", sizeof(void *));
    
    printf("fsqrt by carnack %f\n", SqrtByCarmack(2));


    int x1 = 1;
    int* p = &x1;
    printf("x = %d\n", x1);
    *p = 2;
    printf("x = %d\n", x1);


    int x2 = 0x12345678;
    int *p1 = &x2;
    char *pc = (char*)p1; // force convert
    printf("%x\n", *pc); // little endian

    int second;
    int first = func(second);
    printf("result: %d %d\n", first, second);


    auto a3 = makearr();
    for (int i =0;i<1024;i++)
        a3[i] += 1;
    //free(a3);

    int arr1[6] = {1, 2, 3, 4, 5, 6};
    func(arr1, 6);

    //std::string str1;
    //printf("input str:\n");
    //std::cin >> str1;
    //printf("str: %s\n", str1.c_str());
    //printf("str: %s\n", str1.c_str());
    
    const char* str2 = "Hello world";
    const char* str3 = "Hello world";
    bool equal = str3 == str2;
    printf("%d\n", equal);

    char str4[] = "hello";
    char str5[] = "hello";
    bool equal2 = str5 == str4;
    printf("%d\n", equal2);

    bool equal3 = !strcmp(str5, str4);
    printf("%d\n", equal3);

    return 0;
}
 