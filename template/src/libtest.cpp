#include <iostream>
#include "libtest.h"

template <typename T>
void test_lib(T val){
    printf("I'm lib val is %f \n", val);
}

template void test_lib(float val);