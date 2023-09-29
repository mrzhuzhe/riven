#include <iostream>
#include "animal.hpp"   // include is in the end of file

// template <typename T>
// void print_name(const T* type){
//     printf("cat name is %s \n", type);
// }

void print_name(const char* type){
    printf("cat name is %s \n", type);
}

//template void print_name(const char* type); 

//  extern template void print_name(const char* type);  // not work