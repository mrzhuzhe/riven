#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    
struct AA {
    AA ** array;
};
void nullifyy_array(AA *Ptr, int index, AA *ThisPtr){
    // std::cout << ((*(Ptr->array))+(index--)) << std::endl;
    // std::cout << (*(Ptr->array)+1) << std::endl;
    
    while (((*(Ptr->array))+(index--)) != ThisPtr) {
        AA* _ptr = *(Ptr->array)+(index);
        std::cout << _ptr << " " <<  ThisPtr << std::endl;
        _ptr = NULL;
    };
};

void test_speculation(AA* Ptr, int index, AA* ThisPtr);

#ifdef __cplusplus
}
#endif