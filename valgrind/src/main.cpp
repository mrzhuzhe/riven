#include <iostream>
class A {
    public:
        A(){
            std::cout << "a construct" << std::endl; 
        };
        ~A(){
            std::cout << "a desconstruct" << std::endl; 
        };
        int* bbb;
};

int main(){
    //  valgrind --leak-check=yes ./build/bin/main
    int* x = (int*)malloc(10 * sizeof(int));
    //x[10] = 0;        // problem 1: heap block overrun
                      // problem 2: memory leak -- x not freed
    // wind pointer / core dump / memory disambigulation
    int* a = new int{4};
    delete a;
    std::cout << !a << std::endl;
    a = NULL;
    std::cout << !a << std::endl; 
    A* b = new A;
    b->bbb = new int{5};
    delete b;
    std::cout << b->bbb << " " /* << *(b->bbb) << " " */ << !b->bbb << " " << std::endl;


    return 0;
}