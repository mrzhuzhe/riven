#include <iostream>

class A {    
    public:
        A (int n){
            value = n;
        }
        int value = 1;
        int operator+(int n){
            std::cout << "add " << value + n << std::endl;
            return value + n;
        }
};

int main(){
    A a(2);
    int b;
    b = a
    std::cout << " b " << b << std::endl;
    return 0;
}