#include <iostream>

class A {  
    private:
        int value = 1;  
    public:
        A (){

        }
        A (int n){
            value = n;
        }
        int getVal() {
            return value;
        }
        A operator+(int n){
            std::cout << "add int " << value << "+" << n << std::endl;
            return A(this->value + n);
        }
        A operator+(A n){
            std::cout << "add A " << value << "+" << n.getVal() << std::endl;
            return A(this->value + n.getVal());
        }
        A operator=(A n){
            std::cout << "assign " << n.getVal() << std::endl;
            this->value = n.getVal();
            return *this;
        }

};

int main(){
    A a(2);
    A b;
    b = a+1+a+2+a+3;
    std::cout << " b " << b.getVal() << std::endl;

    std::cout << sizeof(unsigned int) << " " << sizeof(float) << std::endl;

    char ch1 = 'x';
    char chrs[] = "asdasdas";
    std::cout << ch1 << " "<< chrs << " " << sizeof(chrs) << " " << sizeof(ch1) << std::endl;
    printf("%c %s \n", ch1, chrs);
    return 0;
}