#include <iostream>
#include <thread>
#include <mutex>
#include <string>

//thread_local 
static int a = 1;
std::mutex mutex_lock;

class CA {
    public:
        int n = 123;
        virtual void getN(){
            std::cout << n <<std::endl;
        }
};

// int CA::n = 123;

class CB : public CA {
    public:
        static int n;
};

int CB::n = 321;

void inc_a(std::string name){
    std::lock_guard<std::mutex> guard(mutex_lock);
    a++;
    std::cout << name << " " << a << std::endl;
}

int main(){
    std::cout << "multi threads" << std::endl;

    std::thread td_1(inc_a, "a1");
    std::thread td_2(inc_a, "b2");
    {
        std::lock_guard<std::mutex> guard(mutex_lock);
        std::cout << "main thread " << a << std::endl;
    }

    td_1.join();
    td_2.join();

    CB cb;
    cb.n = 333;
    cb.getN();

    return 0;
}