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

class CC {
    private:
        CC(){
            std::cout << "CC create" << std::endl;
        }
        static CC* impl;
    public: 
        int nx;
        CC(const CC& other) = delete;
        static CC* getInstance(){
            if (impl == nullptr) {
                impl = new CC();
            }
            return impl;
        };
};

CC* CC::getInstance() = nullptr;

void inc_a(std::string name){
    a++;
    std::lock_guard<std::mutex> guard(mutex_lock);
    std::cout << name << " " << a << std::endl;
}

int main(){
    // todo not work in g++
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
    // code that builds only under ThreadSanitizer
        std::cout << "thread_sanitizer go go go" << std::endl;
    #  endif
    #endif

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

    CC* cc = CC::getInstance();
    //std::cout << "singleton data " <<  cc->nx << std::endl;
    return 0;
}