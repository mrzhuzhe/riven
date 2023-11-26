#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <pthread.h>
#include <cstring>

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

// class CC {
//     private:
//         CC(){
//             std::cout << "CC create" << std::endl;
//         }
//         static CC* impl;
//     public: 
//         int nx;
//         CC(const CC& other) = delete;
//         static CC* getInstance(){
//             if (impl == nullptr) {
//                 impl = new CC();
//             }
//             return impl;
//         };
// };

// CC* CC::getInstance() = nullptr;

void inc_a(std::string name){
    a++;
    std::lock_guard<std::mutex> guard(mutex_lock);
    std::cout << name << " " << a << std::endl;
}


//  https://man7.org/linux/man-pages/man3/pthread_create.3.html
struct thread_info{
    pthread_t thread_id;
    int thread_num;
    int arg_string;    
};

void* thread_fn(void *arg){
    struct  thread_info *tinfo  = (thread_info *)arg;    
    std::cout << "thread fn " << tinfo->arg_string << std::endl;
    return (void*)tinfo;
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

    // CC* cc = CC::getInstance();
    //std::cout << "singleton data " <<  cc->nx << std::endl;

    int status;
    void* res;
    size_t num_threads = 2;
    pthread_attr_t attr;
    thread_info tinfo[num_threads];

    status = pthread_attr_init(&attr);

    for (size_t tnum = 0; tnum < num_threads; tnum++){
        tinfo[tnum].thread_num = tnum;
        //char str[20] = strcat("asdasdsadasd", (char*)tnum);
        tinfo[tnum].arg_string = 12 + tnum;
        status = pthread_create(&tinfo[tnum].thread_id, &attr, &thread_fn, &tinfo[tnum]);
        if (status) {
            std::cout << "create status " << status << std::endl;
        }
    }

    status = pthread_attr_destroy(&attr);

    for (size_t tnum = 0; tnum < num_threads; tnum++){
        status = pthread_join(tinfo[tnum].thread_id, &res);
        if (status) {
            std::cout << "join " << tinfo[tnum].thread_id << " "  << status << std::endl;
        }
        std::cout << "joined" << ((thread_info *)res)->thread_num << std::endl;
    }
    
    return 0;
}