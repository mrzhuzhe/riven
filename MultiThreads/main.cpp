#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <pthread.h>
#include <cstring>
//  https://stackoverflow.com/questions/4184468/sleep-for-milliseconds
#include <chrono>
#include <thread>

//thread_local int a = 1;
//static int a = 1;
__thread int a = 1;
std::mutex mutex_lock;
pthread_key_t thread_key;

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
    ~thread_info(){
        std::cout << "Thread info destructor "  << this->arg_string << std::endl;
    }
};

pthread_key_t tpskey;

void* thread_fn(void *arg){
    int status;
    thread_info *tinfo  = (thread_info *)arg;
    //pthread_getthreadid_np() is undefined    
    std::this_thread::sleep_for(std::chrono::milliseconds((int)random()%1000));
    std::cout << "thread fn arg" << " " << " "<< tinfo->arg_string << std::endl;
    //tinfo->arg_string = 1231231;
    return tinfo;
}


void* thread_fn2(void *arg){
    int status;
    thread_info *tinfo; 
    tinfo = (thread_info *)arg;

    // thread_info *tinfo3;
    // tinfo3 = (thread_info*)malloc(sizeof(thread_info));
    // tinfo3->thread_num = tinfo->thread_num;
    // tinfo3->arg_string = 100 + tinfo->thread_num;
    status = pthread_setspecific(tpskey, tinfo);
    
    //pthread_getthreadid_np() is undefined    
    std::cout << "thread fn arg" << " " << " "<< tinfo->arg_string << std::endl;
    
    //std::this_thread::sleep_for(std::chrono::milliseconds((int)random()%1000));

    thread_info *tinfo2= (thread_info*)pthread_getspecific(tpskey);
    std::cout << "thread fn " << " " << " "<< tinfo2->arg_string << " " << " "<< tinfo->arg_string << std::endl;

    //tinfo2->arg_string = 456456;

    return NULL;
}

void* thread_fn3(void *arg){
    int status;
    thread_info *tinfo = (thread_info *)arg;
    
    int* a = (int*)malloc(sizeof(int)*10);
    a[0] = (int)random()%1000;    
    std::cout << tinfo->thread_num << " a1 " << a[0] << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(a[0]));
    
    std::cout << tinfo->thread_num << " a2 " << a[0] << std::endl;

    // a[0] = 456;

    // std::cout << " a " << a[0] << " " << (int)random()%3000  << std::endl;


    return NULL;
}


void dataDestructor(void *data) {
    std::cout << "destructor " << ((thread_info*)data)->arg_string << std::endl;
    pthread_setspecific(tpskey, NULL);
    free(data);
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
    
    std::cout << "main thread a final " << a << std::endl;

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
        tinfo[tnum].arg_string = 1000 + tnum;
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
    
    std::cout << "********** pthread_key_create ***************" << std::endl;

    
    status = pthread_key_create(&tpskey, dataDestructor);

    status = pthread_attr_init(&attr);
    
    //status = pthread_setspecific(tpskey, &tinfo[0]);  // no use for main thread
    std::cout << tinfo[0].arg_string << " set all" << std::endl;
    thread_info* singletinfo;
    for (size_t tnum = 0; tnum < num_threads; tnum++){       
        thread_info singletinfo3;
        singletinfo3.thread_num = tnum;
        singletinfo3.arg_string = 12 + tnum;
        std::cout << "This is because struct initial without new will be free in for loop scope so this address must be same " << &singletinfo3 << std::endl;
        
        // status = pthread_create(&tinfo[tnum].thread_id, &attr, &thread_fn2, &singletinfo2);
        
        thread_info *singletinfo2 = new thread_info;
        singletinfo2->thread_num = tnum;
        singletinfo2->arg_string = 100 + tnum;
        status = pthread_create(&tinfo[tnum].thread_id, &attr, &thread_fn2, singletinfo2);

        // singletinfo = (thread_info*)malloc(sizeof(thread_info));
        // singletinfo->thread_num = tnum;
        // singletinfo->arg_string = 100 + tnum;
        // status = pthread_create(&tinfo[tnum].thread_id, &attr, thread_fn2, singletinfo);
        // free(singletinfo);

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
        //std::cout << "joined" << ((thread_info *)res)->thread_num << std::endl;
        //std::cout << "joined " << singletinfo->arg_string << std::endl;
    }
    std::cout << "joined " << singletinfo->arg_string << " notice thread arg 100 and 101 not destructor " << std::endl;
    

    pthread_key_delete(tpskey);


    std::cout << "********** localstorage ***************" << std::endl;

    status = pthread_attr_init(&attr);

    for (size_t tnum = 0; tnum < num_threads; tnum++){           
                
        singletinfo = (thread_info*)malloc(sizeof(thread_info));
        singletinfo->thread_num = tnum;
        singletinfo->arg_string = 100 + tnum;
        status = pthread_create(&tinfo[tnum].thread_id, &attr, &thread_fn3, singletinfo);
        // free(singletinfo);

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
        //std::cout << "joined" << ((thread_info *)res)->thread_num << std::endl;
    }
    

    return 0;
}