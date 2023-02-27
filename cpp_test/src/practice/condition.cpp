#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <shared_mutex> 
#include <condition_variable>
// g++ condition.cpp  -o outputs/condition.o -pthread -std=c++17



template <class T>
class MTQueue {
    std::condition_variable m_cv;
    std::mutex m_mtx;
    std::vector<T> m_arr;
public:
    T pop(){
        std::unique_lock lck(m_mtx);
        m_cv.wait(lck, [this]{
            return !m_arr.empty();
        });
        T ret = std::move(m_arr.back());
        m_arr.pop_back();
        return ret;
    }
    auto pop_hold(){
        std::unique_lock lck(m_mtx);
        m_cv.wait(lck, [this]{ return !m_arr.empty();   });
        T ret = std::move(m_arr.back());
        m_arr.pop_back();
        return std::pair(std::move(ret), std::move(lck));
    }
    void push(T val){
        std::unique_lock lck(m_mtx);
        m_arr.push_back(std::move(val));
        m_cv.notify_one();
    }
    void push_many(std::initializer_list<T> vals){
        std::unique_lock lck(m_mtx);
        std::copy(
            std::move_iterator(vals.begin()),
            std::move_iterator(vals.end()),
            std::back_insert_iterator(m_arr)
        );
        m_cv.notify_all();
    }
};


int main(){
    std::condition_variable cv;
    std::mutex mtx;
    std::vector<int> foods;
    MTQueue<int> foods2;

    std::thread t1(
        [&] {
            for (int i = 0; i < 2; i++) {
                /*
                std::unique_lock lck(mtx);
                cv.wait(lck, [&]{
                    return foods.size() != 0;
                });
                auto food = foods.back();
                foods.pop_back();
                lck.unlock();
                */
                auto food = foods2.pop();
                std::cout << "t1 got food:" << food << std::endl;
            }
        }
    );

    std::thread t2(
        [&] {
            for (int i = 0; i < 2; i++) {
                /*
                std::unique_lock lck(mtx);
                cv.wait(lck, [&]{
                    return foods.size() != 0;
                });
                auto food = foods.back();
                foods.pop_back();
                lck.unlock();
                */
                auto food = foods2.pop();
                std::cout << "t2 got food:" << food << std::endl;
            }
        }
    );

    /*
    foods.push_back(42);
    cv.notify_one();

    foods.push_back(233);
    cv.notify_one();

    foods.push_back(666);
    foods.push_back(4399);
    cv.notify_all();
    */

    foods2.push(42);
    foods2.push(233);
    foods2.push_many({666, 4399});

    t1.join();
    t2.join();


    return 0;
}