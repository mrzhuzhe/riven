// resource acquisition is initialization 

#include <vector>
#include <iostream>
#include <algorithm>
//#include <numeric> // cpp 17 cpp20 ranges module auto coroutine generator format
#include <fstream>
#include <stdexcept>
#include <memory>

int sum3 = 0;
void func(int vi){
    sum3 += vi;
}

void test(){
    std::ofstream fout("a.txt");
    fout << "there is a bean\n";
    throw std::runtime_error("sudden death");
    fout << "call java bean";
}

struct Demo {
    explicit Demo(std::string a, std::string b){
        std::cout << "Demo(" << a << "," << b << ")" << std::endl;
    }
};

///*
struct Pig {
    std::string m_name;
    int m_weight{100};
    Demo m_demo{"Hello", "demo"};
    //Demo m_demo; // can not be initialized without params
    //  explicit for single value
    explicit Pig(std::string name, int weight): m_name("peiqi111" + std::to_string(weight) + "123")
    //,m_weight(888)
    {
        //m_name = "peiqi";
        //m_weight = 80;
    }
    /*
    Pig(Pig const &) = delete;  // forbidden constructor Pig
    Pig &operator=(Pig const &) = delete;   // forbidden copy assign
    */
    Pig()
    {}

    Pig(Pig const &other)
        : m_name(other.m_name)
        , m_weight(other.m_weight)
    {}
    
    Pig &operator=(Pig const &other){
        m_name = other.m_name;
        m_weight = other.m_weight;
        return *this;
    }

    Pig(Pig &&other)
    : m_name(std::move(other.m_name))
    , m_weight(std::move(other.m_weight))
    {}

    Pig &operator=(Pig &&other){
        m_name = std::move(other.m_name);
        m_weight = std::move(other.m_weight);
        return *this;
    }

    ~Pig() {}

};
//*/


// default struct 

struct C {
    C(){
        printf("initialize C\n");
    };    // constructor
    C(C const &c);  // copy construcor
    C(C &&c);   // move construcor
    C &operator=(C const &c);  // copy assgin
    C &operator=(C &&c);    // move assign

    ~C(){
        printf("destruct C\n");
    }; // destructor

    void do_something(){
        printf("inline function do something\n");
    }
};


//void funcp(std::unique_ptr<C> p){
void funcp(C *p){
    p->do_something();
}

struct Vector {
    size_t m_size;
    int *m_data;

    Vector(size_t n) {
        m_size = n;
        m_data = (int *)malloc(n * sizeof(int));
        // memcpy(m_data, other.m_data, m_size * sizeof(int))
        // rreturn *this;
    }

    Vector &operator=(Vector const &other){
        this->~Vector();
        new (this) Vector(other);
        return *this;
    }

    ~Vector(){
        free(m_data);
    }

    size_t size() {
        return m_size;
    }

    void resize(size_t size){
        m_size = size;
        m_data = (int *)realloc(m_data, m_size);
    }

    int &operator[](size_t index) { // sub script 
        return m_data[index];
    }

};

void test_copy(){
    std::vector<int> v1(10);
    std::vector<int> v2(200);
    v1 = v2;
    std::cout << "after copy " << std::endl;
    std::cout << "v1 length" << v1.size() << std::endl;
    std::cout << "v2 length" << v2.size() << std::endl;
}

void test_move(){
    std::vector<int> v1(10);
    std::vector<int> v2(200);

    v1 = std::move(v2);

    std::cout << "after move" << std::endl;
    std::cout << "v1 length" << v1.size() << std::endl;
    std::cout << "v2 length" << v2.size() << std::endl;
}

void test_swap() {

    std::vector<int> v1(10);
    std::vector<int> v2(200);

    std::swap(v1, v2);

    std::cout << "after swap" << std::endl;
    std::cout << "v1 length" << v1.size() << std::endl;
    std::cout << "v2 length" << v2.size() << std::endl;
}

std::vector<std::unique_ptr<C>> objlist;
std::vector<std::shared_ptr<C>> objlist2;

void funcp2(std::unique_ptr<C> p){
    objlist.push_back(std::move(p));
}

void funcp3(std::shared_ptr<C> p){
    objlist2.push_back(std::move(p));
}

int main() {

    size_t nv = 4;
    int *v = (int *)malloc(nv * sizeof(int));
    v[0] = 4;
    v[1] = 3;
    v[2] = 2;
    v[3] = 1;
    int sum = 0;
    for (size_t i =0; i< nv; i++){
        sum += v[i];
    }
    printf("%d\n", sum);
    free(v);


    std::vector<int> v1(4);
    v1[0] = 4;
    v1[1] = 3;
    v1[2] = 2;
    v1[3] = 1;
    v1 = {5, 4, 3, 2, 1};

    int sum1 = 0;
    for (size_t i = 0; i< v1.size(); i++){
        sum1 += v1[i];
    }
    std::cout << sum1 << std::endl;

    int sum2 = 0;
    // range based for-loop
    for (int vi: v1){
        sum2 += vi;
    }
    std::cout << sum2 << std::endl;


    std::for_each(v1.begin(), v1.end(), func);
    std::cout << "sum3 " << sum3 << std::endl;
    
    // lambda phrase
    int sum4 = 0;
    //std::for_each(v1.begin(), v1.end(), [&](int vi){
    std::for_each(v1.begin(), v1.end(), [&](auto vi){
        sum4 += vi;
    });

    std::cout << "sum4 " << sum4 << std::endl;

    //int sum5 = std::reduce(v1.begin(), v1.end());
    //int sum5 = std::reduce(v1.begin(), v1.end(), 0, std::plus{});

    //std::cout << sum5 << std::endl;

    // silly smart 
    std::ifstream f1("f1.txt");
    /*
    if (checkFileContent(f1)){
        printf("bad file 1!\n");
        f1.close();
        return 1;
    };
    */
    std::vector<std::ifstream> files;
    files.push_back(std::ifstream("3.txt"));

    // f1 files will be auto released when excution finished
    for (auto &file: files)
        file.close();

    f1.close();

    try {

    } catch (std::exception const &e) {
        std::cout << "catch exception: " << e.what() << std::endl;
    }


    // Cpp Trap 
    Pig pig{ "fpigggg", 999};
    // Pig pig = 80;
    
    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;

    Pig pig2 = pig;

    //std::cout << "name: " << pig2.m_name << std::endl;
    //std::cout << "weight: " << pig2.m_weight << std::endl;

    Vector v2(2);
    v2[0] = 4;
    v2[1] = 3;
    v2.resize(4);

    v2[2] = 2;
    v2[3] = 1;

    int sum6 = 0;
    for (size_t i = 0; i < v2.size(); i++ ){
        sum6 += v2[i];
    }

    std::cout << sum6 << std::endl;
    test_copy();
    test_move();
    test_swap();

    std::unique_ptr<C> p = std::make_unique<C>();
    /*
    if (1 + 1 == 2){
        printf("something wrong\n");
        return 1;
    }
    */
    funcp(p.get());

    funcp2(std::move(p));

    p->do_something();

    printf("123123\n");

    std::shared_ptr<C> p2= std::make_shared<C>();

    printf("user count = %ld\n", p2.use_count());

    funcp3(p2);
    p2->do_something();
    objlist2.clear();
    p2->do_something();

    printf("user count = %ld\n", p2.use_count());

    std::weak_ptr<C> weak_p = p2;

    printf("use count = %ld\n", p2.use_count());

    funcp3(std::move(p2));

    if (weak_p.expired()){
        printf("weak ptr expired\n");
    } else {
        weak_p.lock()->do_something();
    }

    objlist2.clear();
    if (weak_p.expired()){
        printf("weak ptr expired\n");
    } else {
        weak_p.lock()->do_something();
    }


    return 0;
}