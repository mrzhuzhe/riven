//  gcc -fomit-frame-pointer -fverbose-asm -S  asm.cpp -o outputs/asm.S
//  gcc -O3 -fomit-frame-pointer -fverbose-asm -S  asm.cpp -o outputs/asm.S
int func() {
    return 42;
}

int func2(int a, int b, int c, int d, int e, int f) {
    return a;
}

int func3(int a, int b) {
    return a * b;
}

long long func3(long long a, long long b) {
    return a * b;
}

int func4(int a, int b) {
    return a + b;
}

// arithmatic reduce
int func5(int a, int b) {
    int c = a + b;
    int d = a - b;
    return (c + d)/2;
}

// variable fold
int func6() {
    int a = 32;
    int b = 10;
    return a + b; 
}

// loop evaluate
int func7() {
    int ret = 0;
    for (int i = 1; i <= 100; i++){
        ret += i;
    }
    return ret;
}

#include <vector>
#include <array>
#include <numeric>

int func8(){
    std::vector<int> arr;
    for (int i=0; i<100; i++){
        arr.push_back(i);
    }
    //return std::reduce(arr.begin(), arr.end());
    return 0;
}

int func9(){
    std::array<int, 100> arr;
    for (int i =1; i <= 100; i++){
        arr[i-1] = i;
    }
    int ret = 0;
    for (int i =1; i<= 100; i++){
        ret += arr[i-1];
    }
    return ret;
}


constexpr int func10(){
    std::array<int, 10> arr{};
    for (int i =1; i <= 10; i++){
        arr[i-1] = i;
    }
    int ret = 0;
    for (int i =1; i<= 10; i++){
        ret += arr[i-1];
    }
    return ret;
}

// pointer aliasing 
// func11(&a, &b, &b)
// b = a 
// b = b
void func11(int *a, int *b, int *c){
    *c = *a;
    *c = *b;
}

void func12(int *a){
    a[0] = 123;
    //a[1] = 456;
    a[2] = 456;
}

// SIMD xmm 128 register
void func13(int *a){
    a[0] = 111;
    a[1] = 222;
    a[2] = 333;
    a[3] = 444;
}

// memset
void func14(int *a, int n){
    for (int i = 0; i < n; i ++){
        a[i] = 0;
    }
}

// openmp 
// gcc -fopenmp -O3 -fomit-frame-pointer -fverbose-asm -S  asm.cpp -o outputs/asm.S
void func15(float *a, float *b) {
#pragma omp simd
    for (int i = 0; i < 1024; i++){
        a[i] = b[i] + 1;
    }
}



void func16(){
    
}