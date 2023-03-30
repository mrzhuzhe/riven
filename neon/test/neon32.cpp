#include "iostream"
#include <vector>
#include "TikTok.h"
#include <arm_neon.h>   //  https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/Developing-for-NEON/Intrinsics?lang=en

const uint32_t n = 65536;

//int a[3*n];
//int b[n];

std::vector<uint32_t> a(n*3);
std::vector<uint32_t> b(n);

void fillArray(uint32_t n, uint32_t *src, uint32_t *dest){
    for (int i=0; i < n; i++){
        src[i*3] = 0+3*i;
        src[i*3+1] = 1+3*i;
        src[i*3+2] = 2+3*i;
        dest[i] = i;
    }
}

void showArray(uint32_t n, uint32_t *src, uint32_t *dest){
    printf("\n");
    for (int i=0; i < n; i++){
        int r = *src++;
        int g = *src++;
        int b = *src++;
        printf("%d %d %d %d\n", r, g,  b, dest[i]);        
    }
}

void gray_native(int n, uint32_t *src, uint32_t *dest){
    for (int i=0; i < n; i++){
        int r = *src++;
        int g = *src++;
        int b = *src++;

        int r_ratio = 77;
        int g_ratio = 151;
        int b_ratio = 28;

        int temp = r * r_ratio;
        temp += g * g_ratio;
        temp += (b * b_ratio);

        dest[i] = (temp>>8);        
    }
}

void gray_uint8(int n, uint32_t *src, uint32_t *dest){
    for (int i=0; i < n; i++){
        uint32_t r = *src++;
        uint32_t g = *src++;
        uint32_t b = *src++;

        uint32_t r_ratio = 77;
        uint32_t g_ratio = 151;
        uint32_t b_ratio = 28;

        int temp = r * r_ratio;
        temp += g * g_ratio;
        temp += (b * b_ratio);

        dest[i] = (temp>>8);        
    }
}


//void gray_mla(int n, int *src, int *dest){
void gray_mla(uint32_t *src, uint32_t *dest, int n){

    //uint32x2_t _src = *src;


    n /= 8;
    uint32x2_t r_ratio = vdup_n_u32(77);
    uint32x2_t g_ratio = vdup_n_u32(151);
    uint32x2_t b_ratio = vdup_n_u32(28);

    //printf("12123123 %d", *src);
    ///*
    for (int i=0; i < n; i++){
        uint32x2x3_t rgb = vld3_u32(src);        
        uint32x2_t r = rgb.val[0];
        uint32x2_t g = rgb.val[1];
        uint32x2_t b = rgb.val[2];

        uint64x2_t y = vmull_u32(r, r_ratio);
        y = vmlal_u32(y, g, g_ratio);
        y = vmlal_u32(y, b, b_ratio);
        uint32x2_t ret = vshrn_n_u64(y, 8);

        vst1_u32(dest, ret);
        src += 3*32;
        dest +=32;
    }
    //*/
}

int main(){
    //int *pa = a;
    //int *pb = b;
    uint32_t *pa = a.data();
    uint32_t *pb = b.data();
    fillArray(n, pa, pb);
    showArray(n, pa, pb);
   
    
    Tik();
    gray_native(n, pa, pb);
    showArray(n, pa, pb);
    Tok("native");

    Tik();
    gray_uint8(n, pa, pb);
    showArray(n, pa, pb);
    Tok("uint8");

    Tik();
    float32x4_t v1 = { 1.0, 2.0, 3.0, 4.0 }, v2 = { 1.0, 1.0, 1.0, 1.0 };
    float32x4_t sum = vaddq_f32(v1, v2);
    std::cout << sum[0] << std::endl;
    Tok("neon native");    

    Tik();
    gray_mla(pa, pb, n);
    showArray(n, pa, pb);
    Tok("mal");


    std::cout << "\nneon hello world\n" << std::endl;
    return 0;
}