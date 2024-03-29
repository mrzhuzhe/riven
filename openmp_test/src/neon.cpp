//  bad case on mac m1 neon cannot be build with cmake but canbe build with clang++ 

#include "iostream"
#include <vector>
#include "TikTok.h"
#include <arm_neon.h>   //  https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/Developing-for-NEON/Intrinsics?lang=en

const int n = 10;

//int a[3*n];
//int b[n];

std::vector<int> a(n*3);
std::vector<int> b(n);

void fillArray(int n, int *src, int *dest){
    for (int i=0; i < n; i++){
        src[i*3] = 0+3*i;
        src[i*3+1] = 1+3*i;
        src[i*3+2] = 2+3*i;
        dest[i] = i;
    }
}

void showArray(int n, int *src, int *dest){
    printf("\n");
    for (int i=0; i < n; i++){
        int r = *src++;
        int g = *src++;
        int b = *src++;
        printf("%d %d %d %d\n", r, g,  b, dest[i]);        
    }
}

void gray_native(int n, int *src, int *dest){
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

void gray_uint8(int n, int *src, int *dest){
    for (int i=0; i < n; i++){
        uint8_t r = *src++;
        uint8_t g = *src++;
        uint8_t b = *src++;

        uint8_t r_ratio = 77;
        uint8_t g_ratio = 151;
        uint8_t b_ratio = 28;

        int temp = r * r_ratio;
        temp += g * g_ratio;
        temp += (b * b_ratio);

        dest[i] = (temp>>8);        
    }
}

/*
void gray_mla(int n, int *src, int *dest){
    n /= 8;
    uint8x8_t r_ratio = vdup_n_u8(77);
    uint8x8_t g_ratio = vdup_n_u8(151);
    uint8x8_t b_ratio = vdup_n_u8(28);

    for (int i=0; i < n; i++){
        uint8x8x3_t rgb = vld3_u8(src);
        uint8x8_t r = rgb.val[0];
        uint8x8_t g = rgb.val[1];
        uint8x8_t b = rgb.val[2];

        uint16x8_t y = vmull_u8(r, r_ratio);
        y = vmlal_u8(y, g, g_ratio);
        y = vmlal_u8(y, b, b_ratio);
        uint8x8_t ret = vshrn_n_u16(y, 8);

        vstl_u8(dest, src);
        src += 3*8;
        dest +=8;
    }
}
*/

int main(){
    //int *pa = a;
    //int *pb = b;
    int *pa = a.data();
    int *pb = b.data();
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
    //gray_mla(n, pa, pb);
    //showArray(n, pa, pb);
    float32x4_t v1 = { 1.0, 2.0, 3.0, 4.0 }, v2 = { 1.0, 1.0, 1.0, 1.0 };
    float32x4_t sum = vaddq_f32(v1, v2);
    // => sum = { 2.0, 3.0, 4.0, 5.0 }
    std::cout << sum[0] << std::endl;
    Tok("mal");


    std::cout << "\nneon hello world\n" << std::endl;
    return 0;
}