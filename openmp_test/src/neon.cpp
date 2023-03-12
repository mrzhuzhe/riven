#include "iostream"

void native(int n, int *src, int *dest){
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

int main(){
    std::cout << "neon hello world" << std::endl;
    return 0;
}