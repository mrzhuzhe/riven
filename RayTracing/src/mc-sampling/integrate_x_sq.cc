/*
    One Dimensional MC Integration

    Build command: g++ integrate_x_sq.cc  -o ../outputs/integrate_x_sq
    Run comman: ../outputs/integrate_x_sq

*/

#include "../constant.h"

#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>


inline double pdf(double x) {
    return 0.5*x;
}

inline double pdf2(double x) {
    return 0.5;
}

inline double pdf3(double x) {
    return 3*x*x/8;
}

int main() {
    int N = 1;
    auto sum = 0.0;
    for (int i = 0; i < N; i++) {
        
        /*
        auto x = random_double(0,2);
        sum += x*x;
        */

        /*
        auto x = sqrt(random_double(0,4));
        sum += x*x / pdf(x);
        */

        /*
        auto x = random_double(0,2);
        sum += x*x / pdf2(x);
        */

        auto x = pow(random_double(0,8), 1./3.);
        sum += x*x / pdf3(x);

    }
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "I = " << 2 * sum/N << '\n';
}