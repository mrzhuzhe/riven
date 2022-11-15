#include <stdio.h>
#include <math.h>
#include "TutorialConfig.h"


#ifdef USE_MYMATH
#include "MathFunctions.h"

mysqrt mysqrt;
#endif


int main (int argc, char *argv[])
{
    if ( argc < 2 ){
        fprintf(stdout, "%s Version %d.%d\n", argv[0], Tutorial_VERSION_MAJOR, Tutorial_VERSION_MINOR);
        fprintf(stdout, "Usage: %s number\n", argv[0]);
        return 1;
    }

    double inputValue = atof(argv[1]);
    #ifdef USE_MYMATH
        double outputValue = mysqrt.fn1(inputValue);
    #else
        double outputValue = sqrt(inputValue);
    #endif

    fprintf(stdout, "The square root of %g is %g \n", inputValue, outputValue);
    
    return 0;

}