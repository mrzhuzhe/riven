#include <iostream>
#include "libtest.h"
#include <opencv2/opencv.hpp>

int main(){
    float testval = 9.99f;
    test_lib(testval);
    printf("template go go go \n");  

    int key = 0;
    while(key != 27)
    {
        cv::Mat image(700, 700, CV_32FC3);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        key = cv::waitKey(10);
    }
    return 0;
}