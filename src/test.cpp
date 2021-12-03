#include "include/run_yolo.h"
#include <iostream>

using namespace std;

int main()
{
    cv::Mat test = cv::imread("hi.png");
    cv::imshow("hi", test);

    return 0;

}
