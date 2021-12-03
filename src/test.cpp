#include "include/run_yolo.h"
#include <iostream>

using namespace std;

int main()
{
    cv::VideoCapture capture("rtsp://@192.168.10.17:554/s0");
	if (!capture.isOpened()) 
	{
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    cv::Mat frame;
    
    while(1)
    {
        capture >> frame; 
        cv::imshow("hii",frame);
        cv::waitKey(20);
        double fps = capture.get(cv::CAP_PROP_FPS);
        cout<<fps<<endl;
    }

    return 0;

}
