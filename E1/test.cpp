
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap(0);
    Mat img;

    while (1) {
        cap >> img;
        if (img.empty())
            break;
        namedWindow("img", WINDOW_NORMAL);
        imshow("img", img);
        if (27 == waitKey(20))
            break;
    }

    return 0;
}
