#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

void myDistanceTransform(const Mat &src, Mat &dst){  // 计算每个点到最近的黑色点的最近距离
    dst = Mat::zeros(src.size(), CV_32F);
    dst.setTo(Scalar::all(FLT_MAX)); // 初始化距离为INF

    const uchar* s = src.data;
    float* row = dst.ptr<float>();
    float* lastRow = nullptr;
    // 前向扫描
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (s[x] == 0) {
                row[x] = 0; // 黑色为源点，距离为0
            } else {
                if (lastRow) {
                    row[x] = std::min(row[x], lastRow[x] + 1);
                }
                if (x > 0) {
                    row[x] = std::min(row[x], row[x-1] + 1);
                }
            }
        }
        row += dst.cols;
        s += src.step;
        if (y == 0)
            lastRow = dst.ptr<float>();
        else lastRow += dst.cols;
    }

    row = dst.ptr<float>(dst.rows - 1); // 最后一行开始
    lastRow = nullptr;
    // 逆向扫描
    for (int y = src.rows - 1; y >= 0; y--) {
        for (int x = src.cols - 1; x >= 0; x--) {
            if (lastRow) {
                row[x] = std::min(row[x], lastRow[x] + 1);
            }
            if (x + 1 < src.cols) {
                row[x] = std::min(row[x], row[x + 1] + 1);
            }
        }
        row -= dst.cols;
        if (y + 1 == src.rows)
            lastRow = dst.ptr<float>(dst.rows - 1);
        else
            lastRow -= dst.cols;
    }
}

int main(){
    Mat img = imread("E1\\E6.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Error loading image" << std::endl;
        return -1;
    }
    Mat ans1;
    Mat ans2;

    myDistanceTransform(img, ans1);
    distanceTransform(img, ans2, DIST_L1, 3);

    normalize(ans1, ans1, 0, 255, NORM_MINMAX);
    ans1.convertTo(ans1, CV_8U);
    normalize(ans2, ans2, 0, 255, NORM_MINMAX, CV_8U);  // 归一化

    namedWindow("Original", WINDOW_NORMAL);
    resizeWindow("Original", 600, 600);
    namedWindow("myDT", WINDOW_NORMAL);
    resizeWindow("myDT", 600, 600);
    namedWindow("CV's DT", WINDOW_NORMAL);
    resizeWindow("CV's DT", 600, 600);
    imshow("Original", img);
    imshow("myDT", ans1);
    imshow("CV's DT", ans2);
    waitKey(0);

    return 0;
}