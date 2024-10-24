#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

void myEqualizeHist(Mat &src, Mat &dst){
    int channels = src.channels();
    dst = src.clone();
    int **h = new int *[channels];
    for(int i=0;i<channels;++i){
        h[i] = new int[256];
        memset(h[i], 0, sizeof(int) * 256);
    }
    std::cout << channels;
    uchar *row = src.data;
    for(int y = 0;y < src.rows; ++y, row += src.step){
        for(int x = 0; x < src.cols; ++x){
            for(int i = 0;i < channels; ++i){
                h[i][row[x * channels + i]]++; // 分通道统计颜色数
            }
        }
    }
    for(int i = 0;i < channels; ++i){
        for(int j = 1;j < 256; ++j){
            h[i][j] += h[i][j-1]; // 前缀和
        }
    }
    row = dst.data;
    for (int y = 0; y < src.rows; ++y, row += src.step) {
        for (int x = 0; x < src.cols; ++x) {
            for (int i = 0; i < channels; ++i) {
                float color = 255.0 * h[i][row[channels * x + i]] / (src.rows * src.cols); // p(x) / n * 255   分通道均衡化
                row[channels * x + i] = static_cast<uchar>(color);
            }
        }
    }
}

int main()
{
    Mat img = imread("E1\\test-data\\test-jpg.jpg");
    if (img.empty()) {
        std::cout << "Error loading image" << std::endl;
        return -1;
    }
    Mat ans;
    myEqualizeHist(img, ans);
    namedWindow("Original", WINDOW_NORMAL);
    resizeWindow("Original", 600, 600);
    namedWindow("Equalize hist", WINDOW_NORMAL);
    resizeWindow("Equalize hist", 600, 600);
    imshow("Original", img);
    imshow("Equalize hist", ans);
    waitKey();

    return 0;
}