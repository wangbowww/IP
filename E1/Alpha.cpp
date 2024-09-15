#include <iostream>
#include <opencv2/opencv.hpp>
#include <GetChannel.h>
void alpha(const cv::Mat &image, const cv::Mat &bg, cv::Mat &result){
    // 获取图像属性
    int width = image.cols;
    int height = image.rows;
    int inChannels = image.channels();
    int inStep = image.step;

    // 创建一个单通道的输出图像
    cv::Mat output(height, width, CV_8UC1);
    uchar* outputData = output.ptr();
    int outStep = output.step;

    // 分离出Alpha通道
    getChannel(image.data, width, height, inStep, inChannels, outputData, outStep, 3);

    //Alpha合成
    uchar *row1 = image.data;
    uchar *row2 = bg.data;
    uchar *row3 = result.data;
    for (int y = 0; y < height; ++y, row1 += image.step, row2 += bg.step, row3 += result.step) {
        uchar *p1 = row1;
        uchar *p2 = row2;
        uchar *p3 = row3;
        for (int x = 0; x < width; ++x) {
            // 获取当前像素的Alpha值
            uchar a = *(output.data + y * outStep + x);
            // 计算混合后的像素值
            for(int i=0;i<3;++i){ // 像素的前三个通道
                *p3 = ((*p1) * a + (*p2) * (255 - a)) / 255;
                p1++, p2++, p3++;
            }
            *p3 = *p1;
            p1++,p3++;
        }
    }
}
int main()
{
    // 加载图像
    cv::Mat image = cv::imread("E1/a1.png", cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cout << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }

    // 检查图像是否为四通道图像
    if (image.channels() != 4) {
        std::cout << "Error: Image is not a 4-channel-image!" << std::endl;
        return -1;
    }

    // 加载背景图
    cv::Mat bg = cv::imread("E1/bg.jpg", cv::IMREAD_COLOR); 
    if (bg.empty()) {
        std::cout << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }
    cv::resize(bg, bg, image.size()); // 大小一致
    cv::Mat result = cv::Mat(image.size(), image.type());

    //  应用Alpha混合
    alpha(image, bg, result);

    // 显示结果图像
    cv::namedWindow("Result Image", cv::WINDOW_NORMAL);
    cv::imshow("Result Image", result);
    cv::waitKey(0);

    return 0;
}