#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    // 指定图像文件路径
    const char* imagePath = "E1\\test-data\\test-png.png";
    //const char* imagePath = "E1\\test-data\\test-jpg.jpg";
    //const char* imagePath = "E1\\test-data\\test-bmp.bmp";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR); // 读取图像文件
    if (image.empty()) { // 检查是否正确读取
        std::cout << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }

    cv::namedWindow("Display window", cv::WINDOW_NORMAL); // 创建一个窗口
    cv::resizeWindow("Display window", 800, 600);
    cv::imshow("Display window", image); // 在窗口中显示图像
    cv::waitKey(0); // 等待按键输入

    return 0;
}