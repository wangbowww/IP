#include <GetChannel.h>
#include <opencv.hpp>
int main()
{
    // 加载图像
    cv::Mat image = cv::imread("E1\\test-data\\test-png.png", cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cout << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }

    // 获取图像属性
    int width = image.cols;
    int height = image.rows;
    int inChannels = image.channels();
    int inStep = image.step;

    // 创建一个单通道的输出图像
    cv::Mat output(height, width, CV_8UC1);
    uchar* outputData = output.ptr();
    int outStep = output.step;

    // 分别提取BGR通道
    int channelToGet = 0;
    getChannel(image.data, width, height, inStep, inChannels, outputData, outStep, channelToGet);
    cv::namedWindow("B1", cv::WINDOW_NORMAL);
    cv::imshow("B1", output);

    channelToGet = 1;
    getChannel(image.data, width, height, inStep, inChannels, outputData, outStep, channelToGet);
    cv::namedWindow("G1", cv::WINDOW_NORMAL);
    cv::imshow("G1", output);

    channelToGet = 2;
    getChannel(image.data, width, height, inStep, inChannels, outputData, outStep, channelToGet);
    cv::namedWindow("R1", cv::WINDOW_NORMAL);
    cv::imshow("R1", output);

    // OpenCV方法
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // OpenCV的通道分离结果
    cv::namedWindow("R2", cv::WINDOW_NORMAL);
    cv::imshow("R2", channels[2]);

    cv::namedWindow("G2", cv::WINDOW_NORMAL);
    cv::imshow("G2", channels[1]);

    cv::namedWindow("B2", cv::WINDOW_NORMAL);
    cv::imshow("B2", channels[0]);

    cv::waitKey(0);

    return 0;
}