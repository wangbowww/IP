#include <opencv2/opencv.hpp>
#include <assert.h>
using namespace std;
void meanFilter(const cv::Mat& input, cv::Mat& output, int windowSize)
{
    assert(windowSize % 2 == 1 && windowSize > 0);  // 窗口大小是正奇数

    //创建并建立积分图
    cv::Mat integral(input.size(), CV_32FC3); // 防止溢出
    const float* row1 = input.ptr<float>();
    float* row2 = integral.ptr<float>();
    for (int y = 0; y < input.rows; ++y, row1 += input.step / 4, row2 += integral.step / 4) {
        const float* p1 = row1;
        float* p2 = row2;
        for (int x = 0; x < input.cols; ++x, p1+=3, p2+=3) {
            for (int i = 0; i < 3; ++i) { // 像素的三个通道
                //sum(x,y) = sum(x-1,y) + sum(x,y-1) - sum(x-1,y-1) + a[x][y];
                p2[i] = p1[i];
                if(y>0) p2[i] += (p2 - integral.step / 4)[i];
                if(x>0) p2[i] += (p2 - 3)[i];
                if(x>0 && y>0) p2[i] -= (p2 - integral.step / 4 - 3)[i]; // 每个通道单独算前缀和
            }
        }
    }
    float* row = output.ptr<float>();
    float* p2 = integral.ptr<float>();
    for (int y = 0; y < output.rows; ++y, row += output.step / 4) {
        float* p1 = row;
        for (int x = 0; x < output.cols; ++x, p1 += 3) {
            int x1 = min(output.cols-1, x+windowSize/2), y1 = min(output.rows-1, y+windowSize/2); // 右下角
            int x2 = x-windowSize/2-1, y2 = y-windowSize/2-1; // 左上
            int x3 = x-windowSize/2-1, y3 = y1;// 左下
            int x4 = x1, y4 = y-windowSize/2-1;// 右上
            for (int i = 0; i < 3; ++i) { // 像素的三个通道
                float sum = 0;
                sum += (p2 + integral.step / 4 * y1 + x1 * 3)[i];
                if(x3>=0 && y4>=0) 
                    sum -= (p2 + integral.step / 4 * y3 + x3 * 3)[i];
                if (x4 >= 0 && y4 >= 0)
                    sum -= (p2 + integral.step / 4 * y4 + x4 * 3)[i];
                if (x2 >= 0 && y2 >= 0)
                    sum += (p2 + integral.step / 4 * y2 + x2 * 3)[i];
                p1[i] = sum / (windowSize * windowSize);
            }
        }
    }
}

int main()
{
    // 指定图像文件路径
    const char* imagePath = "E1\\test-data\\test-jpg.jpg";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR); // 读取图像文件
    if (image.empty()) { // 检查是否正确读取
        cerr << "Error: Image cannot be loaded!" << endl;
        return -1;
    }
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    //原始图像
    cv::namedWindow("original image", cv::WINDOW_NORMAL); // 创建一个窗口
    cv::resizeWindow("original image", 600, 600);
    cv::imshow("original image", image); // 在窗口中显示图像

    // 8U3C
    cv::Mat ans = image.clone();
    int windowSize = 21;

    //my filter
    cv::TickMeter tm;
    tm.reset();
    tm.start();
    meanFilter(image, ans, windowSize);
    tm.stop();
    cout << "my filter takes: " << tm.getTimeMilli() << " ms" << endl;

    cv::namedWindow("my filter", cv::WINDOW_NORMAL); // 创建一个窗口
    cv::resizeWindow("my filter", 600, 600);
    cv::imshow("my filter", ans); // 在窗口中显示图像

    //OpenCV's filter
    tm.reset();
    tm.start();
    cv::boxFilter(image, ans, -1, cv::Size(windowSize, windowSize));
    tm.stop();
    cout << "OpenCV's filter takes: " << tm.getTimeMilli() << " ms" << endl;

    cv::namedWindow("OpenCV's filter", cv::WINDOW_NORMAL); // 创建一个窗口
    cv::resizeWindow("OpenCV's filter", 600, 600);
    cv::imshow("OpenCV's filter", ans); // 在窗口中显示图像
    cv::waitKey(0); // 等待按键输入

    return 0;
}