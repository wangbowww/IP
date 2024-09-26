#include <iostream>
#include <opencv2/opencv.hpp>

void transform(const cv::Mat& input, cv::Mat& output, int x, int y, int width, int height)
{
    float cx = width / 2.0; // 中心点
    float cy = height / 2.0;
    float r = 300; // 镜半径
    // 计算原图像位置
    float d = sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
    if(d>r){ // 不在范围内
        output.at<cv::Vec3b>(y,x) = input.at<cv::Vec3b>(y,x);
    }else{
        // 双线性插值
        float ox = (x-cx) * d/r + cx;
        float oy = (y-cy) * d/r + cy;
        float dx = ox - floor(ox);
        float dy = oy - floor(oy);
        for(int i=0;i<3;++i){
            float h1 = input.at<cv::Vec3b>(floor(oy), floor(ox))[i] + dx * (input.at<cv::Vec3b>(floor(oy) + 1, floor(ox))[i] - input.at<cv::Vec3b>(floor(oy), floor(ox))[i]);
            float h2 = input.at<cv::Vec3b>(floor(oy), floor(ox) + 1)[i] + dx * (input.at<cv::Vec3b>(floor(oy) + 1, floor(ox) + 1)[i] - input.at<cv::Vec3b>(floor(oy), floor(ox) + 1)[i]);
            float h = h1 + dy * (h2 - h1);
            output.at<cv::Vec3b>(y, x)[i] = h;
        }
    }
}

int main()
{
    cv::VideoCapture cap(0); // 打开默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "error: fail to open the camera\n";
        return 1;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer;
    writer.open("haha.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(width, height), true);

    cv::Mat frame, output;
    double fps = cap.get(cv::CAP_PROP_FPS);
    double start = (double)cv::getTickCount();
    int frames = 0;

    while (1) {
        cap >> frame; // 获取一帧图像
        if (frame.empty()) {
            break;
        }

        // 输出图像
        output = cv::Mat(frame.size(), frame.type());

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                transform(frame, output, x, y, width, height); // 计算原像素位置并做双线性插值
            }
        }

        frames++;
        double end = (double)cv::getTickCount();
        if ((end - start) / cv::getTickFrequency() > 1) { // 每过一秒计算一次
            fps = frames / ((end - start) / cv::getTickFrequency());
            start = end;
            frames = 0;
        }

        // 显示帧率
        cv::putText(output, cv::format("FPS: %.2f", fps), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        writer << output;

        cv::imshow("output", output);

        if (cv::waitKey(1) >= 0)
            break;
    }

    // 释放资源
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}